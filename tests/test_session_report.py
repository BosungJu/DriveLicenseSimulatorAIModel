import argparse
import contextlib
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from src import report
from src.baseline import train
from src.session_predict import build_section_records, predict_session, summarize_prediction

LOG_COLUMNS = ["time_s", "point_number", "speed_kmh", "accel_mps2", "yaw_rate_dps"]


def log_row(point_number, speed_kmh, accel_mps2=0.0, yaw_rate_dps=0.0):
    return {"time_s": "0", "point_number": str(point_number), "speed_kmh": str(speed_kmh),
            "accel_mps2": str(accel_mps2), "yaw_rate_dps": str(yaw_rate_dps)}


def section_results(sample_rate_hz=50.0):
    return {
        "score": 90, "maxScore": 100, "passingScore": 80, "totalDeduction": 10,
        "isPassed": True, "isDisqualified": False, "recordedSeconds": 0.08, "sampleRateHz": sample_rate_hz,
        "sections": [
            {"pointNumber": 1, "instructionText": "출발", "commands": ["start"], "sampleCount": 2},
            {"pointNumber": 2, "instructionText": "경사로", "commands": ["hill"], "sampleCount": 2},
            {"pointNumber": 3, "instructionText": "종료", "commands": ["end"], "sampleCount": 0},
        ],
        "deductions": [
            {"pointNumber": 2, "command": "hill", "reason": "HillRollback", "points": 10, "isDisqualification": False},
        ],
    }


class SessionPredictTests(unittest.TestCase):
    #region Section records

    def test_BuildSectionRecords_ConsecutivePointNumbers_SplitsSectionsInCheckpointFeatureOrder(self):
        rows = [log_row(1, 10.0, accel_mps2=1.0), log_row(1, 12.0), log_row(2, 5.0, yaw_rate_dps=3.0)]

        records = build_section_records(rows, ["yaw_rate_dps", "speed_kmh"], 50, "run-1")

        self.assertEqual([record["section_id"] for record in records], [1, 2])
        self.assertEqual(records[0]["features"], [[0.0, 10.0], [0.0, 12.0]])
        self.assertEqual(records[1]["features"], [[3.0, 5.0]])
        self.assertEqual(records[0]["feature_names"], ["yaw_rate_dps", "speed_kmh"])
        self.assertEqual((records[0]["run_id"], records[0]["sample_rate_hz"]), ("run-1", 50))

    def test_BuildSectionRecords_MissingFeatureOrEmptyLog_Fails(self):
        with self.assertRaisesRegex(ValueError, "missing checkpoint features"):
            build_section_records([log_row(1, 10.0)], ["steering_input"], 50, "run-1")
        with self.assertRaisesRegex(ValueError, "no samples"):
            build_section_records([], ["speed_kmh"], 50, "run-1")

    #endregion

    #region Prediction summary

    def test_SummarizePrediction_ProbabilityAtThreshold_IsFlagged(self):
        prediction = {"section_id": 1, "probabilities": {"late_stop": 0.5, "unstable_steering": 0.49}}

        summary = summarize_prediction(prediction, 0.5)

        self.assertEqual(summary["flagged_labels"], ["late_stop"])
        self.assertFalse(summary["is_normal"])

    def test_SummarizePrediction_NoLabelAboveThreshold_IsNormal(self):
        summary = summarize_prediction({"section_id": 1, "probabilities": {"late_stop": 0.1}}, 0.5)

        self.assertEqual(summary["flagged_labels"], [])
        self.assertTrue(summary["is_normal"])

    #endregion

    #region Session prediction with a trained checkpoint

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    def train_checkpoint(self):
        schema = {"schema_version": 1, "sample_rate_hz": 50, "feature_names": ["speed_kmh", "yaw_rate_dps"],
                  "label_names": ["late_stop"]}
        records = [dict(schema, run_id="train-a", split="train", features=[[10.0, 0.0], [0.0, 1.0]], labels=[1]),
                   dict(schema, run_id="train-b", split="train", features=[[20.0, 2.0]], labels=[0]),
                   dict(schema, run_id="val", split="val", features=[[5.0, 0.0]], labels=[1])]
        data_path = self.root / "train.jsonl"
        data_path.write_text("\n".join(json.dumps(record) for record in records), encoding="utf-8")
        schema_path = self.root / "schema.json"
        schema_path.write_text(json.dumps(schema), encoding="utf-8")
        checkpoint_path = self.root / "model.pt"
        args = argparse.Namespace(data=str(data_path), schema=str(schema_path), output=str(checkpoint_path), seed=42,
                                  hidden_size=4, learning_rate=0.01, epochs=1, batch_size=2, command="train", device="cpu")
        with contextlib.redirect_stdout(io.StringIO()):
            train(args)
        return checkpoint_path

    def write_session(self, sample_rate_hz=50.0):
        session_dir = self.root / "20261014_120000"
        session_dir.mkdir()
        with open(session_dir / "driving_log.csv", "w", newline="", encoding="utf-8") as log_file:
            writer = csv.DictWriter(log_file, fieldnames=LOG_COLUMNS)
            writer.writeheader()
            writer.writerows([log_row(1, 0.0), log_row(1, 8.0), log_row(2, 3.0), log_row(2, 0.0)])
        # utf-8-sig keeps older Unity exports (Encoding.UTF8 with a BOM) readable.
        (session_dir / "section_results.json").write_text(json.dumps(section_results(sample_rate_hz)), encoding="utf-8-sig")
        return session_dir

    def test_PredictSession_TrainedCheckpoint_WritesPredictionPerReachedSection(self):
        checkpoint_path = self.train_checkpoint()
        session_dir = self.write_session(sample_rate_hz=49.99999)

        result = predict_session(session_dir, checkpoint_path, device="cpu")

        self.assertEqual([section["section_id"] for section in result["sections"]], [1, 2])
        self.assertEqual(result["run_id"], "20261014_120000")
        self.assertTrue((session_dir / "sections.jsonl").exists())
        saved = json.loads((session_dir / "predictions.json").read_text(encoding="utf-8"))
        self.assertEqual(saved, result)

    def test_PredictSession_SampleRateMismatch_Fails(self):
        checkpoint_path = self.train_checkpoint()
        session_dir = self.write_session(sample_rate_hz=60.0)

        with self.assertRaisesRegex(ValueError, "sample rate"):
            predict_session(session_dir, checkpoint_path, device="cpu")

    def test_PredictSession_InvalidThreshold_Fails(self):
        for threshold in (float("nan"), -0.1, 1.1):
            with self.subTest(threshold=threshold), self.assertRaisesRegex(ValueError, "threshold"):
                predict_session(self.root, self.root / "unused.pt", device="cpu", threshold=threshold)

    #endregion


class ReportTests(unittest.TestCase):
    def predictions(self):
        return {"sections": [
            {"section_id": 1, "probabilities": {"late_stop": 0.1}, "flagged_labels": [], "threshold": 0.5},
            {"section_id": 2, "probabilities": {"late_stop": 0.8}, "flagged_labels": ["late_stop"], "threshold": 0.5},
        ]}

    def test_BuildReportInput_MergesDeductionsAndPredictionsAndSkipsUnreachedSections(self):
        report_input = report.build_report_input(section_results(), self.predictions())

        self.assertEqual([section["point_number"] for section in report_input["sections"]], [1, 2])
        hill_section = report_input["sections"][1]
        self.assertEqual(hill_section["section_deduction"], 10)
        self.assertEqual(hill_section["official_deductions"][0]["description"], "경사로 뒤로 밀림")
        self.assertEqual(hill_section["ai_analysis"]["flagged_labels"], ["late_stop"])
        self.assertEqual(report_input["sections"][0]["official_deductions"], [])
        self.assertTrue(report_input["ai_model_available"])

    def test_BuildReportInput_WithoutPredictions_LeavesAnalysisEmpty(self):
        report_input = report.build_report_input(section_results(), None)

        self.assertFalse(report_input["ai_model_available"])
        self.assertTrue(all(section["ai_analysis"] is None for section in report_input["sections"]))

    def test_GenerateReport_WritesModelInputAndReport(self):
        with tempfile.TemporaryDirectory() as directory:
            session_dir = Path(directory)
            (session_dir / "section_results.json").write_text(json.dumps(section_results()), encoding="utf-8-sig")
            (session_dir / "predictions.json").write_text(json.dumps(self.predictions()), encoding="utf-8")

            with mock.patch.object(report, "request_report", return_value="## 총평") as request_report:
                report_path = report.generate_report(session_dir, model="test-model")

            sent_input, sent_model = request_report.call_args.args
            self.assertEqual(sent_model, "test-model")
            self.assertEqual(report_path.read_text(encoding="utf-8"), "## 총평")
            saved_input = json.loads((session_dir / "report_input.json").read_text(encoding="utf-8"))
            self.assertEqual(saved_input, sent_input)

    def test_GenerateReport_MissingPredictions_FailsUnlessAllowed(self):
        with tempfile.TemporaryDirectory() as directory:
            session_dir = Path(directory)
            (session_dir / "section_results.json").write_text(json.dumps(section_results()), encoding="utf-8-sig")

            with self.assertRaises(FileNotFoundError):
                report.generate_report(session_dir)
            with mock.patch.object(report, "request_report", return_value="## 총평"):
                report.generate_report(session_dir, require_predictions=False)


if __name__ == "__main__":
    unittest.main()
