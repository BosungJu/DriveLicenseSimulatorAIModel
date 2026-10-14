import argparse
import contextlib
import copy
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest

from src.baseline import train
from src.session_predict import build_section_records, predict_session, read_driving_log


CSV_HEADER = "time_s,point_number,speed_kmh,accel_mps2,yaw_rate_dps,heading_deg,pos_x,pos_z,steering_input,throttle_input,brake_applied"
CSV_SAMPLES = (
    "0.000,1,4.000,-0.500,3.000,90.00,10.000,20.000,-0.250,-0.500,1\n"
    "0.020,1,0.000,0.000,0.000,90.00,10.000,20.000,0.000,0.500,0\n"
    "0.040,2,8.000,1.000,-2.000,88.00,10.100,20.100,0.250,1.000,0\n"
)


class UnityInputFeatureTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        schema_path = Path(__file__).resolve().parents[1] / "res" / "schema.example.json"
        self.schema = json.loads(schema_path.read_text(encoding="utf-8"))
        self.rows = list(csv.DictReader(io.StringIO(CSV_HEADER + "\n" + CSV_SAMPLES)))

    def test_UpdatedSchema_ExporterColumns_MapsSixFeaturesInSchemaOrder(self):
        self.assertEqual(self.schema["feature_names"], ["speed_kmh", "accel_mps2", "yaw_rate_dps",
                                                       "steering_input", "throttle_input", "brake_applied"])
        self.assertNotIn("brake_applied", self.schema["continuous_features"])
        records = build_section_records(self.rows, self.schema["feature_names"], 50, "run")
        self.assertEqual(records[0]["features"], [[4.0, -0.5, 3.0, -0.25, -0.5, 1.0],
                                                  [0.0, 0.0, 0.0, 0.0, 0.5, 0.0]])
        self.assertEqual(records[1]["features"], [[8.0, 1.0, -2.0, 0.25, 1.0, 0.0]])

    def test_ControlInput_InvalidRangeOrBinaryBrake_Fails(self):
        for name, value in (("steering_input", "1.1"), ("throttle_input", "-1.1"),
                            ("brake_applied", "0.5"), ("brake_applied", "2")):
            rows = copy.deepcopy(self.rows)
            rows[0][name] = value
            with self.subTest(name=name, value=value), self.assertRaisesRegex(ValueError, name):
                build_section_records(rows, self.schema["feature_names"], 50, "run")

    def test_LegacyCsv_MissingInputColumns_FailsInsteadOfFabricatingInputs(self):
        rows = [{name: value for name, value in row.items() if name not in ("steering_input", "throttle_input", "brake_applied")}
                for row in self.rows]
        with self.assertRaisesRegex(ValueError, "missing checkpoint features"):
            build_section_records(rows, self.schema["feature_names"], 50, "run")
        # Old checkpoints can still select only the old columns.
        records = build_section_records(rows, ["speed_kmh", "yaw_rate_dps"], 50, "run")
        self.assertEqual(records[0]["features"][0], [4.0, 3.0])

    def test_UnavailableControlInput_BlankColumns_PreservesLegacyPhysicsAndRejectsNewModel(self):
        rows = copy.deepcopy(self.rows)
        for name in ("steering_input", "throttle_input", "brake_applied"):
            rows[0][name] = ""
        legacy_records = build_section_records(rows, ["speed_kmh", "yaw_rate_dps"], 50, "run")
        self.assertEqual(legacy_records[0]["features"][0], [4.0, 3.0])
        with self.assertRaisesRegex(ValueError, "Unavailable checkpoint features"):
            build_section_records(rows, self.schema["feature_names"], 50, "run")

    def test_ControlInput_TimeGapOrInvalidTime_Fails(self):
        for sample_time in ("0.060", "0.000", "NaN", ""):
            rows = copy.deepcopy(self.rows)
            rows[1]["time_s"] = sample_time
            with self.subTest(time_s=sample_time), self.assertRaisesRegex(ValueError, "time"):
                build_section_records(rows, self.schema["feature_names"], 50, "run")

    def test_NewExporterCsv_TrainAndSessionPredict_UsesAllSixFeatures(self):
        records = []
        for split in ("train", "val", "test"):
            sections = build_section_records(self.rows, self.schema["feature_names"], 50, split + "-run")
            for record in sections:
                records.append(dict(record, split=split, label_names=self.schema["label_names"],
                                    labels=[1, 0] if record["section_id"] == 1 else [0, 1]))
        training_path = self.root / "training.jsonl"
        training_path.write_text("\n".join(json.dumps(record) for record in records), encoding="utf-8")
        schema_path = self.root / "schema.json"
        schema_path.write_text(json.dumps(self.schema), encoding="utf-8")
        checkpoint = self.root / "model.pt"
        args = argparse.Namespace(data=str(training_path), schema=str(schema_path), output=str(checkpoint),
                                  seed=42, hidden_size=4, epochs=1, batch_size=2, learning_rate=0.01, device="cpu")
        with contextlib.redirect_stdout(io.StringIO()):
            baseline = train(args)
        self.assertEqual(baseline.model.lstm.input_size, 6)
        session = self.root / "session"
        session.mkdir()
        (session / "driving_log.csv").write_text(CSV_HEADER + "\n" + CSV_SAMPLES, encoding="utf-8-sig")
        (session / "section_results.json").write_text(json.dumps({"sampleRateHz": 49.99999}), encoding="utf-8")
        self.assertEqual(len(read_driving_log(session / "driving_log.csv")), 3)
        result = predict_session(session, checkpoint, device="cpu")
        self.assertEqual([section["section_id"] for section in result["sections"]], [1, 2])
        prepared = [json.loads(line) for line in (session / "sections.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertEqual(prepared[0]["feature_names"], self.schema["feature_names"])
        for invalid_rate in (float("nan"), float("inf"), 0, "50", True):
            (session / "section_results.json").write_text(json.dumps({"sampleRateHz": invalid_rate}), encoding="utf-8")
            with self.subTest(sample_rate=invalid_rate), self.assertRaisesRegex(ValueError, "sample rate"):
                predict_session(session, checkpoint, device="cpu")


if __name__ == "__main__":
    unittest.main()
