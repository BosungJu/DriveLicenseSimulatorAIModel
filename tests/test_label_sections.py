import argparse
import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest

import torch

from src.baseline import LSTMBaseline, attach_label_masks, load_records, train, validate_label_sections

# rollback is judged only in hill sections, signal_missing only in turn sections, device_error everywhere.
SCHEMA = {
    "schema_version": 1, "sample_rate_hz": 50,
    "feature_names": ["speed", "section_hill", "section_turn"],
    "continuous_features": ["speed"],
    "label_names": ["rollback", "signal_missing", "device_error"],
    "section_features": ["section_hill", "section_turn"],
    "label_sections": {"rollback": ["section_hill"], "signal_missing": ["section_turn"], "device_error": "all"},
}
HILL = [1.0, 0.0]
TURN = [0.0, 1.0]


class LabelSectionTests(unittest.TestCase):
    #region Setup and Test Helpers

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        deterministic_algorithms = torch.are_deterministic_algorithms_enabled()
        self.addCleanup(torch.use_deterministic_algorithms, deterministic_algorithms)

    def record(self, run_id, split, section, labels, speed=1.0):
        return {"schema_version": 1, "sample_rate_hz": 50, "run_id": run_id, "split": split,
                "feature_names": SCHEMA["feature_names"], "label_names": SCHEMA["label_names"],
                "features": [[speed] + section, [speed * 2] + section], "labels": labels}

    def write_json(self, name, content):
        path = self.root / name
        path.write_text(content if isinstance(content, str) else json.dumps(content), encoding="utf-8")
        return path

    def training_records(self, masked_label_value=0):
        """Hill and turn sections per split; masked_label_value fills the labels their section never judges."""
        records = []
        for split in ("train", "val", "test"):
            for index, positive in enumerate((0, 1)):
                records.append(self.record(f"{split}-hill-{index}", split, HILL, [positive, masked_label_value, 0],
                                           speed=-1.0 if positive else 1.0))
                records.append(self.record(f"{split}-turn-{index}", split, TURN, [masked_label_value, positive, 0],
                                           speed=-1.0 if positive else 1.0))
        return records

    def train_model(self, records, name):
        data_path = self.write_json(f"{name}.jsonl", "\n".join(json.dumps(record) for record in records))
        schema_path = self.write_json("schema.json", SCHEMA)
        args = argparse.Namespace(data=str(data_path), schema=str(schema_path), output=str(self.root / f"{name}.pt"),
                                  seed=5, hidden_size=4, learning_rate=0.01, epochs=3, batch_size=2, device="cpu")
        with contextlib.redirect_stdout(io.StringIO()):
            return train(args)

    #endregion

    #region Schema validation

    def test_ValidateLabelSections_WithoutKey_JudgesEveryLabelEverywhere(self):
        schema = {key: value for key, value in SCHEMA.items() if key not in ("section_features", "label_sections")}

        self.assertEqual(validate_label_sections(schema), (None, None))

    def test_ValidateLabelSections_ValidSchema_ReturnsSectionFeatureIndices(self):
        section_features, label_section_indices = validate_label_sections(SCHEMA)

        self.assertEqual(section_features, ["section_hill", "section_turn"])
        self.assertEqual(label_section_indices, [[1], [2], None])

    def test_ValidateLabelSections_InvalidSchema_Fails(self):
        invalid_changes = [
            {"label_sections": {"rollback": ["section_hill"], "signal_missing": ["section_turn"]}},
            {"label_sections": dict(SCHEMA["label_sections"], rollback=[])},
            {"label_sections": dict(SCHEMA["label_sections"], rollback="hill")},
            {"label_sections": dict(SCHEMA["label_sections"], rollback=["section_hill", "section_hill"])},
            {"label_sections": dict(SCHEMA["label_sections"], rollback=["speed"])},
            {"section_features": ["section_hill", "unknown"]},
            {"section_features": ["speed", "section_hill", "section_turn"]},
            {"section_features": None},
        ]
        for changes in invalid_changes:
            schema = dict(copy.deepcopy(SCHEMA), **changes)
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                validate_label_sections(schema)
        schema = {key: value for key, value in SCHEMA.items() if key != "label_sections"}
        with self.assertRaisesRegex(ValueError, "section_features requires label_sections"):
            validate_label_sections(schema)

    #endregion

    #region Label masks

    def test_AttachLabelMasks_SectionKinds_JudgeOnlyTheirLabels(self):
        records = [self.record("hill", "train", HILL, [0, 0, 0]), self.record("turn", "train", TURN, [0, 0, 0]),
                   self.record("plain", "train", [0.0, 0.0], [0, 0, 0])]
        data = "\n".join(json.dumps(record) for record in records).encode("utf-8")
        loaded = load_records(None, SCHEMA["feature_names"], SCHEMA["label_names"], data_bytes=data)

        attach_label_masks(loaded, SCHEMA)

        self.assertEqual([record["label_mask"].tolist() for record in loaded],
                         [[1.0, 0.0, 1.0], [0.0, 1.0, 1.0], [0.0, 0.0, 1.0]])

    def test_AttachLabelMasks_UnjudgedLabels_MaskOnlyThatRecord(self):
        unjudged_hill = dict(self.record("hill-after", "train", HILL, [0, 0, 0]),
                             unjudged_labels=["rollback", "device_error"])
        records = [unjudged_hill, self.record("hill", "train", HILL, [0, 0, 0])]
        data = "\n".join(json.dumps(record) for record in records).encode("utf-8")
        loaded = load_records(None, SCHEMA["feature_names"], SCHEMA["label_names"], data_bytes=data)

        attach_label_masks(loaded, SCHEMA)

        self.assertEqual([record["label_mask"].tolist() for record in loaded], [[0.0, 0.0, 0.0], [1.0, 0.0, 1.0]])

    def test_LoadRecords_InvalidUnjudgedLabels_Fail(self):
        for unjudged_labels, labels in ((["unknown"], [0, 0, 0]), (["rollback", "rollback"], [0, 0, 0]),
                                        ("rollback", [0, 0, 0]), ([["rollback"]], [0, 0, 0]),
                                        (["rollback"], [1, 0, 0])):
            record = dict(self.record("hill", "train", HILL, labels), unjudged_labels=unjudged_labels)
            with self.subTest(unjudged_labels=unjudged_labels, labels=labels), self.assertRaises(ValueError):
                load_records(None, SCHEMA["feature_names"], SCHEMA["label_names"],
                             data_bytes=json.dumps(record).encode("utf-8"))

    #endregion

    #region Training, evaluation and prediction

    def test_Fit_LogsStartAndEveryEpochWithEta(self):
        with self.assertLogs("drive_lstm", level="INFO") as logs:
            self.train_model(self.training_records(), "progress")

        messages = [record.getMessage() for record in logs.records]
        epoch_messages = [message for message in messages if message.startswith("epoch ")]
        self.assertIn("4 train sections x 1 = 4 samples, 2 batches/epoch, 3 epochs, 4 val sections", messages[0])
        self.assertEqual([message.split()[1] for message in epoch_messages], ["1/3", "2/3", "3/3"])
        self.assertTrue(all("val_bce" in message and "eta" in message for message in epoch_messages))
        self.assertIn("*best", epoch_messages[0])
        self.assertTrue(epoch_messages[-1].endswith("eta 0m00s"))
        self.assertTrue(messages[-1].startswith("Label thresholds chosen on val: rollback"))

    def test_Fit_UnjudgedLabelValue_DoesNotChangeTraining(self):
        zero_filled = self.train_model(self.training_records(masked_label_value=0), "zero")
        one_filled = self.train_model(self.training_records(masked_label_value=1), "one")

        for key, value in zero_filled.model.state_dict().items():
            torch.testing.assert_close(value, one_filled.model.state_dict()[key])

    def test_Fit_SplitWithoutJudgedLabels_Fails(self):
        # Without device_error ("all"), plain sections judge no label at all.
        schema = dict(SCHEMA, label_sections=dict(SCHEMA["label_sections"], device_error=["section_hill"]))
        self.write_json("schema.json", schema)
        for unjudged_split in ("train", "val"):
            records = [self.record(f"{split}-{index}", split, HILL if split != unjudged_split else [0.0, 0.0], [0, 0, 0])
                       for split in ("train", "val") for index in range(2)]
            data_path = self.write_json("records.jsonl", "\n".join(json.dumps(record) for record in records))
            args = argparse.Namespace(data=str(data_path), schema=str(self.root / "schema.json"),
                                      output=str(self.root / "model.pt"), seed=5, hidden_size=4, learning_rate=0.01,
                                      epochs=1, batch_size=2, device="cpu")
            with self.subTest(split=unjudged_split), self.assertRaisesRegex(ValueError, f"any {unjudged_split} section"):
                train(args)

    def test_Evaluate_LabelSections_ReportsJudgedCountsAndPerSectionMetrics(self):
        self.train_model(self.training_records(), "model")

        test_metrics = LSTMBaseline.load(self.root / "model.pt", device="cpu").evaluate(self.root / "model.jsonl")
        # Three splits of two hill and two turn sections; device_error is judged in all twelve.
        self.assertEqual([label["judged_count"] for label in test_metrics["per_label"]], [6, 6, 12])
        self.assertEqual(sorted(test_metrics["per_section"]), ["section_hill", "section_turn"])
        hill_metrics = test_metrics["per_section"]["section_hill"]
        self.assertEqual(hill_metrics["samples"], 6)
        self.assertEqual([label["judged_count"] for label in hill_metrics["per_label"]], [6, 0, 6])

    def test_Predict_LabelSections_ReturnsOnlyJudgedLabels(self):
        self.train_model(self.training_records(), "model")
        unlabeled = []
        for run_id, section in (("hill", HILL), ("turn", TURN)):
            record = self.record(run_id, "train", section, [0, 0, 0])
            for key in ("split", "labels", "label_names"):
                del record[key]
            unlabeled.append(record)
        inference_path = self.write_json("inference.jsonl", "\n".join(json.dumps(record) for record in unlabeled))

        predictions = LSTMBaseline.load(self.root / "model.pt", device="cpu").predict(inference_path)

        self.assertEqual([sorted(prediction["probabilities"]) for prediction in predictions],
                         [["device_error", "rollback"], ["device_error", "signal_missing"]])

    #endregion


if __name__ == "__main__":
    unittest.main()
