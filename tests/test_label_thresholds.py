import argparse
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import torch

from src.baseline import DEFAULT_LABEL_THRESHOLD, LSTMBaseline, select_label_thresholds, train
from src.session_predict import summarize_prediction


class SelectLabelThresholdsTests(unittest.TestCase):
    #region Test Helpers

    def select(self, probabilities, actual, judged=None):
        probabilities = torch.tensor(probabilities).unsqueeze(1)
        actual = torch.tensor(actual, dtype=torch.bool).unsqueeze(1)
        judged = torch.ones_like(actual) if judged is None else torch.tensor(judged, dtype=torch.bool).unsqueeze(1)
        return select_label_thresholds(probabilities, actual, judged)[0]

    #endregion

    #region Normal Cases

    def test_Select_SeparableLabel_CutsHalfwayBelowLowestPositive(self):
        threshold = self.select([0.9, 0.7, 0.3, 0.1], [1, 1, 0, 0])

        self.assertAlmostEqual(threshold, 0.5)

    def test_Select_LowProbabilityPositives_LowersThresholdBelowHalf(self):
        threshold = self.select([0.3, 0.25, 0.05, 0.04], [1, 1, 0, 0])

        self.assertAlmostEqual(threshold, 0.15)

    def test_Select_OverlappingLabel_MaximizesF1(self):
        # F1 by lowest flagged probability: 0.9 -> 0.67, 0.7 -> 0.5, 0.6 -> 0.8, 0.5 -> 0.67, 0.1 -> 0.57.
        threshold = self.select([0.9, 0.7, 0.6, 0.5, 0.1], [1, 0, 1, 0, 0])

        self.assertAlmostEqual(threshold, 0.55)

    #endregion

    #region Boundary Cases

    def test_Select_NoPositiveOnlyPositivesOrUnjudged_KeepsDefault(self):
        self.assertEqual(self.select([0.9, 0.1], [0, 0]), DEFAULT_LABEL_THRESHOLD)
        self.assertEqual(self.select([0.9, 0.1], [1, 1]), DEFAULT_LABEL_THRESHOLD)
        self.assertEqual(self.select([0.9, 0.1], [1, 0], judged=[1, 0]), DEFAULT_LABEL_THRESHOLD)

    def test_Select_UnjudgedEntries_AreIgnored(self):
        # The unjudged 0.95 negative would otherwise force a cut above it.
        threshold = self.select([0.95, 0.6, 0.2], [0, 1, 0], judged=[0, 1, 1])

        self.assertAlmostEqual(threshold, 0.4)

    def test_Select_LowestProbabilityIsBestCut_UsesItWithoutMidpoint(self):
        threshold = self.select([0.6, 0.2], [0, 1])

        self.assertAlmostEqual(threshold, 0.2)

    #endregion


class CheckpointThresholdTests(unittest.TestCase):
    #region Setup and Test Helpers

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        deterministic = torch.are_deterministic_algorithms_enabled()
        self.addCleanup(torch.use_deterministic_algorithms, deterministic)

    def train_checkpoint(self):
        schema = {"schema_version": 1, "sample_rate_hz": 50, "feature_names": ["speed"], "label_names": ["late", "fast"]}
        records = []
        for split in ("train", "val", "test"):
            for index, value in enumerate((-2.0, -1.0, 1.0, 2.0)):
                records.append(dict(schema, run_id=f"{split}-{index}", split=split, features=[[value], [value]],
                                    labels=[int(value > 0), 0]))
        data_path = self.root / "data.jsonl"
        data_path.write_text("\n".join(json.dumps(record) for record in records), encoding="utf-8")
        schema_path = self.root / "schema.json"
        schema_path.write_text(json.dumps(schema), encoding="utf-8")
        args = argparse.Namespace(data=str(data_path), schema=str(schema_path), output=str(self.root / "model.pt"),
                                  seed=3, hidden_size=4, learning_rate=0.05, epochs=5, batch_size=4, device="cpu")
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            train(args)
        return args.output, json.loads(output.getvalue())

    #endregion

    def test_Train_SavesValThresholdsAndTunedMetrics(self):
        checkpoint_path, result = self.train_checkpoint()

        self.assertEqual(set(result["label_thresholds"]), {"late", "fast"})
        # "fast" has no val positive, so it keeps the default.
        self.assertEqual(result["label_thresholds"]["fast"], DEFAULT_LABEL_THRESHOLD)
        self.assertEqual(set(result["lstm_tuned_thresholds"]), {"train", "val", "test"})
        loaded = LSTMBaseline.load(checkpoint_path, device="cpu")
        self.assertEqual(loaded.label_thresholds, result["label_thresholds"])

    def test_Load_CheckpointWithoutThresholds_UsesDefaultForEveryLabel(self):
        checkpoint_path, _ = self.train_checkpoint()
        checkpoint = torch.load(checkpoint_path, weights_only=True)
        del checkpoint["label_thresholds"]
        torch.save(checkpoint, checkpoint_path)

        loaded = LSTMBaseline.load(checkpoint_path, device="cpu")

        self.assertEqual(loaded.label_thresholds, {"late": DEFAULT_LABEL_THRESHOLD, "fast": DEFAULT_LABEL_THRESHOLD})


class SummarizePredictionThresholdTests(unittest.TestCase):
    def test_Summarize_PerLabelThresholds_FlagEachLabelAtItsOwnThreshold(self):
        prediction = {"section_id": 1, "probabilities": {"late": 0.3, "fast": 0.3}}

        summary = summarize_prediction(prediction, {"late": 0.2, "fast": 0.5, "unjudged": 0.1})

        self.assertEqual(summary["flagged_labels"], ["late"])
        self.assertEqual(summary["threshold"], {"late": 0.2, "fast": 0.5})
        self.assertFalse(summary["is_normal"])


if __name__ == "__main__":
    unittest.main()
