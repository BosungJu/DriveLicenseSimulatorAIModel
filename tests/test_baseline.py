import argparse
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import torch

from src.baseline import SectionLSTM, fit_normalization, load_records, make_batch, predict, train


class BaselineTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        deterministic_algorithms = torch.are_deterministic_algorithms_enabled()
        self.addCleanup(torch.use_deterministic_algorithms, deterministic_algorithms)
        self.schema = {"schema_version": 1, "sample_rate_hz": 50,
                       "feature_names": ["speed", "brake"], "label_names": ["late_stop"]}

    def record(self, run_id="run-1", split="train"):
        return dict(self.schema, run_id=run_id, split=split,
                    features=[[2.0, 0.0], [0.0, 1.0]], labels=[1])

    def write_records(self, records, name="records.jsonl"):
        path = self.root / name
        path.write_text("\n".join(json.dumps(record) for record in records), encoding="utf-8")
        return path

    def load(self, records):
        return load_records(self.write_records(records), self.schema["feature_names"], self.schema["label_names"])

    def test_LoadRecords_ContractViolations_Fail(self):
        invalid_fields = [
            {"features": []}, {"features": [[1.0]]},
            {"features": [[float("nan"), 0.0]]}, {"labels": [0.5]},
            {"labels": [None]}, {"labels": [1, 0]},
            {"labels": [True]}, {"features": [[True, False]]},
            {"feature_names": ["brake", "speed"]}, {"label_names": ["other"]},
            {"run_id": ""}, {"split": "invalid"},
            {"schema_version": 2}, {"sample_rate_hz": 25},
            {"schema_version": True}, {"sample_rate_hz": True},
        ]
        for fields in invalid_fields:
            with self.subTest(fields=fields), self.assertRaises(ValueError):
                self.load([dict(self.record(), **fields)])
        for missing_key in ("features", "labels"):
            record = self.record()
            del record[missing_key]
            with self.subTest(missing_key=missing_key), self.assertRaises(ValueError):
                self.load([record])

    def test_LoadRecords_RunOrDriverCrossesSplits_Fails(self):
        with self.assertRaisesRegex(ValueError, "run_id crosses"):
            self.load([self.record(), self.record(split="val")])
        with self.assertRaisesRegex(ValueError, "driver_id crosses"):
            self.load([dict(self.record(), driver_id="driver-1"),
                       dict(self.record(run_id="run-2", split="val"), driver_id="driver-1")])

    def test_Normalization_UsesTrainingFramesAndConstantFeatureIsFinite(self):
        training = dict(self.record(), features=[[2.0, 1.0], [4.0, 1.0]])
        validation = dict(self.record(run_id="run-2", split="val"), features=[[1000.0, 5.0]])
        records = self.load([training, validation])
        mean, scale = fit_normalization([record for record in records if record["split"] == "train"])
        torch.testing.assert_close(mean, torch.tensor([3.0, 1.0]))
        torch.testing.assert_close(scale, torch.tensor([1.0, 1.0]))
        normalized, _ = make_batch(records[1:], mean, scale)
        self.assertEqual(normalized[0, 0, 1].item(), 4.0)

    def test_LSTM_PaddingAndBatchOrder_DoNotChangePrediction(self):
        torch.manual_seed(3)
        records = self.load([self.record(), dict(self.record(run_id="run-2"), features=[[0.0, 1.0]] * 7)])
        model = SectionLSTM(2, 1, 4).eval()
        mean, scale = torch.zeros(2), torch.ones(2)
        with torch.no_grad():
            alone = model(*make_batch(records[:1], mean, scale))[0]
            batched = model(*make_batch(records, mean, scale))[0]
            reversed_batch = model(*make_batch(records[::-1], mean, scale))[1]
        torch.testing.assert_close(alone, batched)
        torch.testing.assert_close(alone, reversed_batch)

    def test_Training_SyntheticSeparableSamples_ReducesLoss(self):
        records = []
        for split in ("train", "val", "test"):
            for label in (0, 1):
                records.append(dict(self.record(f"{split}-{label}", split),
                                    features=[[-1.0 if label == 0 else 1.0, 0.0]] * 3,
                                    labels=[label]))
        data_path = self.write_records(records)
        schema_path = self.root / "schema.json"
        schema_path.write_text(json.dumps(self.schema), encoding="utf-8")
        output_path = self.root / "model.pt"
        args = argparse.Namespace(data=str(data_path), schema=str(schema_path), output=str(output_path),
                                  seed=7, hidden_size=8, learning_rate=0.03, epochs=30, batch_size=2, command="train", device="cpu")
        with contextlib.redirect_stdout(io.StringIO()):
            train(args)
        checkpoint = torch.load(output_path, weights_only=True)
        self.assertLess(checkpoint["metrics"]["test"]["bce"],
                        checkpoint["prevalence_baseline"]["test"]["bce"] * 0.5)

    def test_TrainAndPredict_CheckpointRoundTrip_ReusesTrainingStatistics(self):
        records = [self.record("train-positive"), dict(self.record("train-negative"), labels=[0]),
                   dict(self.record("validation", "val"), features=[[100.0, 9.0]]),
                   dict(self.record("test", "test"), features=[[200.0, 8.0]])]
        data = self.write_records(records)
        schema_path = self.root / "schema.json"
        schema_path.write_text(json.dumps(self.schema), encoding="utf-8")
        checkpoint_path = self.root / "model.pt"
        args = argparse.Namespace(data=str(data), schema=str(schema_path), output=str(checkpoint_path),
                                  seed=42, hidden_size=4, learning_rate=0.01, epochs=2, batch_size=2, command="train", device="cpu")
        with contextlib.redirect_stdout(io.StringIO()):
            train(args)
        checkpoint = torch.load(checkpoint_path, weights_only=True)
        torch.testing.assert_close(checkpoint["mean"], torch.tensor([1.0, 0.5]))
        torch.testing.assert_close(checkpoint["scale"], torch.tensor([1.0, 0.5]))
        self.assertEqual(checkpoint["metrics"]["test"]["samples"], 1)
        self.assertIn("prevalence_baseline", checkpoint)
        unlabeled = self.record("inference")
        for key in ("split", "labels", "label_names"):
            del unlabeled[key]
        inference_path = self.write_records([unlabeled], "inference.jsonl")
        inference_args = argparse.Namespace(data=str(inference_path), checkpoint=str(checkpoint_path), device="cpu")
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            predict(inference_args)
        result = json.loads(output.getvalue())
        model = SectionLSTM(2, 1, checkpoint["hidden_size"])
        model.load_state_dict(checkpoint["state_dict"])
        with torch.no_grad():
            expected = model(torch.tensor([unlabeled["features"]]).sub(checkpoint["mean"]).div(checkpoint["scale"]), torch.tensor([2])).sigmoid().item()
        self.assertAlmostEqual(result["probabilities"]["late_stop"], expected, places=6)
        invalid_path = self.write_records([dict(unlabeled, feature_names=["brake", "speed"])], "invalid.jsonl")
        inference_args.data = str(invalid_path)
        with self.assertRaisesRegex(ValueError, "feature names/order mismatch"):
            predict(inference_args)


if __name__ == "__main__":
    unittest.main()
