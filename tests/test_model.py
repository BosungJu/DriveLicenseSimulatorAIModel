import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

from src.baseline import LSTMBaseline, parse_args, resolve_device, training_runtime
from src.model import SectionLSTM


class DeviceTests(unittest.TestCase):
    def test_ResolveDevice_AutoWithoutCuda_UsesCpu(self):
        with patch("torch.cuda.is_available", return_value=False):
            self.assertEqual(resolve_device("auto"), torch.device("cpu"))
            with self.assertRaisesRegex(ValueError, "CUDA requested but unavailable"):
                resolve_device("cuda")

    def test_ResolveDevice_AutoAndIndexedCuda_ChooseAvailableDevice(self):
        with patch.dict(os.environ), patch("torch.cuda.is_available", return_value=True), patch("torch.cuda.device_count", return_value=2):
            self.assertEqual(resolve_device("auto"), torch.device("cuda:0"))
            self.assertEqual(resolve_device("cuda:1"), torch.device("cuda:1"))
            with self.assertRaisesRegex(ValueError, "device 2 unavailable"):
                resolve_device("cuda:2")

    def test_ResolveDevice_InvalidNames_Fail(self):
        for name in ("gpu", "mps", "cuda:-1", "cuda:", "cuda:1.5", True):
            with self.subTest(name=name), self.assertRaises(ValueError):
                resolve_device(name)

    def test_TrainingRuntime_Exception_RestoresDeterminismSettings(self):
        expected = (torch.are_deterministic_algorithms_enabled(),
                    torch.is_deterministic_algorithms_warn_only_enabled(),
                    torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic)
        with self.assertRaisesRegex(ValueError, "training failed"):
            with training_runtime(42, not expected[0]):
                raise ValueError("training failed")
        actual = (torch.are_deterministic_algorithms_enabled(),
                  torch.is_deterministic_algorithms_warn_only_enabled(),
                  torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic)
        self.assertEqual(expected, actual)

    def test_ParseArgs_DeviceAndDeterministicCli_OverrideConfig(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "settings.json"
            path.write_text(json.dumps({"data": "data", "schema": "schema", "output": "model",
                                        "device": "cuda", "deterministic": True, "num_layers": 2}), encoding="utf-8")
            configured = parse_args(["train", "--config", str(path)])
            self.assertEqual(configured.num_layers, 2)
            args = parse_args(["train", "--config", str(path), "--device", "cpu", "--no-deterministic", "--num-layers", "3"])
            self.assertEqual(args.device, "cpu")
            self.assertFalse(args.deterministic)
            self.assertEqual(args.num_layers, 3)


class LSTMBaselineTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.schema = {"schema_version": 1, "sample_rate_hz": 50,
                       "feature_names": ["speed", "brake"], "continuous_features": ["speed"],
                       "label_names": ["improvement"]}
        self.schema_path = self.root / "schema.json"
        self.schema_path.write_text(json.dumps(self.schema), encoding="utf-8")
        records = []
        for split in ("train", "val", "test"):
            for label in (0, 1):
                records.append(dict(self.schema, split=split, run_id=f"{split}-{label}",
                                    features=[[-1 if label == 0 else 1, float(index % 2)] for index in range(2 + label)],
                                    labels=[label]))
        self.data_path = self.root / "data.jsonl"
        self.data_path.write_text("\n".join(json.dumps(record) for record in records), encoding="utf-8")

    def settings(self, device="cpu"):
        return argparse.Namespace(data=str(self.data_path), schema=str(self.schema_path), output=str(self.root / "model.pt"),
                                  seed=42, hidden_size=4, learning_rate=0.01, epochs=2, batch_size=2,
                                  device=device, deterministic=True, command="train",
                                  augmentation={"probability": 1, "features": {"speed": {
                                      "bias_max": 0.1, "noise_max": 0.02, "smoothing": 0.1,
                                      "min": -10, "max": 10, "protected_values": [0]}}})

    def test_UnfittedBaseline_OperationsFailClearly(self):
        baseline = LSTMBaseline("cpu")
        self.assertIsNone(baseline.model)
        for operation in (lambda: baseline.save(self.root / "missing.pt"),
                          lambda: baseline.predict(self.data_path), lambda: baseline.evaluate(self.data_path)):
            with self.assertRaisesRegex(ValueError, "Fit or load"):
                operation()

    def test_FitSaveLoadCpu_ClassApiPreservesPredictionsAndLegacyCheckpoint(self):
        args = self.settings()
        baseline = LSTMBaseline("cpu")
        result = baseline.fit(args)
        self.assertIsInstance(baseline.model, SectionLSTM)
        self.assertEqual(result["device"], "cpu")
        self.assertEqual(baseline.evaluate(self.data_path)["samples"], 6)
        predictions = baseline.predict(self.data_path)
        legacy = torch.load(args.output, weights_only=True)
        self.assertEqual(baseline.model.lstm.num_layers, 1)
        for key in ("device", "cuda_version", "num_layers"):
            del legacy[key]
        legacy_path = self.root / "legacy.pt"
        torch.save(legacy, legacy_path)
        loaded = LSTMBaseline.load(legacy_path, "cpu")
        self.assertEqual(predictions, loaded.predict(self.data_path))
        copy_path = self.root / "copied.pt"
        loaded.save(copy_path)
        self.assertEqual(predictions, LSTMBaseline.load(copy_path, "cpu").predict(self.data_path))

    def test_ClassApi_InvalidEditedSettings_FailBeforeTraining(self):
        baseline = LSTMBaseline("cpu")
        for name, value in (("epochs", 0), ("batch_size", True), ("hidden_size", -1),
                            ("learning_rate", float("nan")), ("deterministic", "true"),
                            ("num_layers", 0), ("num_layers", True), ("num_layers", 1.5)):
            args = self.settings()
            setattr(args, name, value)
            with self.subTest(name=name), self.assertRaises(ValueError):
                baseline.fit(args)
            self.assertIsNone(baseline.model)

    def test_ClassApi_ConstructorDevice_IsRecordedDespiteSettingsDevice(self):
        baseline = LSTMBaseline("cpu")
        args = self.settings("cuda")
        baseline.fit(args)
        checkpoint = torch.load(args.output, weights_only=True)
        self.assertEqual(checkpoint["training_args"]["device"], "cpu")
        with self.assertRaisesRegex(ValueError, "batch_size"):
            baseline.evaluate(self.data_path, batch_size=0)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA-enabled PyTorch and NVIDIA GPU required")
    def test_CudaCli_ConfigTrainAndCpuPredict_UsesSelectedDevices(self):
        config_path = self.root / "config.json"
        settings = vars(self.settings("auto"))
        settings.pop("command")
        config_path.write_text(json.dumps(settings), encoding="utf-8")
        entry_path = Path(__file__).resolve().parents[1] / "src" / "baseline.py"
        training = subprocess.run([sys.executable, str(entry_path), "train", "--config", str(config_path)],
                                  check=True, capture_output=True, text=True, timeout=60)
        self.assertEqual(json.loads(training.stdout)["device"], "cuda:0")
        inference = subprocess.run([sys.executable, str(entry_path), "predict", "--data", str(self.data_path),
                                   "--checkpoint", settings["output"], "--device", "cpu"],
                                  check=True, capture_output=True, text=True, timeout=60)
        predictions = [json.loads(line) for line in inference.stdout.splitlines()]
        self.assertEqual(len(predictions), 6)
        self.assertTrue(all(0 <= prediction["probabilities"]["improvement"] <= 1 for prediction in predictions))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA-enabled PyTorch and NVIDIA GPU required")
    def test_CudaTrainingEvaluationPrediction_UsesGpuAndLoadsCheckpointOnCpu(self):
        args = self.settings("cuda:0")
        args.num_layers = 2
        baseline = LSTMBaseline("cuda:0")
        result = baseline.fit(args)
        self.assertEqual(result["device"], "cuda:0")
        self.assertEqual(next(baseline.model.parameters()).device.type, "cuda")
        self.assertEqual(baseline.evaluate(self.data_path)["samples"], 6)
        checkpoint = torch.load(args.output, weights_only=True)
        self.assertEqual(checkpoint["num_layers"], 2)
        self.assertEqual(baseline.model.lstm.num_layers, 2)
        self.assertTrue(all(value.device.type == "cpu" for value in checkpoint["state_dict"].values()))
        self.assertEqual(checkpoint["mean"].device.type, "cpu")
        gpu_predictions = baseline.predict(self.data_path)
        cpu_predictions = LSTMBaseline.load(args.output, "cpu").predict(self.data_path)
        for gpu, cpu in zip(gpu_predictions, cpu_predictions):
            self.assertAlmostEqual(gpu["probabilities"]["improvement"], cpu["probabilities"]["improvement"], places=5)
        first_state = checkpoint["state_dict"]
        baseline.fit(args)
        second_state = torch.load(args.output, weights_only=True)["state_dict"]
        for key in first_state:
            torch.testing.assert_close(first_state[key], second_state[key], rtol=0, atol=0)
        reloaded_gpu = LSTMBaseline.load(args.output, "cuda:0")
        self.assertEqual(gpu_predictions, reloaded_gpu.predict(self.data_path))

    def test_TwoLayerCpuModel_CheckpointRestoresArchitectureAndPredictions(self):
        args = self.settings()
        args.num_layers = 2
        baseline = LSTMBaseline("cpu")
        baseline.fit(args)
        checkpoint = torch.load(args.output, weights_only=True)
        self.assertEqual(checkpoint["num_layers"], 2)
        self.assertEqual(checkpoint["training_args"]["num_layers"], 2)
        self.assertIn("lstm.weight_ih_l1", checkpoint["state_dict"])
        loaded = LSTMBaseline.load(args.output, "cpu")
        self.assertEqual(loaded.model.lstm.num_layers, 2)
        self.assertEqual(baseline.predict(self.data_path), loaded.predict(self.data_path))

    def test_SectionLSTM_InvalidLayerCounts_Fail(self):
        for num_layers in (0, -1, True, 1.5):
            with self.subTest(num_layers=num_layers), self.assertRaisesRegex(ValueError, "num_layers"):
                SectionLSTM(2, 1, num_layers=num_layers)


if __name__ == "__main__":
    unittest.main()
