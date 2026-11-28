import argparse
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

from src.baseline import LSTMBaseline, iterate_prefetched, parse_args, resolve_device, training_runtime
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

    #region Regularization and pooling

    def test_SectionLSTM_InvalidDropoutOrPooling_Fail(self):
        for dropout in (-0.1, 1.0, float("nan"), True):
            with self.subTest(dropout=dropout), self.assertRaisesRegex(ValueError, "dropout"):
                SectionLSTM(2, 1, dropout=dropout)
        with self.assertRaisesRegex(ValueError, "pooling"):
            SectionLSTM(2, 1, pooling="average")

    def test_SectionLSTM_DefaultSettings_KeepLegacyStateDictLayout(self):
        model = SectionLSTM(2, 3, hidden_size=4)

        self.assertEqual(set(model.state_dict()), {"lstm.weight_ih_l0", "lstm.weight_hh_l0", "lstm.bias_ih_l0",
                                                   "lstm.bias_hh_l0", "classifier.weight", "classifier.bias"})
        self.assertEqual(tuple(model.classifier.weight.shape), (3, 4))

    def test_SectionLSTM_MeanMaxPooling_IgnoresPaddingOfShorterSequence(self):
        torch.manual_seed(0)
        model = SectionLSTM(2, 1, hidden_size=4, pooling="mean_max").eval()
        short = torch.randn(1, 3, 2)
        padded = torch.cat([short, torch.full((1, 4, 2), 50.0)], dim=1)
        longer = torch.randn(1, 7, 2)

        alone = model(short, torch.tensor([3]))
        in_batch = model(torch.cat([padded, longer]), torch.tensor([3, 7]))

        self.assertEqual(tuple(model.classifier.weight.shape), (1, 8))
        torch.testing.assert_close(in_batch[:1], alone)

    def test_ParseArgs_RegularizationSettings_ConfigAndCliOverride(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "settings.json"
            path.write_text(json.dumps({"data": "data", "schema": "schema", "output": "model",
                                        "dropout": 0.2, "weight_decay": 0.001, "pooling": "mean_max"}), encoding="utf-8")
            configured = parse_args(["train", "--config", str(path)])
            overridden = parse_args(["train", "--config", str(path), "--dropout", "0", "--pooling", "last"])
            defaults = parse_args(["train", "--data", "data", "--schema", "schema", "--output", "model"])

        self.assertEqual((configured.dropout, configured.weight_decay, configured.pooling), (0.2, 0.001, "mean_max"))
        self.assertEqual((overridden.dropout, overridden.weight_decay, overridden.pooling), (0.0, 0.001, "last"))
        self.assertEqual((defaults.dropout, defaults.weight_decay, defaults.pooling), (0.0, 0.0, "last"))

    def test_ParseArgs_InvalidRegularizationSettings_Fail(self):
        for option, value in (("--dropout", "1"), ("--dropout", "-0.5"), ("--weight-decay", "-1"), ("--weight-decay", "nan")):
            with self.subTest(option=option, value=value), self.assertRaises(SystemExit):
                with patch("sys.stderr"):
                    parse_args(["train", "--data", "data", "--schema", "schema", "--output", "model", option, value])

    def test_RegularizedMeanMaxModel_CheckpointRestoresArchitectureAndPredictions(self):
        args = self.settings()
        args.dropout = 0.3
        args.weight_decay = 0.01
        args.pooling = "mean_max"
        baseline = LSTMBaseline("cpu")

        baseline.fit(args)

        checkpoint = torch.load(args.output, weights_only=True)
        self.assertEqual((checkpoint["dropout"], checkpoint["pooling"]), (0.3, "mean_max"))
        self.assertEqual(checkpoint["training_args"]["weight_decay"], 0.01)
        loaded = LSTMBaseline.load(args.output, "cpu")
        self.assertEqual(loaded.model.pooling, "mean_max")
        self.assertEqual(baseline.predict(self.data_path), loaded.predict(self.data_path))

    def test_ClassApi_InvalidRegularizationSettings_FailBeforeTraining(self):
        baseline = LSTMBaseline("cpu")
        for name, value in (("dropout", 1.0), ("weight_decay", -0.1), ("pooling", "average")):
            args = self.settings()
            setattr(args, name, value)
            with self.subTest(name=name), self.assertRaises(ValueError):
                baseline.fit(args)
            self.assertIsNone(baseline.model)

    #endregion

    #region Regularization and pooling

    def test_SectionLSTM_MeanMaxPooling_IgnoresPadding(self):
        torch.manual_seed(0)
        model = SectionLSTM(2, 3, hidden_size=4, pooling="mean_max").eval()
        short_sequence = torch.randn(1, 3, 2)
        padded_batch = torch.cat([torch.cat([short_sequence, torch.full((1, 2, 2), 99.0)], 1), torch.randn(1, 5, 2)])

        alone = model(short_sequence, torch.tensor([3]))
        batched = model(padded_batch, torch.tensor([3, 5]))

        torch.testing.assert_close(batched[:1], alone)

    def test_SectionLSTM_MeanMaxPooling_MatchesManualMeanAndMax(self):
        torch.manual_seed(0)
        model = SectionLSTM(2, 1, hidden_size=4, pooling="mean_max").eval()
        sequence = torch.randn(1, 4, 2)

        logits = model(sequence, torch.tensor([4]))

        outputs, _ = model.lstm(sequence)
        readout = torch.cat([outputs.mean(1), outputs.amax(1)], dim=1)
        torch.testing.assert_close(logits, model.classifier(readout))
        self.assertEqual(model.classifier.in_features, 8)

    def test_SectionLSTM_Dropout_ActiveOnlyInTraining(self):
        model = SectionLSTM(2, 1, hidden_size=4, dropout=0.5)
        sequence, lengths = torch.randn(2, 3, 2), torch.tensor([3, 2])

        model.eval()
        self.assertTrue(torch.equal(model(sequence, lengths), model(sequence, lengths)))
        model.train()
        torch.manual_seed(1)
        first = model(sequence, lengths)
        torch.manual_seed(2)
        self.assertFalse(torch.equal(first, model(sequence, lengths)))

    def test_SectionLSTM_DefaultSettings_KeepLegacyStateDictKeys(self):
        legacy_keys = {"lstm.weight_ih_l0", "lstm.weight_hh_l0", "lstm.bias_ih_l0", "lstm.bias_hh_l0",
                       "classifier.weight", "classifier.bias"}

        self.assertEqual(set(SectionLSTM(2, 1).state_dict()), legacy_keys)
        self.assertEqual(set(SectionLSTM(2, 1, dropout=0.3).state_dict()), legacy_keys)

    def test_SectionLSTM_InvalidDropoutOrPooling_Fail(self):
        for dropout in (-0.1, 1.0, float("nan"), "0.1"):
            with self.subTest(dropout=dropout), self.assertRaisesRegex(ValueError, "dropout"):
                SectionLSTM(2, 1, dropout=dropout)
        with self.assertRaisesRegex(ValueError, "pooling"):
            SectionLSTM(2, 1, pooling="attention")

    def test_FitSaveLoad_DropoutWeightDecayMeanMax_CheckpointRestoresArchitecture(self):
        args = self.settings()
        args.dropout, args.weight_decay, args.pooling = 0.2, 0.01, "mean_max"
        baseline = LSTMBaseline("cpu")

        baseline.fit(args)

        checkpoint = torch.load(args.output, weights_only=True)
        self.assertEqual((checkpoint["dropout"], checkpoint["pooling"]), (0.2, "mean_max"))
        self.assertEqual(checkpoint["training_args"]["weight_decay"], 0.01)
        self.assertEqual(checkpoint["state_dict"]["classifier.weight"].shape[1], 8)
        loaded = LSTMBaseline.load(args.output, "cpu")
        self.assertEqual(loaded.model.pooling, "mean_max")
        self.assertEqual(baseline.predict(self.data_path), loaded.predict(self.data_path))

    def test_Fit_InvalidRegularizationSettings_FailBeforeTraining(self):
        baseline = LSTMBaseline("cpu")
        for name, value in (("dropout", 1.0), ("weight_decay", -0.1), ("weight_decay", float("inf")), ("pooling", "first")):
            args = self.settings()
            setattr(args, name, value)
            with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                baseline.fit(args)
            self.assertIsNone(baseline.model)

    def test_Fit_LogEveryBatches_LogsRunningLossExceptAtEpochEnd(self):
        args = self.settings()
        args.batch_size, args.log_every_batches, args.augmentation = 1, 1, None

        with self.assertLogs("drive_lstm", level="INFO") as logs:
            LSTMBaseline("cpu").fit(args)

        batch_messages = [record.getMessage() for record in logs.records if record.getMessage().startswith("  batch ")]
        self.assertEqual([message.split()[1:4] for message in batch_messages],
                         [["1/2", "(epoch", "1/2)"], ["1/2", "(epoch", "2/2)"]])
        self.assertTrue(all("train_bce" in message and "last 1 batches" in message for message in batch_messages))

    def test_Fit_LogEveryBatchesZero_LogsOnlyEpochs(self):
        args = self.settings()
        args.batch_size, args.log_every_batches = 1, 0

        with self.assertLogs("drive_lstm", level="INFO") as logs:
            LSTMBaseline("cpu").fit(args)

        self.assertFalse(any(record.getMessage().startswith("  batch ") for record in logs.records))

    def test_ParseArgs_RegularizationSettings_DefaultConfigAndCliOverride(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "settings.json"
            path.write_text(json.dumps({"data": "data", "schema": "schema", "output": "model",
                                        "dropout": 0.3, "weight_decay": 0.001, "pooling": "mean_max"}), encoding="utf-8")

            defaults = parse_args(["train", "--data", "d", "--schema", "s", "--output", "o"])
            configured = parse_args(["train", "--config", str(path)])
            overridden = parse_args(["train", "--config", str(path), "--dropout", "0", "--weight-decay", "0", "--pooling", "last"])

            self.assertEqual((defaults.dropout, defaults.weight_decay, defaults.pooling), (0.0, 0.0, "last"))
            self.assertEqual((configured.dropout, configured.weight_decay, configured.pooling), (0.3, 0.001, "mean_max"))
            self.assertEqual((overridden.dropout, overridden.weight_decay, overridden.pooling), (0.0, 0.0, "last"))
            self.assertEqual(defaults.log_every_batches, 20)
            self.assertEqual(parse_args(["train", "--config", str(path), "--log-every-batches", "0"]).log_every_batches, 0)
            for option in (["--dropout", "1"], ["--weight-decay", "-1"], ["--log-every-batches", "-1"]):
                with self.subTest(option=option), self.assertRaises(SystemExit), \
                        patch("sys.stderr", new_callable=io.StringIO):
                    parse_args(["train", "--config", str(path), *option])

    #endregion


class PrefetchTests(unittest.TestCase):
    def test_IteratePrefetched_SharedGenerator_MatchesPlainLoopOrder(self):
        def draw(generator, size):
            return torch.rand(size, generator=generator)

        plain_generator = torch.Generator().manual_seed(7)
        expected = [draw(plain_generator, size) for size in (3, 1, 4, 2, 5)]

        prefetch_generator = torch.Generator().manual_seed(7)
        actual = list(iterate_prefetched(lambda size: draw(prefetch_generator, size), [3, 1, 4, 2, 5], 2))

        self.assertEqual(len(actual), len(expected))
        for expected_item, actual_item in zip(expected, actual):
            self.assertTrue(torch.equal(expected_item, actual_item))

    def test_IteratePrefetched_WorkerError_RaisedAtItsTurn(self):
        def make_item(value):
            if value == 2:
                raise ValueError("bad item")
            return value

        received = []
        with self.assertRaisesRegex(ValueError, "bad item"):
            for item in iterate_prefetched(make_item, [0, 1, 2, 3], 2):
                received.append(item)

        self.assertEqual(received, [0, 1])

    def test_IteratePrefetched_EarlyClose_StopsBuildingRemainingItems(self):
        built = []
        items = iterate_prefetched(lambda value: built.append(value) or value, range(100), 2)

        self.assertEqual(next(items), 0)
        items.close()

        self.assertLessEqual(len(built), 4)

    def test_IteratePrefetched_EmptyArguments_YieldsNothing(self):
        self.assertEqual(list(iterate_prefetched(lambda value: value, [], 2)), [])


if __name__ == "__main__":
    unittest.main()
