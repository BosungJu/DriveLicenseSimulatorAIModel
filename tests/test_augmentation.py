import argparse
import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from src.baseline import (AUGMENTATION_STAT_KEYS, augment_features, export_augmented_dataset, load_records, make_batch,
                          make_copy_batch, parse_args, predict, smoothed_noise, train, validate_augmentation)


class AugmentationTests(unittest.TestCase):
    def setUp(self):
        self.schema = {"schema_version": 1, "sample_rate_hz": 50,
                       "feature_names": ["speed", "brake"], "continuous_features": ["speed"],
                       "label_names": ["improvement"]}
        self.config = {"probability": 1.0, "features": {"speed": {
            "bias_max": 0.1, "noise_max": 0.02, "smoothing": 0.1,
            "min": -10.0, "max": 10.0, "protected_values": [0.0]}}}

    def augment(self, features, config=None, seed=7):
        return augment_features(features, self.schema["feature_names"], config or self.config,
                                torch.Generator().manual_seed(seed))

    def test_Augment_BoundedPerturbations_PreservesOriginalAndExcludedFeatures(self):
        features = torch.tensor([[2.0, float(index % 2)] for index in range(30)])
        original = features.clone()
        augmented = self.augment(features)
        torch.testing.assert_close(features, original)
        torch.testing.assert_close(augmented[:, 1], original[:, 1])
        self.assertTrue(torch.any(augmented[:, 0] != original[:, 0]))
        self.assertLessEqual((augmented[:, 0] - original[:, 0]).abs().max().item(), 0.120001)
        self.assertEqual(augmented.shape, original.shape)

    def test_Augment_BiasOnly_OffsetIsConstantAcrossSection(self):
        config = copy.deepcopy(self.config)
        config["features"]["speed"]["noise_max"] = 0
        features = torch.tensor([[1.0, 0.0], [2.0, 1.0], [3.0, 0.0]])
        difference = self.augment(features, config)[:, 0] - features[:, 0]
        torch.testing.assert_close(difference, difference[0].expand_as(difference))

    def test_Augment_SmoothedNoise_AdjacentChangesAreBounded(self):
        config = copy.deepcopy(self.config)
        config["features"]["speed"]["bias_max"] = 0
        features = torch.tensor([[2.0, 0.0]] * 40)
        augmented = self.augment(features, config)
        self.assertLessEqual(torch.diff(augmented[:, 0]).abs().max().item(), 0.004001)

    def test_SmoothedNoise_MatchesRecursiveDefinition(self):
        for smoothing in (1.0, 0.99, 0.5, 0.1, 0.001):
            for length in (1, 2, 7, 300, 1500):
                innovations = torch.rand(length, generator=torch.Generator().manual_seed(length)) * 2 - 1
                # float64 reference, so float32 accumulation error of the loop itself does not count.
                expected = innovations.to(torch.float64)
                for timestep in range(1, length):
                    expected[timestep] = (1 - smoothing) * expected[timestep - 1] + smoothing * expected[timestep]

                with self.subTest(smoothing=smoothing, length=length):
                    torch.testing.assert_close(smoothed_noise(innovations, smoothing), expected.to(torch.float32),
                                               rtol=1e-5, atol=1e-6)

    def test_Augment_Seed_ReproducibleAndDifferentSeedsVary(self):
        features = torch.tensor([[2.0, 0.0]] * 10)
        torch.testing.assert_close(self.augment(features), self.augment(features), rtol=0, atol=0)
        self.assertFalse(torch.equal(self.augment(features), self.augment(features, seed=8)))

    def test_Augment_ZeroProbabilityAndSingleFrame_AreSupported(self):
        features = torch.tensor([[2.0, 1.0]])
        config = dict(self.config, probability=0)
        torch.testing.assert_close(self.augment(features, config), features)
        self.assertEqual(self.augment(features).shape, features.shape)

    def test_Augment_InvalidOriginalBounds_FailsEvenWhenProbabilityIsZero(self):
        with self.assertRaisesRegex(ValueError, "exceeds configured bounds"):
            self.augment(torch.tensor([[20.0, 0.0]]), dict(self.config, probability=0))

    def test_Augment_ThresholdEqualityCrossingAndBounds_KeepOriginalFeature(self):
        # Any changed equality would alter stop/overspeed membership.
        features = torch.tensor([[0.0, 1.0], [2.0, 0.0]])
        torch.testing.assert_close(self.augment(features), features)
        config = copy.deepcopy(self.config)
        config["features"]["speed"].update(bias_max=10, noise_max=0)
        features = torch.tensor([[0.001, 1.0]])
        torch.testing.assert_close(self.augment(features, config), features)
        config["features"]["speed"].update(min=0, max=0.01, protected_values=[])
        torch.testing.assert_close(self.augment(features, config), features)

    def test_Validate_UnknownDiscreteAndInvalidSettings_Fail(self):
        variants = []
        for key, value in (("bias_max", -1), ("noise_max", float("nan")),
                           ("smoothing", 0), ("max", -10), ("protected_values", [True])):
            config = copy.deepcopy(self.config)
            config["features"]["speed"][key] = value
            variants.append(config)
        variants.extend([dict(self.config, probability=2), dict(self.config, probability=True),
                         dict(self.config, features={"brake": self.config["features"]["speed"]}),
                         dict(self.config, features={"unknown": self.config["features"]["speed"]})])
        for config in variants:
            with self.subTest(config=config), self.assertRaises(ValueError):
                validate_augmentation(config, self.schema)


    def test_Augment_FixedValues_KeepRestFramesAndPerturbOthers(self):
        config = copy.deepcopy(self.config)
        config["features"]["speed"].update(min=0.0, protected_values=[], fixed_values=[0.0, 5.0])
        features = torch.tensor([[0.0, 0.0], [2.0, 1.0], [5.0, 0.0], [3.0, 1.0], [0.0, 0.0]])

        augmented = self.augment(features, config)

        torch.testing.assert_close(augmented[[0, 2, 4], 0], features[[0, 2, 4], 0], rtol=0, atol=0)
        self.assertTrue(torch.all(augmented[[1, 3], 0] != features[[1, 3], 0]))
        torch.testing.assert_close(augmented[:, 1], features[:, 1])

    def test_Validate_AugmentedCopiesAndFixedValues_AcceptValidAndRejectInvalid(self):
        valid = copy.deepcopy(self.config)
        valid["augmented_copies"] = 4
        valid["features"]["speed"]["fixed_values"] = [0, 1.5]
        self.assertEqual(validate_augmentation(valid, self.schema), valid)

        invalid_variants = [dict(self.config, augmented_copies=value) for value in (0, -1, 1.5, True, None)]
        for value in ([True], [float("inf")], 0):
            config = copy.deepcopy(self.config)
            config["features"]["speed"]["fixed_values"] = value
            invalid_variants.append(config)
        invalid_variants.append(dict(self.config, unknown=1))
        for config in invalid_variants:
            with self.subTest(config=config), self.assertRaises(ValueError):
                validate_augmentation(config, self.schema)

    def test_Augment_Statistics_ReportAcceptedRejectedAndUnchangedCandidates(self):
        stats = {"speed": {key: 0 for key in AUGMENTATION_STAT_KEYS}}
        generator = torch.Generator().manual_seed(7)
        augment_features(torch.tensor([[2.0, 0.0]]), self.schema["feature_names"], self.config, generator, stats)
        augment_features(torch.tensor([[0.0, 0.0]]), self.schema["feature_names"], self.config, generator, stats)
        disabled = dict(self.config, probability=0)
        augment_features(torch.tensor([[2.0, 0.0]]), self.schema["feature_names"], disabled, generator, stats)
        zero_strength = copy.deepcopy(self.config)
        zero_strength["features"]["speed"].update(bias_max=0, noise_max=0)
        augment_features(torch.tensor([[2.0, 0.0]]), self.schema["feature_names"], zero_strength, generator, stats)
        self.assertEqual(stats["speed"], {"attempted": 3, "applied": 1, "unchanged": 1,
                                          "skipped": 1, "rejected_bounds": 0, "rejected_threshold": 1, "kept_frames": 0})

    def keep_frame_config(self, noise_max=0.5):
        config = copy.deepcopy(self.config)
        config["features"]["speed"].update(bias_max=0, noise_max=noise_max, smoothing=1.0, min=0.0, max=10.0,
                                           protected_values=[1.0], on_violation="keep_frame")
        return config

    def test_Augment_KeepFrameAtProtectedValue_KeepsThatFrameAndPerturbsOthers(self):
        # A frame exactly on a protected value changes membership under any nonzero noise, so it must stay original.
        features = torch.tensor([[5.0, 0.0], [1.0, 1.0], [5.0, 0.0]])
        stats = {"speed": {key: 0 for key in AUGMENTATION_STAT_KEYS}}

        augmented = augment_features(features, self.schema["feature_names"], self.keep_frame_config(),
                                     torch.Generator().manual_seed(7), stats)

        self.assertEqual(augmented[1, 0].item(), 1.0)
        self.assertTrue(torch.all(augmented[[0, 2], 0] != features[[0, 2], 0]))
        torch.testing.assert_close(augmented[:, 1], features[:, 1])
        self.assertEqual(stats["speed"]["kept_frames"], 1)
        self.assertEqual(stats["speed"]["applied"], 1)
        self.assertEqual(stats["speed"]["rejected_threshold"], 0)

    def test_Augment_KeepFrameNearBoundsAndThreshold_PreservesFrameRelations(self):
        # Frames within noise_max of min, max and the protected value violate under some seeds.
        features = torch.tensor([[0.1, 0.0], [0.9, 0.0], [1.1, 0.0], [9.9, 0.0], [5.0, 0.0]] * 20)
        stats = {"speed": {key: 0 for key in AUGMENTATION_STAT_KEYS}}

        for seed in range(5):
            augmented = augment_features(features, self.schema["feature_names"], self.keep_frame_config(),
                                         torch.Generator().manual_seed(seed), stats)
            speed = augmented[:, 0]

            with self.subTest(seed=seed):
                self.assertTrue(torch.all((speed >= 0.0) & (speed <= 10.0)))
                torch.testing.assert_close(speed < 1.0, features[:, 0] < 1.0)
                torch.testing.assert_close(speed > 1.0, features[:, 0] > 1.0)
                self.assertTrue(torch.all(speed[4::5] != features[4::5, 0]))

        self.assertGreater(stats["speed"]["kept_frames"], 0)
        self.assertEqual(stats["speed"]["applied"], 5)
        self.assertEqual(stats["speed"]["rejected_bounds"] + stats["speed"]["rejected_threshold"], 0)

    def test_Augment_RejectFeatureDefault_DropsWholeFeatureOnSingleViolation(self):
        config = self.keep_frame_config()
        del config["features"]["speed"]["on_violation"]
        features = torch.tensor([[5.0, 0.0], [1.0, 1.0], [5.0, 0.0]])

        torch.testing.assert_close(self.augment(features, config), features, rtol=0, atol=0)

    def test_Validate_OnViolation_AcceptsPoliciesAndRejectsUnknown(self):
        for policy in ("reject_feature", "keep_frame"):
            config = copy.deepcopy(self.config)
            config["features"]["speed"]["on_violation"] = policy
            self.assertEqual(validate_augmentation(config, self.schema), config)

        for policy in ("keep", None, 1):
            config = copy.deepcopy(self.config)
            config["features"]["speed"]["on_violation"] = policy
            with self.subTest(policy=policy), self.assertRaises(ValueError):
                validate_augmentation(config, self.schema)


class TrainingConfigurationTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        deterministic = torch.are_deterministic_algorithms_enabled()
        self.addCleanup(torch.use_deterministic_algorithms, deterministic)

    def write_json(self, name, value):
        path = self.root / name
        path.write_text(json.dumps(value), encoding="utf-8")
        return path

    def test_ParseArgs_ConfigPathsAndCliOverrides_AreResolved(self):
        config_path = self.write_json("config.json", {"data": "data.jsonl", "schema": "schema.json",
                                                     "output": "model.pt", "learning_rate": 0.01, "epochs": 3})
        args = parse_args(["train", "--config", str(config_path), "--learning-rate", "0.02", "--output", "override.pt"])
        self.assertEqual(args.learning_rate, 0.02)
        self.assertEqual(args.epochs, 3)
        self.assertEqual(args.data, str(self.root / "data.jsonl"))
        self.assertEqual(args.output, "override.pt")
        self.assertEqual(args.batch_size, 32)
        self.assertEqual(args.num_layers, 1)

    def test_ParseArgs_InvalidHyperparameters_Fail(self):
        for key, value in (("epochs", 0), ("batch_size", True), ("hidden_size", 1.5),
                           ("learning_rate", float("inf")), ("seed", "bad"),
                           ("num_layers", 0), ("num_layers", True), ("num_layers", 1.5),
                           ("augmentation_seed", "bad"), ("device", None),
                           ("deterministic", "true"), ("typo", 1)):
            path = self.write_json("config.json", {"data": "data", "schema": "schema", "output": "output", key: value})
            with self.subTest(key=key), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parse_args(["train", "--config", str(path)])

    def test_ParseArgs_SeparateAugmentationFile_OverridesTrainingConfig(self):
        config_path = self.write_json("config.json", {"data": "data", "schema": "schema", "output": "output",
                                                     "augmentation": {"probability": 0, "features": {}}})
        override = {"probability": 0.5, "features": {"speed": {"bias_max": 0.01}}}
        augmentation_path = self.write_json("augmentation.json", override)
        args = parse_args(["train", "--config", str(config_path), "--augmentation", str(augmentation_path)])
        self.assertEqual(args.augmentation, override)

    def test_Train_ConfigAugmentation_OnlyTrainingChangesAndCheckpointIsReproducible(self):
        schema = {"schema_version": 1, "sample_rate_hz": 50, "feature_names": ["speed", "brake"],
                  "continuous_features": ["speed"], "label_names": ["improvement"]}
        augmentation = {"probability": 1, "features": {"speed": {"bias_max": 0.1, "noise_max": 0.02,
                        "smoothing": 0.1, "min": -10, "max": 10, "protected_values": [0]}}}
        self.write_json("schema.json", schema)
        records = []
        for split, values in (("train", [-1, 1]), ("val", [5]), ("test", [7])):
            for index, value in enumerate(values):
                records.append(dict(schema, run_id=f"{split}-{index}", split=split,
                                    features=[[value, 0], [value, 1]], labels=[int(value > 0)]))
        data_path = self.root / "data.jsonl"
        data_path.write_text("\n".join(json.dumps(record) for record in records), encoding="utf-8")
        settings = {"data": "data.jsonl", "schema": "schema.json", "output": "model.pt", "seed": 42,
                    "device": "cpu",
                    "epochs": 2, "batch_size": 2, "hidden_size": 4, "learning_rate": 0.01,
                    "augmentation": augmentation}
        config_path = self.write_json("config.json", settings)
        args = parse_args(["train", "--config", str(config_path)])
        calls = []

        def observe_batch(batch, mean, scale, augmentation=None, generator=None, augmentation_stats=None):
            originals = [record["features"].clone() for record in batch]
            result = make_batch(batch, mean, scale, augmentation, generator, augmentation_stats)
            calls.extend((record.get("split"), augmentation is not None) for record in batch)
            for record, original in zip(batch, originals):
                torch.testing.assert_close(record["features"], original)
            return result

        with patch("src.baseline.make_batch", side_effect=observe_batch), contextlib.redirect_stdout(io.StringIO()):
            train(args)
        self.assertTrue(any(split == "train" and enabled for split, enabled in calls))
        self.assertTrue(any(split == "val" for split, _ in calls))
        self.assertTrue(any(split == "test" for split, _ in calls))
        self.assertTrue(all(split == "train" for split, enabled in calls if enabled))
        first = torch.load(args.output, weights_only=True)
        torch.testing.assert_close(first["mean"], torch.tensor([0.0, 0.5]))
        torch.testing.assert_close(first["scale"], torch.tensor([1.0, 0.5]))
        self.assertEqual(first["augmentation"], augmentation)
        self.assertEqual(first["augmentation_stats"]["speed"]["attempted"], 4)
        self.assertEqual(first["augmentation_stats"]["speed"]["applied"], 4)
        self.assertNotEqual(first["augmentation_seed"], first["seed"])
        self.assertEqual(first["training_args"]["learning_rate"], 0.01)
        settings["augmentation"]["probability"] = 0
        original_data = data_path.read_text(encoding="utf-8")
        invalid_records = copy.deepcopy(records)
        invalid_records[0]["features"][0][0] = 20
        data_path.write_text("\n".join(json.dumps(record) for record in invalid_records), encoding="utf-8")
        invalid_args = argparse.Namespace(**vars(args))
        invalid_args.augmentation = settings["augmentation"]
        with patch("src.baseline.SectionLSTM") as model_factory, self.assertRaisesRegex(ValueError, "exceeds configured bounds"):
            train(invalid_args)
        model_factory.assert_not_called()
        data_path.write_text(original_data, encoding="utf-8")
        args.augmentation["probability"] = 1
        with contextlib.redirect_stdout(io.StringIO()):
            train(args)
        second = torch.load(args.output, weights_only=True)
        for key in first["state_dict"]:
            torch.testing.assert_close(first["state_dict"][key], second["state_dict"][key], rtol=0, atol=0)
        calls.clear()
        with patch("src.baseline.make_batch", side_effect=observe_batch), contextlib.redirect_stdout(io.StringIO()):
            predict(argparse.Namespace(data=str(data_path), checkpoint=args.output, device="cpu"))
        self.assertTrue(all(not enabled for _, enabled in calls))

    def write_export_inputs(self, augmentation):
        schema = {"schema_version": 1, "sample_rate_hz": 50, "feature_names": ["speed", "brake"],
                  "continuous_features": ["speed"], "label_names": ["improvement"]}
        self.write_json("schema.json", schema)
        records = [dict(schema, run_id="train-0", split="train", features=[[1.5, 0], [2.25, 1], [0, 1]], labels=[1]),
                   dict(schema, run_id="val-0", split="val", features=[[5, 0], [5, 1]], labels=[0])]
        (self.root / "data.jsonl").write_text("\n".join(json.dumps(record) for record in records), encoding="utf-8")
        config_path = self.write_json("config.json", {"data": "data.jsonl", "schema": "schema.json",
                                                      "output": "model.pt", "seed": 7, "augmentation": augmentation})
        return records, config_path

    def test_Augment_ExportsOriginalPlusSeededCopiesAndUnchangedVal(self):
        augmentation = {"probability": 1, "augmented_copies": 3, "features": {"speed": {
            "bias_max": 0.1, "noise_max": 0.02, "smoothing": 0.5, "min": 0, "max": 10, "protected_values": [],
            "fixed_values": [0]}}}
        records, config_path = self.write_export_inputs(augmentation)
        output = self.root / "data" / "augmented.jsonl"
        args = parse_args(["augment", "--config", str(config_path), "--output", str(output)])

        summary = export_augmented_dataset(args)
        exported = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]

        self.assertEqual([(record["split"], record.get("augmentation_copy")) for record in exported],
                         [("train", 0), ("train", 1), ("train", 2), ("train", 3), ("val", None)])
        self.assertEqual(exported[0]["features"], records[0]["features"])
        self.assertEqual(exported[4], records[1])
        for copy in exported[1:4]:
            self.assertEqual([frame[1] for frame in copy["features"]], [0, 1, 1])
            self.assertEqual(copy["features"][2][0], 0)
            self.assertNotEqual(copy["features"][0][0], 1.5)
        self.assertEqual(len({json.dumps(copy["features"]) for copy in exported[1:4]}), 3)
        self.assertEqual(summary["records_by_split"], {"train": 4, "val": 1})
        self.assertEqual(summary["augmentation_stats"]["speed"]["applied"], 3)
        meta = json.loads(output.with_suffix(".meta.json").read_text(encoding="utf-8"))
        self.assertEqual(meta["augmentation_seed"], summary["augmentation_seed"])
        # The export trains as a plain dataset, and the same seed reproduces it byte for byte.
        loaded = load_records(output, ["speed", "brake"], ["improvement"], sample_rate_hz=50)
        self.assertEqual(len(loaded), 5)
        first_export = output.read_bytes()
        export_augmented_dataset(args)
        self.assertEqual(output.read_bytes(), first_export)

    def test_Augment_WithoutCopiesOrOutput_Fails(self):
        augmentation = {"probability": 1, "features": {"speed": {
            "bias_max": 0.1, "noise_max": 0.02, "smoothing": 0.5, "min": 0, "max": 10, "protected_values": []}}}
        _, config_path = self.write_export_inputs(augmentation)
        args = parse_args(["augment", "--config", str(config_path), "--output", str(self.root / "out.jsonl")])

        with self.assertRaisesRegex(ValueError, "augmented_copies"):
            export_augmented_dataset(args)
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parse_args(["augment", "--config", str(config_path)])

    def test_Train_AugmentedCopies_EachEpochSeesOriginalPlusFreshCopies(self):
        schema = {"schema_version": 1, "sample_rate_hz": 50, "feature_names": ["speed", "brake"],
                  "continuous_features": ["speed"], "label_names": ["improvement"]}
        augmentation = {"probability": 1, "augmented_copies": 4, "features": {"speed": {
            "bias_max": 0.1, "noise_max": 0.02, "smoothing": 0.1, "min": -10, "max": 10, "protected_values": [0]}}}
        self.write_json("schema.json", schema)
        originals = {"train-0": [[-1, 0], [-1, 1]], "train-1": [[1, 0], [1, 1]]}
        records = [dict(schema, run_id=run_id, split="train", features=features, labels=[int(features[0][0] > 0)])
                   for run_id, features in originals.items()]
        records.append(dict(schema, run_id="val-0", split="val", features=[[5, 0], [5, 1]], labels=[1]))
        (self.root / "data.jsonl").write_text("\n".join(json.dumps(record) for record in records), encoding="utf-8")
        epochs = 2
        config_path = self.write_json("config.json", {
            "data": "data.jsonl", "schema": "schema.json", "output": "model.pt", "seed": 42, "device": "cpu",
            "epochs": epochs, "batch_size": 3, "hidden_size": 4, "learning_rate": 0.01, "augmentation": augmentation})
        args = parse_args(["train", "--config", str(config_path)])
        train_samples = []

        # Observes training batches only; the final train-split evaluation also calls make_batch with originals.
        def observe_copy_batch(records, entries, augmentation, generator, augmentation_stats):
            batch = make_copy_batch(records, entries, augmentation, generator, augmentation_stats)
            train_samples.extend((record["run_id"], record["features"].clone()) for record in batch)
            return batch

        with patch("src.baseline.make_copy_batch", side_effect=observe_copy_batch), \
                contextlib.redirect_stdout(io.StringIO()):
            train(args)

        for run_id, features in originals.items():
            original = torch.tensor(features, dtype=torch.float32)
            samples = [sample for sample_run_id, sample in train_samples if sample_run_id == run_id]
            unchanged = [sample for sample in samples if torch.equal(sample, original)]
            augmented = [sample for sample in samples if not torch.equal(sample, original)]
            self.assertEqual((len(unchanged), len(augmented)), (epochs, epochs * 4))
            self.assertEqual(len({tuple(sample[:, 0].tolist()) for sample in augmented}), epochs * 4)
        checkpoint = torch.load(args.output, weights_only=True)
        self.assertEqual(checkpoint["augmentation_stats"]["speed"]["attempted"], len(originals) * epochs * 4)
        self.assertEqual(checkpoint["augmentation"]["augmented_copies"], 4)


if __name__ == "__main__":
    unittest.main()
