import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import torch

from src import sweep
from src.sweep import build_grid, main, parse_sweep_args, rank_trials, trial_name


def make_record(run_id, split, labels, value=1.0):
    return {"schema_version": 1, "sample_rate_hz": 50, "run_id": run_id, "split": split,
            "feature_names": ["speed", "brake"], "label_names": ["late", "fast"],
            "features": [[value, 0.0], [value, 1.0]], "labels": labels}


class GridTests(unittest.TestCase):
    def test_BuildGrid_EveryCombinationInValueOrder(self):
        grid = build_grid([32, 64], [1, 2], [0.001])

        self.assertEqual(grid, [(32, 1, 0.001), (32, 2, 0.001), (64, 1, 0.001), (64, 2, 0.001)])

    def test_TrialName_FormatsLearningRateCompactly(self):
        self.assertEqual(trial_name(64, 2, 0.0005), "h64-l2-lr0.0005")

    def test_RankTrials_OrdersByValBceThenName(self):
        trials = [{"name": "b", "val_bce": 0.3}, {"name": "c", "val_bce": 0.2}, {"name": "a", "val_bce": 0.3}]

        self.assertEqual(rank_trials(trials), ["c", "a", "b"])


class SweepRunTests(unittest.TestCase):
    #region Setup and Teardown

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        deterministic = torch.are_deterministic_algorithms_enabled()
        self.addCleanup(torch.use_deterministic_algorithms, deterministic)
        schema = {"schema_version": 1, "sample_rate_hz": 50, "feature_names": ["speed", "brake"],
                  "continuous_features": ["speed"], "label_names": ["late", "fast"]}
        (self.root / "schema.json").write_text(json.dumps(schema), encoding="utf-8")
        records = [make_record("train-1", "train", [1, 0], 1.0), make_record("train-2", "train", [0, 1], -1.0),
                   make_record("val-1", "val", [1, 1], 3.0), make_record("test-1", "test", [1, 0], 9.0)]
        (self.root / "data.jsonl").write_text("\n".join(json.dumps(record) for record in records), encoding="utf-8")
        config = {"data": "data.jsonl", "schema": "schema.json", "output": "model.pt", "device": "cpu",
                  "epochs": 2, "batch_size": 2, "hidden_size": 4, "learning_rate": 0.01}
        self.config_path = self.root / "config.json"
        self.config_path.write_text(json.dumps(config), encoding="utf-8")
        self.results_path = self.root / "sweep.json"

    def run_main(self, *grid_options):
        argv = ["--config", str(self.config_path), "--results", str(self.results_path), "--quiet", *grid_options]
        with contextlib.redirect_stderr(io.StringIO()):
            return main(argv)

    #endregion

    #region Normal Cases

    def test_Main_Grid_TrainsEachTrialAndRanksByValBce(self):
        result = self.run_main("--hidden-sizes", "2", "4", "--layer-counts", "1", "--learning-rates", "0.01")

        saved = json.loads(self.results_path.read_text(encoding="utf-8"))
        self.assertEqual(saved["ranking"], result["ranking"])
        self.assertEqual([trial["name"] for trial in result["trials"]], ["h2-l1-lr0.01", "h4-l1-lr0.01"])
        self.assertEqual(sorted(result["ranking"]), ["h2-l1-lr0.01", "h4-l1-lr0.01"])
        val_bce = {trial["name"]: trial["val_bce"] for trial in result["trials"]}
        self.assertLessEqual(val_bce[result["ranking"][0]], val_bce[result["ranking"][1]])
        for trial in result["trials"]:
            self.assertEqual(len(trial["val_bce_by_epoch"]), 2)
            self.assertEqual(trial["val_bce"], min(trial["val_bce_by_epoch"]))
            self.assertNotIn("test", json.dumps(trial))
        self.assertFalse((self.root / "model.pt").exists())

    def test_Main_ExistingResults_ResumesWithoutRetrainingAndKeepsOtherTrials(self):
        self.run_main("--hidden-sizes", "2", "--layer-counts", "1", "--learning-rates", "0.01")
        trained = []
        original_train_trial = sweep.train_trial

        def recording_train_trial(training_args, hidden_size, num_layers, learning_rate, checkpoint_path):
            trained.append(trial_name(hidden_size, num_layers, learning_rate))
            return original_train_trial(training_args, hidden_size, num_layers, learning_rate, checkpoint_path)

        with mock.patch.object(sweep, "train_trial", side_effect=recording_train_trial):
            result = self.run_main("--hidden-sizes", "4", "2", "--layer-counts", "1", "--learning-rates", "0.01")
            narrowed = self.run_main("--hidden-sizes", "4", "--layer-counts", "1", "--learning-rates", "0.01")

        self.assertEqual(trained, ["h4-l1-lr0.01"])
        self.assertEqual([trial["name"] for trial in result["trials"]], ["h4-l1-lr0.01", "h2-l1-lr0.01"])
        self.assertEqual(sorted(trial["name"] for trial in narrowed["trials"]), ["h2-l1-lr0.01", "h4-l1-lr0.01"])

    #endregion

    #region Failure Cases

    def test_Main_ExistingResultsWithOtherSettings_Fails(self):
        self.run_main("--hidden-sizes", "2", "--layer-counts", "1", "--learning-rates", "0.01")

        with self.assertRaisesRegex(ValueError, "other settings or data"):
            self.run_main("--hidden-sizes", "2", "--layer-counts", "1", "--learning-rates", "0.01", "--epochs", "1")

    def test_ParseArgs_GridOwnedOptionsOrInvalidValues_Fail(self):
        cases = (["--hidden-size", "8"], ["--num-layers", "2"], ["--learning-rate", "0.1"], ["--output", "x.pt"],
                 ["--hidden", "8"], ["--hidden-sizes", "0"], ["--hidden-sizes", "8", "8"],
                 ["--learning-rates", "-0.1"], ["--results", str(Path("data.jsonl"))])
        for options in cases:
            argv = ["--config", str(self.config_path), "--results", str(self.results_path), *options]
            if options[0] == "--results":
                argv = ["--config", str(self.config_path), "--results", str(self.root / "data.jsonl")]
            with self.subTest(options=options), contextlib.redirect_stderr(io.StringIO()), \
                    self.assertRaises(SystemExit):
                parse_sweep_args(argv)

    #endregion

    def test_ParseArgs_Defaults_GridOfSizesLayersAndLearningRates(self):
        args, training_args = parse_sweep_args(["--config", str(self.config_path), "--results", "sweep.json",
                                                "--epochs", "12"])

        self.assertEqual((args.hidden_sizes, args.layer_counts, args.learning_rates),
                         ([32, 64, 128], [1, 2], [0.0005, 0.001, 0.002]))
        self.assertEqual(training_args.epochs, 12)


if __name__ == "__main__":
    unittest.main()
