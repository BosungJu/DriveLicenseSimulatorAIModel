import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import torch

from src.cross_validate import (assign_folds, main, parse_cross_validation_args, summarize_groups,
                                summarize_runs, validate_source_records, write_fold_data)


def make_record(run_id, split, labels, value=1.0, driver_id=None):
    record = {"schema_version": 1, "sample_rate_hz": 50, "run_id": run_id, "split": split,
              "feature_names": ["speed", "brake"], "label_names": ["late", "fast"],
              "features": [[value, 0.0], [value, 1.0]], "labels": labels}
    if driver_id is not None:
        record["driver_id"] = driver_id
    return record


def counts(true_positive=0, false_positive=0, false_negative=0):
    return {"true_positive": true_positive, "false_positive": false_positive, "false_negative": false_negative}


def history(*val_values, label_counts=None):
    """Epoch entries with the given val BCE; label_counts maps an epoch index to its per-label counts."""
    label_counts = label_counts or {}
    return [{"epoch": index + 1, "train_bce": 1.0 / (index + 1), "val_bce": value,
             "val_label_counts": label_counts.get(index, [counts(), counts()])} for index, value in enumerate(val_values)]


class FoldAssignmentTests(unittest.TestCase):
    #region Normal Cases

    def test_AssignFolds_SessionSections_StayInOneFold(self):
        records = [make_record("a", "train", [0, 0]), make_record("a", "train", [1, 0]),
                   make_record("b", "val", [0, 0]), make_record("c", "train", [0, 1]),
                   make_record("d", "train", [0, 0], driver_id="driver-1"),
                   make_record("e", "val", [0, 0], driver_id="driver-1")]
        groups = summarize_groups(records, 2)

        group_folds = assign_folds(groups, 2)

        self.assertEqual(set(groups), {"run:a", "run:b", "run:c", "driver:driver-1"})
        self.assertEqual(groups["run:a"], {"sections": 2, "positives": [1, 0]})
        self.assertEqual(groups["driver:driver-1"]["sections"], 2)
        self.assertEqual(set(group_folds), set(groups))

    def test_SummarizeGroups_LabelMasks_CountOnlyJudgedPositives(self):
        records = [make_record("a", "train", [1, 1]), make_record("a", "train", [1, 0])]

        groups = summarize_groups(records, 2, label_masks=[[1, 0], [0, 1]])

        self.assertEqual(groups["run:a"], {"sections": 2, "positives": [1, 0]})

    def test_AssignFolds_PositiveHeavySessions_SpreadAcrossFolds(self):
        groups = {"run:heavy-1": {"sections": 15, "positives": [7, 5]},
                  "run:heavy-2": {"sections": 15, "positives": [3, 1]},
                  "run:light-1": {"sections": 4, "positives": [1, 0]},
                  "run:light-2": {"sections": 4, "positives": [0, 0]},
                  "run:light-3": {"sections": 4, "positives": [0, 0]},
                  "run:light-4": {"sections": 4, "positives": [0, 0]}}

        group_folds = assign_folds(groups, 3)

        heavy_folds = {group_folds["run:heavy-1"], group_folds["run:heavy-2"], group_folds["run:light-1"]}
        self.assertEqual(len(heavy_folds), 3)
        self.assertEqual(set(group_folds.values()), {0, 1, 2})

    def test_AssignFolds_SameGroups_GiveSameFolds(self):
        groups = {f"run:{index}": {"sections": 4, "positives": [index % 2, 0]} for index in range(7)}
        reordered = dict(reversed(list(groups.items())))

        self.assertEqual(assign_folds(groups, 3), assign_folds(reordered, 3))

    #endregion

    #region Failure Cases

    def test_AssignFolds_FewerSessionsThanFoldsOrOneFold_Fail(self):
        groups = {"run:a": {"sections": 1, "positives": [0]}, "run:b": {"sections": 1, "positives": [0]}}

        for fold_count in (1, 3):
            with self.subTest(fold_count=fold_count), self.assertRaises(ValueError):
                assign_folds(groups, fold_count)

    #endregion


class FoldDataTests(unittest.TestCase):
    def test_WriteFoldData_FoldSessionsBecomeValAndOthersTrain(self):
        records = [make_record("a", "train", [0, 0]), make_record("b", "val", [1, 0]), make_record("b", "val", [0, 0])]
        group_folds = {"run:a": 0, "run:b": 1}

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fold.jsonl"
            write_fold_data(records, group_folds, 0, path)
            written = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]

        self.assertEqual([(record["run_id"], record["split"]) for record in written], [("a", "val"), ("b", "train"), ("b", "train")])
        self.assertEqual(written[0]["features"], records[0]["features"])
        self.assertEqual(records[0]["split"], "train")


class RunSummaryTests(unittest.TestCase):
    def test_SummarizeRuns_SelectsEpochWithLowestMeanValBce(self):
        # Per-run minima are epochs 1 and 3, but the mean curve (0.5, 0.35, 0.4) is lowest at epoch 2.
        runs = [{"fold": 0, "seed": 1, "prevalence_val_bce": 0.6, "epoch_history": history(0.2, 0.3, 0.6)},
                {"fold": 1, "seed": 1, "prevalence_val_bce": 0.8, "epoch_history": history(0.8, 0.4, 0.2)}]

        summary = summarize_runs(runs, ["late", "fast"])

        self.assertEqual(summary["selected_epochs"], 2)
        self.assertAlmostEqual(summary["selected_mean_val_bce"], 0.35)
        self.assertAlmostEqual(summary["selected_std_val_bce"], 0.05)
        self.assertEqual(summary["selected_val_bce_by_fold"], {0: 0.3, 1: 0.4})
        self.assertAlmostEqual(summary["mean_best_epoch_val_bce"], 0.2)
        self.assertAlmostEqual(summary["mean_prevalence_val_bce"], 0.7)
        for actual, expected in zip(summary["mean_val_bce_by_epoch"], (0.5, 0.35, 0.4)):
            self.assertAlmostEqual(actual, expected)

    def test_SummarizeRuns_DifferentEpochCounts_Fail(self):
        runs = [{"fold": 0, "seed": 1, "prevalence_val_bce": 0.5, "epoch_history": history(0.2, 0.3)},
                {"fold": 1, "seed": 1, "prevalence_val_bce": 0.5, "epoch_history": history(0.2)}]

        with self.assertRaises(ValueError):
            summarize_runs(runs, ["late", "fast"])

    def test_SummarizeRuns_F1_PoolsCountsOfAllRunsAtSelectedEpoch(self):
        # Epoch 2 is selected. Pooled late: TP 1+2, FP 1+0, FN 0+1 -> precision 3/4, recall 3/4, F1 6/8.
        # Pooled fast: nothing flagged and 2 missed -> precision None, recall 0, F1 0.
        # Micro: TP 3, FP 1, FN 3 -> F1 6/10. Epoch 1 counts are ignored for the selected scores.
        runs = [{"fold": 0, "seed": 1, "prevalence_val_bce": 0.5,
                 "epoch_history": history(0.5, 0.3, label_counts={0: [counts(9, 9, 9), counts()],
                                                                   1: [counts(1, 1, 0), counts(0, 0, 2)]})},
                {"fold": 1, "seed": 1, "prevalence_val_bce": 0.5,
                 "epoch_history": history(0.5, 0.3, label_counts={1: [counts(2, 0, 1), counts()]})}]

        summary = summarize_runs(runs, ["late", "fast"])

        late, fast = summary["selected_val_per_label"]["late"], summary["selected_val_per_label"]["fast"]
        self.assertEqual((late["true_positive"], late["false_positive"], late["false_negative"]), (3, 1, 1))
        self.assertAlmostEqual(late["precision"], 0.75)
        self.assertAlmostEqual(late["recall"], 0.75)
        self.assertAlmostEqual(late["f1"], 0.75)
        self.assertIsNone(fast["precision"])
        self.assertEqual((fast["recall"], fast["f1"]), (0.0, 0.0))
        self.assertAlmostEqual(summary["selected_val_micro"]["f1"], 0.6)
        self.assertEqual(summary["selected_val_micro_f1_by_fold"], {0: 2 / 5, 1: 4 / 5})
        self.assertAlmostEqual(summary["pooled_val_micro_f1_by_epoch"][0], 0.5)

    def test_SummarizeRuns_NoPositivesNoFlags_ScoresAreNone(self):
        runs = [{"fold": 0, "seed": 1, "prevalence_val_bce": 0.5, "epoch_history": history(0.3)}]

        summary = summarize_runs(runs, ["late", "fast"])

        self.assertEqual(summary["selected_val_micro"], {"true_positive": 0, "false_positive": 0, "false_negative": 0,
                                                         "precision": None, "recall": None, "f1": None})


class CrossValidationRunTests(unittest.TestCase):
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
                   make_record("train-3", "train", [0, 0], 2.0), make_record("val-1", "val", [1, 1], 3.0),
                   make_record("test-1", "test", [1, 0], 9.0)]
        (self.root / "data.jsonl").write_text("\n".join(json.dumps(record) for record in records), encoding="utf-8")
        config = {"data": "data.jsonl", "schema": "schema.json", "output": "model.pt", "device": "cpu",
                  "epochs": 3, "batch_size": 2, "hidden_size": 4, "learning_rate": 0.01}
        self.config_path = self.root / "config.json"
        self.config_path.write_text(json.dumps(config), encoding="utf-8")

    #endregion

    def test_Main_FoldsAndSeeds_TrainEachPairWithoutTestSections(self):
        results_path = self.root / "cv.json"

        with contextlib.redirect_stderr(io.StringIO()):
            result = main(["--config", str(self.config_path), "--folds", "2", "--seeds", "1", "2",
                           "--results", str(results_path), "--epochs", "2"])

        self.assertEqual(json.loads(results_path.read_text(encoding="utf-8"))["summary"]["selected_epochs"],
                         result["summary"]["selected_epochs"])
        self.assertEqual(result["excluded_test_sections"], 1)
        self.assertEqual(sorted(group for fold in result["folds"] for group in fold["groups"]),
                         ["run:train-1", "run:train-2", "run:train-3", "run:val-1"])
        self.assertEqual([(run["fold"], run["seed"]) for run in result["runs"]], [(0, 1), (0, 2), (1, 1), (1, 2)])
        self.assertTrue(all(len(run["epoch_history"]) == 2 for run in result["runs"]))
        self.assertTrue(all(len(entry["val_label_counts"]) == 2 for run in result["runs"] for entry in run["epoch_history"]))
        self.assertEqual(set(result["summary"]["selected_val_per_label"]), {"late", "fast"})
        # "late" is positive in train-1 and val-1; each of the 2 seeds validates every session once, test-1 never.
        late = result["summary"]["selected_val_per_label"]["late"]
        self.assertEqual(late["true_positive"] + late["false_negative"], 2 * 2)
        self.assertEqual(result["settings"]["epochs"], 2)
        self.assertFalse((self.root / "model.pt").exists())

    def test_Main_SameSeedTwice_GivesSameCurves(self):
        argv = ["--config", str(self.config_path), "--folds", "2", "--seeds", "3", "--results", str(self.root / "cv.json")]

        with contextlib.redirect_stderr(io.StringIO()):
            first = main(argv)
            second = main(argv)

        self.assertEqual([run["epoch_history"] for run in first["runs"]], [run["epoch_history"] for run in second["runs"]])

    def test_ParseArgs_SeedOrOutputOption_Fails(self):
        for option in (["--seed", "1"], ["--output", "x.pt"], ["--seed=1"], ["--see", "1"], ["--out", "x.pt"]):
            argv = ["--config", str(self.config_path), "--results", "cv.json", *option]
            with self.subTest(option=option), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parse_cross_validation_args(argv)

    def test_ParseArgs_DuplicateSeedsOrResultsOverInput_Fails(self):
        augmentation_path = self.root / "augmentation.json"
        augmentation_path.write_text(json.dumps({"probability": 0, "features": {}}), encoding="utf-8")
        for options in (["--seeds", "1", "1", "--results", "cv.json"],
                        ["--results", str(self.root / "data.jsonl")],
                        ["--results", str(self.root / "schema.json")],
                        ["--results", str(self.config_path)],
                        ["--augmentation", str(augmentation_path), "--results", str(augmentation_path)]):
            argv = ["--config", str(self.config_path), *options]
            with self.subTest(options=options), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parse_cross_validation_args(argv)

    def test_ParseArgs_Defaults_UseTwoDistinctSeeds(self):
        args, _ = parse_cross_validation_args(["--config", str(self.config_path), "--results", "cv.json"])

        self.assertEqual(args.seeds, [42, 43])
        self.assertEqual(args.folds, 5)


class SourceValidationTests(unittest.TestCase):
    def test_ValidateSourceRecords_OriginalSplits_Pass(self):
        validate_source_records([make_record("a", "train", [0, 0]), make_record("b", "val", [0, 0]),
                                 make_record("c", "test", [0, 0]), dict(make_record("d", "train", [0, 0]), augmentation_copy=0)])

    def test_ValidateSourceRecords_InvalidSplitCopiesOrTestLeak_Fail(self):
        cases = {"missing split": [{key: value for key, value in make_record("a", "train", [0, 0]).items() if key != "split"}],
                 "unknown split": [make_record("a", "holdout", [0, 0])],
                 "augmented copy": [dict(make_record("a", "train", [0, 0]), augmentation_copy=1)],
                 "run in test and train": [make_record("a", "test", [0, 0]), make_record("a", "train", [0, 0])],
                 "driver in test and val": [make_record("a", "test", [0, 0], driver_id="d"),
                                            make_record("b", "val", [0, 0], driver_id="d")],
                 "run with driver on some rows": [make_record("a", "test", [0, 0], driver_id="d"),
                                                  make_record("a", "train", [0, 0])],
                 "run with driver on some pooled rows": [make_record("a", "train", [0, 0], driver_id="d"),
                                                         make_record("a", "val", [0, 0])]}

        for name, records in cases.items():
            with self.subTest(case=name), self.assertRaises(ValueError):
                validate_source_records(records)


if __name__ == "__main__":
    unittest.main()
