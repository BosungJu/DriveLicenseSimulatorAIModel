"""Session-grouped k-fold cross-validation of LSTM training settings on the train and val sections.

Input: the training config of `baseline.py train` with its CLI overrides, the fold count and the seeds.
Processing: test sections are left out. The train and val sections are pooled and their sessions (a driver, or a run
without a driver) are dealt into folds so no session is split and positive labels spread over the folds. Every
fold x seed trains a fresh model with that fold as val and records the val BCE after each epoch.
Output: JSON with the folds, each run's epoch curve, and the epoch count whose val BCE averaged over runs is lowest.
"""

import argparse
import copy
import hashlib
import json
import logging
import math
from pathlib import Path
import sys
import tempfile
import time
import warnings

if __package__:
    from .baseline import LSTMBaseline, attach_label_masks, format_duration, load_records, parse_args
    from .deduction_labels import split_group
else:
    from baseline import LSTMBaseline, attach_label_masks, format_duration, load_records, parse_args
    from deduction_labels import split_group

LOGGER = logging.getLogger("drive_cv")
DEFAULT_FOLD_COUNT = 5
DEFAULT_SEEDS = (42, 43)
# Training options the cross-validation sets itself for every run.
RESERVED_TRAINING_OPTIONS = ("--seed", "--output")
NO_TEST_SPLIT_WARNING = "No held-out test split"
COUNT_KEYS = ("true_positive", "false_positive", "false_negative")


def group_key(record):
    kind, identifier = split_group(record)
    return f"{kind}:{identifier}"


def validate_source_records(records):
    """Checks the original splits before test is dropped, since the fold data no longer carries them.

    Pre-augmented copies (`baseline.py augment`) would enter val folds and give their sessions extra weight.
    A session in both test and train/val would leak held-out data into the folds.
    """
    is_test_by_session = {}
    is_train_only_by_session = {}
    driver_by_run = {}
    for line_number, record in enumerate(records, 1):
        # A run's sessions key on driver_id, so rows of one run with different driver_ids would split it across folds.
        if driver_by_run.setdefault(record.get("run_id"), record.get("driver_id")) != record.get("driver_id"):
            raise ValueError(f"Record {line_number}: run {record.get('run_id')!r} mixes driver_id values")
        split = record.get("split")
        if split not in ("train", "val", "test"):
            raise ValueError(f"Record {line_number}: split must be train/val/test")
        if record.get("train_only") and split != "train":
            raise ValueError(f"Record {line_number}: a train_only section must have the train split")
        if record.get("augmentation_copy"):
            raise ValueError("Cross-validation needs the original sections; pass augmentation settings instead of augmented copies")
        # Runs and drivers are checked separately, as the loader does, so a run with a driver_id on only some rows is caught.
        sessions = [f"run:{record.get('run_id')}"] + ([f"driver:{record['driver_id']}"] if record.get("driver_id") else [])
        for session in sessions:
            is_test_by_session.setdefault(session, set()).add(split == "test")
            is_train_only_by_session.setdefault(session, set()).add(bool(record.get("train_only")))
    leaked_sessions = sorted(session for session, is_test in is_test_by_session.items() if len(is_test) > 1)
    if leaked_sessions:
        raise ValueError(f"Sessions in both test and train/val: {leaked_sessions}")
    # A held-out fold would validate the session's other rows while its train_only rows still train.
    mixed_sessions = sorted(session for session, flags in is_train_only_by_session.items() if len(flags) > 1)
    if mixed_sessions:
        raise ValueError(f"Sessions with both train_only and foldable sections: {mixed_sessions}")


def judged_label_masks(data_path, schema, data_bytes):
    """Per-record 0/1 list of the labels the scorer judges, with the same label mask training applies."""
    records = load_records(data_path, schema["feature_names"], schema["label_names"],
                           sample_rate_hz=schema["sample_rate_hz"], data_bytes=data_bytes)
    attach_label_masks(records, schema)
    return [[int(value) for value in record["label_mask"].tolist()] for record in records]


def summarize_groups(records, label_count, label_masks=None):
    """Group key -> section count and positive count per label, over the given labeled records.

    With label_masks (one 0/1 list per record), only judged positives count, as the metrics count them.
    """
    groups = {}
    for record_index, record in enumerate(records):
        summary = groups.setdefault(group_key(record), {"sections": 0, "positives": [0] * label_count})
        summary["sections"] += 1
        is_judged = label_masks[record_index] if label_masks is not None else [1] * label_count
        for index, label in enumerate(record["labels"]):
            summary["positives"][index] += int(label == 1 and is_judged[index] == 1)
    return groups


def assign_folds(groups, fold_count):
    """Returns group key -> fold index.

    Groups with the most positives are placed first, each into the fold that so far holds the fewest positives of
    the labels the group has, then the fewest sections. Ties go to the lower fold and the group key orders equal
    groups, so the same groups always give the same folds.
    """
    if type(fold_count) is not int or fold_count < 2:
        raise ValueError("Cross-validation needs at least 2 folds")
    if len(groups) < fold_count:
        raise ValueError(f"{len(groups)} sessions cannot fill {fold_count} folds")

    label_count = len(next(iter(groups.values()))["positives"])
    fold_positives = [[0] * label_count for _ in range(fold_count)]
    fold_sections = [0] * fold_count
    group_folds = {}
    placement_order = sorted(groups, key=lambda key: (-sum(groups[key]["positives"]), -groups[key]["sections"], key))
    for key in placement_order:
        positives = groups[key]["positives"]
        positive_labels = [index for index, count in enumerate(positives) if count]

        def placement_cost(fold):
            return sum(fold_positives[fold][index] for index in positive_labels), fold_sections[fold], fold

        fold = min(range(fold_count), key=placement_cost)
        group_folds[key] = fold
        fold_sections[fold] += groups[key]["sections"]
        for index, count in enumerate(positives):
            fold_positives[fold][index] += count
    return group_folds


def describe_folds(groups, group_folds, fold_count, label_names):
    folds = []
    for fold in range(fold_count):
        members = sorted(key for key, assigned in group_folds.items() if assigned == fold)
        positives = [sum(groups[key]["positives"][index] for key in members) for index in range(len(label_names))]
        folds.append({"fold": fold, "groups": members, "sections": sum(groups[key]["sections"] for key in members),
                      "positives": dict(zip(label_names, positives))})
    return folds


def write_fold_data(records, group_folds, fold, path):
    """Writes the pooled records with the fold's sessions as val and the rest, including train_only, as train."""
    with open(path, "w", encoding="utf-8", newline="\n") as output_file:
        for record in records:
            is_fold_val = not record.get("train_only") and group_folds[group_key(record)] == fold
            split = "val" if is_fold_val else "train"
            output_file.write(json.dumps(dict(record, split=split), ensure_ascii=False) + "\n")


def train_fold(training_args, data_path, checkpoint_path, seed):
    """Trains one fresh model on the fold data and returns its result."""
    args = copy.copy(training_args)
    args.data = str(data_path)
    args.output = str(checkpoint_path)
    args.seed = seed
    # Derived from the seed, so each seed also draws different augmented copies.
    args.augmentation_seed = None
    with warnings.catch_warnings():
        # Test sections are left out on purpose; the fold's val is the evaluation here.
        warnings.filterwarnings("ignore", message=NO_TEST_SPLIT_WARNING)
        return LSTMBaseline(args.device).fit(args)


def mean_and_std(values):
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return mean, math.sqrt(variance)


def classification_scores(true_positive, false_positive, false_negative):
    """Counts with precision, recall and F1; a score whose denominator is zero is None (nothing flagged, no positives)."""
    flagged = true_positive + false_positive
    positives = true_positive + false_negative
    return {"true_positive": true_positive, "false_positive": false_positive, "false_negative": false_negative,
            "precision": true_positive / flagged if flagged else None,
            "recall": true_positive / positives if positives else None,
            "f1": 2 * true_positive / (flagged + positives) if flagged + positives else None}


def format_score(score):
    return "n/a" if score is None else f"{score:.3f}"


def pooled_label_counts(runs, epoch_index):
    """Per-label TP/FP/FN of the val sections at one epoch, summed over the runs."""
    label_count = len(runs[0]["epoch_history"][epoch_index]["val_label_counts"])
    totals = [dict.fromkeys(COUNT_KEYS, 0) for _ in range(label_count)]
    for run in runs:
        for total, counts in zip(totals, run["epoch_history"][epoch_index]["val_label_counts"]):
            for key in COUNT_KEYS:
                total[key] += counts[key]
    return totals


def pooled_micro_scores(runs, epoch_index):
    label_totals = pooled_label_counts(runs, epoch_index)
    return classification_scores(**{key: sum(total[key] for total in label_totals) for key in COUNT_KEYS})


def summarize_runs(runs, label_names):
    """Averages the val BCE curves of all runs and picks the epoch count with the lowest mean.

    Every run weighs the same regardless of its fold size. The per-run best epoch BCE is reported too, but it is
    chosen on the same val it is measured on, so it is optimistic.
    F1, precision and recall use probability 0.5 and pool the TP/FP/FN counts of all runs before dividing, because
    a fold holds only a few positives; tuning thresholds on the fold's own val would make them optimistic.
    """
    epoch_count = len(runs[0]["epoch_history"])
    if any(len(run["epoch_history"]) != epoch_count for run in runs):
        raise ValueError("Every run must train the same number of epochs")

    mean_val_by_epoch, std_val_by_epoch, mean_train_by_epoch = [], [], []
    for epoch_index in range(epoch_count):
        val_mean, val_std = mean_and_std([run["epoch_history"][epoch_index]["val_bce"] for run in runs])
        mean_val_by_epoch.append(val_mean)
        std_val_by_epoch.append(val_std)
        mean_train_by_epoch.append(mean_and_std([run["epoch_history"][epoch_index]["train_bce"] for run in runs])[0])
    selected_index = min(range(epoch_count), key=lambda index: (mean_val_by_epoch[index], index))

    runs_by_fold = {}
    for run in runs:
        runs_by_fold.setdefault(run["fold"], []).append(run)
    selected_label_scores = [classification_scores(**counts) for counts in pooled_label_counts(runs, selected_index)]
    return {
        "selected_epochs": selected_index + 1,
        "selected_mean_val_bce": mean_val_by_epoch[selected_index],
        "selected_std_val_bce": std_val_by_epoch[selected_index],
        "selected_val_bce_by_fold": {fold: mean_and_std([run["epoch_history"][selected_index]["val_bce"] for run in fold_runs])[0]
                                     for fold, fold_runs in sorted(runs_by_fold.items())},
        "selected_val_micro": pooled_micro_scores(runs, selected_index),
        "selected_val_per_label": dict(zip(label_names, selected_label_scores)),
        "selected_val_micro_f1_by_fold": {fold: pooled_micro_scores(fold_runs, selected_index)["f1"]
                                          for fold, fold_runs in sorted(runs_by_fold.items())},
        "mean_prevalence_val_bce": mean_and_std([run["prevalence_val_bce"] for run in runs])[0],
        "mean_best_epoch_val_bce": mean_and_std([min(entry["val_bce"] for entry in run["epoch_history"]) for run in runs])[0],
        "mean_val_bce_by_epoch": mean_val_by_epoch,
        "std_val_bce_by_epoch": std_val_by_epoch,
        "mean_train_bce_by_epoch": mean_train_by_epoch,
        "pooled_val_micro_f1_by_epoch": [pooled_micro_scores(runs, epoch_index)["f1"] for epoch_index in range(epoch_count)],
    }


def cross_validate(training_args, fold_count, seeds):
    schema = json.loads(Path(training_args.schema).read_text(encoding="utf-8"))
    label_names = schema["label_names"]
    data_bytes = Path(training_args.data).read_bytes()
    all_records = [json.loads(line) for line in data_bytes.decode("utf-8").splitlines() if line.strip()]
    validate_source_records(all_records)
    # load_records skips the same blank lines, so the masks line up with all_records.
    all_label_masks = judged_label_masks(training_args.data, schema, data_bytes)
    is_pooled = [record["split"] != "test" for record in all_records]
    records = [record for record, pooled in zip(all_records, is_pooled) if pooled]
    label_masks = [mask for mask, pooled in zip(all_label_masks, is_pooled) if pooled]
    if any(record.get("label_names") != label_names for record in records):
        raise ValueError("Every record must use the schema's label_names")

    # train_only sections (deduction_labels.py --train-only) train every fold and are never validated on.
    is_foldable = [not record.get("train_only") for record in records]
    foldable_records = [record for record, foldable in zip(records, is_foldable) if foldable]
    foldable_masks = [mask for mask, foldable in zip(label_masks, is_foldable) if foldable]
    groups = summarize_groups(foldable_records, len(label_names), foldable_masks)
    group_folds = assign_folds(groups, fold_count)
    folds = describe_folds(groups, group_folds, fold_count, label_names)
    for fold in folds:
        LOGGER.info("fold %d: %d sessions, %d sections, positives %s", fold["fold"], len(fold["groups"]), fold["sections"],
                    {name: count for name, count in fold["positives"].items() if count})

    runs = []
    run_count = fold_count * len(seeds)
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="drive-cv-") as work_directory:
        data_path = Path(work_directory) / "fold.jsonl"
        checkpoint_path = Path(work_directory) / "fold.pt"
        for fold in range(fold_count):
            write_fold_data(records, group_folds, fold, data_path)
            for seed in seeds:
                # Marks where the epoch lines of each run start when --verbose logs them.
                LOGGER.info("run %d/%d fold %d seed %d: training", len(runs) + 1, run_count, fold, seed)
                result = train_fold(training_args, data_path, checkpoint_path, seed)
                runs.append({"fold": fold, "seed": seed, "best_epoch": result["best_epoch"],
                             "prevalence_val_bce": result["prevalence_baseline"]["val"]["bce"],
                             "epoch_history": result["epoch_history"]})
                elapsed = time.perf_counter() - started
                best_epoch_scores = pooled_micro_scores(runs[-1:], result["best_epoch"] - 1)
                LOGGER.info("run %d/%d fold %d seed %d: best epoch %d val_bce %.4f (prevalence %.4f) val_f1 %s  elapsed %s  eta %s",
                            len(runs), run_count, fold, seed, result["best_epoch"],
                            min(entry["val_bce"] for entry in result["epoch_history"]),
                            result["prevalence_baseline"]["val"]["bce"], format_score(best_epoch_scores["f1"]),
                            format_duration(elapsed), format_duration(elapsed / len(runs) * (run_count - len(runs))))

    summary = summarize_runs(runs, label_names)
    micro = summary["selected_val_micro"]
    LOGGER.info("selected %d epochs: mean val_bce %.4f +- %.4f (prevalence %.4f), pooled val F1 %s precision %s recall %s",
                summary["selected_epochs"], summary["selected_mean_val_bce"], summary["selected_std_val_bce"],
                summary["mean_prevalence_val_bce"], format_score(micro["f1"]), format_score(micro["precision"]),
                format_score(micro["recall"]))
    for name, scores in summary["selected_val_per_label"].items():
        if scores["true_positive"] + scores["false_negative"] + scores["false_positive"]:
            LOGGER.info("  %-26s F1 %s precision %s recall %s (TP %d FP %d FN %d)", name, format_score(scores["f1"]),
                        format_score(scores["precision"]), format_score(scores["recall"]), scores["true_positive"],
                        scores["false_positive"], scores["false_negative"])
    settings = {key: value for key, value in vars(training_args).items() if key not in ("command", "output", "seed", "augmentation_seed")}
    return {"settings": settings, "data_sha256": hashlib.sha256(data_bytes).hexdigest(),
            "excluded_test_sections": len(all_records) - len(records),
            "train_only_sections": len(records) - len(foldable_records), "fold_count": fold_count, "seeds": list(seeds),
            "label_names": label_names, "folds": folds, "summary": summary, "runs": runs}


def parse_cross_validation_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, allow_abbrev=False, formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Other options (--config, --hidden-size, --learning-rate, --epochs, ...) are passed to `baseline.py train`.")
    parser.add_argument("--folds", type=int, default=DEFAULT_FOLD_COUNT, help="Number of session-grouped folds")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS), help="Training seeds per fold")
    parser.add_argument("--results", required=True, help="Cross-validation result JSON path")
    parser.add_argument("--verbose", action="store_true", help="Also log every epoch of every run")
    args, training_options = parser.parse_known_args(argv)
    if args.folds < 2:
        parser.error("--folds must be at least 2")
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("--seeds must be distinct; a repeated seed only reweights the same run")
    if any(is_reserved_option(option) for option in training_options):
        parser.error("--seed and --output are set per run; use --seeds instead")
    # The checkpoint of each run is temporary, so a placeholder satisfies the train parser.
    training_args = parse_args(["train", *training_options, "--output", "cross-validation.pt"])
    input_paths = [training_args.data, training_args.schema, training_args.config]
    if "--augmentation" in training_options:
        input_paths.append(training_options[training_options.index("--augmentation") + 1])
    if Path(args.results).resolve() in {Path(path).resolve() for path in input_paths if path}:
        parser.error("--results must not overwrite the input data, schema, config or augmentation file")
    return args, training_args


def is_reserved_option(option, reserved_options=RESERVED_TRAINING_OPTIONS):
    """True for a reserved option (--seed/--output by default) and the abbreviations the train parser would expand to it."""
    name = option.split("=")[0]
    return name.startswith("--") and len(name) > 2 and any(reserved.startswith(name) for reserved in reserved_options)


def main(argv=None):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S", stream=sys.stderr)
    args, training_args = parse_cross_validation_args(argv)
    logging.getLogger("drive_lstm").setLevel(logging.INFO if args.verbose else logging.WARNING)
    result = cross_validate(training_args, args.folds, args.seeds)
    results_path = Path(args.results)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    main()
