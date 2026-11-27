"""Grid search of the LSTM size and learning rate on the fixed train/val split.

Input: the training config of `baseline.py train` with its CLI overrides, and the grid values (--hidden-sizes,
--layer-counts, --learning-rates).
Processing: every grid combination trains a fresh model with the same seed, data and split, keeping its lowest val BCE
epoch. Checkpoints are temporary. A trial already in the results file with the same base settings and data is not
trained again, so an interrupted sweep resumes.
Output: JSON with each trial's settings, best epoch, val BCE / micro F1 and epoch curve, and the trials ranked by val
BCE. Test metrics are left out on purpose: choose on val, then evaluate test once with the final settings.
"""

import argparse
import copy
import hashlib
import itertools
import json
import logging
from pathlib import Path
import sys
import tempfile
import time

if __package__:
    from .baseline import LSTMBaseline, format_duration, parse_args
    from .cross_validate import is_reserved_option
else:
    from baseline import LSTMBaseline, format_duration, parse_args
    from cross_validate import is_reserved_option

LOGGER = logging.getLogger("drive_sweep")
DEFAULT_HIDDEN_SIZES = (32, 64, 128)
DEFAULT_LAYER_COUNTS = (1, 2)
DEFAULT_LEARNING_RATES = (0.0005, 0.001, 0.002)
# Set per trial by the grid, or per trial as a temporary checkpoint.
RESERVED_TRAINING_OPTIONS = ("--hidden-size", "--num-layers", "--learning-rate", "--output")
# Base settings that differ per trial or only name files, so they are not part of the resume match.
TRIAL_SETTING_KEYS = ("command", "output", "hidden_size", "num_layers", "learning_rate")


def trial_name(hidden_size, num_layers, learning_rate):
    return f"h{hidden_size}-l{num_layers}-lr{learning_rate:g}"


def build_grid(hidden_sizes, layer_counts, learning_rates):
    """Returns every (hidden_size, num_layers, learning_rate) combination in the given value order."""
    return list(itertools.product(hidden_sizes, layer_counts, learning_rates))


def base_settings(training_args):
    """The settings shared by every trial, as they read back from JSON (tuples become lists) for the resume match."""
    settings = {key: value for key, value in vars(training_args).items() if key not in TRIAL_SETTING_KEYS}
    return json.loads(json.dumps(settings))


def load_completed_trials(results_path, settings, data_sha256):
    """Returns the trials of an earlier run with the same settings and data, keyed by name; none without a file."""
    if not results_path.is_file():
        return {}

    previous = json.loads(results_path.read_text(encoding="utf-8"))
    if previous.get("settings") != settings or previous.get("data_sha256") != data_sha256:
        raise ValueError(f"{results_path} holds a sweep with other settings or data; pass another --results path")
    return {trial["name"]: trial for trial in previous.get("trials", [])}


def train_trial(training_args, hidden_size, num_layers, learning_rate, checkpoint_path):
    """Trains one fresh model with the trial's size and learning rate and returns its training result."""
    args = copy.copy(training_args)
    args.hidden_size = hidden_size
    args.num_layers = num_layers
    args.learning_rate = learning_rate
    args.output = str(checkpoint_path)
    return LSTMBaseline(args.device).fit(args)


def summarize_trial(hidden_size, num_layers, learning_rate, result, elapsed_seconds):
    val_metrics = result["lstm"]["val"]
    return {
        "name": trial_name(hidden_size, num_layers, learning_rate),
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "learning_rate": learning_rate,
        "best_epoch": result["best_epoch"],
        "val_bce": val_metrics["bce"],
        "val_micro_f1": val_metrics["micro_f1"],
        "val_per_label_f1": [label["f1"] for label in val_metrics["per_label"]],
        "val_bce_by_epoch": [entry["val_bce"] for entry in result["epoch_history"]],
        "train_bce_by_epoch": [entry["train_bce"] for entry in result["epoch_history"]],
        "elapsed_seconds": round(elapsed_seconds, 1),
    }


def rank_trials(trials):
    """Names ordered by val BCE, ties by name, so the same results always give the same ranking."""
    return [trial["name"] for trial in sorted(trials, key=lambda trial: (trial["val_bce"], trial["name"]))]


def write_results(results_path, settings, data_sha256, grid, label_names, trials):
    result = {"settings": settings, "data_sha256": data_sha256,
              "grid": [trial_name(*combination) for combination in grid], "label_names": label_names,
              "trials": trials, "ranking": rank_trials(trials)}
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def run_sweep(training_args, grid, results_path):
    results_path = Path(results_path)
    settings = base_settings(training_args)
    data_sha256 = hashlib.sha256(Path(training_args.data).read_bytes()).hexdigest()
    schema = json.loads(Path(training_args.schema).read_text(encoding="utf-8"))
    completed = load_completed_trials(results_path, settings, data_sha256)
    # Finished trials outside this grid stay in the file and the ranking instead of being dropped.
    grid_names = {trial_name(*combination) for combination in grid}
    other_trials = [trial for name, trial in completed.items() if name not in grid_names]

    trials = []
    started = time.perf_counter()
    trained_count = 0
    with tempfile.TemporaryDirectory(prefix="drive-sweep-") as work_directory:
        checkpoint_path = Path(work_directory) / "trial.pt"
        for index, (hidden_size, num_layers, learning_rate) in enumerate(grid, 1):
            name = trial_name(hidden_size, num_layers, learning_rate)
            if name in completed:
                LOGGER.info("trial %d/%d %s: already in %s, skipped", index, len(grid), name, results_path)
                trials.append(completed[name])
                continue

            LOGGER.info("trial %d/%d %s: training", index, len(grid), name)
            trial_started = time.perf_counter()
            result = train_trial(training_args, hidden_size, num_layers, learning_rate, checkpoint_path)
            trial = summarize_trial(hidden_size, num_layers, learning_rate, result, time.perf_counter() - trial_started)
            trials.append(trial)
            # Saved after every trial so an interrupted sweep keeps the finished ones.
            write_results(results_path, settings, data_sha256, grid, schema["label_names"], trials + other_trials)

            trained_count += 1
            elapsed = time.perf_counter() - started
            remaining = sum(1 for combination in grid[index:] if trial_name(*combination) not in completed)
            LOGGER.info("trial %d/%d %s: best epoch %d val_bce %.4f val_f1 %.3f  elapsed %s  eta %s", index, len(grid),
                        name, trial["best_epoch"], trial["val_bce"], trial["val_micro_f1"], format_duration(elapsed),
                        format_duration(elapsed / trained_count * remaining))

    trials.extend(other_trials)
    result = write_results(results_path, settings, data_sha256, grid, schema["label_names"], trials)
    trials_by_name = {trial["name"]: trial for trial in trials}
    LOGGER.info("ranking by val BCE (test is not used for choosing):")
    for rank, name in enumerate(result["ranking"], 1):
        trial = trials_by_name[name]
        LOGGER.info("  %2d. %-16s val_bce %.4f  val_f1 %.3f  best epoch %d", rank, name, trial["val_bce"],
                    trial["val_micro_f1"], trial["best_epoch"])
    return result


def parse_positive_values(parser, option, values, value_type):
    if len(set(values)) != len(values):
        parser.error(f"{option} values must be distinct")
    if any(value <= 0 for value in values):
        parser.error(f"{option} values must be positive")
    return [value_type(value) for value in values]


def parse_sweep_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, allow_abbrev=False, formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Other options (--config, --epochs, --batch-size, ...) are passed to `baseline.py train`.")
    parser.add_argument("--hidden-sizes", type=int, nargs="+", default=list(DEFAULT_HIDDEN_SIZES))
    parser.add_argument("--layer-counts", type=int, nargs="+", default=list(DEFAULT_LAYER_COUNTS))
    parser.add_argument("--learning-rates", type=float, nargs="+", default=list(DEFAULT_LEARNING_RATES))
    parser.add_argument("--results", required=True, help="Sweep result JSON path; an existing one is resumed")
    parser.add_argument("--quiet", action="store_true", help="Log only trial results, not every epoch")
    args, training_options = parser.parse_known_args(argv)
    args.hidden_sizes = parse_positive_values(parser, "--hidden-sizes", args.hidden_sizes, int)
    args.layer_counts = parse_positive_values(parser, "--layer-counts", args.layer_counts, int)
    args.learning_rates = parse_positive_values(parser, "--learning-rates", args.learning_rates, float)
    if any(is_reserved_option(option, RESERVED_TRAINING_OPTIONS) for option in training_options):
        parser.error("--hidden-size, --num-layers and --learning-rate are set by the grid; --output is temporary")

    # The checkpoint of each trial is temporary, so a placeholder satisfies the train parser.
    training_args = parse_args(["train", *training_options, "--output", "sweep.pt"])
    input_paths = [training_args.data, training_args.schema, training_args.config]
    if Path(args.results).resolve() in {Path(path).resolve() for path in input_paths if path}:
        parser.error("--results must not overwrite the input data, schema or config")
    return args, training_args


def main(argv=None):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S", stream=sys.stderr)
    args, training_args = parse_sweep_args(argv)
    logging.getLogger("drive_lstm").setLevel(logging.WARNING if args.quiet else logging.INFO)
    grid = build_grid(args.hidden_sizes, args.layer_counts, args.learning_rates)
    return run_sweep(training_args, grid, args.results)


if __name__ == "__main__":
    main()
