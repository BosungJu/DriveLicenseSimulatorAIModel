"""Labels Unity training sections with the official deductions of the same session.

Input  (session folders): training_sections.jsonl (TrainingDataCollector), section_results.json (ReportManager).
A session recorded without TrainingDataCollector is read from its driving_log.csv instead.
An OBS 2.1 package run (runs/<id>/ with label.json and the Unity session in reports/) contributes only the sections
of its label.json scope, under its run folder name.
Output: one merged JSONL of data contract v1 training records with split, label_names and labels.
Sessions given as train-only always get the train split and are left out of the val/test split assignment.

A label is 1 when the official scorer deducted or disqualified for that category in that section, and 0 when it
did not. Only deduction reasons the model can observe in the recorded features are labels; the other reasons are
listed with the missing information. Every DrivingDeductionReason of the Unity simulator (DrivingScoreSheet.cs)
must appear in exactly one table, so a new or renamed reason stops the labeling instead of becoming a negative.
"""

import argparse
import json
from pathlib import Path
import random

if __package__:
    from .session_predict import (DRIVING_LOG_FILE, SECTION_RESULT_FILE, build_section_records, read_driving_log,
                                  write_jsonl)
else:
    from session_predict import (DRIVING_LOG_FILE, SECTION_RESULT_FILE, build_section_records, read_driving_log,
                                 write_jsonl)

TRAINING_SECTION_FILE = "training_sections.jsonl"
# OBS 2.1 package layout: <package>/runs/<run id>/{label.json, reports/<Unity session files>}.
OBS_RUNS_DIR = "runs"
OBS_LABEL_FILE = "label.json"
OBS_REPORT_DIR = "reports"
# driving_log.csv columns before the model features (Unity DrivingFeatureColumns): time, route section and pose.
DRIVING_LOG_METADATA_COLUMNS = ("time_s", "point_number", "heading_deg", "pos_x", "pos_z")
# Unity TrainingDataCollector writes sample_rate_hz with "0.###", so 50.00002 Hz becomes 50.
SAMPLE_RATE_DECIMALS = 3
DEFAULT_VAL_RATIO = 0.2
DEFAULT_TEST_RATIO = 0.0
DEFAULT_SPLIT_SEED = 42

# Label name -> Unity DrivingDeductionReason names. The order is the label order of the dataset and schema.
LABEL_REASONS = {
    "stop_failure": ("HillNotStopped", "EmergencyStopLate"),
    "hill_rollback": ("HillRollback", "HillRollbackOver"),
    "over_speed": ("OverSpeed", "AccelerationEntryTooFast", "AccelerationExitTooFast"),
    "insufficient_acceleration": ("AccelerationTooSlow",),
    "section_time_over": ("StartTimeOver", "HillPassTimeOver"),
    "turn_signal_missing": ("TurnSignalMissing", "StartSignalMissing", "FinishSignalMissing",
                            "StartTurnSignalNotOperated"),
    "hazard_light_error": ("EmergencyHazardLate", "EmergencyHazardNotOff", "EmergencyHazardEarly"),
    "vehicle_device_error": ("SeatBeltNotFastened", "SeatBeltNotWorn", "EngineNotStarted", "HeadlightNotOperated",
                             "EngineStalled", "GearNotOperated"),
}

# Deduction reasons kept out of the labels, with what the section features lack to observe them.
UNOBSERVABLE_REASONS = {
    "LaneKeepingLineTouch": "lane lines and vehicle position are not model features",
    "ParkingLineTouch": "parking lines and vehicle position are not model features",
    "SignalViolation": "the traffic light state is not a model feature",
    "SignalLongStop": "waiting on a red light is exempt, and the traffic light state is not a model feature",
    "ParkingTimeOver": "judged over the whole T course, which spans several sections",
    "ParkingBrakeNotApplied": "judged over the whole T course, which spans several sections",
    "CourseTimeOver": "judged on the whole course time, not on one section",
    "WiperNotOperated": "the wiper state is not recorded",
}

LABEL_NAMES = list(LABEL_REASONS)
REASON_LABELS = {reason: label for label, reasons in LABEL_REASONS.items() for reason in reasons}

# Reasons Unity records through DrivingDeductionRecorder.Disqualify. DrivingScoreSheet keeps only the first
# disqualification, while ordinary deductions keep being recorded after it.
DISQUALIFYING_REASONS = frozenset({"StartTimeOver", "HillNotStopped", "HillRollbackOver", "HillPassTimeOver",
                                   "SeatBeltNotWorn", "SignalViolation"})
# From the disqualified section on, a 0 of these labels is unknown: their disqualifying fault could go unrecorded.
DISQUALIFIABLE_LABELS = [label for label in LABEL_NAMES if DISQUALIFYING_REASONS & set(LABEL_REASONS[label])]

# FinishSectionJudge deducts on entering the finish point, which completes the course before any sample is
# recorded there. It judges the approach section's turn signal, so the label belongs to the section before it.
APPROACH_SECTION_REASONS = frozenset({"FinishSignalMissing"})


def collect_section_labels(section_results):
    """Maps every route point number to the labels of its official deductions.

    Also returns the point numbers of the first disqualified section and every later one. The score sheet keeps only
    the first disqualification, so in those sections a later disqualifying fault leaves no record.
    """
    route_points = [section["pointNumber"] for section in section_results["sections"]]
    if len(set(route_points)) != len(route_points):
        raise ValueError("section_results.json has duplicate route point numbers")

    route_indices = {point_number: index for index, point_number in enumerate(route_points)}
    labels_by_point = {point_number: set() for point_number in route_points}
    disqualified_index = None
    for deduction in section_results["deductions"]:
        reason = deduction["reason"]
        if reason not in REASON_LABELS and reason not in UNOBSERVABLE_REASONS:
            raise ValueError(f"Unknown deduction reason {reason!r}; add it to deduction_labels.py")

        point_number = deduction["pointNumber"]
        if point_number not in route_indices:
            raise ValueError(f"Deduction {reason} refers to point {point_number}, which is not a route section")

        section_index = route_indices[point_number]
        if deduction["isDisqualification"] and (disqualified_index is None or section_index < disqualified_index):
            disqualified_index = section_index

        if reason in APPROACH_SECTION_REASONS:
            if section_index == 0:
                raise ValueError(f"Deduction {reason} at the first route point has no approach section")
            section_index -= 1

        label = REASON_LABELS.get(reason)
        if label is not None:
            labels_by_point[route_points[section_index]].add(label)

    untrusted_points = set()
    if disqualified_index is not None:
        untrusted_points = set(route_points[disqualified_index:])
    return labels_by_point, untrusted_points


def read_jsonl(path):
    with open(path, encoding="utf-8-sig") as jsonl_file:
        return [json.loads(line) for line in jsonl_file if line.strip()]


def read_session_sections(session_dir, section_results, run_id):
    """Returns the unlabeled section records of one session folder.

    Prefers training_sections.jsonl, whose records carry Unity's own run_id. Without it, builds the records from
    driving_log.csv under run_id the way Unity TrainingDataCollector does, except that an invalid sample fails the
    whole session instead of skipping its section.
    """
    training_section_path = session_dir / TRAINING_SECTION_FILE
    if training_section_path.is_file():
        return read_jsonl(training_section_path)

    log_rows = read_driving_log(session_dir / DRIVING_LOG_FILE)
    if not log_rows:
        raise ValueError(f"{run_id}: {DRIVING_LOG_FILE} has no samples")

    feature_names = [name for name in log_rows[0] if name not in DRIVING_LOG_METADATA_COLUMNS]
    sample_rate_hz = round(section_results["sampleRateHz"], SAMPLE_RATE_DECIMALS)
    try:
        return build_section_records(log_rows, feature_names, sample_rate_hz, run_id=run_id)
    except ValueError as error:
        raise ValueError(f"{run_id}: {error}") from error


def is_obs_run_dir(path):
    """An OBS 2.1 package run keeps its Unity session in reports/ and its target route scope in label.json."""
    return (path / OBS_LABEL_FILE).is_file() and is_session_dir(path / OBS_REPORT_DIR)


def read_obs_scope_points(run_dir):
    """Returns the route points of an OBS run's label.json scope [start, end); the end point is only the exit.

    The run drives from the course start to reach the scope, so the sections before it repeat nearly the same
    automated driving in every run and are not its labeled task.
    """
    label = json.loads((run_dir / OBS_LABEL_FILE).read_text(encoding="utf-8-sig"))
    scope = label.get("scope")
    is_valid_scope = (isinstance(scope, list) and len(scope) == 2
                      and all(isinstance(point, int) and not isinstance(point, bool) for point in scope)
                      and scope[0] < scope[1])
    if not is_valid_scope:
        raise ValueError(f"{run_dir.name}: {OBS_LABEL_FILE} scope must be [start, end) route points, got {scope!r}")
    return set(range(scope[0], scope[1]))


def label_session(session_dir):
    """Adds label_names/labels to the training sections of one session folder or OBS package run folder.

    From the first disqualified section on, the 0s of DISQUALIFIABLE_LABELS are listed in unjudged_labels, so
    training leaves them out; the other labels and every observed 1 are kept. An OBS run keeps only the sections
    of its scope, labeled from the deductions of the whole run.
    Returns the labeled records, the number of sections with unjudged labels, and the number of sections with a
    positive label but no training record (Unity skips sections with missing or uneven samples).
    """
    session_dir = Path(session_dir)
    run_id = session_dir.name
    scope_points = None
    if is_obs_run_dir(session_dir):
        scope_points = read_obs_scope_points(session_dir)
        session_dir = session_dir / OBS_REPORT_DIR

    section_results = json.loads((session_dir / SECTION_RESULT_FILE).read_text(encoding="utf-8-sig"))
    labels_by_point, untrusted_points = collect_section_labels(section_results)
    if scope_points is not None:
        labels_by_point = {point: labels for point, labels in labels_by_point.items() if point in scope_points}

    labeled_records = []
    recorded_points = set()
    masked_after_disqualification = 0
    for record in read_session_sections(session_dir, section_results, run_id):
        section_id = record["section_id"]
        if scope_points is not None and section_id not in scope_points:
            continue
        if section_id not in labels_by_point:
            raise ValueError(f"{run_id}: training section {section_id} is not in {SECTION_RESULT_FILE}")

        recorded_points.add(section_id)
        section_labels = labels_by_point[section_id]
        labeled_record = dict(record, label_names=list(LABEL_NAMES),
                              labels=[int(name in section_labels) for name in LABEL_NAMES])
        if scope_points is not None:
            # An OBS run is identified by its package folder whichever file its sections come from.
            labeled_record["run_id"] = run_id
        if section_id in untrusted_points:
            unjudged_labels = [label for label in DISQUALIFIABLE_LABELS if label not in section_labels]
            if unjudged_labels:
                labeled_record["unjudged_labels"] = unjudged_labels
                masked_after_disqualification += 1
        labeled_records.append(labeled_record)

    if scope_points is not None and not labeled_records:
        raise ValueError(f"{run_id}: no recorded section in its {OBS_LABEL_FILE} scope {sorted(scope_points)}")

    unrecorded_positive_sections = sum(
        1 for point_number, section_labels in labels_by_point.items()
        if section_labels and point_number not in recorded_points)
    return labeled_records, masked_after_disqualification, unrecorded_positive_sections


def split_group(record):
    """Records of one anonymous driver, or of one run without a driver_id, always share a split."""
    driver_id = record.get("driver_id")
    if driver_id:
        return ("driver", driver_id)
    return ("run", record["run_id"])


def assign_splits(records, val_ratio=DEFAULT_VAL_RATIO, test_ratio=DEFAULT_TEST_RATIO, seed=DEFAULT_SPLIT_SEED):
    """Returns copies of the records with a split assigned per driver or run group, reproducible for one seed."""
    if not 0 < val_ratio < 1 or not 0 <= test_ratio < 1 or val_ratio + test_ratio >= 1:
        raise ValueError("val_ratio must be in (0, 1), test_ratio in [0, 1), and their sum below 1")

    # Unity writes one driver_id per session, so a run with mixed driver metadata was merged or edited by hand;
    # grouping it would split the run across splits.
    run_groups = {}
    for record in records:
        if run_groups.setdefault(record["run_id"], split_group(record)) != split_group(record):
            raise ValueError(f"run_id {record['run_id']!r} has inconsistent driver_id values")

    groups = sorted({split_group(record) for record in records})
    random.Random(seed).shuffle(groups)
    val_count = max(1, round(len(groups) * val_ratio))
    test_count = max(1, round(len(groups) * test_ratio)) if test_ratio > 0 else 0
    if len(groups) - val_count - test_count < 1:
        raise ValueError(f"{len(groups)} driver/run groups cannot fill train, val and the requested test split")

    group_splits = {}
    for index, group in enumerate(groups):
        if index < test_count:
            group_splits[group] = "test"
        elif index < test_count + val_count:
            group_splits[group] = "val"
        else:
            group_splits[group] = "train"
    return [dict(record, split=group_splits[split_group(record)]) for record in records]


def is_session_dir(path):
    """A session has its deductions and either Unity training sections or the driving log to build them from."""
    has_sections = (path / TRAINING_SECTION_FILE).is_file() or (path / DRIVING_LOG_FILE).is_file()
    return has_sections and (path / SECTION_RESULT_FILE).is_file()


def is_labelable_dir(path):
    return is_session_dir(path) or is_obs_run_dir(path)


def find_session_dirs(paths):
    """Accepts session or OBS run folders, folders (such as AIReport) whose direct subfolders are sessions, and
    OBS package folders whose runs/ subfolders are runs. A folder found more than once is returned once."""
    session_dirs = []
    for path in map(Path, paths):
        # An OBS run's reports/ folder is a plain session; label it as its run so the scope and run_id apply.
        # samefile also matches other spellings of the folder, such as REPORTS on a case-insensitive file system.
        if is_obs_run_dir(path.parent) and path.is_dir() and path.samefile(path.parent / OBS_REPORT_DIR):
            path = path.parent
        if is_labelable_dir(path):
            session_dirs.append(path)
            continue

        child_sessions = []
        for parent in (path, path / OBS_RUNS_DIR):
            if parent.is_dir():
                child_sessions.extend(sorted(child for child in parent.iterdir()
                                             if child.is_dir() and is_labelable_dir(child)))
        if not child_sessions:
            raise FileNotFoundError(f"No session ({SECTION_RESULT_FILE} with {TRAINING_SECTION_FILE} or "
                                    f"{DRIVING_LOG_FILE}, or an OBS run with {OBS_LABEL_FILE}) in {path}, "
                                    f"its direct subfolders or its {OBS_RUNS_DIR} subfolders")
        session_dirs.extend(child_sessions)

    unique_session_dirs = {}
    for session_dir in session_dirs:
        unique_session_dirs.setdefault(session_dir.resolve(), session_dir)
    return list(unique_session_dirs.values())


def label_sessions(session_dirs):
    """Returns the labeled records of the sessions and the summed unjudged / unrecorded positive section counts."""
    labeled_records = []
    masked_after_disqualification = 0
    unrecorded_positive_sections = 0
    for session_dir in session_dirs:
        session_records, session_masked, session_unrecorded = label_session(session_dir)
        labeled_records.extend(session_records)
        masked_after_disqualification += session_masked
        unrecorded_positive_sections += session_unrecorded
    return labeled_records, masked_after_disqualification, unrecorded_positive_sections


def find_shared_split_groups(split_records, train_only_records):
    """Returns the run/driver groups present on both sides, which would leak a train-only run into val or test."""
    split_groups = {split_group(record) for record in split_records}
    split_run_ids = {record["run_id"] for record in split_records}
    shared_groups = set()
    for record in train_only_records:
        group = split_group(record)
        if group in split_groups or record["run_id"] in split_run_ids:
            shared_groups.add(group)
    return shared_groups


def count_positives(records):
    positives = {}
    for record in records:
        split_positives = positives.setdefault(record["split"], dict.fromkeys(LABEL_NAMES, 0))
        for name, value in zip(LABEL_NAMES, record["labels"]):
            split_positives[name] += value
    return positives


def build_dataset(paths, output, val_ratio=DEFAULT_VAL_RATIO, test_ratio=DEFAULT_TEST_RATIO,
                  seed=DEFAULT_SPLIT_SEED, train_only_paths=()):
    """Labels the sessions of paths into train/val(/test) and those of train_only_paths into train only.

    Train-only sessions are for sources without independent evaluation data, such as automated runs of one
    controller, so val/test keep measuring the split sessions. They do not change the split of the other sessions.
    """
    session_dirs = find_session_dirs(paths)
    train_only_dirs = find_session_dirs(train_only_paths) if train_only_paths else []
    overlapping_dirs = {path.resolve() for path in session_dirs} & {path.resolve() for path in train_only_dirs}
    if overlapping_dirs:
        raise ValueError(f"Sessions given both for splitting and as train-only: {sorted(map(str, overlapping_dirs))}")

    labeled_records, masked_after_disqualification, unrecorded_positive_sections = label_sessions(session_dirs)
    train_only_records, train_only_masked, train_only_unrecorded = label_sessions(train_only_dirs)
    if not labeled_records:
        raise ValueError("No labeled training sections were produced")

    shared_groups = find_shared_split_groups(labeled_records, train_only_records)
    if shared_groups:
        raise ValueError(f"Train-only sessions share run/driver groups with split sessions: {sorted(shared_groups)}")

    split_records = assign_splits(labeled_records, val_ratio, test_ratio, seed)
    # train_only tells cross_validate.py to keep these sections out of its val folds as well.
    split_records.extend(dict(record, split="train", train_only=True) for record in train_only_records)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(output, split_records)
    return {
        "output": str(output),
        "session_count": len(session_dirs) + len(train_only_dirs),
        "train_only_session_count": len(train_only_dirs),
        "section_count": len(split_records),
        "train_only_section_count": len(train_only_records),
        "masked_after_disqualification": masked_after_disqualification + train_only_masked,
        "unrecorded_positive_sections": unrecorded_positive_sections + train_only_unrecorded,
        "label_names": LABEL_NAMES,
        "positives_by_split": count_positives(split_records),
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Label Unity training sections with official deductions")
    parser.add_argument("sessions", nargs="+", help="Session folders, or folders whose subfolders are sessions")
    parser.add_argument("--output", required=True, help="Merged labeled JSONL to write (overwritten)")
    parser.add_argument("--val-ratio", type=float, default=DEFAULT_VAL_RATIO)
    parser.add_argument("--test-ratio", type=float, default=DEFAULT_TEST_RATIO)
    parser.add_argument("--seed", type=int, default=DEFAULT_SPLIT_SEED, help="Seed of the driver/run split order")
    parser.add_argument("--train-only", nargs="+", action="extend", default=[], metavar="PATH",
                        help="Session/OBS folders always put in train, e.g. automated runs without independent "
                             "val/test data")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    summary = build_dataset(args.sessions, args.output, args.val_ratio, args.test_ratio, args.seed,
                            train_only_paths=args.train_only)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
