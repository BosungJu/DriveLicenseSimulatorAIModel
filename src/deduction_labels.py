"""Labels Unity training sections with the official deductions of the same session.

Input  (session folders): training_sections.jsonl (TrainingDataCollector), section_results.json (ReportManager).
A session recorded without TrainingDataCollector is read from its driving_log.csv instead.
Output: one merged JSONL of data contract v1 training records with split, label_names and labels.

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


def read_session_sections(session_dir, section_results):
    """Returns the unlabeled section records of one session folder.

    Prefers training_sections.jsonl. Without it, builds the records from driving_log.csv the way Unity
    TrainingDataCollector does, except that an invalid sample fails the whole session instead of skipping its section.
    """
    training_section_path = session_dir / TRAINING_SECTION_FILE
    if training_section_path.is_file():
        return read_jsonl(training_section_path)

    log_rows = read_driving_log(session_dir / DRIVING_LOG_FILE)
    if not log_rows:
        raise ValueError(f"{session_dir.name}: {DRIVING_LOG_FILE} has no samples")

    feature_names = [name for name in log_rows[0] if name not in DRIVING_LOG_METADATA_COLUMNS]
    sample_rate_hz = round(section_results["sampleRateHz"], SAMPLE_RATE_DECIMALS)
    try:
        return build_section_records(log_rows, feature_names, sample_rate_hz, run_id=session_dir.name)
    except ValueError as error:
        raise ValueError(f"{session_dir.name}: {error}") from error


def label_session(session_dir):
    """Adds label_names/labels to the training sections of one session folder.

    From the first disqualified section on, the 0s of DISQUALIFIABLE_LABELS are listed in unjudged_labels, so
    training leaves them out; the other labels and every observed 1 are kept.
    Returns the labeled records, the number of sections with unjudged labels, and the number of sections with a
    positive label but no training record (Unity skips sections with missing or uneven samples).
    """
    session_dir = Path(session_dir)
    section_results = json.loads((session_dir / SECTION_RESULT_FILE).read_text(encoding="utf-8-sig"))
    labels_by_point, untrusted_points = collect_section_labels(section_results)

    labeled_records = []
    recorded_points = set()
    masked_after_disqualification = 0
    for record in read_session_sections(session_dir, section_results):
        section_id = record["section_id"]
        if section_id not in labels_by_point:
            raise ValueError(f"{session_dir.name}: training section {section_id} is not in {SECTION_RESULT_FILE}")

        recorded_points.add(section_id)
        section_labels = labels_by_point[section_id]
        labeled_record = dict(record, label_names=list(LABEL_NAMES),
                              labels=[int(name in section_labels) for name in LABEL_NAMES])
        if section_id in untrusted_points:
            unjudged_labels = [label for label in DISQUALIFIABLE_LABELS if label not in section_labels]
            if unjudged_labels:
                labeled_record["unjudged_labels"] = unjudged_labels
                masked_after_disqualification += 1
        labeled_records.append(labeled_record)

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


def find_session_dirs(paths):
    """Accepts session folders, or folders (such as AIReport) whose direct subfolders are sessions."""
    session_dirs = []
    for path in map(Path, paths):
        if is_session_dir(path):
            session_dirs.append(path)
            continue

        child_sessions = []
        if path.is_dir():
            child_sessions = sorted(child for child in path.iterdir() if child.is_dir() and is_session_dir(child))
        if not child_sessions:
            raise FileNotFoundError(f"No session ({SECTION_RESULT_FILE} with {TRAINING_SECTION_FILE} or "
                                    f"{DRIVING_LOG_FILE}) in {path} or its direct subfolders")
        session_dirs.extend(child_sessions)
    return session_dirs


def count_positives(records):
    positives = {}
    for record in records:
        split_positives = positives.setdefault(record["split"], dict.fromkeys(LABEL_NAMES, 0))
        for name, value in zip(LABEL_NAMES, record["labels"]):
            split_positives[name] += value
    return positives


def build_dataset(paths, output, val_ratio=DEFAULT_VAL_RATIO, test_ratio=DEFAULT_TEST_RATIO,
                  seed=DEFAULT_SPLIT_SEED):
    session_dirs = find_session_dirs(paths)
    labeled_records = []
    masked_after_disqualification = 0
    unrecorded_positive_sections = 0
    for session_dir in session_dirs:
        session_records, session_masked, session_unrecorded = label_session(session_dir)
        labeled_records.extend(session_records)
        masked_after_disqualification += session_masked
        unrecorded_positive_sections += session_unrecorded

    if not labeled_records:
        raise ValueError("No labeled training sections were produced")

    split_records = assign_splits(labeled_records, val_ratio, test_ratio, seed)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(output, split_records)
    return {
        "output": str(output),
        "session_count": len(session_dirs),
        "section_count": len(split_records),
        "masked_after_disqualification": masked_after_disqualification,
        "unrecorded_positive_sections": unrecorded_positive_sections,
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
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    summary = build_dataset(args.sessions, args.output, args.val_ratio, args.test_ratio, args.seed)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
