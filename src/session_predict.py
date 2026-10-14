"""Runs the section LSTM on one Unity driving session exported by ReportManager.

Input  (session folder): driving_log.csv (one row per physics step), section_results.json (deductions).
Output (session folder): sections.jsonl (inference records, data contract v1), predictions.json.
"""

import argparse
import csv
import json
import math
from pathlib import Path

if __package__:
    from .baseline import LSTMBaseline
else:
    from baseline import LSTMBaseline

DRIVING_LOG_FILE = "driving_log.csv"
SECTION_RESULT_FILE = "section_results.json"
SECTION_RECORD_FILE = "sections.jsonl"
PREDICTION_FILE = "predictions.json"
POINT_NUMBER_COLUMN = "point_number"
# Unity writes 1 / Time.fixedDeltaTime as float, so 50 Hz may arrive as 49.99999.
SAMPLE_RATE_TOLERANCE_HZ = 0.01
DEFAULT_LABEL_THRESHOLD = 0.5
CONTROL_INPUT_RANGES = {"steering_input": (-1.0, 1.0), "throttle_input": (-1.0, 1.0)}
# Unity exports time_s to three decimals, giving at most 0.001 seconds of interval rounding.
SAMPLE_TIME_TOLERANCE_SECONDS = 0.0011


def read_driving_log(path):
    with open(path, newline="", encoding="utf-8-sig") as log_file:
        return list(csv.DictReader(log_file))


def build_section_records(log_rows, feature_names, sample_rate_hz, run_id):
    """Splits the log into consecutive point_number runs and keeps only the checkpoint's features, in order."""
    if not log_rows:
        raise ValueError("Driving log has no samples")
    missing_features = [name for name in feature_names if name not in log_rows[0]]
    if missing_features:
        raise ValueError(f"Driving log is missing checkpoint features: {missing_features}")

    records = []
    previous_sample_time = None
    uses_control_inputs = any(name in CONTROL_INPUT_RANGES or name == "brake_applied" for name in feature_names)
    for row in log_rows:
        section_id = int(row[POINT_NUMBER_COLUMN])
        unavailable_features = [name for name in feature_names if row.get(name) is None or not str(row[name]).strip()]
        if unavailable_features:
            raise ValueError(f"Unavailable checkpoint features at section {section_id}: {unavailable_features}")
        if uses_control_inputs:
            try:
                sample_time = float(row["time_s"])
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError("Control input samples require numeric time_s") from error
            if not math.isfinite(sample_time) or sample_time < 0:
                raise ValueError("Sample time_s must be finite and nonnegative")
            if previous_sample_time is not None:
                interval = sample_time - previous_sample_time
                if interval <= 0 or abs(interval - 1 / sample_rate_hz) > SAMPLE_TIME_TOLERANCE_SECONDS:
                    raise ValueError("Control input samples have a time gap or do not match the fixed sample rate")
            previous_sample_time = sample_time
        frame = [float(row[name]) for name in feature_names]
        if not all(math.isfinite(value) for value in frame):
            raise ValueError(f"Non-finite feature value at section {section_id}")
        for name, value in zip(feature_names, frame):
            if name in CONTROL_INPUT_RANGES:
                minimum, maximum = CONTROL_INPUT_RANGES[name]
                if not minimum <= value <= maximum:
                    raise ValueError(f"Control input {name} at section {section_id} must be in [{minimum}, {maximum}]")
            if name == "brake_applied" and value not in (0.0, 1.0):
                raise ValueError(f"brake_applied at section {section_id} must be 0 or 1")

        if not records or records[-1]["section_id"] != section_id:
            records.append({"schema_version": 1, "sample_rate_hz": sample_rate_hz, "run_id": run_id,
                            "section_id": section_id, "feature_names": list(feature_names), "features": []})
        records[-1]["features"].append(frame)
    return records


def write_jsonl(path, records):
    with open(path, "w", encoding="utf-8") as jsonl_file:
        for record in records:
            jsonl_file.write(json.dumps(record, ensure_ascii=False) + "\n")


def summarize_prediction(prediction, threshold):
    """Adds the labels at or above the threshold; a section without any flagged label is treated as normal."""
    flagged_labels = [label for label, probability in prediction["probabilities"].items() if probability >= threshold]
    return dict(prediction, threshold=threshold, flagged_labels=flagged_labels, is_normal=not flagged_labels)


def predict_session(session_dir, checkpoint_path, device="auto", threshold=DEFAULT_LABEL_THRESHOLD):
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("threshold must be a finite probability between 0 and 1")

    session_dir = Path(session_dir)
    baseline = LSTMBaseline.load(checkpoint_path, device=device)
    # The checkpoint schema fixes feature order and sample rate for inference.
    schema = baseline.schema

    section_results = json.loads((session_dir / SECTION_RESULT_FILE).read_text(encoding="utf-8-sig"))
    recorded_rate_hz = section_results["sampleRateHz"]
    if type(recorded_rate_hz) not in (int, float) or not math.isfinite(recorded_rate_hz) or recorded_rate_hz <= 0:
        raise ValueError("Session sample rate must be finite and positive")
    if abs(recorded_rate_hz - schema["sample_rate_hz"]) > SAMPLE_RATE_TOLERANCE_HZ:
        raise ValueError(f"Session sample rate {recorded_rate_hz} Hz does not match checkpoint {schema['sample_rate_hz']} Hz")

    log_rows = read_driving_log(session_dir / DRIVING_LOG_FILE)
    records = build_section_records(log_rows, schema["feature_names"], schema["sample_rate_hz"], run_id=session_dir.name)
    record_path = session_dir / SECTION_RECORD_FILE
    write_jsonl(record_path, records)

    predictions = [summarize_prediction(prediction, threshold) for prediction in baseline.predict(record_path)]
    result = {"run_id": session_dir.name, "label_names": schema["label_names"], "threshold": threshold,
              "sections": predictions}
    (session_dir / PREDICTION_FILE).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Predict improvement labels for each section of a Unity session")
    parser.add_argument("session_dir", help="Session folder exported by ReportManager")
    parser.add_argument("--checkpoint", required=True, help="LSTM checkpoint (.pt) saved by LSTMBaseline")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--threshold", type=float, default=DEFAULT_LABEL_THRESHOLD,
                        help="Probability at or above which a label is flagged")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    result = predict_session(args.session_dir, args.checkpoint, args.device, args.threshold)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
