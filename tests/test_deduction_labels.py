import json
import os
from pathlib import Path
import re
import tempfile
import unittest

from src import report
from src.baseline import load_records, validate_label_sections
from src.deduction_labels import (APPROACH_SECTION_REASONS, DISQUALIFIABLE_LABELS, DISQUALIFYING_REASONS,
                                  LABEL_NAMES, OBS_LABEL_FILE, OBS_REPORT_DIR, OBS_RUNS_DIR, REASON_LABELS,
                                  TRAINING_SECTION_FILE, UNOBSERVABLE_REASONS, assign_splits, build_dataset,
                                  collect_section_labels, find_session_dirs, label_session, parse_args)
from src.session_predict import DRIVING_LOG_FILE, SECTION_FEATURES, SECTION_RESULT_FILE

PROJECT_ROOT = Path(__file__).resolve().parents[1]
# Set DRIVER_LICENSE_SIMULATOR_ROOT when the Unity repository is not next to this project.
SIMULATOR_ROOT = Path(os.environ.get("DRIVER_LICENSE_SIMULATOR_ROOT", PROJECT_ROOT.parent / "DriverLicenseSimulator"))
SCORE_SHEET_PATH = SIMULATOR_ROOT / "Assets" / "Scripts" / "Level" / "DrivingScoreSheet.cs"


def read_unity_deduction_reasons():
    source = SCORE_SHEET_PATH.read_text(encoding="utf-8-sig")
    enum_body = re.search(r"public enum DrivingDeductionReason\s*\{(?P<body>[^}]*)\}", source).group("body")
    enum_body = re.sub(r"//[^\n]*", "", enum_body)
    return [name.strip() for name in enum_body.split(",") if name.strip()]


def route_section(point_number):
    return {"pointNumber": point_number, "instructionText": "", "commands": [], "sampleCount": 1}


def deduction(point_number, reason, points=5, is_disqualification=False):
    return {"pointNumber": point_number, "command": "", "reason": reason, "points": points,
            "isDisqualification": is_disqualification}


def section_results(point_numbers, deductions):
    return {"sections": [route_section(point_number) for point_number in point_numbers], "deductions": deductions}


def training_record(run_id, section_id, driver_id=None):
    record = {"schema_version": 1, "sample_rate_hz": 50, "run_id": run_id, "section_id": section_id,
              "start_time_s": 0.0, "feature_names": ["speed_kmh"], "features": [[0.0], [1.0]]}
    if driver_id is not None:
        record["driver_id"] = driver_id
    return record


def write_session(session_dir, results, records):
    session_dir.mkdir(parents=True)
    (session_dir / SECTION_RESULT_FILE).write_text(json.dumps(results), encoding="utf-8")
    lines = "".join(json.dumps(record) + "\n" for record in records)
    (session_dir / TRAINING_SECTION_FILE).write_text(lines, encoding="utf-8")


def write_log_session(session_dir, results, samples):
    """Writes a session recorded without TrainingDataCollector; samples are (time_s, point_number, speed_kmh)."""
    session_dir.mkdir(parents=True)
    (session_dir / SECTION_RESULT_FILE).write_text(json.dumps(results), encoding="utf-8")
    rows = [f"{time_s},{point_number},90.0,1.0,2.0,{speed_kmh}" for time_s, point_number, speed_kmh in samples]
    log_text = "\n".join(["time_s,point_number,heading_deg,pos_x,pos_z,speed_kmh"] + rows) + "\n"
    (session_dir / DRIVING_LOG_FILE).write_text(log_text, encoding="utf-8")


def write_obs_run(run_dir, scope, results, samples):
    """Writes an OBS 2.1 package run: label.json with its scope and the Unity session under reports/."""
    write_log_session(run_dir / OBS_REPORT_DIR, results, samples)
    (run_dir / OBS_LABEL_FILE).write_text(json.dumps({"run": run_dir.name, "scope": scope}), encoding="utf-8")


def obs_samples(point_numbers):
    """Two samples per route point at 50 Hz, so every listed point becomes one recorded section."""
    samples = []
    for point_number in point_numbers:
        for _ in range(2):
            samples.append((round(len(samples) * 0.02, 3), point_number, float(point_number)))
    return samples


class DeductionReasonSyncTests(unittest.TestCase):
    """Every table keyed by Unity DrivingDeductionReason must cover the enum exactly."""

    def setUp(self):
        if not SCORE_SHEET_PATH.is_file():
            self.skipTest(f"Unity DrivingScoreSheet.cs not found at {SCORE_SHEET_PATH}")
        self.unity_reasons = read_unity_deduction_reasons()

    def test_LabelTables_EveryUnityReason_IsLabeledOrExcludedExactlyOnce(self):
        self.assertEqual(set(REASON_LABELS) | set(UNOBSERVABLE_REASONS), set(self.unity_reasons))
        self.assertFalse(set(REASON_LABELS) & set(UNOBSERVABLE_REASONS))
        self.assertTrue(APPROACH_SECTION_REASONS <= set(REASON_LABELS))

    def test_DisqualifyingReasons_MatchUnityDisqualifyCalls(self):
        scripts_root = SIMULATOR_ROOT / "Assets" / "Scripts"
        disqualify_call = re.compile(r"Disqualify\(\s*DrivingDeductionReason\.(\w+)")
        unity_disqualifying = {match for path in scripts_root.rglob("*.cs")
                               for match in disqualify_call.findall(path.read_text(encoding="utf-8-sig"))}

        self.assertEqual(DISQUALIFYING_REASONS, unity_disqualifying)

    def test_ReportReasonNames_MatchUnityEnumInOrder(self):
        self.assertEqual(list(report.DEDUCTION_REASON_NAMES), self.unity_reasons)

    def test_ReportCategories_EveryUnityReasonBelongsToExactlyOneCategory(self):
        categorized_reasons = [reason for reasons in report.CATEGORY_REASONS.values() for reason in reasons]

        # calculate_category_scores rejects uncategorized reasons, so a gap would stop report generation.
        self.assertEqual(sorted(categorized_reasons), sorted(self.unity_reasons))


class DeductionLabelTests(unittest.TestCase):
    def test_SchemaExample_UsesDeductionLabelNames(self):
        schema = json.loads((PROJECT_ROOT / "res" / "schema.example.json").read_text(encoding="utf-8"))

        self.assertEqual(schema["label_names"], LABEL_NAMES)

    def test_SchemaExample_LabelSectionsCoverEveryLabelAndUnitySectionFeature(self):
        schema = json.loads((PROJECT_ROOT / "res" / "schema.example.json").read_text(encoding="utf-8"))

        _, label_section_indices = validate_label_sections(schema)

        self.assertEqual(schema["section_features"], list(SECTION_FEATURES))
        self.assertEqual(len(label_section_indices), len(LABEL_NAMES))

    def test_ReportInstructions_DescribeEveryLabel(self):
        for label_name in LABEL_NAMES:
            with self.subTest(label=label_name):
                self.assertIn(f"- {label_name}:", report.REPORT_INSTRUCTIONS)

    #region Section labels

    def test_CollectSectionLabels_GroupsReasonsByDeductedSection(self):
        results = section_results([1, 2, 3], [
            deduction(1, "StartSignalMissing"),
            deduction(1, "SeatBeltNotFastened"),
            deduction(2, "HillRollback", 10),
            deduction(2, "OverSpeed", 3),
            deduction(2, "LaneKeepingLineTouch", 15),
        ])

        labels_by_point, untrusted_points = collect_section_labels(results)

        self.assertEqual(labels_by_point, {1: {"turn_signal_missing", "vehicle_device_error"},
                                           2: {"hill_rollback", "over_speed"}, 3: set()})
        self.assertEqual(untrusted_points, set())

    def test_CollectSectionLabels_FinishSignalMissing_LabelsApproachSection(self):
        results = section_results([1, 2, 3], [deduction(3, "FinishSignalMissing")])

        labels_by_point, _ = collect_section_labels(results)

        self.assertEqual(labels_by_point, {1: set(), 2: {"turn_signal_missing"}, 3: set()})

    def test_CollectSectionLabels_Disqualification_UntrustsDisqualifiedAndLaterSections(self):
        results = section_results([1, 2, 3, 4], [
            deduction(2, "HillNotStopped", 0, is_disqualification=True),
            deduction(3, "OverSpeed", 3),
        ])

        labels_by_point, untrusted_points = collect_section_labels(results)

        self.assertEqual(labels_by_point[2], {"stop_failure"})
        self.assertEqual(labels_by_point[3], {"over_speed"})
        self.assertEqual(untrusted_points, {2, 3, 4})

    def test_CollectSectionLabels_UnknownReasonOrPoint_Fails(self):
        with self.assertRaisesRegex(ValueError, "Unknown deduction reason 'HillShortStop'"):
            collect_section_labels(section_results([1], [deduction(1, "HillShortStop")]))
        with self.assertRaisesRegex(ValueError, "not a route section"):
            collect_section_labels(section_results([1], [deduction(-1, "OverSpeed")]))
        with self.assertRaisesRegex(ValueError, "no approach section"):
            collect_section_labels(section_results([1], [deduction(1, "FinishSignalMissing")]))

    def test_LabelSession_AddsLabelVectorsAndCountsUnrecordedPositives(self):
        with tempfile.TemporaryDirectory() as directory:
            session_dir = Path(directory) / "session"
            results = section_results([1, 2, 3], [
                deduction(1, "EmergencyHazardLate", 10),
                deduction(2, "AccelerationTooSlow", 10),
            ])
            write_session(session_dir, results, [training_record("run-1", 1), training_record("run-1", 3)])

            records, masked, unrecorded = label_session(session_dir)

        self.assertEqual([record["section_id"] for record in records], [1, 3])
        self.assertEqual(records[0]["label_names"], LABEL_NAMES)
        self.assertEqual(records[0]["labels"], [int(name == "hazard_light_error") for name in LABEL_NAMES])
        self.assertEqual(records[1]["labels"], [0] * len(LABEL_NAMES))
        self.assertNotIn("unjudged_labels", records[0])
        self.assertEqual((masked, unrecorded), (0, 1))

    def test_LabelSession_AfterDisqualification_KeepsSectionsAndUnjudgesDisqualifiableZeros(self):
        with tempfile.TemporaryDirectory() as directory:
            session_dir = Path(directory) / "session"
            results = section_results([1, 2, 3, 4], [
                deduction(2, "HillNotStopped", 0, is_disqualification=True),
                deduction(3, "OverSpeed", 3),
                deduction(3, "EngineStalled", 5),
                deduction(4, "TurnSignalMissing", 5),
            ])
            write_session(session_dir, results, [training_record("run-1", point) for point in (1, 2, 3, 4)])

            records, masked, unrecorded = label_session(session_dir)

        labels = {record["section_id"]: dict(zip(LABEL_NAMES, record["labels"])) for record in records}
        unjudged = {record["section_id"]: record.get("unjudged_labels") for record in records}
        self.assertEqual(sorted(labels), [1, 2, 3, 4])
        self.assertIsNone(unjudged[1])
        # The recorded disqualification and observed deductions stay positive.
        self.assertEqual((labels[2]["stop_failure"], labels[3]["over_speed"], labels[3]["vehicle_device_error"],
                          labels[4]["turn_signal_missing"]), (1, 1, 1, 1))
        # Only the 0s of labels that a suppressed disqualification could belong to become unjudged.
        self.assertEqual(unjudged[2], ["hill_rollback", "section_time_over", "vehicle_device_error"])
        self.assertEqual(unjudged[3], ["stop_failure", "hill_rollback", "section_time_over"])
        self.assertEqual(unjudged[4], DISQUALIFIABLE_LABELS)
        self.assertEqual((masked, unrecorded), (3, 0))

    def test_LabelSession_WithoutTrainingSections_BuildsRecordsFromDrivingLog(self):
        with tempfile.TemporaryDirectory() as directory:
            session_dir = Path(directory) / "20261125_155430_878"
            results = dict(section_results([0, 1], [deduction(0, "GearNotOperated")]), sampleRateHz=50.00001907348633)
            write_log_session(session_dir, results, [(0.0, 0, 0.0), (0.02, 0, 1.5), (0.04, 1, 3.0)])

            records, dropped, unrecorded = label_session(session_dir)

        self.assertEqual([record["section_id"] for record in records], [0, 1])
        self.assertEqual({record["run_id"] for record in records}, {"20261125_155430_878"})
        self.assertEqual({record["sample_rate_hz"] for record in records}, {50.0})
        self.assertEqual(records[0]["feature_names"], ["speed_kmh"])
        self.assertEqual(records[0]["features"], [[0.0], [1.5]])
        self.assertEqual(records[0]["labels"], [int(name == "vehicle_device_error") for name in LABEL_NAMES])
        self.assertEqual(records[1]["labels"], [0] * len(LABEL_NAMES))
        self.assertEqual((dropped, unrecorded), (0, 0))

    def test_LabelSession_InvalidDrivingLog_FailsWithSessionName(self):
        with tempfile.TemporaryDirectory() as directory:
            session_dir = Path(directory) / "broken-session"
            write_log_session(session_dir, dict(section_results([0], []), sampleRateHz=50.0), [(0.0, 0, "nan")])

            with self.assertRaisesRegex(ValueError, "broken-session"):
                label_session(session_dir)

    #endregion

    #region OBS package runs

    def test_LabelSession_ObsRun_KeepsScopeSectionsUnderRunFolderName(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "obs21-replacement-hill01-010"
            results = dict(section_results([0, 1, 2, 3], [deduction(0, "OverSpeed"), deduction(2, "HillRollback")]),
                           sampleRateHz=50.00001907348633)
            write_obs_run(run_dir, [1, 3], results, obs_samples([0, 1, 2]))

            records, masked, unrecorded = label_session(run_dir)

        self.assertEqual([record["section_id"] for record in records], [1, 2])
        self.assertEqual({record["run_id"] for record in records}, {"obs21-replacement-hill01-010"})
        self.assertEqual(records[0]["labels"], [0] * len(LABEL_NAMES))
        self.assertEqual(records[1]["labels"], [int(name == "hill_rollback") for name in LABEL_NAMES])
        # The section 0 over-speed lies outside the scope, so it is neither a record nor an unrecorded positive.
        self.assertEqual((masked, unrecorded), (0, 0))

    def test_LabelSession_ObsRunDisqualifiedInScope_UnjudgesDisqualifiableZeros(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "obs21-train-signal11-001"
            results = dict(section_results([10, 11, 12], [deduction(11, "SignalViolation", 0, True)]),
                           sampleRateHz=50.0)
            write_obs_run(run_dir, [11, 12], results, obs_samples([10, 11]))

            records, masked, _ = label_session(run_dir)

        self.assertEqual([record["section_id"] for record in records], [11])
        self.assertEqual(records[0]["labels"], [0] * len(LABEL_NAMES))
        self.assertEqual(records[0]["unjudged_labels"], DISQUALIFIABLE_LABELS)
        self.assertEqual(masked, 1)

    def test_LabelSession_ObsRunInvalidOrUnrecordedScope_Fails(self):
        with tempfile.TemporaryDirectory() as directory:
            results = dict(section_results([0, 1, 2], []), sampleRateHz=50.0)
            for name, scope in (("reversed", [2, 1]), ("missing", None), ("float", [0.0, 1.0])):
                run_dir = Path(directory) / name
                write_obs_run(run_dir, scope, results, obs_samples([0, 1]))
                with self.assertRaisesRegex(ValueError, f"{name}: label.json scope"):
                    label_session(run_dir)

            unrecorded_run = Path(directory) / "unrecorded"
            write_obs_run(unrecorded_run, [2, 3], results, obs_samples([0, 1]))
            with self.assertRaisesRegex(ValueError, "unrecorded: no recorded section"):
                label_session(unrecorded_run)

    def test_FindSessionDirs_ObsPackageRootRunsFolderOrRun_FindsRuns(self):
        with tempfile.TemporaryDirectory() as directory:
            package_root = Path(directory) / "Unity-OBS21"
            results = dict(section_results([0, 1], []), sampleRateHz=50.0)
            run_dirs = [package_root / OBS_RUNS_DIR / f"run-{index}" for index in range(2)]
            for run_dir in run_dirs:
                write_obs_run(run_dir, [0, 1], results, obs_samples([0]))
            (package_root / "tools").mkdir()

            from_root = find_session_dirs([package_root])
            from_runs = find_session_dirs([package_root / OBS_RUNS_DIR])
            from_run = find_session_dirs([run_dirs[0]])

        self.assertEqual(from_root, run_dirs)
        self.assertEqual(from_runs, run_dirs)
        self.assertEqual(from_run, run_dirs[:1])

    def test_FindSessionDirs_ObsReportsFolderOrRepeatedRun_ReturnsEachRunOnce(self):
        with tempfile.TemporaryDirectory() as directory:
            package_root = Path(directory) / "Unity-OBS21"
            run_dir = package_root / OBS_RUNS_DIR / "run-0"
            write_obs_run(run_dir, [0, 1], dict(section_results([0, 1], []), sampleRateHz=50.0), obs_samples([0]))

            found = find_session_dirs([run_dir / OBS_REPORT_DIR, package_root, run_dir])
            # Windows and macOS file systems are case-insensitive, where another spelling is the same folder.
            upper_reports = run_dir / OBS_REPORT_DIR.upper()
            found_upper = find_session_dirs([upper_reports]) if upper_reports.is_dir() else [run_dir]

        self.assertEqual(found, [run_dir])
        self.assertEqual(found_upper, [run_dir])

    def test_LabelSession_ObsRunWithTrainingSections_UsesRunFolderName(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "obs21-train-left10-001"
            write_session(run_dir / OBS_REPORT_DIR, section_results([0, 1], []),
                          [training_record("unity-run-x", 0), training_record("unity-run-x", 1)])
            (run_dir / OBS_LABEL_FILE).write_text(json.dumps({"scope": [1, 2]}), encoding="utf-8")

            records, _, _ = label_session(run_dir)

        self.assertEqual([(record["run_id"], record["section_id"]) for record in records],
                         [("obs21-train-left10-001", 1)])

    #endregion

    #region Splits

    def test_AssignSplits_KeepsDriverAndRunGroupsTogetherReproducibly(self):
        records = [training_record("run-a1", 1, "driver-a"), training_record("run-a2", 1, "driver-a"),
                   training_record("run-b", 1), training_record("run-b", 2), training_record("run-c", 1),
                   training_record("run-d", 1)]

        first = assign_splits(records, val_ratio=0.25, seed=7)
        second = assign_splits(records, val_ratio=0.25, seed=7)

        self.assertEqual([record["split"] for record in first], [record["split"] for record in second])
        self.assertEqual(first[0]["split"], first[1]["split"])
        self.assertEqual(first[2]["split"], first[3]["split"])
        self.assertEqual({record["split"] for record in first}, {"train", "val"})

    def test_AssignSplits_TooFewGroupsInvalidRatioOrMixedDriverMetadata_Fails(self):
        with self.assertRaisesRegex(ValueError, "cannot fill"):
            assign_splits([training_record("run-1", 1)])
        with self.assertRaisesRegex(ValueError, "val_ratio"):
            assign_splits([training_record("run-1", 1), training_record("run-2", 1)], val_ratio=0.6, test_ratio=0.4)
        with self.assertRaisesRegex(ValueError, "inconsistent driver_id"):
            assign_splits([training_record("run-1", 1, "driver-a"), training_record("run-1", 2),
                           training_record("run-2", 1)])

    def test_BuildDataset_SessionRootFolder_WritesTrainableLabeledJsonl(self):
        with tempfile.TemporaryDirectory() as directory:
            report_root = Path(directory) / "AIReport"
            for index in range(3):
                write_session(report_root / f"session-{index}",
                              section_results([1, 2], [deduction(2, "OverSpeed", 3)]),
                              [training_record(f"run-{index}", 1), training_record(f"run-{index}", 2)])
            output = Path(directory) / "train" / "sections.jsonl"

            summary = build_dataset([report_root], output, val_ratio=0.34, seed=1)
            # The production training loader validates labels, splits and run/driver split consistency.
            loaded = load_records(output, ["speed_kmh"], LABEL_NAMES, sample_rate_hz=50)

        self.assertEqual((summary["session_count"], summary["section_count"]), (3, 6))
        self.assertEqual(len(loaded), 6)
        self.assertEqual({record["split"] for record in loaded}, {"train", "val"})
        self.assertEqual(sum(counts["over_speed"] for counts in summary["positives_by_split"].values()), 3)

    #endregion

    #region Train-only sessions

    def write_split_sessions(self, report_root, count):
        for index in range(count):
            write_session(report_root / f"session-{index}", section_results([1, 2], [deduction(2, "OverSpeed", 3)]),
                          [training_record(f"run-{index}", 1), training_record(f"run-{index}", 2)])

    def test_BuildDataset_TrainOnlyObsPackage_AlwaysTrainAndKeepsOtherSplits(self):
        with tempfile.TemporaryDirectory() as directory:
            report_root = Path(directory) / "AIReport"
            self.write_split_sessions(report_root, 5)
            package_root = Path(directory) / "Unity-OBS21"
            results = dict(section_results([0, 1, 2], [deduction(1, "TurnSignalMissing")]), sampleRateHz=50.0)
            for index in range(3):
                write_obs_run(package_root / OBS_RUNS_DIR / f"obs-{index}", [1, 2], results, obs_samples([0, 1]))
            without_output = Path(directory) / "without.jsonl"
            with_output = Path(directory) / "with.jsonl"

            build_dataset([report_root], without_output, val_ratio=0.2, test_ratio=0.2, seed=3)
            summary = build_dataset([report_root], with_output, val_ratio=0.2, test_ratio=0.2, seed=3,
                                    train_only_paths=[package_root])
            without = load_records(without_output, ["speed_kmh"], LABEL_NAMES, sample_rate_hz=50)
            loaded = load_records(with_output, ["speed_kmh"], LABEL_NAMES, sample_rate_hz=50)

        split_by_run = {record["run_id"]: record["split"] for record in loaded}
        self.assertEqual({run_id: split_by_run[run_id] for run_id in (r["run_id"] for r in without)},
                         {record["run_id"]: record["split"] for record in without})
        self.assertEqual({split_by_run[f"obs-{index}"] for index in range(3)}, {"train"})
        self.assertEqual({record["run_id"] for record in loaded if record.get("train_only")},
                         {f"obs-{index}" for index in range(3)})
        self.assertEqual((summary["session_count"], summary["train_only_session_count"]), (8, 3))
        self.assertEqual((summary["section_count"], summary["train_only_section_count"]), (13, 3))
        self.assertEqual(summary["positives_by_split"]["train"]["turn_signal_missing"], 3)

    def test_BuildDataset_TrainOnlyOverlappingSessionOrRunId_Fails(self):
        with tempfile.TemporaryDirectory() as directory:
            report_root = Path(directory) / "AIReport"
            self.write_split_sessions(report_root, 3)
            output = Path(directory) / "sections.jsonl"
            with self.assertRaisesRegex(ValueError, "both for splitting and as train-only"):
                build_dataset([report_root], output, train_only_paths=[report_root / "session-0"])

            duplicate_run = Path(directory) / "copied" / "session-copy"
            write_session(duplicate_run, section_results([1], []), [training_record("run-1", 1)])
            with self.assertRaisesRegex(ValueError, "share run/driver groups"):
                build_dataset([report_root], output, train_only_paths=[duplicate_run])

            obs_run = Path(directory) / "Unity-OBS21" / OBS_RUNS_DIR / "obs-0"
            write_obs_run(obs_run, [0, 1], dict(section_results([0, 1], []), sampleRateHz=50.0), obs_samples([0]))
            with self.assertRaisesRegex(ValueError, "both for splitting and as train-only"):
                build_dataset([report_root, obs_run / OBS_REPORT_DIR], output,
                              train_only_paths=[obs_run.parent.parent])

    def test_ParseArgs_TrainOnly_CollectsRepeatedPaths(self):
        args = parse_args(["a", "b", "--output", "out.jsonl", "--train-only", "c", "d", "--train-only", "e"])

        self.assertEqual(args.sessions, ["a", "b"])
        self.assertEqual(args.train_only, ["c", "d", "e"])
        self.assertEqual(parse_args(["a", "--output", "out.jsonl"]).train_only, [])

    #endregion


if __name__ == "__main__":
    unittest.main()
