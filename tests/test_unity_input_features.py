import argparse
import contextlib
import copy
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest

from src.baseline import load_records, train
from src.session_predict import build_section_records, predict_session, read_driving_log


# Column layout written by DrivingReportCsv in the Unity project.
CSV_HEADER = ("time_s,point_number,heading_deg,pos_x,pos_z,speed_kmh,accel_mps2,yaw_rate_dps,steering_input,throttle_input,"
              "brake_input,gear_park,gear_reverse,gear_neutral,gear_drive,parking_brake_on,engine_on,left_signal_on,"
              "right_signal_on,hazard_on,headlight_on,high_beam_on,seat_belt_on,horn_on")
CSV_SAMPLES = (
    "0.000,1,90.00,10.000,20.000,4.000,-0.500,3.000,-0.250,0.500,0.750,0,0,0,1,0,1,1,0,0,1,0,1,0\n"
    "0.020,1,90.00,10.000,20.000,0.000,0.000,0.000,0.000,0.000,1.000,1,0,0,0,1,1,0,0,0,0,0,1,0\n"
    "0.040,2,88.00,10.100,20.100,8.000,1.000,-2.000,0.250,1.000,0.000,0,1,0,0,0,1,0,1,1,1,1,1,1\n"
)
PRE_ANALOG_CSV = (
    "time_s,point_number,speed_kmh,accel_mps2,yaw_rate_dps,heading_deg,pos_x,pos_z,steering_input,throttle_input,brake_applied\n"
    "0.000,1,4.000,-0.500,3.000,90.00,10.000,20.000,-0.250,-0.500,1\n"
)
PHYSICS_FEATURES = ["speed_kmh", "accel_mps2", "yaw_rate_dps"]
CONTROL_FEATURES = ("steering_input", "throttle_input", "brake_input", "gear_park", "gear_reverse", "gear_neutral",
                    "gear_drive", "parking_brake_on")
VEHICLE_STATE_FEATURES = ("engine_on", "left_signal_on", "right_signal_on", "hazard_on", "headlight_on", "high_beam_on",
                          "seat_belt_on", "horn_on")


class UnityInputFeatureTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        schema_path = Path(__file__).resolve().parents[1] / "res" / "schema.example.json"
        self.schema = json.loads(schema_path.read_text(encoding="utf-8"))
        self.rows = list(csv.DictReader(io.StringIO(CSV_HEADER + "\n" + CSV_SAMPLES)))

    def test_UpdatedSchema_ExporterColumns_MapsNineteenFeaturesInSchemaOrder(self):
        self.assertEqual(self.schema["feature_names"], PHYSICS_FEATURES + list(CONTROL_FEATURES + VEHICLE_STATE_FEATURES))
        self.assertEqual(self.schema["continuous_features"], PHYSICS_FEATURES + ["steering_input", "throttle_input", "brake_input"])
        records = build_section_records(self.rows, self.schema["feature_names"], 50, "run")
        self.assertEqual(records[0]["features"], [
            [4.0, -0.5, 3.0, -0.25, 0.5, 0.75, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]])
        self.assertEqual(records[1]["features"], [
            [8.0, 1.0, -2.0, 0.25, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]])

    def test_ControlInput_InvalidRangeOrBinaryFlag_Fails(self):
        for name, value in (("steering_input", "1.1"), ("throttle_input", "-0.1"), ("throttle_input", "1.1"),
                            ("brake_input", "-0.1"), ("brake_input", "1.5"), ("gear_drive", "0.5"),
                            ("parking_brake_on", "2"), ("engine_on", "0.5"), ("hazard_on", "-1"), ("horn_on", "2")):
            rows = copy.deepcopy(self.rows)
            rows[0][name] = value
            with self.subTest(name=name, value=value), self.assertRaisesRegex(ValueError, name):
                build_section_records(rows, self.schema["feature_names"], 50, "run")

    def test_GearOneHot_NoneOrMultipleActive_Fails(self):
        for name, value in (("gear_drive", "0"), ("gear_park", "1")):
            rows = copy.deepcopy(self.rows)
            rows[0][name] = value
            with self.subTest(name=name, value=value), self.assertRaisesRegex(ValueError, "exactly one active gear"):
                build_section_records(rows, self.schema["feature_names"], 50, "run")

    def test_PreAnalogCsv_MissingNewColumns_FailsInsteadOfFabricatingInputs(self):
        rows = list(csv.DictReader(io.StringIO(PRE_ANALOG_CSV)))
        with self.assertRaisesRegex(ValueError, "missing checkpoint features"):
            build_section_records(rows, self.schema["feature_names"], 50, "run")
        # Physics-only checkpoints can still select the columns both layouts share.
        records = build_section_records(rows, PHYSICS_FEATURES, 50, "run")
        self.assertEqual(records[0]["features"][0], [4.0, -0.5, 3.0])

    def test_PreAnalogCheckpoint_SignedReverseThrottle_StillAccepted(self):
        rows = list(csv.DictReader(io.StringIO(PRE_ANALOG_CSV)))
        pre_analog_features = PHYSICS_FEATURES + ["steering_input", "throttle_input", "brake_applied"]
        records = build_section_records(rows, pre_analog_features, 50, "run")
        self.assertEqual(records[0]["features"][0], [4.0, -0.5, 3.0, -0.25, -0.5, 1.0])
        rows[0]["throttle_input"] = "-1.1"
        with self.assertRaisesRegex(ValueError, "throttle_input"):
            build_section_records(rows, pre_analog_features, 50, "run")

    def test_UnavailableVehicleState_BlankColumns_KeepsPhysicsModelAndRejectsNewModel(self):
        rows = copy.deepcopy(self.rows)
        for name in VEHICLE_STATE_FEATURES:
            rows[0][name] = ""
        records = build_section_records(rows, PHYSICS_FEATURES, 50, "run")
        self.assertEqual(records[0]["features"][0], [4.0, -0.5, 3.0])
        with self.assertRaisesRegex(ValueError, "Unavailable checkpoint features"):
            build_section_records(rows, self.schema["feature_names"], 50, "run")

    def test_UnavailableControlInput_BlankColumns_KeepsPhysicsModelAndRejectsNewModel(self):
        rows = copy.deepcopy(self.rows)
        for name in CONTROL_FEATURES:
            rows[0][name] = ""
        records = build_section_records(rows, PHYSICS_FEATURES, 50, "run")
        self.assertEqual(records[0]["features"][0], [4.0, -0.5, 3.0])
        with self.assertRaisesRegex(ValueError, "Unavailable checkpoint features"):
            build_section_records(rows, self.schema["feature_names"], 50, "run")

    def test_UnityTrainingRecord_LabeledOffline_LoadsWithSchema(self):
        # Same layout as DrivingTrainingDataset.Build in the Unity project; a reviewer adds split and labels later.
        feature_names_json = json.dumps(self.schema["feature_names"], separators=(",", ":"))
        unity_line = ('{"schema_version":1,"sample_rate_hz":50,"run_id":"20261015_101500_000_ab12cd34",'
                      '"driver_id":"driver-a","section_id":1,"start_time_s":0.000,"feature_names":' + feature_names_json +
                      ',"features":[[4.000,-0.500,3.000,-0.250,0.500,1.000,0,0,0,1,0,1,1,0,0,1,0,1,0],'
                      '[0.000,0.000,0.000,0.000,0.000,0.000,0,1,0,0,1,1,0,0,0,0,0,0,0]]}')
        unlabeled = load_records(None, self.schema["feature_names"], self.schema["label_names"], labeled=False,
                                 data_bytes=unity_line.encode("utf-8"))
        self.assertEqual(tuple(unlabeled[0]["features"].shape), (2, 19))

        labeled_record = dict(json.loads(unity_line), split="train", label_names=self.schema["label_names"], labels=[1, 0])
        labeled = load_records(None, self.schema["feature_names"], self.schema["label_names"],
                               data_bytes=json.dumps(labeled_record).encode("utf-8"))
        self.assertEqual(labeled[0]["features"][1].tolist(),
                         [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.assertEqual(labeled[0]["driver_id"], "driver-a")

    def test_ControlInput_TimeGapOrInvalidTime_Fails(self):
        for sample_time in ("0.060", "0.000", "NaN", ""):
            rows = copy.deepcopy(self.rows)
            rows[1]["time_s"] = sample_time
            with self.subTest(time_s=sample_time), self.assertRaisesRegex(ValueError, "time"):
                build_section_records(rows, self.schema["feature_names"], 50, "run")

    def test_NewExporterCsv_TrainAndSessionPredict_UsesAllNineteenFeatures(self):
        records = []
        for split in ("train", "val", "test"):
            sections = build_section_records(self.rows, self.schema["feature_names"], 50, split + "-run")
            for record in sections:
                records.append(dict(record, split=split, label_names=self.schema["label_names"],
                                    labels=[1, 0] if record["section_id"] == 1 else [0, 1]))
        training_path = self.root / "training.jsonl"
        training_path.write_text("\n".join(json.dumps(record) for record in records), encoding="utf-8")
        schema_path = self.root / "schema.json"
        schema_path.write_text(json.dumps(self.schema), encoding="utf-8")
        checkpoint = self.root / "model.pt"
        args = argparse.Namespace(data=str(training_path), schema=str(schema_path), output=str(checkpoint),
                                  seed=42, hidden_size=4, epochs=1, batch_size=2, learning_rate=0.01, device="cpu")
        with contextlib.redirect_stdout(io.StringIO()):
            baseline = train(args)
        self.assertEqual(baseline.model.lstm.input_size, 19)
        session = self.root / "session"
        session.mkdir()
        (session / "driving_log.csv").write_text(CSV_HEADER + "\n" + CSV_SAMPLES, encoding="utf-8-sig")
        (session / "section_results.json").write_text(json.dumps({"sampleRateHz": 49.99999}), encoding="utf-8")
        self.assertEqual(len(read_driving_log(session / "driving_log.csv")), 3)
        result = predict_session(session, checkpoint, device="cpu")
        self.assertEqual([section["section_id"] for section in result["sections"]], [1, 2])
        prepared = [json.loads(line) for line in (session / "sections.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertEqual(prepared[0]["feature_names"], self.schema["feature_names"])
        for invalid_rate in (float("nan"), float("inf"), 0, "50", True):
            (session / "section_results.json").write_text(json.dumps({"sampleRateHz": invalid_rate}), encoding="utf-8")
            with self.subTest(sample_rate=invalid_rate), self.assertRaisesRegex(ValueError, "sample rate"):
                predict_session(session, checkpoint, device="cpu")


if __name__ == "__main__":
    unittest.main()
