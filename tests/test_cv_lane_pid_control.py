import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from smartcar.whalesbot.tools.lane_collect import (
    CvLanePidConfig,
    CvLanePidController,
    LaneAnalysisResult,
)
from smartcar.whalesbot.tools.lane_collect.ssh_test import OpenCVLaneSshTest


def analysis(valid=True, lateral=0.25, heading=1.0, reason=None, metrics=None):
    return LaneAnalysisResult(
        valid=valid,
        error_y=None,
        error_angle=None,
        raw_lateral=lateral if valid else None,
        raw_heading=heading if valid else None,
        confidence=0.8 if valid else 0.0,
        cross_status="none",
        cross_score=0.0,
        reason=reason,
        work_size=(320, 240),
        metrics=metrics or {},
    )


def centered_track():
    image = np.full((240, 320, 3), 210, dtype=np.uint8)
    polygon = np.array([[118, 239], [145, 35], [175, 35], [202, 239]],
                       dtype=np.int32)
    cv2.fillPoly(image, [polygon], (35, 35, 35))
    return image


class FakeCamera:
    def __init__(self, image):
        self.image = image

    def read(self):
        return self.image


class StaleFakeCamera(FakeCamera):
    frame_timestamp = 0.0


class FakeCar:
    def __init__(self):
        self.commands = []

    def set_velocity(self, x, y, z):
        self.commands.append((x, y, z))


class CvLanePidControllerTests(unittest.TestCase):
    def test_straight_uses_maximum_forward_speed(self):
        command = CvLanePidController().compute(analysis(heading=0.0))
        self.assertAlmostEqual(command.forward_speed, 0.20, places=6)
        self.assertAlmostEqual(command.target_forward_speed, 0.20, places=6)
        self.assertEqual(command.steering_demand, 0.0)

    def test_forward_speed_decreases_with_steering_demand(self):
        speeds = []
        demands = []
        for heading in (0.0, 0.25, 0.5, 1.0):
            command = CvLanePidController(CvLanePidConfig(
                heading_ema_alpha=1.0)).compute(analysis(heading=heading))
            speeds.append(command.target_forward_speed)
            demands.append(command.steering_demand)
        self.assertEqual(speeds, sorted(speeds, reverse=True))
        self.assertEqual(demands, sorted(demands))
        self.assertGreaterEqual(min(speeds), 0.08)
        self.assertLessEqual(max(speeds), 0.20)

    def test_bend_deceleration_is_faster_than_exit_acceleration(self):
        controller = CvLanePidController(CvLanePidConfig(
            heading_ema_alpha=1.0))
        straight = controller.compute(analysis(heading=0.0))
        bend = controller.compute(analysis(heading=1.0))
        exit_command = controller.compute(analysis(heading=0.0))
        self.assertAlmostEqual(
            straight.forward_speed - bend.forward_speed, 0.03, places=6)
        self.assertAlmostEqual(
            exit_command.forward_speed - bend.forward_speed, 0.005, places=6)

    def test_heading_scale_reverses_cv_sign(self):
        controller = CvLanePidController()
        command = controller.compute(analysis(heading=1.0))
        self.assertTrue(command.valid)
        self.assertAlmostEqual(command.error_angle, -0.40, places=6)
        self.assertLess(command.angular_speed, 0.0)

    def test_lateral_control_is_disabled_for_first_test(self):
        command = CvLanePidController().compute(analysis(lateral=0.8))
        self.assertEqual(command.error_y, 0.0)
        self.assertEqual(command.lateral_speed, 0.0)

    def test_heading_output_is_limited(self):
        controller = CvLanePidController()
        command = None
        for _ in range(20):
            command = controller.compute(analysis(heading=10.0))
        self.assertLessEqual(abs(command.angular_speed), 0.60)

    def test_new_heading_waits_for_travel_distance(self):
        controller = CvLanePidController(CvLanePidConfig(
            startup_straight_distance_m=0.0,
            heading_onset_delay_m=0.10))
        first = controller.compute(analysis(heading=1.0), distance_m=0.0)
        before = controller.compute(analysis(heading=1.0), distance_m=0.09)
        released = controller.compute(analysis(heading=1.0), distance_m=0.10)
        self.assertEqual(first.angular_speed, 0.0)
        self.assertEqual(before.angular_speed, 0.0)
        self.assertLess(released.angular_speed, 0.0)

    def test_only_session_start_is_held_straight_for_ten_centimeters(self):
        controller = CvLanePidController()
        first = controller.compute(analysis(heading=1.0), distance_m=0.0)
        before = controller.compute(analysis(heading=1.0), distance_m=0.099)
        released = controller.compute(analysis(heading=1.0), distance_m=0.10)
        later = CvLanePidController().compute(
            analysis(heading=-1.0), distance_m=0.50)
        self.assertEqual(first.angular_speed, 0.0)
        self.assertEqual(first.reason, "startup_straight")
        self.assertEqual(before.angular_speed, 0.0)
        self.assertLess(released.angular_speed, 0.0)
        self.assertGreater(later.angular_speed, 0.0)
        self.assertNotEqual(later.reason, "startup_straight")

    def test_default_controller_uses_near_field_raw_heading(self):
        command = CvLanePidController().compute(analysis(
            heading=1.0,
            metrics={"perspective_k": -0.02, "perspective_b": -0.12},
        ))
        self.assertLess(command.angular_speed, 0.0)
        self.assertEqual(command.reason, "")

    def test_perspective_kb_drives_heading_without_lateral_speed(self):
        controller = CvLanePidController(CvLanePidConfig(
            startup_straight_distance_m=0.0,
            perspective_kb_enabled=True))
        command = controller.compute(analysis(
            heading=-1.0,
            lateral=0.8,
            metrics={
                "perspective_k": 0.02,
                "perspective_b": 0.12,
            },
        ))
        self.assertLess(command.angular_speed, 0.0)
        self.assertEqual(command.lateral_speed, 0.0)
        self.assertEqual(command.reason, "perspective_kb")

    def test_invalid_analysis_commands_stop(self):
        command = CvLanePidController().compute(
            analysis(valid=False, reason="no lane"))
        self.assertFalse(command.valid)
        self.assertEqual(
            (command.forward_speed, command.lateral_speed, command.angular_speed),
            (0.0, 0.0, 0.0),
        )


class OpenCVLaneSshTestTests(unittest.TestCase):
    def test_cross_state_is_bypassed_while_disabled(self):
        with tempfile.TemporaryDirectory() as directory:
            car = FakeCar()
            runner = OpenCVLaneSshTest(
                FakeCamera(centered_track()), car, output_root=directory)
            runner.commands.put("start")
            runner._consume_commands()

            class DisabledCrossState:
                def update(self, *args, **kwargs):
                    raise AssertionError("disabled crossing state was called")

            runner.cross_state = DisabledCrossState()
            runner._control_once()
            self.assertTrue(runner.running)
            runner._disarm("test complete")

    def test_start_saves_raw_camera_size_and_stop_commands_zero(self):
        with tempfile.TemporaryDirectory() as directory:
            car = FakeCar()
            runner = OpenCVLaneSshTest(
                FakeCamera(centered_track()), car, output_root=directory)
            runner.commands.put("start")
            runner._consume_commands()
            runner._control_once()
            session_dir = runner.writer.session_dir
            runner.commands.put("stop")
            runner._consume_commands()

            saved = cv2.imread(str(session_dir / "000000.jpg"))
            records = json.loads((session_dir / "data.json").read_text())
            metadata = json.loads((session_dir / "session.json").read_text())
            self.assertEqual(saved.shape, (240, 320, 3))
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["teacher"], "opencv_pid_command")
            self.assertFalse(records[0]["held"])
            self.assertEqual(records[0]["command_source"], "standard")
            self.assertIn("control_reason", records[0])
            self.assertIn("steering_demand", records[0])
            self.assertIn("target_forward_speed", records[0])
            self.assertEqual(records[0]["state"], records[0]["control"])
            self.assertEqual(
                metadata["state_fields"],
                ["forward_speed", "lateral_speed", "angular_speed"],
            )
            self.assertTrue(metadata["usable_for_training"])
            self.assertEqual(
                metadata["controller"]["max_forward_speed"], 0.20)
            self.assertEqual(
                metadata["controller"]["min_forward_speed"], 0.08)
            self.assertIn("speed_control", metadata)
            self.assertEqual(car.commands[-1], (0.0, 0.0, 0.0))

    def test_invalid_frame_disarms_and_stops(self):
        with tempfile.TemporaryDirectory() as directory:
            car = FakeCar()
            image = np.full((240, 320, 3), 35, dtype=np.uint8)
            runner = OpenCVLaneSshTest(
                FakeCamera(image), car, output_root=directory)
            runner.commands.put("start")
            runner._consume_commands()
            runner._control_once()
            self.assertFalse(runner.running)
            self.assertEqual(car.commands[-1], (0.0, 0.0, 0.0))

    def test_ten_invalid_frames_hold_then_eleventh_stops(self):
        with tempfile.TemporaryDirectory() as directory:
            car = FakeCar()
            camera = FakeCamera(centered_track())
            runner = OpenCVLaneSshTest(
                camera, car, output_root=directory)
            runner.commands.put("start")
            runner._consume_commands()
            runner._control_once()
            last_valid = car.commands[-1]

            camera.image = np.full((240, 320, 3), 35, dtype=np.uint8)
            for _ in range(10):
                runner._control_once()
                self.assertTrue(runner.running)
                self.assertEqual(car.commands[-1], last_valid)

            held_records = runner.writer.records[-10:]
            self.assertTrue(all(row["held"] for row in held_records))
            self.assertTrue(all(row["state"] == list(last_valid)
                                for row in held_records))

            runner._control_once()
            self.assertFalse(runner.running)
            self.assertEqual(car.commands[-1], (0.0, 0.0, 0.0))

    def test_stale_camera_frame_disarms_and_stops(self):
        with tempfile.TemporaryDirectory() as directory:
            car = FakeCar()
            runner = OpenCVLaneSshTest(
                StaleFakeCamera(centered_track()), car, output_root=directory)
            runner.commands.put("start")
            runner._consume_commands()
            runner._control_once()
            self.assertFalse(runner.running)
            self.assertEqual(car.commands[-1], (0.0, 0.0, 0.0))


class CollectionIsolationTests(unittest.TestCase):
    def test_main_runtime_does_not_import_collection_cv(self):
        root = Path(__file__).resolve().parents[1]
        for name in ("car_wrap_2026.py", "car_task_function.py", "car_start_2026.py"):
            source = (root / name).read_text(encoding="utf-8")
            self.assertNotIn("lane_collect", source)
            self.assertNotIn("OpenCVLaneAnalyzer", source)


if __name__ == "__main__":
    unittest.main()
