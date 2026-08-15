import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from smartcar.whalesbot.tools.lane_collect import (
    CvLanePidController,
    LaneAnalysisResult,
)
from smartcar.whalesbot.tools.lane_collect.ssh_test import OpenCVLaneSshTest


def analysis(valid=True, lateral=0.25, heading=1.0, reason=None):
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
    def test_heading_scale_reverses_cv_sign(self):
        controller = CvLanePidController()
        command = controller.compute(analysis(heading=1.0))
        self.assertTrue(command.valid)
        self.assertAlmostEqual(command.error_angle, -0.316, places=6)
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

    def test_invalid_analysis_commands_stop(self):
        command = CvLanePidController().compute(
            analysis(valid=False, reason="no lane"))
        self.assertFalse(command.valid)
        self.assertEqual(
            (command.forward_speed, command.lateral_speed, command.angular_speed),
            (0.0, 0.0, 0.0),
        )


class OpenCVLaneSshTestTests(unittest.TestCase):
    def test_start_saves_exact_cnn_size_and_stop_commands_zero(self):
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
            self.assertEqual(saved.shape, (128, 128, 3))
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["teacher"], "opencv_pid_command")
            self.assertFalse(records[0]["held"])
            self.assertEqual(records[0]["command_source"], "standard")
            self.assertEqual(records[0]["state"], records[0]["control"])
            self.assertEqual(
                metadata["state_fields"],
                ["forward_speed", "lateral_speed", "angular_speed"],
            )
            self.assertTrue(metadata["usable_for_training"])
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
