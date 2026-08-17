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
from smartcar.whalesbot.tools.curvature_control import CurvatureSpeedController


def analysis(valid=True, lateral=0.25, heading=1.0, reason=None, metrics=None):
    return LaneAnalysisResult(
        valid=valid,
        error_y=lateral if valid else None,
        error_angle=heading if valid else None,
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
    def test_cnn_speed_demand_is_independent_from_action_curvature(self):
        controller = CurvatureSpeedController(
            max_speed=0.30, min_speed=0.12,
            deceleration_rate=0.0, acceleration_rate=0.0)
        vx, vy, wz, target = controller.command(
            0.5, speed_demand=1.0, dt_s=0.05)
        self.assertAlmostEqual(target, 0.12)
        self.assertAlmostEqual(vx, 0.12)
        self.assertEqual(vy, 0.0)
        self.assertAlmostEqual(wz, vx * 0.5)

    def test_launch_guard_default_is_fifteen_centimetres(self):
        self.assertAlmostEqual(
            CvLanePidConfig().initial_straight_distance_m, 0.15)

    def test_pure_pursuit_converts_curvature_to_yaw_rate(self):
        controller = CvLanePidController(CvLanePidConfig(
            steering_mode="pure_pursuit", max_heading_rate=0.0,
            pure_pursuit_entry_rate=0.0, max_heading_release_rate=0.0,
            max_deceleration_mps2=0.0))
        result = analysis(heading=0.0, metrics={
            "pure_pursuit_curvature_m_inv": 2.0,
        })
        command = controller.compute(result)
        self.assertTrue(command.valid)
        self.assertEqual(command.reason, "ipm_pure_pursuit")
        self.assertAlmostEqual(
            command.angular_speed, command.forward_speed * 2.0, places=6)
        self.assertAlmostEqual(
            -command.error_angle * controller.config.heading_kp,
            command.target_forward_speed * 2.0, places=6)

    def test_pure_pursuit_builds_turn_faster_than_it_releases(self):
        controller = CvLanePidController(CvLanePidConfig(
            steering_mode="pure_pursuit", max_forward_speed=0.20,
            min_forward_speed=0.20, pure_pursuit_entry_rate=2.0,
            max_heading_release_rate=0.8))
        bend = analysis(heading=0.0, metrics={
            "pure_pursuit_curvature_m_inv": 5.0,
        })
        straight = analysis(heading=0.0, metrics={
            "pure_pursuit_curvature_m_inv": 0.0,
        })
        first = controller.compute(bend)
        second = controller.compute(bend)
        release = controller.compute(straight)
        self.assertAlmostEqual(first.angular_speed, 0.10)
        self.assertAlmostEqual(second.angular_speed, 0.20)
        self.assertAlmostEqual(release.angular_speed, 0.16)

    def test_pure_pursuit_reverses_faster_than_same_direction_release(self):
        controller = CvLanePidController(CvLanePidConfig(
            steering_mode="pure_pursuit", max_forward_speed=0.20,
            min_forward_speed=0.20, pure_pursuit_entry_rate=2.0,
            max_heading_release_rate=0.8,
            max_heading_reverse_rate=2.0))
        right = analysis(heading=0.0, metrics={
            "pure_pursuit_curvature_m_inv": 5.0,
        })
        left = analysis(heading=0.0, metrics={
            "pure_pursuit_curvature_m_inv": -5.0,
        })
        controller.compute(right)
        controller.compute(right)
        reverse = controller.compute(left)
        self.assertAlmostEqual(reverse.angular_speed, 0.10)

    def test_right_turn_geometric_command_bypasses_frame_slew(self):
        controller = CvLanePidController(CvLanePidConfig(
            steering_mode="pure_pursuit", max_forward_speed=0.20,
            min_forward_speed=0.20, pure_pursuit_entry_rate=2.0))
        result = analysis(heading=0.0, metrics={
            "pure_pursuit_curvature_m_inv": -6.0,
            "route_right_turn_override": True,
        })
        command = controller.compute(result)
        self.assertEqual(command.reason, "route_right_turn")
        self.assertAlmostEqual(command.angular_speed, -1.20)

    def test_pure_pursuit_requires_curvature_metric(self):
        controller = CvLanePidController(CvLanePidConfig(
            steering_mode="pure_pursuit"))
        command = controller.compute(analysis(heading=0.0))
        self.assertFalse(command.valid)
        self.assertIn("curvature", command.reason)

    def test_straight_uses_maximum_forward_speed(self):
        command = CvLanePidController().compute(analysis(heading=0.0))
        self.assertAlmostEqual(command.forward_speed, 0.30, places=6)
        self.assertAlmostEqual(command.target_forward_speed, 0.30, places=6)
        self.assertEqual(command.steering_demand, 0.0)

    def test_forward_speed_decreases_with_steering_demand(self):
        speeds = []
        demands = []
        for heading in (0.0, 0.25, 0.5, 1.0):
            command = CvLanePidController().compute(analysis(heading=heading))
            speeds.append(command.target_forward_speed)
            demands.append(command.steering_demand)
        self.assertEqual(speeds, sorted(speeds, reverse=True))
        self.assertEqual(demands, sorted(demands))
        self.assertGreaterEqual(min(speeds), 0.12)
        self.assertLessEqual(max(speeds), 0.30)

    def test_bend_deceleration_is_faster_than_exit_acceleration(self):
        controller = CvLanePidController()
        real_car_dt = 0.07741451263427734
        straight = controller.compute(analysis(heading=0.0), dt_s=real_car_dt)
        bend = controller.compute(analysis(heading=1.0), dt_s=real_car_dt)
        exit_command = controller.compute(
            analysis(heading=0.0), dt_s=real_car_dt)
        self.assertAlmostEqual(
            straight.forward_speed - bend.forward_speed,
            controller.config.max_deceleration_mps2 * real_car_dt,
            places=6)
        self.assertAlmostEqual(
            exit_command.forward_speed - bend.forward_speed,
            controller.config.max_acceleration_mps2 * real_car_dt,
            places=6)

    def test_pid_keeps_single_frame_labels_while_vy_is_disabled(self):
        controller = CvLanePidController(CvLanePidConfig(max_heading_rate=0.0))
        command = controller.compute(analysis(lateral=0.05, heading=-0.10))
        self.assertTrue(command.valid)
        self.assertAlmostEqual(command.error_y, 0.05, places=6)
        self.assertAlmostEqual(command.error_angle, -0.10, places=6)
        self.assertEqual(command.lateral_speed, 0.0)
        self.assertAlmostEqual(command.angular_speed, -0.195, places=6)
        self.assertEqual(command.reason, "ipm_lane_pid")

    def test_heading_output_is_limited(self):
        controller = CvLanePidController(CvLanePidConfig(max_heading_rate=0.0))
        command = controller.compute(analysis(heading=10.0))
        self.assertLessEqual(abs(command.angular_speed), 1.50)

    def test_distance_does_not_delay_current_frame_heading(self):
        command = CvLanePidController().compute(
            analysis(heading=-0.20), distance_m=0.0)
        self.assertLess(command.angular_speed, 0.0)
        self.assertAlmostEqual(command.error_angle, -0.20, places=6)

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
            self.assertEqual(
                records[0]["teacher"], "opencv_ipm_pure_pursuit")
            self.assertFalse(records[0]["held"])
            self.assertEqual(records[0]["command_source"], "standard")
            self.assertIn("control_reason", records[0])
            self.assertIn("steering_demand", records[0])
            self.assertIn("target_forward_speed", records[0])
            self.assertEqual(records[0]["state"], records[0]["control"])
            self.assertEqual(
                records[0]["model_target"]["speed_demand"],
                records[0]["steering_demand"])
            self.assertAlmostEqual(
                records[0]["model_target"]["kappa_action"],
                records[0]["control"][2] /
                max(abs(records[0]["control"][0]), 0.12),
                places=6)
            self.assertEqual(records[0]["target_mask"], [1.0, 1.0])
            self.assertEqual(
                records[0]["legacy_pid_state"][1],
                records[0]["cv"]["error_y"])
            self.assertEqual(
                metadata["state_fields"],
                ["forward_speed", "lateral_speed", "angular_speed"],
            )
            self.assertEqual(
                metadata["model_output_fields"],
                ["speed_demand", "kappa_action"])
            self.assertTrue(metadata["usable_for_training"])
            self.assertEqual(
                metadata["controller"]["max_forward_speed"], 0.30)
            self.assertEqual(
                metadata["controller"]["min_forward_speed"], 0.12)
            self.assertEqual(
                metadata["controller"]["steering_mode"], "pure_pursuit")
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
            first_state = list(runner.writer.records[-1]["state"])

            camera.image = np.full((240, 320, 3), 35, dtype=np.uint8)
            for _ in range(10):
                runner._control_once()
                self.assertTrue(runner.running)
                self.assertEqual(car.commands[-1], last_valid)

            held_records = runner.writer.records[-10:]
            self.assertTrue(all(row["held"] for row in held_records))
            self.assertTrue(all(row["state"] == first_state
                                for row in held_records))
            self.assertTrue(all(row["control"] == list(last_valid)
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
