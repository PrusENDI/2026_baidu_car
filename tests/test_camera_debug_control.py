import importlib.util
import pathlib
import unittest
from unittest import mock


MODULE_PATH = (
    pathlib.Path(__file__).resolve().parents[1]
    / "smartcar"
    / "whalesbot"
    / "tools"
    / "camera_debug_control.py"
)
spec = importlib.util.spec_from_file_location("camera_debug_control", MODULE_PATH)
camera_debug_control = importlib.util.module_from_spec(spec)
spec.loader.exec_module(camera_debug_control)


class CameraDebugControlTest(unittest.TestCase):
    def test_parse_size_accepts_width_x_height(self):
        self.assertEqual(camera_debug_control.parse_size("320x240"), (320, 240))

    def test_default_camera_specs_match_collect_control(self):
        args = camera_debug_control.parse_args([])

        specs = camera_debug_control.build_camera_specs(args)

        self.assertEqual(
            specs,
            [
                camera_debug_control.CameraSpec("cam1", "1", 320, 240),
                camera_debug_control.CameraSpec("cam2", "2", 640, 480),
            ],
        )

    def test_auto_camera_specs_use_first_two_project_camera_devices(self):
        args = camera_debug_control.parse_args(["--cam2", "auto"])

        specs = camera_debug_control.build_camera_specs(
            args,
            device_lister=lambda: ["/dev/cam1", "/dev/cam4"],
        )

        self.assertEqual(
            specs,
            [
                camera_debug_control.CameraSpec("cam1", "1", 320, 240),
                camera_debug_control.CameraSpec("cam2", "4", 640, 480),
            ],
        )

    def test_single_cam1_only_selects_cam1(self):
        args = camera_debug_control.parse_args(["--single", "cam1"])

        specs = camera_debug_control.build_camera_specs(args)

        self.assertEqual(specs, [camera_debug_control.CameraSpec("cam1", "1", 320, 240)])

    def test_overrides_camera_index_and_size(self):
        args = camera_debug_control.parse_args(
            ["--cam1", "0", "--cam1-size", "800x600", "--single", "cam1"]
        )

        specs = camera_debug_control.build_camera_specs(args)

        self.assertEqual(specs, [camera_debug_control.CameraSpec("cam1", "0", 800, 600)])

    def test_accepts_explicit_video_device_source(self):
        args = camera_debug_control.parse_args(
            ["--cam2-source", "/dev/video1", "--single", "cam2"]
        )

        specs = camera_debug_control.build_camera_specs(args)

        self.assertEqual(
            specs,
            [camera_debug_control.CameraSpec("cam2", "/dev/video1", 640, 480)],
        )

    def test_resolve_camera_source_converts_project_index_to_dev_cam(self):
        self.assertEqual(camera_debug_control.resolve_camera_source("2"), "/dev/cam2")
        self.assertEqual(
            camera_debug_control.resolve_camera_source("/dev/video1"), "/dev/video1"
        )

    def test_camera_servo_is_disabled_by_default(self):
        args = camera_debug_control.parse_args([])

        self.assertFalse(args.camera_servo)
        self.assertEqual(args.camera_servo_angles, [-42, 165])

    def test_list_devices_returns_before_streamer_import(self):
        calls = []

        with mock.patch.object(
            camera_debug_control,
            "list_camera_devices",
            side_effect=lambda: calls.append("listed"),
        ):
            camera_debug_control.main(["--list-devices"])

        self.assertEqual(calls, ["listed"])

    def test_camera_servo_toggle_cycles_collect_control_angles(self):
        servo = FakeServo()
        controller = camera_debug_control.CameraServoController(
            servo_factory=lambda: servo,
            angles=[-42, 165],
            speed=100,
        )

        self.assertEqual(controller.initialize(), -42)
        self.assertEqual(controller.toggle(), 165)
        self.assertEqual(controller.toggle(), -42)
        self.assertEqual(
            servo.calls,
            [("angle", -42, 100), ("angle", 165, 100), ("angle", -42, 100)],
        )


class FakeServo:
    def __init__(self):
        self.calls = []

    def set_angle(self, angle, speed=100):
        self.calls.append(("angle", angle, speed))


if __name__ == "__main__":
    unittest.main()
