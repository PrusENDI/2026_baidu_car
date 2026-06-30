import importlib.util
import pathlib
import sys
import unittest


MODULE_PATH = (
    pathlib.Path(__file__).resolve().parents[1]
    / "tools"
    / "usb_burst_capture.py"
)
spec = importlib.util.spec_from_file_location("usb_burst_capture", MODULE_PATH)
usb_burst_capture = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = usb_burst_capture
spec.loader.exec_module(usb_burst_capture)


class UsbBurstCaptureTest(unittest.TestCase):
    def test_parse_size_accepts_width_x_height(self):
        self.assertEqual(usb_burst_capture.parse_size("640x480"), (640, 480))
        self.assertEqual(usb_burst_capture.parse_size("640*480"), (640, 480))

    def test_default_output_dir_is_class_session_folder(self):
        args = usb_burst_capture.parse_args(
            ["--class-name", "ball_blue", "--session", "desk_near", "--dry-run"]
        )

        session = usb_burst_capture.build_session(args)

        self.assertEqual(
            session.output_dir,
            pathlib.Path("dataset") / "quick_capture" / "ball_blue" / "desk_near",
        )

    def test_rejects_unknown_class_name(self):
        with self.assertRaises(SystemExit):
            usb_burst_capture.parse_args(["--class-name", "unknown"])

    def test_build_filename_contains_class_session_and_sequence(self):
        path = usb_burst_capture.build_image_path(
            pathlib.Path("out"), "ball_yellow", "edge_light", 7
        )

        self.assertEqual(path, pathlib.Path("out") / "ball_yellow_edge_light_000007.jpg")

    def test_capture_interval_from_fps(self):
        args = usb_burst_capture.parse_args(
            ["--class-name", "water_l1", "--fps", "2", "--dry-run"]
        )

        self.assertEqual(usb_burst_capture.capture_interval(args), 0.5)

    def test_dry_run_session_does_not_require_count(self):
        args = usb_burst_capture.parse_args(["--class-name", "water_l1", "--dry-run"])

        session = usb_burst_capture.build_session(args)

        self.assertEqual(session.count, 0)


if __name__ == "__main__":
    unittest.main()
