import importlib.util
import io
import pathlib
import sys
import types
import unittest
from unittest import mock


MODULE_PATH = (
    pathlib.Path(__file__).resolve().parents[1]
    / "smartcar"
    / "whalesbot"
    / "tools"
    / "collect_data_safe_start.py"
)


def load_module():
    spec = importlib.util.spec_from_file_location("collect_data_safe_start", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class CollectDataSafeStartTest(unittest.TestCase):
    def test_default_args_match_collect_data(self):
        module = load_module()

        args = module.parse_args([])

        self.assertEqual(args.cam1, "1")
        self.assertEqual(args.cam2, "2")
        self.assertEqual(args.cam1_size, (320, 240))
        self.assertEqual(args.cam2_size, (640, 480))
        self.assertEqual(args.dir1, "dataset/image_set_lane")
        self.assertEqual(args.dir2, "dataset/image_set_object")

    def test_install_lightweight_packages_prevents_smartcar_root_import(self):
        module = load_module()
        original_import = __import__

        def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "smartcar":
                raise AssertionError("smartcar package root should not be imported")
            return original_import(name, globals, locals, fromlist, level)

        with mock.patch("builtins.__import__", side_effect=guarded_import):
            module.install_lightweight_packages(pathlib.Path(__file__).resolve().parents[1])
            imported = __import__("smartcar.whalesbot.tools", fromlist=["__name__"])

        self.assertEqual(imported.__name__, "smartcar.whalesbot.tools")

    def test_install_tools_logger_exposes_logger_on_tools_package(self):
        module = load_module()
        root = pathlib.Path(__file__).resolve().parents[1]

        module.install_lightweight_packages(root)
        with mock.patch.dict(
            sys.modules,
            {"yaml": types.SimpleNamespace(safe_load=lambda content: {})},
        ):
            logger = module.install_tools_logger(root)
        tools_package = sys.modules["smartcar.whalesbot.tools"]

        self.assertIs(tools_package.logger, logger)
        self.assertTrue(hasattr(tools_package, "PID"))
        self.assertTrue(hasattr(tools_package, "CountRecord"))
        self.assertTrue(hasattr(tools_package, "limit_val"))

    def test_preflight_reports_missing_camera(self):
        module = load_module()

        errors = module.preflight(
            cam_sources=["/dev/cam1", "/dev/cam2"],
            path_exists=lambda path: path == "/dev/cam1",
            port_checker=lambda port: False,
            ports=[5000],
        )

        self.assertIn("missing camera device: /dev/cam2", errors)

    def test_main_preflight_uses_resolved_camera_paths(self):
        module = load_module()

        with mock.patch.object(module, "project_root_from_file", return_value=MODULE_PATH.parents[3]):
            with mock.patch.object(module, "load_module") as load_module_mock:
                camera_module = mock.Mock()
                camera_module.resolve_camera_path.side_effect = ["/dev/cam1", "/dev/cam4"]
                load_module_mock.return_value = camera_module
                with mock.patch.object(module, "preflight", return_value=[]) as preflight_mock:
                    with mock.patch("sys.stdout", new=io.StringIO()):
                        module.main(["--preflight-only"])

        preflight_mock.assert_called_once()
        self.assertEqual(
            preflight_mock.call_args.kwargs["cam_sources"]
            if "cam_sources" in preflight_mock.call_args.kwargs
            else preflight_mock.call_args.args[0],
            ["/dev/cam1", "/dev/cam4"],
        )


if __name__ == "__main__":
    unittest.main()
