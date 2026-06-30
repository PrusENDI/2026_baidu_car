import importlib.util
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
    / "camera.py"
)
spec = importlib.util.spec_from_file_location(
    "smartcar.whalesbot.tools.camera", MODULE_PATH
)
camera_module = importlib.util.module_from_spec(spec)
camera_module.__package__ = "smartcar.whalesbot.tools"
sys.modules.setdefault("cv2", types.SimpleNamespace())
with mock.patch.dict(
    sys.modules,
    {
        "smartcar": types.SimpleNamespace(),
        "smartcar.whalesbot": types.SimpleNamespace(),
        "smartcar.whalesbot.tools": types.SimpleNamespace(),
        "smartcar.whalesbot.tools.log_wrap": types.SimpleNamespace(
            logger=types.SimpleNamespace(
                error=lambda *args, **kwargs: None,
                info=lambda *args, **kwargs: None,
            )
        ),
    },
):
    spec.loader.exec_module(camera_module)


class CameraPathResolutionTest(unittest.TestCase):
    def test_resolves_missing_cam2_to_second_existing_project_camera(self):
        existing = {"/dev/cam1", "/dev/cam4"}

        with mock.patch.object(camera_module.platform, "platform", return_value="Linux"):
            with mock.patch.object(camera_module.os.path, "exists", side_effect=existing.__contains__):
                with mock.patch.object(camera_module.glob, "glob", return_value=["/dev/cam1", "/dev/cam4"]):
                    self.assertEqual(camera_module.resolve_camera_path(2), "/dev/cam4")


if __name__ == "__main__":
    unittest.main()
