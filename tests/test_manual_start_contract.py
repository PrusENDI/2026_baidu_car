import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class ManualStartContractTest(unittest.TestCase):
    def test_key_start_runtime_assets_are_absent(self):
        assets = (
            "scripts/start_with_key.sh",
            "scripts/wait_for_start_key.py",
            "scripts/inference_backend_probe.py",
            "systemd/baidu-smart-key-start.service",
        )
        self.assertEqual([], [path for path in assets if (ROOT / path).exists()])

    def test_main_entry_has_no_supervisor_disconnect_protocol(self):
        source = (ROOT / "car_start_2026.py").read_text(encoding="utf-8")
        self.assertNotIn("ControllerDisconnected", source)
        self.assertNotIn("MC602_DISCONNECTED_EXIT", source)
        self.assertIn('if __name__ == "__main__":', source)
        self.assertIn("main()", source)


if __name__ == "__main__":
    unittest.main()
