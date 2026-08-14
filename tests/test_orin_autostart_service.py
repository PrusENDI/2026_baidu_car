import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SERVICE = ROOT / "systemd" / "baidu-smart.service"
INSTALLER = ROOT / "scripts" / "install_orin_service.sh"
REMOTE_ROOT = "/home/jetson/workspaces/baidu_smart_2026_7_17"


class OrinAutostartServiceTest(unittest.TestCase):
    def test_service_starts_main_program_directly(self):
        source = SERVICE.read_text(encoding="utf-8")

        self.assertIn("User=jetson", source)
        self.assertIn(f"WorkingDirectory={REMOTE_ROOT}", source)
        self.assertIn(
            f"ExecStart=/usr/bin/python3 -u {REMOTE_ROOT}/car_start_2026.py",
            source,
        )
        self.assertIn("Restart=on-failure", source)
        self.assertIn("KillMode=control-group", source)
        self.assertIn("WantedBy=multi-user.target", source)
        self.assertNotIn("start_with_key", source)

    def test_installer_enables_and_starts_service(self):
        source = INSTALLER.read_text(encoding="utf-8")

        self.assertIn(f'EXPECTED_ROOT="{REMOTE_ROOT}"', source)
        self.assertIn('install -m 0644 "${SERVICE_SOURCE}" "${SERVICE_TARGET}"', source)
        self.assertIn("systemctl daemon-reload", source)
        self.assertIn('systemctl enable --now "${SERVICE_NAME}"', source)


if __name__ == "__main__":
    unittest.main()
