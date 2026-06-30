import importlib.util
import pathlib
import unittest


MODULE_PATH = (
    pathlib.Path(__file__).resolve().parents[1]
    / "smartcar"
    / "whalesbot"
    / "tools"
    / "servo_calibration_control.py"
)
spec = importlib.util.spec_from_file_location("servo_calibration_control", MODULE_PATH)
servo_calibration_control = importlib.util.module_from_spec(spec)
spec.loader.exec_module(servo_calibration_control)


class FakeServo:
    def __init__(self, port):
        self.port = port
        self.calls = []

    def set_angle(self, angle, speed=100):
        self.calls.append(("angle", angle, speed))

    def set_speed(self, speed):
        self.calls.append(("speed", speed))


class ServoCalibrationControlTest(unittest.TestCase):
    def make_session(self):
        self.pwm = {}
        self.bus = {}

        def pwm_factory(port):
            self.pwm.setdefault(port, FakeServo(port))
            return self.pwm[port]

        def bus_factory(port):
            self.bus.setdefault(port, FakeServo(port))
            return self.bus[port]

        return servo_calibration_control.ServoCalibrationSession(
            pwm_factory=pwm_factory,
            bus_factory=bus_factory,
            home_pwm={1: -42, 2: -37},
            home_bus={2: 0},
        )

    def test_parse_home_list_accepts_port_angle_pairs(self):
        self.assertEqual(
            servo_calibration_control.parse_home_list("1:-42,2:-37"),
            {1: -42, 2: -37},
        )

    def test_select_pwm_and_set_angle(self):
        session = self.make_session()

        result = session.handle_line("pwm 1")
        self.assertIn("selected pwm 1", result)
        session.handle_line("angle -42")

        self.assertEqual(self.pwm[1].calls, [("angle", -42, 100)])

    def test_step_updates_current_angle(self):
        session = self.make_session()
        session.handle_line("bus 2")
        session.handle_line("angle 10")
        session.handle_line("step -5")

        self.assertEqual(
            self.bus[2].calls,
            [("angle", 10, 100), ("angle", 5, 100)],
        )

    def test_center_uses_current_servo_home_angle(self):
        session = self.make_session()
        session.handle_line("pwm 2")
        session.handle_line("center")

        self.assertEqual(self.pwm[2].calls, [("angle", -37, 100)])

    def test_home_all_dispatches_default_pwm_and_bus_targets(self):
        session = self.make_session()

        result = session.handle_line("home-all")

        self.assertIn("homed 3 servos", result)
        self.assertEqual(self.pwm[1].calls, [("angle", -42, 100)])
        self.assertEqual(self.pwm[2].calls, [("angle", -37, 100)])
        self.assertEqual(self.bus[2].calls, [("angle", 0, 100)])


if __name__ == "__main__":
    unittest.main()
