import math
import importlib.util
import pathlib
import unittest

MODULE_PATH = (
    pathlib.Path(__file__).resolve().parents[1]
    / "smartcar"
    / "whalesbot"
    / "tools"
    / "chassis_pad_control.py"
)
spec = importlib.util.spec_from_file_location("chassis_pad_control", MODULE_PATH)
chassis_pad_control = importlib.util.module_from_spec(spec)
spec.loader.exec_module(chassis_pad_control)

EXIT_BUTTON_MASK = chassis_pad_control.EXIT_BUTTON_MASK
is_disconnected_pad = chassis_pad_control.is_disconnected_pad
pad_to_velocity = chassis_pad_control.pad_to_velocity
should_exit = chassis_pad_control.should_exit


class ChassisPadControlTest(unittest.TestCase):
    def test_disconnected_pad_value_is_detected(self):
        self.assertTrue(is_disconnected_pad([-1, -1, -1, -1, 0]))

    def test_non_disconnected_pad_value_is_not_detected(self):
        self.assertFalse(is_disconnected_pad([0.0, 0.0, 0.0, 0.0, 0]))

    def test_l1_and_l2_exit_mask_is_detected(self):
        self.assertTrue(should_exit([0.0, 0.0, 0.0, 0.0, EXIT_BUTTON_MASK]))

    def test_partial_exit_mask_does_not_exit(self):
        self.assertFalse(should_exit([0.0, 0.0, 0.0, 0.0, 1 << 14]))

    def test_default_pad_to_velocity_uses_low_speed_mapping(self):
        velocity = pad_to_velocity([0.5, 0.25, -0.5, 0.0, 0])

        self.assertEqual(velocity, (0.0375, -0.075, math.pi * 0.125))

    def test_pad_to_velocity_accepts_low_speed_overrides(self):
        velocity = pad_to_velocity(
            [1.0, -1.0, 0.5, 0.0, 0],
            x_scale=0.12,
            y_scale=0.08,
            yaw_scale=0.2,
        )

        self.assertEqual(velocity, (-0.12, -0.08, -math.pi * 0.1))


if __name__ == "__main__":
    unittest.main()
