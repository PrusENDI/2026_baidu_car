import importlib.util
from pathlib import Path
import unittest


path = Path(__file__).parents[1] / "tools" / "safe_lane_test.py"
spec = importlib.util.spec_from_file_location("safe_lane_test", path)
safe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(safe)


class SafetyTests(unittest.TestCase):
    def test_argument_limits_allow_300_seconds_but_reject_longer(self):
        self.assertIsNone(safe.validate_limits(0.10, 300.0))
        self.assertEqual(
            "speed 0..0.10; duration 0..300",
            safe.validate_limits(0.10, 300.1),
        )
        self.assertEqual(
            "speed 0..0.10; duration 0..300",
            safe.validate_limits(0.11, 15.0),
        )

    def test_rejects_non_finite_or_excessive_model_output(self):
        self.assertEqual("invalid_model_output", safe.validate_output([float("nan"), 0.0], 1.0))
        self.assertEqual("model_output_limit", safe.validate_output([1.1, 0.0], 1.0))
        self.assertIsNone(safe.validate_output([0.2, -0.3], 1.0))

    def test_rejects_wrong_length_lane_output(self):
        self.assertEqual("invalid_model_output", safe.validate_output([0.1], 1.0))
        self.assertEqual("invalid_model_output", safe.validate_output([0.1, 0.2, 0.3], 1.0))

    def test_wait_ready_times_out_without_server(self):
        class NeverReady:
            def ready(self, timeout_ms):
                return False

        self.assertFalse(safe.wait_ready(NeverReady(), timeout=0.0, sleep=lambda _: None))

    def test_cold_start_preflight_allows_five_seconds(self):
        class Client:
            timeout_ms = None

            def infer(self, frame, timeout_ms):
                self.timeout_ms = timeout_ms
                return [0.1, -0.2]

        client = Client()
        self.assertEqual([0.1, -0.2], safe.preflight_infer(client, object()))
        self.assertEqual(5000, client.timeout_ms)

    def test_normal_lane_loop_keeps_duration_elapsed_reason(self):
        class Cap:
            def read(self):
                return object()

        class Stream:
            def update_frame(self, frame, name):
                pass

        class Lane:
            def infer(self, frame):
                return [0.1, -0.2]

        class Car:
            calls = 0

            def set_velocity(self, speed, y, angle):
                self.calls += 1

        ticks = iter([0.0, 0.0, 1.0])
        car = Car()
        reason = safe.run_lane_loop(
            duration=0.1,
            speed=0.05,
            output_limit=1.0,
            cap=Cap(),
            stream=Stream(),
            lane=Lane(),
            car=car,
            py=lambda value: value,
            pa=lambda value: value,
            monotonic=lambda: next(ticks),
        )
        self.assertEqual("duration_elapsed", reason)
        self.assertEqual(1, car.calls)


if __name__ == "__main__":
    unittest.main()
