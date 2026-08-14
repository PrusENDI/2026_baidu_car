import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class FakeEvent:
    def __init__(self):
        self.wait_called = False
        self.clear_called = False

    def wait(self):
        self.wait_called = True

    def clear(self):
        self.clear_called = True


class ManualKeyGateTest(unittest.TestCase):
    def test_wait_for_start_key_waits_then_clears_event(self):
        from manual_start import wait_for_start_key

        event = FakeEvent()
        wait_for_start_key(event)

        self.assertTrue(event.wait_called)
        self.assertTrue(event.clear_called)

    def test_key_thread_sets_start_event_only_for_key_one(self):
        source = (ROOT / "car_wrap_2026.py").read_text(encoding="utf-8")
        self.assertIn("self._start_key_event", source)
        self.assertIn("if key_val == 1:", source)

    def test_main_waits_before_default_task(self):
        source = (ROOT / "car_start_2026.py").read_text(encoding="utf-8")
        main_source = source[source.index("def main"):]
        self.assertLess(main_source.index("wait_for_start_key"), main_source.index("auto_lane_tracing"))

    def test_main_rearms_start_gate_for_next_run(self):
        source = (ROOT / "car_start_2026.py").read_text(encoding="utf-8")
        main_source = source[source.index("def main"):]
        self.assertIn("while True:", main_source)
        self.assertIn("_stop_flag = False", main_source)
        self.assertIn("_start_key_event.clear()", main_source)

    def test_key_thread_keeps_reading_after_task_stop(self):
        source = (ROOT / "car_wrap_2026.py").read_text(encoding="utf-8")
        thread_source = source[source.index("def key_thread_func"):source.index("# 根据某个值", source.index("def key_thread_func"))]
        self.assertIn("key_val = self.key.get_key()", thread_source)
        self.assertNotIn("if not self._stop_flag:", thread_source)


if __name__ == "__main__":
    unittest.main()
