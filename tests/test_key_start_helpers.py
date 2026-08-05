import importlib.util
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


def load_script(name):
    path = ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class SequenceKey:
    def __init__(self, values):
        self.values = iter(values)
        self.read_count = 0

    def get_key(self):
        self.read_count += 1
        return next(self.values)


class RecordingDisplay:
    def __init__(self):
        self.messages = []

    def show(self, message):
        self.messages.append(message)


class KeyStartHelpersTest(unittest.TestCase):
    def test_listener_adds_project_root_to_import_path(self):
        module = load_script("wait_for_start_key.py")
        search_path = []

        project_root = module.add_project_root_to_import_path(search_path=search_path)

        self.assertEqual(Path(project_root), ROOT)
        self.assertEqual(search_path, [project_root])

    def test_wait_for_start_ignores_other_keys_and_starts_once(self):
        module = load_script("wait_for_start_key.py")
        key = SequenceKey([0, 4, 8, 12, 12])
        display = RecordingDisplay()
        sleeps = []

        module.wait_for_start(key, display, sleep=sleeps.append, poll_interval=0.1)

        self.assertEqual(key.read_count, 4)
        self.assertEqual(display.messages, ["wait to start", "started!!!"])
        self.assertEqual(sleeps, [0.1, 0.1, 0.1, 0.3])

    def test_all_services_ready_checks_every_configured_port(self):
        module = load_script("inference_backend_probe.py")
        calls = []

        def probe(port, timeout_ms):
            calls.append((port, timeout_ms))
            return port != 5002

        self.assertFalse(module.all_services_ready(probe=probe, timeout_ms=300))
        self.assertEqual(calls, [(5001, 300), (5002, 300), (5005, 300)])

    def test_wait_until_ready_retries_until_every_service_answers(self):
        module = load_script("inference_backend_probe.py")
        attempts = iter([False, False, True])
        sleeps = []

        self.assertTrue(module.wait_until_ready(
            ready=lambda: next(attempts),
            sleep=sleeps.append,
            interval=0.2,
        ))
        self.assertEqual(sleeps, [0.2, 0.2])


if __name__ == "__main__":
    unittest.main()
