import importlib.util
from pathlib import Path
import unittest


path = Path(__file__).parents[1] / "tools" / "safe_lane_test.py"
spec = importlib.util.spec_from_file_location("safe_lane_test", path)
safe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(safe)


class SafetyTests(unittest.TestCase):
    def test_rejects_non_finite_or_excessive_model_output(self):
        self.assertEqual("invalid_model_output", safe.validate_output(float("nan"), 0.0, 1.0))
        self.assertEqual("model_output_limit", safe.validate_output(1.1, 0.0, 1.0))
        self.assertIsNone(safe.validate_output(0.2, -0.3, 1.0))


if __name__ == "__main__":
    unittest.main()
