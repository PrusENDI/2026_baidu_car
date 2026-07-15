import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


MODULE_PATH = Path(__file__).parents[1] / "tools" / "evaluate_lane_models.py"
SPEC = importlib.util.spec_from_file_location("evaluate_lane_models", MODULE_PATH)
evaluate_lane_models = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(evaluate_lane_models)


class DatasetTests(unittest.TestCase):
    def test_load_dataset_ignores_missing_images_and_utf8_bom(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "0000.jpg").write_bytes(b"image")
            records = [
                {"img_path": "0000.jpg", "state": [0.15, 0.0, 0.2]},
                {"img_path": "0001.jpg", "state": [0.15, 0.0, -0.2]},
            ]
            (root / "data.json").write_text(json.dumps(records), encoding="utf-8-sig")

            samples, missing = evaluate_lane_models.load_dataset(root)

            self.assertEqual(1, len(samples))
            self.assertEqual(["0001.jpg"], missing)
            self.assertEqual("0000.jpg", samples[0]["img_path"])


if __name__ == "__main__":
    unittest.main()
