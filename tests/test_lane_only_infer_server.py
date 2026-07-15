import importlib.util
from pathlib import Path
import unittest


PROJECT_ROOT = Path(__file__).parents[1]
path = PROJECT_ROOT / "tools" / "lane_only_infer_server.py"
spec = importlib.util.spec_from_file_location("lane_only_infer_server", path)
server = importlib.util.module_from_spec(spec)
spec.loader.exec_module(server)


class LaneOnlyInferServerTests(unittest.TestCase):
    def test_model_paths_select_2026_lane_weights(self):
        model, params = server.model_paths(PROJECT_ROOT)
        self.assertEqual("cnn_lane.pdmodel", model.name)
        self.assertEqual("cnn_lane.pdiparams", params.name)
        self.assertIn("smartcar/paddlebaidu/models/lane_model", str(model).replace("\\", "/"))


if __name__ == "__main__":
    unittest.main()
