"""蔬菜专用推理通道的静态集成检查，不连接实车、摄像头或 Paddle。"""

import ast
from pathlib import Path
import unittest

import yaml


ROOT = Path(__file__).resolve().parents[1]
CAR_WRAP_PATH = ROOT / "car_wrap_2026.py"
TASK_PATH = ROOT / "car_task_function.py"
CONFIG_PATH = ROOT / "config_car.yml"
MODEL_DIR = ROOT / "smartcar" / "paddlebaidu" / "models" / "goods_8_2"

VEGETABLE_LABELS = {
    "h_dou_jiao",
    "h_fan_qie",
    "h_jin_zhen_gu",
    "h_mo_gu",
    "h_qin_cai",
    "h_qing_jiao",
    "h_tu_dou",
    "h_xi_lan_hua",
    "h_you_cai",
}


def function_node(path, name):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return next(
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == name
    )


class DedicatedGoodsDetectorTest(unittest.TestCase):
    def test_goods_inference_service_has_independent_model_and_port(self):
        config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
        goods = next(item for item in config["infer_cfg"] if item["name"] == "goods")
        self.assertEqual(goods["infer_type"], "YoloeInfer")
        self.assertEqual(goods["model_dir"], "goods_8_2")
        self.assertEqual(goods["port"], 5005)
        self.assertEqual(goods["img_size"], [640, 640])
        self.assertEqual(goods["run_mode"], "paddle")

    def test_goods_model_files_and_preprocessing_match_submission(self):
        for filename in ("model.pdmodel", "model.pdiparams", "infer_cfg.yml"):
            model_file = MODEL_DIR / filename
            self.assertTrue(model_file.is_file(), filename)
            self.assertGreater(model_file.stat().st_size, 0, filename)

        infer_config = yaml.safe_load(
            (MODEL_DIR / "infer_cfg.yml").read_text(encoding="utf-8")
        )
        normalize = next(
            item
            for item in infer_config["Preprocess"]
            if item["type"] == "NormalizeImage"
        )
        self.assertEqual(normalize["norm_type"], "mean_std")
        self.assertEqual(normalize["mean"], [0.485, 0.456, 0.406])
        self.assertEqual(normalize["std"], [0.229, 0.224, 0.225])
        self.assertTrue(VEGETABLE_LABELS.issubset(infer_config["label_list"]))

    def test_car_initializes_goods_client(self):
        method = function_node(CAR_WRAP_PATH, "paddle_infer_init")
        goods_assignments = [
            node
            for node in ast.walk(method)
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Attribute)
                and isinstance(target.value, ast.Name)
                and target.value.id == "self"
                and target.attr == "goods_det"
                for target in node.targets
            )
            and isinstance(node.value, ast.Call)
            and node.value.args
            and isinstance(node.value.args[0], ast.Constant)
            and node.value.args[0].value == "goods"
        ]
        self.assertEqual(len(goods_assignments), 1)

    def test_alignment_forwards_selected_detector(self):
        method = function_node(CAR_WRAP_PATH, "move_to_detection_target")
        self.assertIn("detector", [argument.arg for argument in method.args.args])
        detection_calls = [
            node
            for node in ast.walk(method)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "get_detection_results"
        ]
        self.assertEqual(len(detection_calls), 1)
        detector_keyword = next(
            keyword
            for keyword in detection_calls[0].keywords
            if keyword.arg == "detector"
        )
        self.assertIsInstance(detector_keyword.value, ast.Name)
        self.assertEqual(detector_keyword.value.id, "detector")

    def test_find_goods_uses_goods_detector_for_every_retry(self):
        function = function_node(TASK_PATH, "find_goods")
        alignment_calls = [
            node
            for node in ast.walk(function)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "move_to_detection_target"
        ]
        self.assertEqual(len(alignment_calls), 4)
        for call in alignment_calls:
            detector = next(
                keyword.value for keyword in call.keywords if keyword.arg == "detector"
            )
            self.assertIsInstance(detector, ast.Attribute)
            self.assertEqual(detector.attr, "goods_det")
            self.assertIsInstance(detector.value, ast.Name)
            self.assertEqual(detector.value.id, "my_car")


if __name__ == "__main__":
    unittest.main()
