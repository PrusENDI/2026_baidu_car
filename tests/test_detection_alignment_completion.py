"""视觉对齐完成条件回归测试，不导入车辆和串口模块。"""

import ast
from pathlib import Path
import unittest


SOURCE_PATH = Path(__file__).resolve().parents[1] / "car_wrap_2026.py"
CONFIG_PATH = Path(__file__).resolve().parents[1] / "config_car.yml"


def load_alignment_helper():
    tree = ast.parse(SOURCE_PATH.read_text(encoding="utf-8"))
    helper = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
        and node.name == "_alignment_axis_reached"
    )
    helper.decorator_list = []
    namespace = {}
    exec(compile(ast.Module(body=[helper], type_ignores=[]), str(SOURCE_PATH), "exec"), namespace)
    return namespace["_alignment_axis_reached"]


def load_camera_calibration_helper():
    tree = ast.parse(SOURCE_PATH.read_text(encoding="utf-8"))
    helper = next((
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
        and node.name == "_camera_calibrated_delta_x"
    ), None)
    if helper is None:
        return None
    helper.decorator_list = []
    namespace = {}
    exec(compile(ast.Module(body=[helper], type_ignores=[]), str(SOURCE_PATH), "exec"), namespace)
    return namespace["_camera_calibrated_delta_x"]


class DetectionAlignmentCompletionTest(unittest.TestCase):
    def test_side_camera_physical_left_offset_is_configurable(self):
        config_source = CONFIG_PATH.read_text(encoding="utf-8")
        self.assertIn("camera_calibration:", config_source)
        self.assertIn("side_center_offset_x_m: -0.06", config_source)

    def test_physical_left_offset_moves_effective_dx_target_right(self):
        calibrated = load_camera_calibration_helper()
        self.assertIsNotNone(calibrated)
        self.assertAlmostEqual(calibrated(0.0, -0.025), 0.025 / 0.33)
        self.assertAlmostEqual(calibrated(-0.1, -0.025), -0.1 + 0.025 / 0.33)
        self.assertEqual(calibrated(-0.1, 0.0), -0.1)

    def test_nonzero_target_offset_is_compared_against_setpoint(self):
        reached = load_alignment_helper()
        self.assertTrue(reached(-0.105, -0.1, 0.04))
        self.assertTrue(reached(-0.048, -0.05, 0.02))
        self.assertFalse(reached(0.0, -0.1, 0.04))

    def test_success_branch_returns_detection_before_timeout(self):
        tree = ast.parse(SOURCE_PATH.read_text(encoding="utf-8"))
        method = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef)
            and node.name == "move_to_detection_target"
        )
        success_if = next(
            node
            for node in ast.walk(method)
            if isinstance(node, ast.If)
            and isinstance(node.test, ast.BoolOp)
            and all(isinstance(v, ast.Name) for v in node.test.values)
            and {v.id for v in node.test.values} == {"flag_x", "flag_y"}
        )
        self.assertTrue(
            any(isinstance(node, ast.Return) for node in ast.walk(success_if)),
            "对齐成功分支必须立即返回检测结果，不能继续循环到 timeout",
        )


if __name__ == "__main__":
    unittest.main()
