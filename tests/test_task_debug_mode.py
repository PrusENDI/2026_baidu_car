"""分离式任务 debug 模式的静态回归检查，不连接实车硬件。"""

import ast
from pathlib import Path
import unittest


TASK_SOURCE = Path(__file__).resolve().parents[1] / "car_task_function.py"
TASK_DISTANCES = {
    "auto_seeding": (0.5, 0.85),
    "target_shooting_detection": (0.40, 1.45),
    "water_tower_task": (0.6, 2.0),
    "target_shooting": (0.70, 3.0),
    "crop_harvesting": (0.70, 2.3),
    "sort_and_store": (0.55, 2.0),
    "get_order": (0.60, 1.5),
    "order_delivery": (0.60, 3.25),
}


def task_functions():
    tree = ast.parse(TASK_SOURCE.read_text(encoding="utf-8"))
    return {
        node.name: node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in TASK_DISTANCES
    }


class TaskDebugModeTest(unittest.TestCase):
    def test_each_task_exposes_opt_in_debug_parameter(self):
        for name, function in task_functions().items():
            with self.subTest(task=name):
                argument_names = [argument.arg for argument in function.args.args]
                self.assertIn("debug", argument_names)
                self.assertTrue(function.args.defaults)
                self.assertIs(function.args.defaults[-1].value, False)

    def test_each_task_selects_debug_and_normal_entry_distances(self):
        for name, function in task_functions().items():
            with self.subTest(task=name):
                distance_assignment = next(
                    node
                    for node in function.body
                    if isinstance(node, ast.Assign)
                    and any(
                        isinstance(target, ast.Name)
                        and target.id == "task_distance"
                        for target in node.targets
                    )
                )
                self.assertIsInstance(distance_assignment.value, ast.IfExp)
                self.assertEqual(
                    ast.literal_eval(distance_assignment.value.body),
                    TASK_DISTANCES[name][0],
                )
                self.assertEqual(
                    ast.literal_eval(distance_assignment.value.orelse),
                    TASK_DISTANCES[name][1],
                )
                distance_calls = [
                    node
                    for node in ast.walk(function)
                    if isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "lane_dis_offset"
                    and any(
                        keyword.arg == "dis_hold"
                        and isinstance(keyword.value, ast.Name)
                        and keyword.value.id == "task_distance"
                        for keyword in node.keywords
                    )
                ]
                self.assertEqual(len(distance_calls), 1)

    def test_each_debug_task_returns_to_odometry_origin(self):
        for name, function in task_functions().items():
            with self.subTest(task=name):
                debug_guards = [
                    node
                    for node in ast.walk(function)
                    if isinstance(node, ast.If)
                    and isinstance(node.test, ast.Name)
                    and node.test.id == "debug"
                ]
                origin_moves = [
                    call
                    for guard in debug_guards
                    for call in ast.walk(guard)
                    if isinstance(call, ast.Call)
                    and isinstance(call.func, ast.Attribute)
                    and call.func.attr == "move_to_position"
                    and call.args
                    and isinstance(call.args[0], ast.List)
                    and ast.literal_eval(call.args[0]) == [0.0, 0.0, 0.0]
                ]
                self.assertEqual(len(origin_moves), 1)


if __name__ == "__main__":
    unittest.main()
