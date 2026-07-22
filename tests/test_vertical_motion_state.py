"""竖直轴连续移动状态回归检查，不连接真实机械臂。"""

import ast
from pathlib import Path
import unittest


SOURCE = Path(__file__).resolve().parents[1] / "smartcar/whalesbot/vehicle/arm/arm_base.py"


class VerticalMotionStateTest(unittest.TestCase):
    @staticmethod
    def _move_y_position_method():
        tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
        return next(
            node for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == "move_y_position"
        )

    def test_move_y_position_resets_state_before_new_target(self):
        method = self._move_y_position_method()
        setpoint_index = next(
            index for index, node in enumerate(method.body)
            if isinstance(node, ast.Assign)
            and isinstance(node.targets[0], ast.Attribute)
            and node.targets[0].attr == "setpoint"
        )
        assignments = {
            node.targets[0].attr: node.value
            for node in method.body[:setpoint_index]
            if isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Attribute)
            and isinstance(node.targets[0].value, ast.Name)
            and node.targets[0].value.id == "self"
        }

        required_assignments = {
            "y_stop_flag",
            "y_pid_flag",
            "y_pose_now",
            "y_pose_last",
            "y_distance_change",
        }
        self.assertTrue(
            required_assignments.issubset(assignments),
            f"缺少竖直轴状态重置: "
            f"{sorted(required_assignments - assignments.keys())}",
        )
        self.assertEqual(
            ast.dump(assignments["y_stop_flag"]),
            "Call(func=Name(id='CountRecord', ctx=Load()), "
            "args=[Constant(value=10)], keywords=[])",
        )
        self.assertEqual(
            ast.dump(assignments["y_pid_flag"]),
            "Call(func=Name(id='CountRecord', ctx=Load()), "
            "args=[Constant(value=5)], keywords=[])",
        )
        self.assertEqual(
            ast.dump(assignments["y_pose_now"]),
            "Call(func=Attribute(value=Name(id='self', ctx=Load()), "
            "attr='y_get_position', ctx=Load()), args=[], keywords=[])",
        )
        self.assertEqual(
            ast.dump(assignments["y_pose_last"]),
            "Attribute(value=Name(id='self', ctx=Load()), "
            "attr='y_pose_now', ctx=Load())",
        )
        self.assertEqual(ast.dump(assignments["y_distance_change"]), "Constant(value=0)")

    def test_move_y_position_paces_each_unfinished_iteration(self):
        method = self._move_y_position_method()
        loop = next(node for node in method.body if isinstance(node, ast.While))
        final_statement = loop.body[-1]

        self.assertIsInstance(final_statement, ast.Expr)
        self.assertEqual(
            ast.dump(final_statement.value),
            "Call(func=Attribute(value=Name(id='time', ctx=Load()), "
            "attr='sleep', ctx=Load()), args=[Constant(value=0.05)], keywords=[])",
        )


if __name__ == "__main__":
    unittest.main()
