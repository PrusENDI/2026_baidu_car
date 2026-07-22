"""水平轴编码器移动与播种回零保护回归检查，不连接真实机械臂。"""

import ast
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
ARM_SOURCE = ROOT / "smartcar/whalesbot/vehicle/arm/arm_base.py"
TASK_SOURCE = ROOT / "car_task_function.py"
WRAP_SOURCE = ROOT / "car_wrap_2026.py"


def method_node(name):
    tree = ast.parse(ARM_SOURCE.read_text(encoding="utf-8"))
    return next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == name
    )


class HorizontalMotionSafetyTest(unittest.TestCase):
    def test_all_right_seed_targets_apply_fixed_pre_detection_offset_with_odometry_logs(self):
        tree = ast.parse(TASK_SOURCE.read_text(encoding="utf-8"))
        method = next(
            node for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == "auto_seeding"
        )
        assignments = {
            target.id: ast.literal_eval(node.value)
            for node in method.body
            if isinstance(node, ast.Assign)
            for target in node.targets
            if isinstance(target, ast.Name)
            and target.id in {
                "RIGHT_SEEDING_PRE_DETECTION_OFFSET",
                "RIGHT_SEEDING_ARM_X_BOUNDS",
            }
        }

        self.assertEqual(
            assignments.get("RIGHT_SEEDING_PRE_DETECTION_OFFSET"),
            [0.0, -0.06, 0.0],
        )
        self.assertEqual(assignments["RIGHT_SEEDING_ARM_X_BOUNDS"], (0.24, 0.260))

        grasp_loop = [node for node in method.body if isinstance(node, ast.For)][1]
        grasp_source = ast.get_source_segment(
            TASK_SOURCE.read_text(encoding="utf-8"), grasp_loop
        )
        base_move_index = grasp_source.index("my_car.move_to_position(base_pose)")
        offset_move_index = grasp_source.index(
            "my_car.move_for(RIGHT_SEEDING_PRE_DETECTION_OFFSET)"
        )
        detection_index = grasp_source.index("my_car.move_to_detection_target(")
        self.assertLess(base_move_index, offset_move_index)
        self.assertLess(offset_move_index, detection_index)
        self.assertIn("right_pre_detection_offset", grasp_source)
        self.assertIn("right_post_detection_offset", grasp_source)
        self.assertIn(
            'f"offset={RIGHT_SEEDING_PRE_DETECTION_OFFSET}', grasp_source
        )

    def test_normal_move_never_redefines_encoder_zero(self):
        method = method_node("move_x_position")
        assigned_attributes = {
            target.attr
            for node in ast.walk(method)
            if isinstance(node, ast.Assign)
            for target in node.targets
            if isinstance(target, ast.Attribute)
        }

        self.assertNotIn("x_pose_start", assigned_attributes)

    def test_normal_move_resets_per_command_state_and_restores_speed_limit(self):
        method = method_node("move_x_position")
        source = ast.dump(method)

        self.assertIn("attr='x_stop_flag'", source)
        self.assertIn("attr='x_pid_flag'", source)
        self.assertIn("attr='x_pose_now'", source)
        self.assertIn("attr='x_pose_last'", source)
        self.assertIn("attr='x_distance_change'", source)
        self.assertIn("attr='x_velocity_limit'", source)

    def test_normal_move_has_success_failure_and_final_motor_stop(self):
        method = method_node("move_x_position")
        returns = [
            node.value.value
            for node in ast.walk(method)
            if isinstance(node, ast.Return)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, bool)
        ]
        finalizers = [
            node.finalbody
            for node in ast.walk(method)
            if isinstance(node, ast.Try)
        ]

        self.assertIn(True, returns)
        self.assertIn(False, returns)
        self.assertTrue(finalizers, "move_x_position() 必须用 finally 停止水平轴")
        finalizer_source = " ".join(ast.dump(node) for node in finalizers[0])
        self.assertIn("attr='x_speed'", finalizer_source)
        self.assertIn("Constant(value=0", finalizer_source)

    def test_only_reset_x_redefines_encoder_zero(self):
        reset_method = method_node("reset_x")
        self.assertIn("attr='x_pose_start'", ast.dump(reset_method))

    def test_auto_seeding_rehomes_on_right_and_guards_left_rotation(self):
        tree = ast.parse(TASK_SOURCE.read_text(encoding="utf-8"))
        method = next(
            node for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == "auto_seeding"
        )
        source = ast.dump(method)

        retract_index = source.rindex("attr='reset_x', ctx=Load())")
        left_index = source.index("value='LEFT'", retract_index)
        self.assertLess(retract_index, left_index)
        self.assertIn("UnaryOp(op=Not(), operand=Call", source[retract_index - 200:retract_index + 300])
        self.assertIn("Raise", source[retract_index - 200:retract_index + 500])

    def test_alignment_adjustment_returns_horizontal_move_result(self):
        tree = ast.parse(WRAP_SOURCE.read_text(encoding="utf-8"))
        method = next(
            node for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == "adjust_arm_position"
        )
        guarded_returns = [
            node for node in ast.walk(method)
            if isinstance(node, ast.Return)
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Attribute)
            and node.value.func.attr == "move_x_position"
        ]

        self.assertEqual(len(guarded_returns), 2)

    def test_auto_seeding_only_guards_left_placement_adjustment(self):
        tree = ast.parse(TASK_SOURCE.read_text(encoding="utf-8"))
        method = next(
            node for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == "auto_seeding"
        )
        guarded_adjustments = [
            node for node in ast.walk(method)
            if isinstance(node, ast.If)
            and isinstance(node.test, ast.UnaryOp)
            and isinstance(node.test.op, ast.Not)
            and isinstance(node.test.operand, ast.Call)
            and isinstance(node.test.operand.func, ast.Attribute)
            and node.test.operand.func.attr == "adjust_arm_position"
        ]

        self.assertEqual(len(guarded_adjustments), 1)

    def test_auto_seeding_rehomes_before_right_rotation_then_extends(self):
        tree = ast.parse(TASK_SOURCE.read_text(encoding="utf-8"))
        method = next(
            node for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == "auto_seeding"
        )
        loops = [node for node in method.body if isinstance(node, ast.For)]
        grasp_loop = loops[1]
        calls = sorted(
            (
                node.lineno,
                node,
            )
            for node in ast.walk(grasp_loop)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
        )

        right_rotation_line = next(
            line for line, node in calls
            if node.func.attr == "set_arm_pose"
            and any(
                keyword.arg == "arm"
                and isinstance(keyword.value, ast.Constant)
                and keyword.value.value == "RIGHT"
                for keyword in node.keywords
            )
        )
        pre_rotation_home_lines = [
            line for line, node in calls
            if node.func.attr == "reset_x"
            and not node.args
            and line < right_rotation_line
        ]
        self.assertTrue(
            pre_rotation_home_lines,
            "旋转 RIGHT 前必须通过 reset_x() 确认物理回收端并重新归零",
        )
        pre_rotation_home_line = pre_rotation_home_lines[0]
        post_rotation_preset_line = next(
            line for line, node in calls
            if node.func.attr == "move_x_position"
            and len(node.args) == 1
            and isinstance(node.args[0], ast.Name)
            and node.args[0].id == "RIGHT_SEEDING_ARM_X_PRESET"
        )

        self.assertLess(pre_rotation_home_line, right_rotation_line)
        self.assertLess(right_rotation_line, post_rotation_preset_line)

        preset_assignment = next(
            node for node in method.body
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name)
                and target.id == "RIGHT_SEEDING_ARM_X_PRESET"
                for target in node.targets
            )
        )
        self.assertEqual(preset_assignment.value.value, 0.25)

    def test_release_telemetry_reads_offsets_for_current_label(self):
        source = TASK_SOURCE.read_text(encoding="utf-8")
        release_section = source.split(
            'if not my_car.adjust_arm_position():', 1
        )[1].split('my_car.arm.move_y_position(0.04)', 1)[0]

        self.assertIn("my_car.get_detection_results()", release_section)
        self.assertIn("det[2] == label", release_section)
        self.assertIn("dx=place_dx", release_section)
        self.assertIn("dy=place_dy", release_section)


if __name__ == "__main__":
    unittest.main()
