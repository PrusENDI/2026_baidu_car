import ast
import unittest
from pathlib import Path


SOURCE_PATH = Path(__file__).parents[1] / "car_task_function_all_test.py"


class ShootingFunctionPortTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tree = ast.parse(SOURCE_PATH.read_text(encoding="utf-8"))
        cls.functions = {
            node.name: node
            for node in cls.tree.body
            if isinstance(node, ast.FunctionDef)
        }
        cls.assignments = {
            target.id
            for node in cls.tree.body
            if isinstance(node, ast.Assign)
            for target in node.targets
            if isinstance(target, ast.Name)
        }

    @staticmethod
    def call_names(function):
        return {
            ast.unparse(node.func)
            for node in ast.walk(function)
            if isinstance(node, ast.Call)
        }

    @staticmethod
    def keyword_names(function):
        return {
            keyword.arg
            for node in ast.walk(function)
            if isinstance(node, ast.Call)
            for keyword in node.keywords
            if keyword.arg is not None
        }

    def test_current_shooting_logic_is_ported_without_detailed_logs(self):
        self.assertIn("TARGET_SHOOTING_DETECTION_POSES", self.assignments)
        self.assertIn("TARGET_SHOOTING_POSES", self.assignments)

        detection = self.functions["target_shooting_detection"]
        shooting = self.functions["target_shooting"]

        self.assertEqual(ast.unparse(detection.args), "debug=False")
        self.assertEqual(
            ast.unparse(shooting.args),
            "animal_list=[1, 1, 1, 0], debug=False, shooting_delta_x=None",
        )

        shooting_keywords = self.keyword_names(shooting)
        self.assertTrue(
            {
                "fixed_order_num",
                "fixed_order_direction",
                "expected_detection_count",
                "require_alignment",
                "target_index",
            }.issubset(shooting_keywords)
        )

        for function in (detection, shooting):
            calls = self.call_names(function)
            self.assertNotIn("shooting_event", calls)
            self.assertFalse(any(name.startswith("logger.") for name in calls))
            self.assertEqual(
                sum(name == "print" for name in calls),
                1,
                f"{function.name} should retain only its final result print",
            )


if __name__ == "__main__":
    unittest.main()
