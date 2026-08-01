"""后端进程识别测试：避免误认其他 worktree 的 infer_back_end.py。"""

from smartcar.paddlebaidu.infer_cs.base.infer_front import (
    command_contains_script,
    is_process_in_project,
)


def test_command_contains_script_requires_a_real_script_argument():
    assert command_contains_script(
        ["python3", "/opt/app/infer_back_end.py", "--port", "5001"],
        "infer_back_end.py",
    )
    assert not command_contains_script(
        ["python3", "-c", "print('infer_back_end.py')"],
        "infer_back_end.py",
    )


def test_process_must_belong_to_current_project_root():
    project_root = "/home/jetson/workspaces/baidu_car_2026_official_run_copy"
    assert is_process_in_project(
        project_root,
        "/home/jetson/workspaces/baidu_car_2026_official_run_copy/smartcar/paddlebaidu/infer_cs/base",
    )
    assert not is_process_in_project(
        project_root,
        "/home/jetson/workspaces/baidu_smart_2026_7_17/smartcar/paddlebaidu/infer_cs/base",
    )
