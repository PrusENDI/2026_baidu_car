import importlib


def test_chassis_import_does_not_initialize_hardware():
    module = importlib.import_module("debug.chassis_check")
    assert hasattr(module, "main")


def test_arm_import_does_not_initialize_hardware():
    module = importlib.import_module("debug.arm_check")
    assert hasattr(module, "main")


def test_chassis_move_is_dry_run_without_apply(capsys):
    from debug import chassis_check

    code = chassis_check.main(["move", "--x", "0.1", "--seconds", "0.1"])

    assert code == 0
    assert "DRY-RUN chassis command" in capsys.readouterr().out


def test_arm_reset_is_dry_run_without_apply(capsys):
    from debug import arm_check

    code = arm_check.main(["reset"])

    assert code == 0
    assert "DRY-RUN arm command" in capsys.readouterr().out
