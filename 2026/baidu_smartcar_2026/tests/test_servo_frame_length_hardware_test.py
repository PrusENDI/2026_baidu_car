"""MC602 舵机帧长度硬件测试脚本的无硬件回归测试。"""

import sys
import subprocess
from pathlib import Path

import pytest

import testangle


class FakeStructData:
    def __init__(self, format_name):
        self.format_name = format_name


class FakeServoBus2:
    def __init__(self):
        self.data_struct = None
        self.last_data = [1, 60, -92]


class FakeServo:
    def __init__(self, events, fail=False):
        self.events = events
        self.fail = fail
        self.calls = []
        self.servo_bus_2 = FakeServoBus2()

    def set_angle(self, angle, speed):
        self.calls.append((angle, speed))
        self.events.append(("set_angle", angle, speed))
        if self.fail:
            raise RuntimeError("servo send failed")


class FakeArm:
    def __init__(self, reset_result=True, move_result=True, servo_fail=False):
        self.events = []
        self.reset_result = reset_result
        self.move_result = move_result
        self.stop_calls = []
        self.arm_servo = FakeServo(self.events, fail=servo_fail)

    def reset_x(self, **_kwargs):
        self.events.append("reset_x")
        return self.reset_result

    def move_x_position(self, target, **_kwargs):
        self.events.append(("move_x_position", target))
        return self.move_result

    def x_get_position(self):
        self.events.append("x_get_position")
        return 0.10

    def x_speed(self, speed):
        self.stop_calls.append(speed)
        self.events.append(("x_speed", speed))


def run_with_fakes(angle_bytes, arm, controller_name="mc602", answer="YES"):
    outputs = []

    def confirm(_prompt):
        arm.events.append("confirm")
        return answer

    def struct_factory(format_name):
        arm.events.append(("struct", format_name))
        return FakeStructData(format_name)

    result = testangle.run_test(
        angle_bytes,
        arm,
        struct_factory,
        controller_name,
        confirm=confirm,
        output=outputs.append,
    )
    return result, outputs


def test_import_does_not_initialize_vehicle_hardware():
    project_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys, testangle; "
                "assert not any(name.startswith('smartcar.whalesbot.vehicle') "
                "for name in sys.modules)"
            ),
        ],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_angle8_expected_frame():
    assert testangle.build_expected_frame(1) == bytes.fromhex(
        "77 68 0a 06 02 02 01 3c a4 0a"
    )


def test_angle16_expected_frame():
    assert testangle.build_expected_frame(2) == bytes.fromhex(
        "77 68 0b 06 02 02 01 3c a4 ff 0a"
    )


def test_unsupported_angle_size_is_rejected():
    with pytest.raises(ValueError, match="unsupported angle byte count"):
        testangle.build_expected_frame(3)


@pytest.mark.parametrize("argv", [[], ["--angle-bytes", "3"]])
def test_cli_requires_a_supported_angle_size(argv):
    with pytest.raises(SystemExit) as exc_info:
        testangle.parse_args(argv)
    assert exc_info.value.code == 2


def test_non_mc602_controller_is_rejected_before_motion():
    arm = FakeArm()

    result, _outputs = run_with_fakes(2, arm, controller_name="mc601")

    assert result != 0
    assert arm.events == [("x_speed", 0)]
    assert arm.arm_servo.calls == []
    assert arm.stop_calls == [0]


def test_homing_failure_does_not_send_servo_command():
    arm = FakeArm(reset_result=False)

    result, _outputs = run_with_fakes(2, arm)

    assert result != 0
    assert arm.events == ["reset_x", ("x_speed", 0)]
    assert arm.arm_servo.calls == []
    assert arm.stop_calls == [0]


def test_x_move_failure_does_not_send_servo_command():
    arm = FakeArm(move_result=False)

    result, _outputs = run_with_fakes(2, arm)

    assert result != 0
    assert arm.events == [
        "reset_x",
        ("move_x_position", 0.10),
        ("x_speed", 0),
    ]
    assert arm.arm_servo.calls == []
    assert arm.stop_calls == [0]


def test_operator_cancellation_does_not_send_servo_command():
    arm = FakeArm()

    result, _outputs = run_with_fakes(2, arm, answer="no")

    assert result != 0
    assert arm.events == [
        "reset_x",
        ("move_x_position", 0.10),
        "x_get_position",
        "confirm",
        ("x_speed", 0),
    ]
    assert arm.arm_servo.calls == []
    assert arm.stop_calls == [0]


@pytest.mark.parametrize(
    ("angle_bytes", "format_name", "expected_tx"),
    [
        (1, "bbbbb", "77 68 0a 06 02 02 01 3c a4 0a"),
        (2, "bbbbh", "77 68 0b 06 02 02 01 3c a4 ff 0a"),
    ],
)
def test_success_uses_selected_format_and_sends_once(
    angle_bytes, format_name, expected_tx
):
    arm = FakeArm()

    result, outputs = run_with_fakes(angle_bytes, arm)

    assert result == 0
    assert arm.events == [
        "reset_x",
        ("move_x_position", 0.10),
        "x_get_position",
        "confirm",
        ("struct", format_name),
        ("set_angle", -92, 60),
        "x_get_position",
        ("x_speed", 0),
    ]
    assert arm.arm_servo.calls == [(-92, 60)]
    assert arm.arm_servo.servo_bus_2.data_struct.format_name == format_name
    assert arm.stop_calls == [0]
    assert any(expected_tx in line for line in outputs)
    assert any("[1, 60, -92]" in line for line in outputs)


def test_servo_exception_still_stops_x_axis():
    arm = FakeArm(servo_fail=True)

    with pytest.raises(RuntimeError, match="servo send failed"):
        run_with_fakes(2, arm)

    assert arm.arm_servo.calls == [(-92, 60)]
    assert arm.stop_calls == [0]
    assert arm.events[-1] == ("x_speed", 0)
