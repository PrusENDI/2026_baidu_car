"""机械臂复位时序回归测试，不连接真实串口或舵机。"""

import smartcar.whalesbot.vehicle.arm.arm_base as arm_base


def test_right_angle_command_is_repeated_without_initial_settle_delay(monkeypatch):
    """总线舵机应立即发送首帧，并以短间隔重复同一角度命令。"""
    class FakeServo:
        def __init__(self):
            self.calls = []

        def set_angle(self, angle, speed):
            self.calls.append((angle, speed))

    arm = object.__new__(arm_base.ArmController)
    arm.side = "LEFT"
    arm.hand_angle_list = {"RIGHT": -93}
    arm.arm_servo = FakeServo()
    sleeps = []
    monkeypatch.setattr(arm_base.time, "sleep", sleeps.append)

    arm.set_arm_angle("RIGHT")

    assert arm.arm_servo.calls == [(-93, 80)] * 3
    assert sleeps == [0.5] * 2
