#!/usr/bin/env python3
"""MC602 水平轴运动期间的 angle16 总线舵机通信测试。"""

import argparse
import struct
from threading import Event, Thread


TARGET_X = 0.10
MOVE_TIMEOUT = 15.0
SERVO_TRIGGER_X = 0.03
TRIGGER_POLL_INTERVAL = 0.01
SERVO_PORT = 2
SERVO_SPEED = 60
SERVO_START_ANGLE = 0
SERVO_ANGLE = -92
FRAME_FORMATS = {
    1: ("angle8", "bbbbb"),
    2: ("angle16", "bbbbh"),
}


def build_expected_frame(angle_bytes, angle=SERVO_ANGLE):
    """构造本次测试应由 MC602 USB 驱动发送的完整帧。"""
    if angle_bytes not in FRAME_FORMATS:
        raise ValueError(f"unsupported angle byte count: {angle_bytes}")

    angle_format = "b" if angle_bytes == 1 else "h"
    payload = struct.pack(
        "<BBBBB",
        0x06,  # servo_bus device ID
        0x02,  # set operation
        SERVO_PORT,
        0x01,  # position mode
        SERVO_SPEED,
    )
    payload += struct.pack(f"<{angle_format}", angle)
    return b"\x77\x68" + bytes([len(payload) + 4]) + payload + b"\x0a"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "MC602 并发测试：X 移向 0.10 m，经过 0.03 m 时单次发送 "
            "angle16 RIGHT=-92°。"
        )
    )
    parser.add_argument(
        "--angle-bytes",
        type=int,
        choices=(2,),
        required=True,
        help="舵机角度字段长度；当前固件只允许 2=angle16",
    )
    return parser.parse_args(argv)


def run_test(
    angle_bytes,
    arm,
    struct_factory,
    controller_name,
    confirm=input,
    output=print,
):
    """执行一次受保护的硬件测试；依赖由调用方注入以支持无硬件测试。"""
    try:
        if angle_bytes not in FRAME_FORMATS:
            raise ValueError(f"unsupported angle byte count: {angle_bytes}")

        output(f"控制器: {controller_name}")
        if "mc602" not in str(controller_name).lower():
            output("错误: 当前控制器不是 MC602，测试终止。")
            return 2

        format_label, struct_format = FRAME_FORMATS[angle_bytes]
        expected_mid_tx = build_expected_frame(angle_bytes, SERVO_START_ANGLE)
        expected_tx = build_expected_frame(angle_bytes, SERVO_ANGLE)
        start_x = arm.x_get_position()
        output(f"起始 X 编码器位置: {start_x:.6f} m")
        output(
            f"测试格式: {format_label}, 端口: {SERVO_PORT}, "
            f"速度: {SERVO_SPEED}, 角度: {SERVO_ANGLE}°"
        )
        output(f"MID 预期 TX: {expected_mid_tx.hex(' ')}")
        output(f"并发 RIGHT 预期 TX: {expected_tx.hex(' ')}")
        output(
            "安全提示: 运行前必须人工确认水平轴位于回收端；本脚本不寻零，"
            "初始化会把当前编码器位置作为参考。确认 X=0.10 m 全程旋转无碰撞。"
        )

        answer = confirm("确认 X 起点和旋转空间安全后输入 YES，将舵机置于 MID: ")
        if answer.strip() != "YES":
            output("未收到 YES，取消测试。")
            return 5

        servo_bus_2 = arm.arm_servo.servo_bus_2
        servo_bus_2.data_struct = struct_factory(struct_format)
        servo_bus_2.last_data = None
        arm.arm_servo.set_angle(SERVO_START_ANGLE, SERVO_SPEED)
        output(f"MID last_data: {servo_bus_2.last_data!r}")

        answer = confirm("确认舵机已到 MID，输入 YES 开始 X 与舵机并发测试: ")
        if answer.strip() != "YES":
            output("未收到 YES，取消并发测试。")
            return 5

        stop_trigger = Event()
        servo_state = {
            "sent": False,
            "trigger_x": None,
            "error": None,
        }

        def send_servo_when_x_reaches_trigger():
            while not stop_trigger.wait(TRIGGER_POLL_INTERVAL):
                current_x = arm.x_pose_now
                if current_x < SERVO_TRIGGER_X:
                    continue
                servo_state["sent"] = True
                servo_state["trigger_x"] = current_x
                output(
                    f"X={current_x:.6f} m 到达触发点，"
                    f"单次发送舵机 {SERVO_ANGLE}°"
                )
                try:
                    servo_bus_2.last_data = None
                    arm.arm_servo.set_angle(SERVO_ANGLE, SERVO_SPEED)
                except BaseException as exc:
                    servo_state["error"] = exc
                return

        servo_thread = Thread(
            target=send_servo_when_x_reaches_trigger,
            name="servo-during-x-motion",
            daemon=True,
        )
        servo_thread.start()
        try:
            move_ok = arm.move_x_position(TARGET_X, out_time=MOVE_TIMEOUT)
        finally:
            stop_trigger.set()
            servo_thread.join()

        output(
            f"move_x_position({TARGET_X}, out_time={MOVE_TIMEOUT}): {move_ok}"
        )
        if servo_state["error"] is not None:
            raise servo_state["error"]
        if not servo_state["sent"]:
            output("错误: X 未到达 0.03 m 触发点，未发送并发舵机命令。")
            return 4
        if not move_ok:
            output("错误: 并发舵机命令已发送，但水平轴未到达 0.10 m。")
            return 4

        output(f"舵机触发 X 位置: {servo_state['trigger_x']:.6f} m")
        output(f"并发舵机 last_data: {servo_bus_2.last_data!r}")
        output(f"并发测试结束 X 位置: {arm.x_get_position():.6f} m")
        output("测试结束；不会自动回收水平轴。")
        return 0
    finally:
        arm.x_speed(0)


def load_hardware():
    """延迟导入项目硬件栈，避免导入本测试模块时扫描串口。"""
    from smartcar.whalesbot.vehicle.arm.arm_base import ArmController
    from smartcar.whalesbot.vehicle.base.controller_wrap import serial_wrap
    from smartcar.whalesbot.vehicle.base.mc602_ctl2 import StructData

    return ArmController, StructData, serial_wrap.dev.name


def main(argv=None):
    args = parse_args(argv)
    arm_factory, struct_factory, controller_name = load_hardware()
    arm = arm_factory()
    return run_test(
        args.angle_bytes,
        arm,
        struct_factory,
        controller_name,
    )


if __name__ == "__main__":
    raise SystemExit(main())
