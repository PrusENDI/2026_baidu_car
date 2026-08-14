#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
机械臂控制模块

该模块实现了机械臂的运动控制, 包括竖直方向、水平方向的移动, 以及手部的控制。
"""

import math
import time
import numpy as np
import yaml
import os
import sys
from typing import Union

# 添加上本地目录
dir_this = os.path.abspath(os.path.dirname(__file__))
sys.path.append(dir_this)
# 添加上两层目录
dir_root = os.path.abspath(os.path.join(dir_this, '..', '..'))
sys.path.append(dir_root)

# 导入自定义模块
from ...tools import get_yaml, limit_val, CountRecord, PID, logger
from .. import (
    AnalogInput, MotorWrap, Key4Btn, ServoPwm,
    ServoBus, StepperWrap, PoutD
)
from ..base.controller_wrap import AnalogInput2

# 常量定义



POSITION_ERROR_THRESHOLD = 1e-3 # 位置误差阈值
STOP_CHECK_THRESHOLD = 1e-4 # 停止检查阈值
ARM_SERVO_COMMAND_RETRIES = 3
ARM_SERVO_COMMAND_RETRY_DELAY = 0.5


def get_path_relative(*args):
    """
    获取相对路径

    Args:
        *args: 路径组件

    Returns:
        str: 完整的绝对路径
    """
    local_dir = os.path.abspath(os.path.dirname(__file__))
    return os.path.join(local_dir, *args)


class ArmController:
    """
    机械臂控制类, 负责机械臂的运动控制和状态管理

    Attributes:
        config: 配置参数
        motor_y: 竖直方向步进电机
        motor_x: 水平方向电机
        hand_servo: 手部舵机
        arm_servo: 手臂舵机
        pump: 气泵控制
        valve: 阀门控制
        y_pose_now: 当前竖直位置
        x_pose_now: 当前水平位置
        side: 机械臂方向
    """

    def __init__(self) -> None:
        """
        初始化机械臂控制类
        """
        # 初始化对象状态。
        self.yaml_path = get_path_relative("arm_cfg.yaml")

        with open(self.yaml_path, 'r') as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)

        
        '''机械臂的长度'''
        self.arm_length: float = self.config["arm_length"]
        # 初始化各部分参数
        self.y_params_init(**self.config["vert_cfg"])
        self.x_params_init(**self.config["horiz_cfg"])
        self.hand_params_init(**self.config["hand_cfg"])
        self.position_params_init(**self.config["pos_cfg"])


    def y_params_init(self, motor, limit_port, pid, threshold):
        """
        初始化竖直方向电机参数

        Args:
            motor: 电机配置
            limit_port: 限位传感器端口
            pid: PID参数
            threshold: 位置阈值
        """
        # 初始化相关资源。
        self.motor_y = StepperWrap(**motor)
        self.y_limit_sensor = AnalogInput(limit_port)

        self.y_pose_start = self.motor_y.get_dis()
        self.y_pose_now = 0
        self.y_pid = PID(**pid)
        self.y_velocity_limit = pid['output_limits']
        self.y_distance_change = 0
        self.y_threshold = threshold  # 竖直位置阈值
        self.y_pose_last = 0

        self.y_pid_flag = CountRecord(5)
        self.y_stop_flag = CountRecord(10)

    def y_reset_check(self):
        """
        检查竖直方向是否到达限位

        Returns:
            bool: 是否到达限位
        """
        # 复位相关状态。
        return self.y_limit_sensor.read() > 1000  # 磁敏传感器的值大于1000时, 则认为到达限位位置

    def y_stop_check(self):
        """
        检查竖直方向是否停止

        Returns:
            bool: 是否停止
        """
        # 停止相关流程。
        return self.y_stop_flag(
            abs(self.y_distance_change) < STOP_CHECK_THRESHOLD
        )
    def y_get_position(self):
        # 获取相关数据。
        self.y_pose_now = (
            self.motor_y.get_dis() - self.y_pose_start
        )
        return self.y_pose_now

    def y_pid_moveto(self, target_pose):
        """
        使用PID控制竖直方向移动

        Args:
            target_pose: 目标位置 (单位: m)

        Returns:
            bool: 是否到达目标位置
        """
        # 记录当前位置, 并更新上次的位置
        self.y_pose_now = (
            self.motor_y.get_dis() - self.y_pose_start
        )
        self.y_distance_change = (
            self.y_pose_now - self.y_pose_last
        )
        self.y_pose_last = self.y_pose_now

        error = target_pose - self.y_pose_now
        velocity = self.y_pid(self.y_pose_now)

        self.y_speed(velocity)

        if self.y_pid_flag(abs(error) < POSITION_ERROR_THRESHOLD):
            return True
        else:
            return False

    def reset_y(self):
        """
        重置竖直方向位置
        """
        # 复位相关状态。
        self.y_pid.setpoint = -0.25
        while True:
            if self.y_pid_moveto(-0.25):
                break
            if self.y_reset_check():
                self.y_pose_start = self.motor_y.get_dis()
                self.y_pose_now = 0
                break
        self.y_speed(0)

    def move_y_position(self, target):
        """
        移动竖直方向指定距离

        Args:
            target: 目标位置
        """
        # 控制运动到目标状态。
        self.y_pid.setpoint = target
        while True:
            if self.y_pid_moveto(target):
                logger.info(f"移动到高度{target}")
                break
            if self.y_stop_check():
                logger.info(f"移到高度{target}过程中检测到停止")
        self.y_speed(0)

    def x_params_init(self, motor, pid, threshold, homing):
        """初始化水平轴及微动开关归零参数。"""
        self.motor_x = MotorWrap(**motor)
        self.x_pid = PID(**pid)
        self.x_velocity_limit = pid['output_limits']
        self.x_pose_start = self.motor_x.get_dis()
        self.x_pose_now = 0
        self.x_threshold = threshold
        self.x_pose_last = 0
        self.x_distance_change = 0
        self.x_stop_flag = CountRecord(10)
        self.x_pid_flag = CountRecord(5)

        limit_port = homing.get("limit_port")
        active = homing.get("active")
        stable_samples = homing.get("stable_samples")
        try:
            trigger_threshold = float(homing.get("threshold"))
            homing_speed = float(homing.get("speed"))
            homing_timeout = float(homing.get("timeout"))
            safe_position = float(homing.get("safe_position"))
        except (TypeError, ValueError) as exc:
            raise ValueError("X 轴归零数值配置无效") from exc

        if isinstance(limit_port, bool) or not isinstance(limit_port, int) or not 1 <= limit_port <= 3:
            raise ValueError("X 轴归零端口必须是 AI1、AI2 或 AI3")
        if active not in ("above", "below"):
            raise ValueError("X 轴归零 active 必须是 above 或 below")
        if not math.isfinite(trigger_threshold):
            raise ValueError("X 轴归零阈值必须是有限数值")
        if isinstance(stable_samples, bool) or not isinstance(stable_samples, int) or stable_samples <= 0:
            raise ValueError("X 轴归零 stable_samples 必须是正整数")
        if not math.isfinite(homing_speed) or homing_speed <= 0:
            raise ValueError("X 轴归零速度必须是正的有限数值")
        if self.x_velocity_limit[0] >= 0 or homing_speed > abs(self.x_velocity_limit[0]):
            raise ValueError("X 轴归零速度超出电机负向速度范围")
        if not math.isfinite(homing_timeout) or homing_timeout <= 0:
            raise ValueError("X 轴归零超时必须是正的有限数值")
        if (not math.isfinite(safe_position)
                or safe_position <= self.x_threshold[0]
                or safe_position > self.x_threshold[1]):
            raise ValueError("X 轴安全回收位置必须位于有效行程内并离开机械零点")

        self.x_limit_port = limit_port
        self.x_limit_active = active
        self.x_limit_threshold = trigger_threshold
        self.x_limit_stable_samples = stable_samples
        self.x_homing_speed = homing_speed
        self.x_homing_timeout = homing_timeout
        self.x_safe_position = safe_position
        self.x_limit_sensor = AnalogInput2(limit_port)
        self.x_zero_valid = False

    def _read_x_limit_fresh(self):
        """读取一次新鲜限位样本。"""
        mc602_sensor = getattr(self.x_limit_sensor, "sensor_2", None)
        if mc602_sensor is None or not hasattr(mc602_sensor, "last_data"):
            raise RuntimeError("X 轴限位传感器不支持新鲜采样检查")
        mc602_sensor.last_data = None
        try:
            raw = float(self.x_limit_sensor.read())
        except (TypeError, ValueError) as exc:
            if mc602_sensor.last_data is None:
                raise RuntimeError("MC602 未返回新的 X 轴限位样本") from exc
            raise RuntimeError("X 轴限位样本无法转换为数值") from exc
        if mc602_sensor.last_data is None:
            raise RuntimeError("MC602 未返回新的 X 轴限位样本")
        if not math.isfinite(raw):
            raise RuntimeError(f"X 轴限位样本不是有限数值 raw={raw!r}")
        return raw

    def _x_limit_is_triggered(self, raw):
        """根据配置判断 X 轴微动开关是否触发。"""
        if self.x_limit_active == "above":
            return raw >= self.x_limit_threshold
        return raw <= self.x_limit_threshold

    def x_stop_check(self):
        """
        检查水平方向是否停止

        Returns:
            bool: 是否停止
        """
        # 停止相关流程。
        return self.x_stop_flag(
            abs(self.x_distance_change) < STOP_CHECK_THRESHOLD
        )
    def x_get_position(self):
        # 获取相关数据。
        self.x_pose_now = self.motor_x.get_dis() - self.x_pose_start
        return self.x_pose_now

    def x_pid_moveto(self, target_pose):
        """
        使用PID控制水平方向移动

        Args:
            target_pose: 目标位置

        Returns:
            bool: 是否到达目标位置
        """
        # 处理 PID 控制。
        self.x_pose_now = (
            self.motor_x.get_dis() - self.x_pose_start
        )
        self.x_distance_change = (
            self.x_pose_now - self.x_pose_last
        )
        self.x_pose_last = self.x_pose_now
        error = target_pose - self.x_pose_now

        velocity = self.x_pid(self.x_pose_now)

        self.x_speed(velocity)

        if self.x_pid_flag(abs(error) < POSITION_ERROR_THRESHOLD):
            return True
        else:
            return False

    def move_x_position(self, target, out_time=6.0) -> bool:
        """按编码器位置移动水平轴，明确返回是否到位。"""
        x_min, x_max = self.x_threshold
        if target < x_min or target > x_max:
            self.x_speed(0)
            logger.error(f"水平目标越界 target={target}, allowed=[{x_min}, {x_max}]")
            return False

        self.x_stop_flag = CountRecord(10)
        self.x_pid_flag = CountRecord(5)
        self.x_pose_now = self.x_get_position()
        self.x_pose_last = self.x_pose_now
        self.x_distance_change = 0
        self.x_pid.output_limits = self.x_velocity_limit
        self.x_pid.reset()
        self.x_pid.setpoint = target
        logger.info(f"移动到水平位置{target}")
        end_time = time.time() + out_time
        try:
            while True:
                if time.time() > end_time:
                    logger.error(
                        f"水平移动超时 target={target}, actual={self.x_get_position():.6f}"
                    )
                    return False
                if self.x_pid_moveto(target):
                    return True
                if self.x_stop_check():
                    logger.error(
                        f"水平移动停滞 target={target}, actual={self.x_get_position():.6f}"
                    )
                    return False
                time.sleep(0.05)
        finally:
            self.x_speed(0)

    def reset_x(self, out_time=None, speed_limit=None) -> bool:
        """低速向机械端点移动，编码器连续停滞后建立水平零点。"""
        self.x_zero_valid = False
        try:
            self.x_speed(0)
            try:
                homing_timeout = self.x_homing_timeout if out_time is None else float(out_time)
                homing_speed = self.x_homing_speed if speed_limit is None else float(speed_limit)
            except (TypeError, ValueError):
                logger.error(
                    f"X 轴归零参数无效 out_time={out_time!r}, speed_limit={speed_limit!r}"
                )
                return False
            if not math.isfinite(homing_timeout) or homing_timeout <= 0:
                logger.error(f"X 轴归零超时参数无效 timeout={homing_timeout!r}")
                return False
            if (not math.isfinite(homing_speed) or homing_speed <= 0
                    or self.x_velocity_limit[0] >= 0
                    or homing_speed > abs(self.x_velocity_limit[0])):
                logger.error(f"X 轴归零速度参数无效 speed={homing_speed!r}")
                return False

            self.x_stop_flag = CountRecord(10)
            self.x_pid_flag = CountRecord(5)
            self.x_pose_now = self.x_get_position()
            self.x_pose_last = self.x_pose_now
            self.x_distance_change = 0
            self.x_pid.output_limits = (-homing_speed, homing_speed)
            self.x_pid.reset()
            end_time = time.time() + homing_timeout

            self.x_speed(-homing_speed)
            while True:
                if time.time() > end_time:
                    logger.error(f"水平轴堵转寻零超时 actual={self.x_get_position():.6f}")
                    return False
                self.x_speed(-homing_speed)
                self.x_pose_now = self.x_get_position()
                self.x_distance_change = self.x_pose_now - self.x_pose_last
                self.x_pose_last = self.x_pose_now
                if self.x_stop_check():
                    break
                time.sleep(0.05)

            self.x_speed(0)
            self.x_pose_start = self.motor_x.get_dis()
            self.x_pose_now = 0
            self.x_pose_last = 0
            self.x_distance_change = 0
            self.x_zero_valid = True
            logger.info("X 轴编码器堵转归零完成 x=0")
            return True
        except Exception as exc:
            logger.error(f"X 轴归零异常 type={type(exc).__name__}, error={exc}")
            return False
        finally:
            self.x_speed(0)
            self.x_pid.output_limits = self.x_velocity_limit
            self.x_pid.reset()

    def retract_x_safe(self, out_time=6.0) -> bool:
        """移动到安全位置，并确认微动开关稳定释放。"""
        try:
            self.x_speed(0)
            if not self.x_zero_valid:
                logger.error("X 轴零点无效，禁止执行安全回收")
                return False
            try:
                move_timeout = float(out_time)
            except (TypeError, ValueError):
                logger.error(f"X 轴安全回收超时参数无效 out_time={out_time!r}")
                return False
            if not math.isfinite(move_timeout) or move_timeout <= 0:
                logger.error(f"X 轴安全回收超时参数无效 timeout={move_timeout!r}")
                return False
            if not self.move_x_position(self.x_safe_position, out_time=move_timeout):
                logger.error(f"X 轴未能到达安全回收位置 target={self.x_safe_position}")
                return False

            last_raw = None
            for sample_index in range(self.x_limit_stable_samples):
                last_raw = self._read_x_limit_fresh()
                if self._x_limit_is_triggered(last_raw):
                    logger.error(
                        f"X 轴到达安全位置后开关未释放 port=AI{self.x_limit_port}, "
                        f"raw={last_raw:.1f}, sample={sample_index + 1}/"
                        f"{self.x_limit_stable_samples}"
                    )
                    return False
                if sample_index + 1 < self.x_limit_stable_samples:
                    time.sleep(0.02)
            return True
        except Exception as exc:
            logger.error(f"X 轴安全回收异常 type={type(exc).__name__}, error={exc}")
            return False
        finally:
            self.x_speed(0)

    def hand_params_init(self, hand, hand2, grap):
        """
        初始化手部参数

        Args:
            hand: 手臂舵机配置
            hand2: 手部舵机配置
            grap: 抓取机构配置
        """
        # 初始化相关资源。
        self.hand_servo = ServoPwm(hand2["port"], mode=hand2["mode"])
        self.hand_angle_list2 = hand2["angle_list"]
        self.arm_servo = ServoBus(hand["port"])
        self.hand_angle_list = hand["angle_list"]
        self.pump = PoutD(grap["port_pump"])
        self.valve = PoutD(grap["port_valve"])

    def grasp(self, value: bool):
        """
        控制抓取机构

        Args:
            value: 抓取状态, True为抓取, False为释放
        """
        # 控制夹爪或吸取动作。
        self.pump.set(value)
        self.valve.set(not value)


    def position_params_init(self, pose_enable, pose_horiz, pose_vert, side):
        """
        初始化位置参数

        Args:
            pose_enable: 是否启用位置
            pose_horiz: 水平位置
            pose_vert: 竖直位置
            side: 方向
        """
        # 初始化相关资源。
        self.pose_enable = pose_enable
        self.y_pose_start = (
            self.motor_y.get_dis() - pose_vert
        )
        self.y_pose_now = pose_vert
        self.x_pose_start = (
            self.motor_x.get_dis() - pose_horiz
        )
        self.x_pose_now = pose_horiz
        self.side = side

    def save_config(self, pose_enable=True):
        """
        保存配置到YAML文件

        Args:
            pose_enable: 是否启用位置
        """
        # 保存数据。
        self.config["pos_cfg"] = {
            "pose_enable": pose_enable,
            "pose_horiz": self.x_pose_now,
            "pose_vert": self.y_pose_now,
            "side": self.side
        }
        with open(self.yaml_path, 'w') as stream:
            yaml.dump(self.config, stream, sort_keys=False)

    def y_speed(self, velocity):
        """
        设置竖直方向速度

        Args:
            velocity: 速度值
        """
        # 处理速度控制。
        velocity = limit_val(velocity, *self.y_velocity_limit)
        self.motor_y.set_velocity(velocity)

    def x_speed(self, velocity):
        """
        设置水平方向速度

        Args:
            velocity: 速度值
        """
        # 处理速度控制。
        velocity = limit_val(velocity, *self.x_velocity_limit)
        self.motor_x.set_linear(velocity)

    def set_position_start(self, y_position):
        """
        设置起始位置

        Args:
            y_position: 竖直位置
        """
        # 启动相关流程。
        self.y_pose_start = self.y_pose_now
        self.x_pose_start = self.x_pose_now
        self.save_config()

    def set_manually(self):
        """
        使用【4键】控制机械臂
        """
        # 设置相关参数。
        self.key = Key4Btn(4)
        logger.info("Using 4 keys to control arm...")
        while True:
            value = self.key.get_key()
            if value == 1:
                self.y_speed(0.1)  # 向上
            elif value == 3:
                self.y_speed(-0.1)  # 向下
            elif value == 4:
                self.x_speed(0.1)  # 向右
            elif value == 2:
                self.x_speed(-0.1)  # 向左
            else:
                self.x_speed(0)
                self.y_speed(0)

    def reset_position(self, rehome_x=True):
        """先归零并安全回收 X 轴，再复位 Y 轴。"""
        self.set_hand_angle("UP")
        self.set_arm_angle("RIGHT")

        #if rehome_x and not self.reset_x():
            #raise RuntimeError("X 轴重置失败 stage=mechanical_homing")
        #if not self.retract_x_safe():
            #stage = "safe_retract_after_homing" if rehome_x else "safe_retract"
            #raise RuntimeError(f"X 轴重置失败 stage={stage}")
        self.reset_x()
        self.x = 0.02
        time.sleep(0.2)
        self.reset_y()
        self.y = 0.01
        self.x_get_position()
        self.save_config()
        return True

    def switch_side(self, side):
        """
        切换机械臂方向

        Args:
            side: 机械臂的方向, LEFT、RIGHT或MID
        """
        # 执行该方法的核心功能。
        if self.side != side:
            self.side = side
            logger.info(f"Changing side to {self.side}")
        else:
            return
        angle_target = self.hand_angle_list[side]
        self.set_arm_angle(angle_target, 80)
        time.sleep(0.5)

    
    
    def set_arm_angle(self, angle: Union[str, int] = "RIGHT", speed=60):
        """
        设置机械臂角度

        Args:
            angle: 目标角度，可以是字符串（"LEFT", "MID", "RIGHT"）或数字
            speed: 速度
        """
        self.side = angle
        requested_angle = angle
        target_side = angle if isinstance(angle, str) else None
        resolved_angle = angle
        if isinstance(resolved_angle, str):
            assert resolved_angle in ("LEFT", "MID", "RIGHT"), "Direction should be LEFT, MID, or RIGHT"
            resolved_angle = self.hand_angle_list[resolved_angle]

        response_received = False
        last_response = None
        for attempt in range(1, ARM_SERVO_COMMAND_RETRIES + 1):
            result = self.arm_servo.set_angle(resolved_angle, speed)
            if result is not None:
                response_received = True
                return True
                last_response = result
                logger.info(
                    f"机械臂翻转指令已获新响应 requested={requested_angle}, "
                    f"resolved={resolved_angle}, speed={speed}, "
                    f"attempt={attempt}/{ARM_SERVO_COMMAND_RETRIES}, "
                    f"response={result!r}"
                )
            else:
                logger.warning(
                    f"机械臂翻转指令无新响应 requested={requested_angle}, "
                    f"resolved={resolved_angle}, speed={speed}, "
                    f"attempt={attempt}/{ARM_SERVO_COMMAND_RETRIES}"
                )
            if attempt < ARM_SERVO_COMMAND_RETRIES:
                time.sleep(ARM_SERVO_COMMAND_RETRY_DELAY)

        if response_received:
            #if target_side is not None:
                #self.side = target_side
            self._arm_angle_last = resolved_angle
            logger.info(
                f"机械臂翻转重复发送完成 requested={requested_angle}, "
                f"resolved={resolved_angle}, speed={speed}, "
                f"attempts={ARM_SERVO_COMMAND_RETRIES}, "
                f"last_response={last_response!r}"
            )
            return True

        logger.error(
            f"机械臂翻转指令连续无新响应 requested={requested_angle}, "
            f"resolved={resolved_angle}, speed={speed}, "
            f"attempts={ARM_SERVO_COMMAND_RETRIES}"
        )
        return False

    def set_hand_angle(self, angle: Union[str, int] = "UP", speed=80):
        """
        设置机械臂手角度

        Args:
            angle: 目标角度，可以是字符串（"UP", "MID", "DOWN"）或数字
            speed: 速度
        """
        # 设置相关参数。
        if isinstance(angle, str):
            assert angle in ("UP","MID","DOWN"), "Direction should be UP, MID, or DOWN"
            angle = self.hand_angle_list2[angle]
        self._hand_angle_last = angle
        self.hand_servo.set_angle(angle, speed)

    def go_for(self, x_offset, y_offset, time_run=None, speed=[0.15, 0.04]):
        """
        移动机械臂到当前位置的相对量

        Args:
            x_offset: 水平偏移
            y_offset: 竖直偏移
            time_run: 运行时间
            speed: 速度 [水平速度, 竖直速度]
        """
        # 执行该方法的核心功能。
        x_pos = self.x_pose_now + x_offset
        y_pos = self.y_pose_now + y_offset
        self.goto_position(x_pos, y_pos, time_run, speed)
    
    def goto_position(self, x=None, y=None,time_run=None, speed= [0.15, 0.04]):
        """
        移动到指定机械臂位置

        Args:
            x: 水平位置
            y: 竖直位置
            time_run: 运行时间
            speed: 速度 [水平速度, 竖直速度]
        """

        # 控制上下限
        x_pos = limit_val(
            x,
            self.x_threshold[0],
            self.x_threshold[1]
        )
        y_pos = limit_val(
            y,
            self.y_threshold[0],
            self.y_threshold[1]
        )

        # 获取结束时间和对应速度
        time_start = time.time()
        if time_run is not None:
            assert isinstance(time_run, (int, float)), "Time must be a number"
            # 根据时间求速度
            time_end = time_start + time_run
            y_time = time_run
            x_time = time_run
        elif speed is not None:
            # 根据速度求时间
            if isinstance(speed, (int, float)):
                speed_x = speed
                speed_y = speed
            elif isinstance(speed, (list, tuple)):
                speed_x = speed[0]
                speed_y = speed[1]
            else:
                logger.error("Invalid speed argument")
                return
            x_time = abs(
                x_pos - self.x_pose_now
            ) / speed_x
            y_time = abs(
                y_pos - self.y_pose_now
            ) / speed_y
            time_run = max(x_time, y_time)
        else:
            logger.error("Either time_run or speed must be provided")
            return
        # 超时时间
        time_end = time_start + time_run

        # 定义结束标志和到达位置标记量
        if y is None:
            y_flag = True
        else:
            y_flag = False
        
        if x is None:
            x_flag = True
        else:
            x_flag = False

        # 获取对应的速度和pid位置
        if y_time < 0.1:
            speed_y = 0.1
            y_flag = True
        else:
            speed_y = abs(
                y_pos - self.y_pose_now
            ) / y_time

        self.y_pid.setpoint = y_pos
        self.y_pid.output_limits = (-speed_y, speed_y)

        if x_time < 0.1:
            speed_x = 0.1
            x_flag = True
        else:
            speed_x = abs(
                x_pos - self.x_pose_now
            ) / x_time

        self.x_pid.setpoint = x_pos
        self.x_pid.output_limits = (
            -speed_x, speed_x
        )

        # 开始移动前, 位置信息定义, 如果中间中断此时位置信息无用
        self.save_config(pose_enable=False)

        while True:
            # 到达结束标志结束
            if y_flag and x_flag:
                break
            # 获取剩余时间
            time_remain = time_end - time.time()
            # 超时处理
            if time_remain < -3:
                logger.warning("Timeout")
                # 超时停止
                self.x_speed(0)
                self.y_speed(0)
                break
            if not y_flag:
                if self.y_pid_moveto(y_pos):
                    self.y_speed(0)
                    y_flag = True

                # 重置初始化位置
                if self.y_reset_check():
                    if self.y_pid.setpoint <= self.y_pose_now:
                        y_flag = True
                        self.y_speed(0)
                    self.y_pose_start = self.motor_y.get_dis()
                    self.y_pose_now = 0
                    self.save_config()

            if not x_flag:
                if self.x_pid_moveto(x_pos):
                    self.x_speed(0)
                    x_flag = True

        self.save_config()
        # logger.debug(
        #     f"机械臂移动完成，当前位置状态: x: {self.x_pose_now:.4f}, y: {self.y_pose_now:.4f}, hand: {self.side}。 "
        # )
    def set_arm_pose(self,x=None,y=None,arm = None,hand = None):
        '''
        设置机械臂的位位姿

        Args:
            x: 水平位置
            y: 竖直位置
            arm: 手臂角度，可以是字符串（"LEFT", "MID", "RIGHT"）或数字
            hand: 手部角度，可以是字符串（"UP", "MID", "DOWN"）或数字
        
        '''
        # 设置相关参数。
        self.goto_position(x, y)
        # time.sleep(0.2)
        if arm is not None:
            self.set_arm_angle(arm)
            time.sleep(0.2)
        if hand is not None:
            self.set_hand_angle(hand)

    # ==================== 便捷属性接口 ====================
    @property
    def y(self) -> float:
        """获取当前竖直位置（单位：mm）"""
        # 执行该方法的核心功能。
        return self.y_get_position() * 1000.0

    @y.setter
    def y(self, mm: float):
        """设置目标竖直位置（单位：mm）"""
        # 执行该方法的核心功能。
        self.move_y_position(mm / 1000.0)

    @property
    def x(self) -> float:
        """获取当前水平位置（单位：mm）"""
        # 执行该方法的核心功能。
        return self.x_get_position() * 1000.0

    @x.setter
    def x(self, mm: float):
        """设置目标水平位置（单位：mm）"""
        # 执行该方法的核心功能。
        self.move_x_position(mm / 1000.0)

    @property
    def angle(self) -> float:
        """获取手臂舵机当前角度"""
        # 执行该方法的核心功能。
        return self._arm_angle_last if hasattr(self, '_arm_angle_last') else 0

    @angle.setter
    def angle(self, val: Union[str, int]):
        """设置手臂舵机角度"""
        # 执行该方法的核心功能。
        self.set_arm_angle(val)

    @property
    def hand_angle(self) -> float:
        """获取手部舵机当前角度"""
        # 执行该方法的核心功能。
        return self._hand_angle_last if hasattr(self, '_hand_angle_last') else 0

    @hand_angle.setter
    def hand_angle(self, val: Union[str, int]):
        """设置手部舵机角度"""
        # 执行该方法的核心功能。
        self.set_hand_angle(val)


if __name__ == '__main__':
    arm = ArmController()
    print(f"机械臂长度: {arm.arm_length}")
    arm.reset_x()
    arm.x_move_to_position(0.1)
    print(arm.x_get_positon())
    arm.x_move_to_position(0.2)
    print(arm.x_get_positon())
    arm.x_move_to_position(0.3)
    print(arm.x_get_positon())   

    # start_time = time.time()
    # # arm.grasp(True)
    # arm.reset_position()
    # arm.goto_position(0.15, 0.1)
    # # time.sleep(1)
    # arm.set_arm_angle("LEFT")
    # time.sleep(1)
    # arm.set_hand_angle("DOWN")
    # # arm.grasp(False)
    
    # print(f"移动时间: {time.time() - start_time:.4f}秒")
    # print(f"x: {arm.x_pose_now:.4f}, y: {arm.y_pose_now:.4f}")
