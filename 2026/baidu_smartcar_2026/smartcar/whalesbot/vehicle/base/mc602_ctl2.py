#!/usr/bin/python3
# -*- coding: utf-8 -*-
# 开始编码格式和运行环境选择

import os
from queue import Queue
from serial.tools import list_ports
from threading import Lock, Thread
from multiprocessing import Process 
import time, math
import struct
import sys
# 添加上本地目录
sys.path.append(os.path.abspath(os.path.dirname(__file__))) 
# 添加上两层目录
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from ...tools import logger
# from pydownload import Scratch_Download_MC602P
from smartcar.whalesbot.vehicle.base.serial_wrap import serial_wrap

serial_mc602 = serial_wrap
# def set_serial_mc602(ser:SerialWrap):
#     global serial_mc602
#     serial_mc602 = ser

ctl602_dev_list = {
    "motor4":{"dev_id":0x01, "format":"bbbbb"},
    "motor":{"dev_id":0x02, "format":"bbb"},
    "encoder4":{"dev_id":0x03, "format":"biiii"},
    "encoder":{"dev_id":0x04, "format":"bbi"},
    "servo_pwm":{"dev_id":0x05, "format":"bbBB"},
    "servo_bus":{"dev_id":0x06, "format":"bbbbh"},
    "sensor_analog":{"dev_id":0x07, "mode":0, "format":"bbH"},
    "sensor_infrared":{"dev_id":0x07, "mode":1, "format":"bbH"},
    "sensor_touch":{"dev_id":0x07, "mode":2, "format":"bbH"},
    "sensor_ultrasonic":{"dev_id":0x07, "mode":3, "format":"bbH"},
    "sensor_ambient_light":{"dev_id":0x07, "mode":4, "format":"bbH"},
    "sensor_analog_a":{"dev_id":0x08, "mode":0, "format":"bbH"},
    "bluetooth":{"dev_id":0x09, "format":"BBBBi"},
    "beep":{"dev_id":0x0a, "format":"BBB"},
    "led_show":{"dev_id":0x0b, "format":"b"*101},
    "power":{"dev_id":0x0c, "format":"bi"},
    "board_key":{"dev_id":0x0d, "format":"bbb"},
    "led_light":{"dev_id":0x0e, "format":"bbBBBB"},
    "nixietube":{"dev_id":0x0f, "format":"bbi"},
    "dout":{"dev_id":0x10, "format":"bbb"},
    "stepper":{"dev_id":0x11, "format":"bbii"}
}

# 导入自定义log模块
from ...tools import logger

class StructData():
    def __init__(self, format=None) -> None:
        # 初始化对象状态。
        if format is None:
            format=''
        self.format = '<b'+ format
        self.size = struct.calcsize(self.format)
        self.len = len(self.format)-1
        
    def set_format(self, format):
        # 设置相关参数。
        self.format = '<b'+ format
        self.size = struct.calcsize(self.format)
        self.len = len(self.format)-1
        
    def __sizeof__(self) -> int:
        # 执行该方法的核心功能。
        return self.size
    
    def unpack_data(self, data, index_start):
        # 执行该方法的核心功能。
        try:
            s = index_start
            e = index_start + self.size
            
            # print(data[s:e])
            re_list = list(struct.unpack(self.format, data[s:e]))
        except Exception as e:
            pass
            return []
        return re_list

    def pack_data(self, data):
        # 执行该方法的核心功能。
        bytes_t = struct.pack(self.format, *data)
        return bytes_t

    # 定义len函数的定义
    def __len__(self):
        # 执行该方法的核心功能。
        return self.len

class DevCmdInterface:
    def __init__(self, dev_id=None, mode=None, port_id=None, format='bb') -> None:
        # 初始化对象状态。
        global serial_mc602
        self.ser = serial_mc602
        self.data_struct = StructData(format)
        self.dev_id = dev_id
        self.mode = mode
        self.port_id = port_id

        self.time_out = 0.2
        self.last_data = None
        # 参数保存位置
        self.arg_reg = 1

    def set_time_out(self, time_out):
        # 设置相关参数。
        self.time_out = time_out

    def set_port(self, port_id):
        # 设置相关参数。
        self.port_id = port_id

    def _identity_field_count(self, mode=None, port_id=None):
        """返回本次请求实际包含的响应身份字段数量。"""
        # dev_id 始终存在；操作码和端口只有在设备或本次调用明确提供时
        # 才属于身份字段。不能直接固定比较前三字节，因为部分无端口设备
        # 会把第二字节开始的位置用于业务参数。
        count = 1
        if mode is not None or self.mode is not None:
            count += 1
        if port_id is not None or self.port_id is not None:
            count += 1
        return count
    
    def get_bytes(self, *args, mode=None, port_id=None):
        # 根据参数补充所有参数
        # 获取相关数据。
        data = []
        # print(args)
        data.append(self.dev_id)
        self.arg_reg = 3
        # 根据需要添加操作参数
        if mode is not None:
            data.append(mode)
        elif self.mode is not None:
            data.append(self.mode)
        else:
            self.arg_reg -= 1
            data.append(0)
        # 根据需要添加端口参数
        if port_id is not None:
            data.append(port_id)
        elif self.port_id is not None:
            data.append(self.port_id)
        else:
            self.arg_reg -= 1
        d_len = len(self.data_struct) - len(data)
        args_list= list(args)
        # 根据情况去除参数或者补齐参数
        while True:
            if len(args_list) > d_len:
                args_list.pop(0)
            elif len(args_list) < d_len:
                args_list.append(0)
            else:
                break
        data = data + args_list
        return self.data_struct.pack_data(data)
    
    def get_result(self, bytes_all, index=0, identity_field_count=None):
        # 获取相关数据。
        # 单设备事务显式传入本次请求的身份字段数量，避免依赖对象上可能被
        # 其他调用改写的 arg_reg；批量接口暂时保留原有 arg_reg 行为。
        result_offset = (
            self.arg_reg
            if identity_field_count is None
            else identity_field_count
        )
        data = self.data_struct.unpack_data(bytes_all, index)[result_offset:]
        # 如果只有一个结果
        if len(data) == 1:
            data = data[0]
        return data
    
    def send_get(self, bytes_tmp:bytes, identity_field_count):
        # 每条命令只允许返回本次通讯的新响应。旧实现会在本次超时时返回
        # last_data，运动控制可能把陈旧的步数/编码器值误判为当前位置。
        self.last_data = None
        ret = self.ser.get_anwser(bytes_tmp, self.time_out)
        if ret is None:
            return None

        # MC602 单设备响应按同一数据结构回传。长度不一致时禁止继续解析，
        # 否则截断帧或其他设备的等长/异长迟到响应可能被当成本次 ACK。
        if len(ret) != len(bytes_tmp):
            logger.warning(
                "MC602 响应长度不匹配 "
                f"expected_len={len(bytes_tmp)}, actual_len={len(ret)}, "
                f"expected={bytes_tmp.hex(' ')}, actual={ret.hex(' ')}"
            )
            return None

        # 比较本次请求实际存在的 dev_id、操作码和端口字段。只有身份完全
        # 匹配的响应才允许进入业务数据解析并更新 last_data。
        expected_identity = bytes_tmp[:identity_field_count]
        actual_identity = ret[:identity_field_count]
        if actual_identity != expected_identity:
            logger.warning(
                "MC602 响应身份不匹配 "
                f"expected={expected_identity.hex(' ')}, "
                f"actual={actual_identity.hex(' ')}, "
                f"response={ret.hex(' ')}"
            )
            return None

        result = self.get_result(
            ret,
            identity_field_count=identity_field_count,
        )
        if result == []:
            return None
        self.last_data = result
        return self.last_data
    
    def act_mode(self, *args, mode=None, port_id=None):
        # 执行该方法的核心功能。
        identity_field_count = self._identity_field_count(
            mode=mode,
            port_id=port_id,
        )
        data_bytes = self.get_bytes(*args, mode=mode, port_id=port_id)
        return self.send_get(data_bytes, identity_field_count)
    
    def reset(self, *args, port_id=None):
        # 复位相关状态。
        identity_field_count = self._identity_field_count(
            mode=3,
            port_id=port_id,
        )
        data_bytes = self.get_bytes(*args, mode=3, port_id=port_id)
        return self.send_get(data_bytes, identity_field_count)
    
    # 设置操作
    def set(self, *args, port_id=None):
        # print(args)
        # 设置相关参数。
        identity_field_count = self._identity_field_count(
            mode=2,
            port_id=port_id,
        )
        data_bytes = self.get_bytes(*args, mode=2, port_id=port_id)
        # print(data_bytes.hex(" "))
        return self.send_get(data_bytes, identity_field_count)
    
    # 获取操作
    def get(self, *args, port_id=None):
        # 获取相关数据。
        identity_field_count = self._identity_field_count(
            mode=1,
            port_id=port_id,
        )
        data_bytes = self.get_bytes(*args, mode=1, port_id=port_id)
        # print(data_bytes)
        return self.send_get(data_bytes, identity_field_count)
    
    # 没有操作符号时
    def no_act(self, port_id=None):
        # 执行该方法的核心功能。
        identity_field_count = self._identity_field_count(port_id=port_id)
        data_bytes = self.get_bytes(port_id=port_id)
        # print(data_bytes)
        return self.send_get(data_bytes, identity_field_count)
    
    def act_default(self, *args, port_id=None):
        # 执行该方法的核心功能。
        data_bytes = self.get_bytes(*args, port_id=port_id)
        return data_bytes

class DevListWrap:
    def __init__(self, dev_list=None) -> None:
        # 初始化对象状态。
        if dev_list is None:
            self.dev_list = []
        else:
            self.dev_list = dev_list

    def get_all(self, args, mode=1):
        # 获取相关数据。
        bytes_all = b''
        for i in range(len(self.dev_list)):
            bytes_all += self.dev_list[i].get_bytes(args[i], mode=mode)
            # bytes_all += self.dev_list[i].act_default(args[i])
        # print(bytes_all.hex(' '))
        res = serial_mc602.get_anwser(bytes_all)
        data_ret = []
        if res is not None:
            index = 0
            for i in range(len(self.dev_list)):
                data = self.dev_list[i].get_result(res, index)
                index += self.dev_list[i].data_struct.size
                data_ret.append(data)
        else:
            return [0,0,0,0]
        return data_ret
    def __getattr__(self, name):
        # 获取相关数据。
        return getattr(self.dev_list, name)
    
class Buzzer_2(DevCmdInterface):
    def __init__(self) -> None:
        # 初始化对象状态。
        super().__init__(**ctl602_dev_list["beep"])

    def rings(self, freq=262, duration=0.2):
        # 音调hz 时间s
        # 执行该方法的核心功能。
        res = super().set(int(freq/2), int(duration*20))
        return res
    
class Motor_2(DevCmdInterface):
    def __init__(self, port_id=None, reverse=1) -> None:
        # 初始化对象状态。
        super().__init__(**ctl602_dev_list["motor"], port_id=port_id)
        self.reverse = reverse
    
    def set_dir(self, reverse):
        # 设置相关参数。
        self.reverse = reverse
        
    def set_speed(self, *args):
        # 设置相关参数。
        args = list(args)
        if len(args) == 2:
            args[1] = int(args[1] * self.reverse)
        else:
            args[0] = int(args[0] * self.reverse)
        # print(args)
        return self.set(*args)
    
class AnalogInput_2(DevCmdInterface):
    def __init__(self, port_id=None) -> None:
        # 初始化对象状态。
        super().__init__(**ctl602_dev_list["sensor_analog"], port_id=port_id)

# 红外传感器
class Infrared_2(DevCmdInterface):
    def __init__(self, port_id=None) -> None:
        # 初始化对象状态。
        super().__init__(**ctl602_dev_list["sensor_infrared"], port_id=port_id)

class Sensor_Analog2_2(DevCmdInterface):
    def __init__(self, port_id=None):
        # 初始化对象状态。
        super().__init__(**ctl602_dev_list["sensor_analog_a"], port_id=port_id)
    def read(self):
        # 读取数据。
        return self.no_act()

class BluetoothPad_2(DevCmdInterface):
    def __init__(self) -> None:
        # 调用父对象初始化
        # 初始化对象状态。
        super().__init__(**ctl602_dev_list["bluetooth"])
        self.throsheld_mid = [97, 97, 97, 97, 0]
        self.stick_min = 40
        self.stick_max = 160
        self.divisor_min = [42, 42, 42, 42]
        self.divisor_max = [56, 56, 56, 56]
        
        self.margin = 6

    def calibrate(self):
        # 执行该方法的核心功能。
        info_tmp = self.no_act()

        # print(info_tmp)
        for i in range(4):
            if abs(info_tmp[i] - self.throsheld_mid[i]) < 10:
                self.throsheld_mid[i] = info_tmp[i]
        # logger.info(str(self.throsheld_mid))
        for i in range(4):
            self.divisor_max[i] = self.stick_max - self.throsheld_mid[i] - self.margin
            self.divisor_min[i] = self.throsheld_mid[i] - self.stick_min - self.margin

    def get_stick(self):
        # 获取相关数据。
        data = self.no_act()
        # print(data)
        re_data = []
        tmp = 0.0
        for i in range(4):
            tmp = data[i] - self.throsheld_mid[i]
            if abs(tmp) < self.margin:
                tmp = 0
            if tmp > 0:
                tmp = (tmp-self.margin) / self.divisor_max[i]
            elif tmp<0:
                tmp = (tmp+self.margin) / self.divisor_min[i]
            tmp = min(1, max(-1, tmp))
            re_data.append(tmp)
        if data[4] == 49152:
            self.calibrate()
        re_data.append(data[4])
        return re_data


class BoardKey_2(DevCmdInterface):
    def __init__(self) -> None:
        # 初始化对象状态。
        super().__init__(**ctl602_dev_list["board_key"])
    
    def no_act(self):
        # 执行该方法的核心功能。
        return super().no_act()[1:]

class LedLight_2(DevCmdInterface):
    def __init__(self, port_id=None) -> None:
        # 初始化对象状态。
        super().__init__(**ctl602_dev_list["led_light"], port_id=port_id)
    
    def set_light(self, led_id, r, g, b, port_id=None):
        # 设置相关参数。
        return super().set(led_id, r, g, b, port_id=port_id)
    
    def set(self, *args, port_id=None):
        # 设置相关参数。
        return super().set(*args, port_id=port_id)

class Key4Btn_2(AnalogInput_2):
    btn_sta = []
    state = []
    limit = 0.05
    stop_time = 0.05
    bak_time = 0.0
    long_time = 0.7
    short_time = 0.4
    
    def __init__(self, port_id=None) -> None:
        # 初始化对象状态。
        super().__init__(port_id=port_id)
        self.key_map = {3:355,1:1366,2:2137, 4:2988}
        self.threshold = 0.1

        for i in range(5):
            # 按键状态  按下时间  最后一次按下的时间
            self.btn_sta.append([False, 0.0, 0.0])

    def key_map_btn(self, val):
        # 执行该方法的核心功能。
        r_key = 0
        diff = 1
        for key, value in self.key_map.items():
            try:
                tmp = abs(value - val) / value
                if tmp < self.threshold and tmp < diff:
                    r_key = key
                    diff = tmp
            except:
                pass
        return r_key
    
    def get_key(self, port_id=None):
        # 获取相关数据。
        val = self.no_act(port_id=port_id)
        # print(val)
        return self.key_map_btn(val)
    
    def get_btn(self, port_id=None):
        # 获取相关数据。
        self.event()
        time.sleep(0.01)
        if len(self.state) > 0:
            key_v, key_state = self.state[0][0], self.state[0][1]
            del self.state[0]
            return key_v + 1 + (key_state-1)*4
        else:
            return 0

    def event(self):
        # 执行该方法的核心功能。
        self.bak_time = time.time()
            
        index = 0
        while len(self.state) > index:
            for i in range(len(self.state)):
                if self.bak_time - self.state[index][2] > self.limit:
                    del self.state[index]
                    continue
            index = + 1

        button_num = self.get_key()
        if button_num != 0:
            index = button_num - 1
        else:
            index = 4
            
        # 对应的按键按下，更新状态
        if self.btn_sta[index][0]:
            # 更新按键按下时间
            self.btn_sta[index][1] += (self.bak_time - self.btn_sta[index][2])
            # 发送连续按下
            if self.btn_sta[index][1] > self.long_time and index != 4:
                self.state.append([index, 3, self.bak_time])
        else:
            self.btn_sta[index][0] = True
            self.btn_sta[index][1] = 0
        self.btn_sta[index][2] = self.bak_time
        # print(self.btn_sta)
        for i in range(4):
            btn_state, time_dur, time_last = self.btn_sta[i][0], self.btn_sta[i][1], self.btn_sta[i][2]
            # 如果有记录按下
            if btn_state:
                # 如果长时间没有更新
                if self.bak_time - time_last > self.stop_time:
                    if self.limit < time_dur < self.long_time:
                        self.state.append([i, 1, time_last])
                    elif time_dur > self.long_time:
                        self.state.append([i, 2, time_last])
                    self.btn_sta[i][0] = False
                    self.btn_sta[i][1] = 0.0

class NixieTube_2(DevCmdInterface):
    def __init__(self, port_id=None) -> None:
        # 初始化对象状态。
        super().__init__(**ctl602_dev_list["nixietube"], port_id=port_id)
        
    def set_number(self, num, port_id=None):
        # 设置相关参数。
        return super().set(num, port_id=port_id)
    
class Motor4_2(DevCmdInterface):
    def __init__(self) -> None:
        # 初始化对象状态。
        super().__init__(**ctl602_dev_list["motor4"])
    
    def set_speed(self, speeds):
        # 设置相关参数。
        return super().set(*speeds)

class Motors_2():
    def __init__(self, ports, reverse=False) -> None:
        # 初始化对象状态。
        self.moto_ports = ports
        self.motors = []
        self.encoders = []
        self.args_none = []
        self.reverse = reverse
        for i in ports:
            self.motors.append(Motor_2(i))
            self.encoders.append(EncoderMotor_2(i))
            self.args_none.append(0)
        self.motors_wrap = DevListWrap(self.motors)
        self.encoders_wrap = DevListWrap(self.encoders)
        
    # 设置速度
    def set_speed(self, speeds):
        # 设置相关参数。
        if not self.reverse:
            speeds = [-i for i in speeds]
        # print(speeds)
        return self.motors_wrap.get_all(speeds, mode=2)

    def get_speed(self):
        # 获取相关数据。
        speed = self.motors_wrap.get_all(self.args_none, mode=1)
        if self.reverse:
            speed = [-i for i in speed]
        return speed
    
    def get_encoder(self):
        # 获取相关数据。
        encoders = self.encoders_wrap.get_all(self.args_none, mode=1)
        if isinstance(encoders[0], list):
            encoders = encoders[0]  # 解开嵌套列表

        if not self.reverse:
            encoders = [-i for i in encoders]
        return encoders

    def reset_encoder(self):
        # 复位相关状态。
        return self.encoders_wrap.get_all(self.args_none, mode=3)
    
    def reset(self):
        # 复位相关状态。
        self.motors_wrap.get_all(self.args_none, mode=3)
        return self.encoders_wrap.get_all(self.args_none, mode=3)
    
class EncoderMotor_2(DevCmdInterface):
    def __init__(self, port_id=None, reverse=-1) -> None:
        # 初始化对象状态。
        self.reverse = reverse
        super().__init__(**ctl602_dev_list["encoder"], port_id=port_id)
    
    def get_encoder(self):
        # 获取相关数据。
        return self.get()*self.reverse

class EncoderMotors4_2(DevCmdInterface):
    def __init__(self) -> None:
        # 初始化对象状态。
        super().__init__(**ctl602_dev_list["encoder4"])

class ServoPwm_2(DevCmdInterface):
    def __init__(self, port_id=None) -> None:
        # 初始化对象状态。
        super().__init__(**ctl602_dev_list["servo_pwm"], port_id=port_id)

    def set_angle(self, angle, speed=100):
        # 设置相关参数。
        self.set(int(speed), int(angle))

class ServoBus_2(DevCmdInterface):
    def __init__(self,port_id=None) -> None:
        # 初始化对象状态。
        super().__init__(**ctl602_dev_list["servo_bus"], port_id=port_id)
        # 串口层现在会真正执行设备超时；保持历史有效的 0.2 s 上限，
        # 禁止一次舵机等待独占 MC602 总线超过 600 ms 轴速度看门狗。
        self.set_time_out(0.2)
    
    def set_angle(self, angle, speed=100):
        # 只接受本次舵机命令对应的新响应，避免沿用上一条命令的缓存结果。
        self.last_data = None
        return self.act_mode(1, speed, angle, mode=2)

    def set_speed(self, speed):
        # 设置相关参数。
        self.act_mode(2, speed, mode=2)

class ScreenShow_2(DevCmdInterface):
    def __init__(self) -> None:
        # 初始化对象状态。
        super().__init__(**ctl602_dev_list["led_show"])
    
    def show(self, args):
        # 显示相关画面。
        if type(args) != str:
            args = str(args)
        int_values = [ord(arg) for arg in args]
        int_values = tuple(int_values)
        self.set(*int_values)
    
class Battry_2(DevCmdInterface):
    def __init__(self) -> None:
        # 初始化对象状态。
        super().__init__(**ctl602_dev_list["power"])

    def read(self):
        # 读取数据。
        res = super().get()
        bat = float(res) / 1000
        return bat
    
class PoutD_2(DevCmdInterface):
    def __init__(self, port_id=1) -> None:
        # 初始化对象状态。
        super().__init__(**ctl602_dev_list["dout"], port_id=port_id)
    
    def set(self, *args):
        # 设置相关参数。
        super().set(*args)
        
class Stepper_2(DevCmdInterface):
    def __init__(self, port_id=1) -> None:
        # 初始化对象状态。
        super().__init__(**ctl602_dev_list["stepper"], port_id=port_id)

    def set_pwm(self, freq):
        # 设置相关参数。
        super().set(int(freq))
    
    def get_step(self):
        # 步数反馈必须来自本次新响应，禁止使用上一次 SET/GET 的缓存结果。
        result = super().get()
        if not isinstance(result, (list, tuple)) or len(result) < 2:
            raise RuntimeError(f"MC602 未返回有效步进电机位置 result={result!r}")
        return result[1]

def beep_test():
    beep = Buzzer_2()
    for i in range(10):
        beep.set(200, 10)
        time.sleep(0.5)

def motor_test():
    motor = Motor_2(1)
    for i in range(10):
        motor.set(10)
        time.sleep(1)
        motor.set(0)
        time.sleep(1)

def motors_test():
    motors = Motors_2([1, 4], True)
    motors.reset_encoder()
    for i in range(10):
        motors.set_speed([10,10])
        time.sleep(0.1)
        # print(motors.get_encoder())
        # motors.set_speed([0, 0, 0])
        # time.sleep(1)
    motors.set_speed([0,0])
    
def motor4_test():
    motor4 = Motor4_2()
    while True:
        for i in range(10):
            motor4.set_speed([32, -16, -16, 30])
            time.sleep(1)
        motor4.set_speed([0, 0, 0, 0])
        time.sleep(1)

def encoders_test():
    encoders = EncoderMotors4_2()
    while True:
        res = encoders.get()
        print(res)
        time.sleep(1)

def sensor_anolog_test():
    sensor4 = AnalogInput_2(1)
    while(1):
        print(sensor4.no_act())
        time.sleep(1)

def sensor_infrared_test():
    infrared1 = Infrared_2(1)
    while(1):
        print(infrared1.no_act())
        time.sleep(1)

def board_key_test():
    key = BoardKey_2()
    while True:
        res = key.no_act()
        print(res)
        time.sleep(0.1)

def show_test():
    show = ScreenShow_2()
    show.show("my_test\n\nok")

def nixie_tube_test():
    nixie = NixieTube_2(1)
    nixie.set_number(1111)

def servo_bus_test():
    servo_bus = ServoBus_2(2)
    while(1):
        servo_bus.set_angle(100, 60)
        time.sleep(1)
        servo_bus.set_angle(50, 60)
        time.sleep(1)

def servo_pwm_test():
    servo_pwm = ServoPwm_2(2)
    while(1):
        servo_pwm.set_angle(60, 60)
        time.sleep(1)
        servo_pwm.set_angle(70, 60)
        time.sleep(1)

def led_light_test():
    led_light = LedLight_2(1)
    while(1):
        led_light.set_light(1, 255, 255, 0)
        time.sleep(1)
        led_light.set_light(1, 0, 0, 0)
        time.sleep(1)

def key_test():
    key = Key4Btn_2(7)
    last_time = time.time()
    while True:

        res = key.get_btn()
        # res = key.get_key()
        # print(res)
        if res!= 0:
            print(res)
        now = time.time()
        try:
            fps = (1 / (now - last_time))
        except ZeroDivisionError:
            fps = 100
        last_time = now
        # time.sleep(0.)
        # print("fps:", fps)
        # print(res)
        # print(res)
        
def dev_list_test():

    motor1 = Motor_2(1)
    motor1.mode = 2
    key_val = Key4Btn_2(4)
    blue_pad = BluetoothPad_2()
    sensor6 = AnalogInput_2(6)
    # 四个同时控制
    dev_list = [sensor6, motor1, blue_pad, key_val]
    dev_wrap = DevListWrap(dev_list)
    while True:
        res = dev_wrap.get_all([0, 10, 0, 0])
        logger.info(res)
        time.sleep(1)

def bluetooth_pad_test():
    blue_pad = BluetoothPad_2()
    while True:
        res = blue_pad.get_stick()
        print(res)
        time.sleep(0.2)

def dout_test():
    dout = PoutD_2(1)
    dout.set(0)
    # time.sleep

def stepper_test():
    stepper = Stepper_2(2)
    stepper.set(2000)
    time.sleep(1)
    stepper.set(0)

if __name__ == "__main__":
    serial_mc602.assert_dev("mc602")
    beep = Buzzer_2()
    beep.rings()
    # bluetooth_pad_test()
    # dev_list_test()
    # board_key_test()
    # led_light_test()
    # beep_test()
    # motor_test()
    # motors_test()
    # dout = Dout_2()
    # stepper_test()
    # dout_test()
    # motor4_test()
    # encoders_test()
    # sensor_anolog_test()
    # sensor_infrared_test()
    # show_test()
    # key_test()
    # servo_bus_test()
    # nixie_tube_test()
    # servo_pwm_test()

