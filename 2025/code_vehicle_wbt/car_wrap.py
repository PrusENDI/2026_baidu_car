#!/usr/bin/python
# -*- coding: utf-8 -*-
import time
import threading
import os
import platform
import signal
from camera import Camera
import numpy as np
from vehicle import ArmBase, ScreenShow, Key4Btn, Infrared, LedLight,CarBase, Beep
from simple_pid import PID
import difflib
import cv2, math
from task_func import MyTask
from infer_cs import ClintInterface, Bbox
from ernie_bot import ErnieBotWrap, ActionPrompt, HumAttrPrompt
from tools import CountRecord, get_yaml, IndexWrap
import sys, os

# For independent ernie function
import erniebot

# 添加上本地目录
sys.path.append(os.path.abspath(os.path.dirname(__file__))) 
from log_info import logger

def sellect_program(programs, order, win_order):
    dis_str = ''
    start_index = 0
    
    start_index = order - win_order
    for i, program in enumerate(programs):
        if i < start_index:
            continue

        now = str(program)
        if i == order:
            now = '>>> ' + now
        else:
            now = str(i+1) + '.' + now
        if len(now) >= 19:
            now = now[:19]
        else:
            now = now + '\n'
        dis_str += now
        if i-start_index == 4:
            break
    return dis_str

def kill_other_python():
    import psutil
    pid_me = os.getpid()
    # logger.info("my pid ", pid_me, type(pid_me))
    python_processes = []
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                if 'python' in proc.info['name'].lower() and len(proc.info['cmdline']) > 1 and len(proc.info['cmdline'][1]) < 30:
                    python_processes.append(proc.info)
            # 出现异常的时候捕获 不存在的异常，权限不足的异常， 僵尸进程
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                pass
    for process in python_processes:
        # logger.info(f"PID: {process['pid']}, Name: {process['name']}, Cmdline: {process['cmdline']}")
        # logger.info("this", process['pid'], type(process['pid']))
        if int(process['pid']) != pid_me:
            os.kill(int(process['pid']), signal.SIGKILL)
            time.sleep(0.3)
            
def limit(value, value_range):
    return max(min(value, value_range), 0-value_range)

# 两个pid集合成一个
class PidCal2():
    def __init__(self, cfg_pid_y=None, cfg_pid_angle=None):
        self.pid_y = PID(**cfg_pid_y)
        self.pid_angle = PID(**cfg_pid_angle)
    
    def get_out(self, error_y, error_angle):
        pid_y_out = self.pid_y(error_y)
        pid_angle_out = self.pid_angle(error_angle)
        return pid_y_out, pid_angle_out

class LanePidCal():
    def __init__(self, cfg_pid_y=None, cfg_pid_angle=None):
        # y_out_limit = 0.7
        # self.pid_y = PID(5, 0, 0)
        # self.pid_y.setpoint = 0
        # self.pid_y.output_limits = (-y_out_limit, y_out_limit)
        # print(cfg_pid_y)
        # print(cfg_pid_angle)
        self.pid_y = PID(**cfg_pid_y)
        # print(self.pid_y)

        angle_out_limit = 1.5
        self.pid_angle = PID(3, 0, 0)
        self.pid_angle.setpoint = 0
        self.pid_angle.output_limits = (-angle_out_limit, angle_out_limit)
    
    def get_out(self, error_y, error_angle):
        pid_y_out = self.pid_y(error_y)
        pid_angle_out = self.pid_angle(error_angle)
        return pid_y_out, pid_angle_out
    
class DetPidCal():
    def __init__(self, cfg_pid_y=None, cfg_pid_angle=None):
        y_out_limit = 0.7
        self.pid_y = PID(0.3, 0, 0)
        self.pid_y.setpoint = 0
        self.pid_y.output_limits = (-y_out_limit, y_out_limit)

        angle_out_limit = 1.5
        self.pid_angle = PID(2, 0, 0)
        self.pid_angle.setpoint = 0
        self.pid_angle.output_limits = (-angle_out_limit, angle_out_limit)
    
    def get_out(self, error_y, error_angle):
        pid_y_out = self.pid_y(error_y)
        pid_angle_out = self.pid_angle(error_angle)
        return pid_y_out, pid_angle_out
    

class LocatePidCal():
    def __init__(self):
        y_out_limit = 0.3
        self.pid_y = PID(0.5, 0, 0)
        self.pid_y.setpoint = 0
        self.pid_y.output_limits = (-y_out_limit, y_out_limit)

        x_out_limit = 0.3
        self.pid_x = PID(0.5, 0, 0)
        self.pid_x.setpoint = 0
        self.pid_x.output_limits = (-x_out_limit, x_out_limit)
    
    def set_target(self, x, y):
        self.pid_y.setpoint = y
        self.pid_x.setpoint = x

    def get_out(self, error_x, error_y):
        pid_y_out = self.pid_y(error_y)
        pid_x_out = self.pid_x(error_x)
        return pid_x_out, pid_y_out

class MyCar(CarBase):
    STOP_PARAM = True
    def __init__(self):
        # 调用继承的初始化
        start_time = time.time()
        super(MyCar, self).__init__()
        logger.info("my car init ok {}".format(time.time() - start_time))
        # 任务
        self.task = MyTask()
        # 显示
        self.display = ScreenShow()
        
        # 获取自己文件所在的目录路径
        self.path_dir = os.path.abspath(os.path.dirname(__file__))
        self.yaml_path = os.path.join(self.path_dir, "config_car.yml")
        # 获取配置
        cfg = get_yaml(self.yaml_path)
        # 根据配置设置sensor
        self.sensor_init(cfg)

        self.car_pid_init(cfg)
        self.ring = Beep()
        self.camera_init(cfg)
        # paddle推理初始化
        self.paddle_infer_init()
        # 文心一言分析初始化
        # self.ernie_bot_init()

        # 相关临时变量设置
        # 程序结束标志
        self._stop_flag = False
        # 按键线程结束标志
        self._end_flag = False
        self.thread_key = threading.Thread(target=self.key_thread_func)
        self.thread_key.setDaemon(True)
        self.thread_key.start()
        
        self.beep()
    
    def beep(self):
        self.ring.rings()
        time.sleep(0.2)

    def sensor_init(self, cfg):
        cfg_sensor = cfg['io']
        # print(cfg_sensor)
        self.key = Key4Btn(cfg_sensor['key'])
        self.light = LedLight(cfg_sensor['light'])
        self.left_sensor = Infrared(cfg_sensor['left_sensor'])
        self.right_sensor = Infrared(cfg_sensor['right_sensor'])
    
    def car_pid_init(self, cfg):
        # lane_pid_cfg = cfg['lane_pid']
        # self.pid_y = PID(lane_pid_cfg['y'], 0, 0)
        # self.lane_pid = LanePidCal(**cfg['lane_pid'])
        # self.det_pid = DetPidCal(**cfg['det_pid'])
        self.lane_pid = PidCal2(**cfg['lane_pid'])
        self.det_pid = PidCal2(**cfg['det_pid'])

    def camera_init(self, cfg):
        # 初始化前后摄像头设置
        self.cap_front = Camera(cfg['camera']['front'])
        # 侧面摄像头
        self.cap_side = Camera(cfg['camera']['side'])

    def paddle_infer_init(self):
        self.cruise = ClintInterface('lane')
        # 前置左右方向识别
        self.front_det = ClintInterface('front')
        # 任务识别
        self.task_det = ClintInterface('task')
        # ocr识别
        self.ocr_rec = ClintInterface('ocr')
        # 识别为None
        self.last_det = None

    def ernie_bot_init(self):
        self.hum_analysis = ErnieBotWrap()
        self.hum_analysis.set_promt(str(HumAttrPrompt()))

        self.action_bot = ErnieBotWrap()
        self.action_bot.set_promt(str(ActionPrompt()))

    @staticmethod
    def get_cfg(path):
        from yaml import load, Loader
        # 把配置文件读取到内存
        with open(path, 'r') as stream:
            yaml_dict = load(stream, Loader=Loader)
        port_list = yaml_dict['port_io']
        # 转化为int
        for port in port_list:
            port['port'] = int(port['port'])
        # print(yaml_dict)

    # 延时函数
    def delay(self, time_hold):
        start_time = time.time()
        while True:
            if self._stop_flag:
                return
            if time.time() - start_time > time_hold:
                break
            
    # 按键检测线程
    def key_thread_func(self):
        while True:
            if not self._stop_flag:
                if self._end_flag:
                    return
                key_val = self.key.get_key()
                # print(key_val)
                if key_val == 3:
                    self._stop_flag = True
                time.sleep(0.2)
    
    
    # 根据某个值获取列表中匹配的结果
    @staticmethod
    def get_list_by_val(list, index, val):
        for det in list:
            if det[index] == val:
                return det
        return None
    
    def move_base(self, sp, end_fuction, stop=STOP_PARAM):
        self.set_velocity(sp[0], sp[1], sp[2])
        while True:
            if self._stop_flag:
                return
            if end_fuction():
                break
            self.set_velocity(sp[0], sp[1], sp[2])
        if stop:
            self.set_velocity(0, 0, 0)


    #  高级移动，按着给定速度进行移动，直到满足条件
    def move_advance(self, sp, value_h=None, value_l=None, times=1, sides=1, dis_out=0.2, stop=STOP_PARAM):
        if value_h is None:
            value_h = 1200
        if value_l is None:
            value_l = 0
        _sensor_usr = self.left_sensor
        if sides == -1:
            _sensor_usr = self.right_sensor
        # 用于检测开始过渡部分的标记
        flag_start = False
        def end_fuction():
            nonlocal flag_start
            val_sensor = _sensor_usr.read()
            # print("val:", val_sensor)
            if val_sensor < value_h and val_sensor > value_l:
                return flag_start
            else:
                flag_start = True
                return False
        for i in range(times):
            self.move_base(sp, end_fuction, stop=False)
        if stop:
            self.stop()

    
    def move_time(self, sp, dur_time=1, stop=STOP_PARAM):
        end_time = time.time() + dur_time
        end_func = lambda: time.time() > end_time
        self.move_base(sp, end_func, stop)

    def move_distance(self, sp, dis=0.1, stop=STOP_PARAM):
        end_dis = self.get_dis_traveled() + dis
        end_func = lambda: self.get_dis_traveled() > end_dis
        self.move_base(sp, end_func, stop)

    # 计算两个坐标的距离
    def calculation_dis(self, pos_dst, pos_src):
        return math.sqrt((pos_dst[0] - pos_src[0])**2 + (pos_dst[1] - pos_src[1])**2)
    
    def det2pose(self, det, w_r=0.06): 
        # r 真实  v 成像  f 焦点
        # rf 真实到焦点的距离  vf 相到焦点的距离
        vf_dis = 1.445
        x_v, y_v, w_v, h_v = det
        
        rf_dis = vf_dis * w_r / w_v
        x_r = x_v * rf_dis / vf_dis
        y_r = y_v * rf_dis / vf_dis
        return x_r, y_r, rf_dis
    
    # 侧面摄像头进行位置定位
    def lane_det_location(self, speed, pts_tar=[[0, 70, 'text_det',  0, 0, 0, 0.70, 0.70]], dis_out=0.05, side=1, time_out=5, det='task'):
        end_time = time.time() + time_out
        infer = self.task_det
        loc_pid = get_yaml(self.yaml_path)["location_pid"]
        pid_x = PID(**loc_pid["pid_x"])
        pid_x.output_limits = (-speed, speed)
        pid_y = PID(**loc_pid["pid_y"])
        pid_y.output_limits = (-0.15, 0.15)
        # pid_w = PID(1.0, 0, 0.00, setpoint=0, output_limits=(-0.15, 0.15))

        # 用于相同记录结果的计数类
        x_count = CountRecord(5)
        dis_count = CountRecord(5)
        
        out_x = speed
        out_y = 0
        
        # 此时设置相对初始位置
        # self.set_pos_relative()
        # self.dis_tra_st = self.get_dis_traveled()
        x_st, y_st, _ = self.get_odometry()
        find_tar = False
        tar = []
        for pt_tar in pts_tar:
            # id, 物体宽度，置信度, 归一化bbox[x_c, y_c, w, h]
            tar_id, tar_width, tar_label, tar_score, tar_bbox = pt_tar[0], pt_tar[1], pt_tar[2], pt_tar[3], pt_tar[4:]
            tar_width *= 0.001
            tar_x, tar_y, tar_dis = self.det2pose(tar_bbox, tar_width)
            tar.append([tar_id, tar_width, tar_x, tar_y, tar_dis])
        # logger.info("tar x:{} dis:{}".format(tar_x, tar_dis))
        tar_id, tar_width, tar_x, tar_y, tar_dis = tar[0]
        pid_x.setpoint = tar_x
        pid_y.setpoint = tar_dis
        tar_index = 0
        flag_location = False
        while True:
            if self._stop_flag:
                return
            if time.time() > end_time:
                logger.info("time out")
                self.set_velocity(0, 0, 0)
                return False
            _pos_x, _pos_y, _pos_omage = self.get_odometry() # 用来计算距离

            if abs(_pos_x-x_st) > dis_out or abs(_pos_y-y_st) > dis_out:
                if not find_tar:
                    logger.info("task location dis out")
                    self.set_velocity(0, 0, 0)
                    return False
            img_side = self.cap_side.read()
            dets_ret = infer(img_side)
            
            # dets_ret = self.mot_hum(img_side)
            # cv2.imshow("side", img_side)
            # cv2.waitKey(1)
            
            # 进行排序，此处排列按照自中心由近及远的顺序
            dets_ret.sort(key=lambda x: (x[4])**2 + (x[5])**2)
            # print(dets_ret)
            # # 找到最近对应的类别，类别存在第一个位置
            # det = self.get_list_by_val(dets_ret, 2, tar_label)
            
            # 如果没有，就重新获取
            if len(dets_ret) > 0:
                det = dets_ret[0]
                # 结果分解
                det_id, obj_id , det_label, det_score, det_bbox = det[0], det[1], det[2], det[3], det[4:]
                # if find_tar is False:
                    # tar_index = 0
                    # for tar_pt in tar:
                for index, tar_pt in enumerate(tar):
                    if det_id == tar_pt[0]:
                        tar_index = index
                        tar_id, tar_width, tar_x, tar_y, tar_dis = tar_pt
                        pid_x.setpoint = tar_x
                        pid_y.setpoint = tar_dis
                        find_tar = True
                        # print("find tar", tar_id)
                        break
                        
                if det_id == tar_id:
                    _x, _y, _dis = self.det2pose(det_bbox, tar_width)
                    out_x = pid_x(_x) * side
                    out_y = pid_y(_dis) * side
                    # out_y = pid_y(_dis)
                    # out_y = pid_w(bbox_error[2])
                    # 检测偏差值连续小于阈值时，跳出循环
                    # print(bbox_error)
                    # print("err x:{:.2}, dis:{:.2}, tar x:{:.2}, tar dis:{:.2}".format(_x, _dis, tar_x, tar_dis))
                    flag_x = x_count(abs(_x - tar_x) < 0.01)
                    flag_dis = dis_count(abs(_dis - tar_dis) < 0.01)
                    if flag_x:
                        out_x = 0
                    if flag_dis:
                        out_y = 0
                    if flag_x and flag_dis:
                        logger.info("location{} ok".format(tar_id))
                        # flag_location = True
                        # 停止
                        self.set_velocity(0, 0, 0)
                        return tar_index
                
                # print("error_x:{:.2}, error_y:{:.2}, out_x:{:.2}, out_y:{:2}".format(bbox_error[0], bbox_error[2], out_x, out_y))
            else:
                x_count(False)
                dis_count(False)
            self.set_velocity(out_x, out_y, 0)

    def special_lane_det_location(self, speed, pts_tar=[[0, 70, 'text_det',  0, 0, 0, 0.70, 0.70]], dis_out=0.05, side=1, time_out=5, det='task'):
        end_time = time.time() + time_out
        infer = self.task_det
        loc_pid = get_yaml(self.yaml_path)["special_location_pid"]
        pid_x = PID(**loc_pid["pid_x"])
        pid_x.output_limits = (-speed, speed)
        pid_y = PID(**loc_pid["pid_y"])
        pid_y.output_limits = (-0.15, 0.15)
        # pid_w = PID(1.0, 0, 0.00, setpoint=0, output_limits=(-0.15, 0.15))

        # 用于相同记录结果的计数类
        x_count = CountRecord(5)
        dis_count = CountRecord(5)
        
        out_x = speed
        out_y = 0
        
        # 此时设置相对初始位置
        # self.set_pos_relative()
        # self.dis_tra_st = self.get_dis_traveled()
        x_st, y_st, _ = self.get_odometry()
        find_tar = False
        tar = []
        for pt_tar in pts_tar:
            # id, 物体宽度，置信度, 归一化bbox[x_c, y_c, w, h]
            tar_id, tar_width, tar_label, tar_score, tar_bbox = pt_tar[0], pt_tar[1], pt_tar[2], pt_tar[3], pt_tar[4:]
            tar_width *= 0.001
            tar_x, tar_y, tar_dis = self.det2pose(tar_bbox, tar_width)
            tar.append([tar_id, tar_width, tar_x, tar_y, tar_dis])
        # logger.info("tar x:{} dis:{}".format(tar_x, tar_dis))
        tar_id, tar_width, tar_x, tar_y, tar_dis = tar[0]
        pid_x.setpoint = tar_x
        pid_y.setpoint = tar_dis
        tar_index = 0
        flag_location = False
        while True:
            if self._stop_flag:
                return
            if time.time() > end_time:
                logger.info("time out")
                self.set_velocity(0, 0, 0)
                return False
            _pos_x, _pos_y, _pos_omage = self.get_odometry() # 用来计算距离

            if abs(_pos_x-x_st) > dis_out or abs(_pos_y-y_st) > dis_out:
                if not find_tar:
                    logger.info("task location dis out")
                    self.set_velocity(0, 0, 0)
                    return False
            img_side = self.cap_side.read()
            dets_ret = infer(img_side)
            
            # dets_ret = self.mot_hum(img_side)
            # cv2.imshow("side", img_side)
            # cv2.waitKey(1)
            
            # 进行排序，此处排列按照自中心由近及远的顺序
            dets_ret.sort(key=lambda x: (x[4])**2 + (x[5])**2)
            # print(dets_ret)
            # # 找到最近对应的类别，类别存在第一个位置
            # det = self.get_list_by_val(dets_ret, 2, tar_label)
            
            # 如果没有，就重新获取
            if len(dets_ret) > 0:
                det = dets_ret[0]
                # 结果分解
                det_id, obj_id , det_label, det_score, det_bbox = det[0], det[1], det[2], det[3], det[4:]
                # if find_tar is False:
                    # tar_index = 0
                    # for tar_pt in tar:
                for index, tar_pt in enumerate(tar):
                    if det_id == tar_pt[0]:
                        tar_index = index
                        tar_id, tar_width, tar_x, tar_y, tar_dis = tar_pt
                        pid_x.setpoint = tar_x
                        pid_y.setpoint = tar_dis
                        find_tar = True
                        # print("find tar", tar_id)
                        break
                        
                if det_id == tar_id:
                    _x, _y, _dis = self.det2pose(det_bbox, tar_width)
                    out_x = pid_x(_x) * side
                    out_y = pid_y(_dis) * side
                    # out_y = pid_y(_dis)
                    # out_y = pid_w(bbox_error[2])
                    # 检测偏差值连续小于阈值时，跳出循环
                    # print(bbox_error)
                    # print("err x:{:.2}, dis:{:.2}, tar x:{:.2}, tar dis:{:.2}".format(_x, _dis, tar_x, tar_dis))
                    flag_x = x_count(abs(_x - tar_x) < 0.01)
                    flag_dis = dis_count(abs(_dis - tar_dis) < 0.01)
                    if flag_x:
                        out_x = 0
                    if flag_dis:
                        out_y = 0
                    if flag_x and flag_dis:
                        logger.info("location{} ok".format(tar_id))
                        # flag_location = True
                        # 停止
                        self.set_velocity(0, 0, 0)
                        return tar_index
                
                # print("error_x:{:.2}, error_y:{:.2}, out_x:{:.2}, out_y:{:2}".format(bbox_error[0], bbox_error[2], out_x, out_y))
            else:
                x_count(False)
                dis_count(False)
            self.set_velocity(out_x, out_y, 0)
            
    def lane_base(self, speed, end_fuction, stop=STOP_PARAM):
        while True:
            if self._stop_flag:
                return
            image = self.cap_front.read()
            error_y, error_angle = self.cruise(image)
            y_speed, angle_speed = self.lane_pid.get_out(-error_y, -error_angle)
            # speed_dy, angle_speed = process(image)
            self.set_velocity(speed, y_speed, angle_speed)
            if end_fuction():
                break
        if stop:
            self.stop()

    def lane_det_base(self, speed, end_fuction, stop=STOP_PARAM):
        # 初始化速度和角度速度
        y_speed = 0
        angle_speed = 0
        w_r=0.06
        # 无限循环
        while True:
            # 读取前摄像头图像
            image = self.cap_front.read()
            dets_ret = self.front_det(image)
            # 此处检测简单不需要排序
            # dets_ret.sort(key=lambda x: x[4]**2 + (x[5])**2)
            if len(dets_ret)>0:
                det = dets_ret[0]
                det_cls, det_id, det_label, det_score, det_bbox = det[0], det[1], det[2], det[3], det[4:]
                _x, _y, _dis = self.det2pose(det_bbox, w_r)
                # error_y = det_bbox[0]
                # dis_x = 1 - det_bbox[1]
                if end_fuction(_dis):
                    break
                error_angle = _x /_dis
                y_speed, angle_speed = self.det_pid.get_out(_x, error_angle)
                # print("_x:{:.2}, _angle:{:.2}, y_vel:{:.2}, angle_vel:{:.2}, dis{:.2}".format(_x, error_angle, y_speed, angle_speed, _dis))
            self.set_velocity(speed, y_speed, angle_speed)
            # if end_fuction(0):
            #     break
        if stop:
            self.stop()
            
    def lane_det_time(self, speed, time_dur, stop=STOP_PARAM):
        time_end = time.time() + time_dur
        end_fuction = lambda x: time.time() > time_end
        self.lane_det_base(speed, end_fuction, stop=stop)

    def lane_det_dis2pt(self, speed, dis_end, stop=STOP_PARAM):
        # lambda定义endfunction
        end_fuction = lambda x: x < dis_end and x != 0
        self.lane_det_base(speed, end_fuction, stop=stop)

    def lane_time(self, speed, time_dur, stop=STOP_PARAM):
        time_end = time.time() + time_dur
        end_fuction = lambda: time.time() > time_end
        self.lane_base(speed, end_fuction, stop=stop)
    
    # 巡航一段路程
    def lane_dis(self, speed, dis_end, stop=STOP_PARAM):
        # lambda重新endfunction
        end_fuction = lambda: self.get_dis_traveled() > dis_end
        self.lane_base(speed, end_fuction, stop=stop)

    def lane_dis_offset(self, speed, dis_hold, stop=STOP_PARAM):
        dis_start = self.get_dis_traveled()
        dis_stop = dis_start + dis_hold
        self.lane_dis(speed, dis_stop, stop=stop)

    def lane_sensor(self, speed, value_h=None, value_l=None, dis_offset=0.0, times=1, sides=1, stop=STOP_PARAM):
        if value_h is None:
            value_h = 1200
        if value_l is None:
            value_l = 0
        _sensor_usr = self.left_sensor
        if sides == -1:
            _sensor_usr = self.right_sensor
        # 用于检测开始过渡部分的标记
        flag_start = False
        def end_fuction():
            nonlocal flag_start
            val_sensor = _sensor_usr.read()
            # print("val:", val_sensor)
            if val_sensor < value_h and val_sensor > value_l:
                return flag_start
            else:
                flag_start = True
                return False

        for i in range(times):
            self.lane_base(speed, end_fuction, stop=False)
        # 根据需要是否巡航
        self.lane_dis_offset(speed, dis_offset, stop=stop)

    def get_card_side(self):
        # 检测卡片左右指示
        count_side = CountRecord(3)
        while True:
            if self._stop_flag:
                return
            image = self.cap_front.read()
            dets_ret = self.front_det(image)
            if len(dets_ret) == 0:
                count_side(-1)
                continue
            det = dets_ret[0]
            det_cls, det_id, det_label, det_score, det_bbox = det[0], det[1], det[2], det[3], det[4:]
            # 联系检测超过3次
            if count_side(det_label):
                if det_label == 'turn_right':
                    return -1
                elif det_label == 'turn_left':
                    return 1
    

    def get_ocr(self, time_out=5):
        time_stop = time.time() + time_out
        # 简单滤波,三次检测到相同的值，认为稳定并返回
        text_count = CountRecord(3)
        text_out = None
        # img_counter = 0
        while True:
            if self._stop_flag:
                return
            if time.time() > time_stop:
                return None
            img = self.cap_side.read()
            response = self.task_det(img)
            print(f"Task detection result : {response}")
            time.sleep(0.1)
            if len(response) > 0:
                for det in response:
                    det_cls_id, det_id, det_label, det_score, det_bbox = det[0], det[1], det[2], det[3], det[4:]
                    if det_cls_id == 0:
                        x1, y1, w, h = det_bbox
                        # print(img.shape)
                        # print(x1, y1, w, h)
                        x1 = img.shape[1] * (1+x1) / 2 - img.shape[1] * w / 4
                        x2 = x1 + img.shape[1] * w / 2
                        y1 = img.shape[0] * (1+y1) / 2 - img.shape[0] * w / 4
                        y2 = y1 + img.shape[0] * h / 2
                        x1 = 0 if x1 < 0 else int(x1)
                        x2 = img.shape[1] if x2 > img.shape[1] else int(x2)
                        y1 = 0 if y1 < 0 else int(y1)
                        y2 = img.shape[0] if y2 > img.shape[0] else int(y2)
                        print(f"Detected text region : {x1, x2, y1, y2}")
                        img_txt = img[y1:y2, x1:x2]
                        
                        # Save corped text image for debugging
                        '''
                        if img_counter == 0:
                            img_filename = f"text_img_{img_counter:03d}.jpg"
                            img_path = os.path.join(self.path_dir, img_filename)
                            cv2.imwrite(img_path, img_txt)
                            print(f"Saved text image: {img_path}")
                            img_counter += 1
                        '''

                        time.sleep(0.5)
                        text = self.ocr_rec(img_txt)
                        if text_out==None:
                            text_out = text
                        else:
                            # 文本相似度比较
                            matcher = difflib.SequenceMatcher(None, text_out, text).ratio()
                            if text_count(matcher > 0.85):
                                return text_out
                            else:
                                text_out = text
                            # if matcher > 0.85:
                            #     text_count(T)
                        # print(text)
                        # print(res.bbox)
                        # print(text)
                        # if text_count(text):
                        #     return text

    # Define a new ernie interact function
    def ernieInteract(self, taskName, text = None, a = None, b = None, c = None, d = None, timeout = 10, ing_side = 'left', default_value=None):
        import json
        import concurrent.futures

        def _ernie_call():
            erniebot.api_type = "aistudio"
            erniebot.access_token = "eeac7c51a9cfaecccb9483e052a029906d445a2e"
            if taskName == 'bmi':
                response = erniebot.ChatCompletion.create(model = "ernie-4.0", messages = [{"role" : "user", "content" : f"计算下面这个人的BMI: {text}, 并且参考我给出的索引：1、过轻；2、正常；3、过重；4、肥胖，你的回答必须以这个格式开头：这个人的BMI是...(你的计算结果)。仅当如果给到的信息无法计算，返回None，其中第一个N大写。"}])
            elif taskName == 'ingredients':
                response = erniebot.ChatCompletion.create(model = "ernie-4.0", messages = [{"role" : "user", "content" : f"阅读以下描述: {text}, 这可能是什么食材？所有特征都是食材在未经烹饪、最原始状态下的。你的回答只能是我给出选项中所对应的一个数字：[豆腐-1]，[西红柿-2]，[辣椒-3]，[鸡肉-4]，[牛肉-5]，[芹菜-6]，[鸡蛋-7]，[蘑菇-8]，[绿豆-9]，[土豆-10]，[花椰菜-11]，[青菜-12]，[包菜-16]，[冬瓜-17]，[排骨-18]，[茄子-19]，在回答的一开始就返回数字索引，然后解释为什么是这个食材。如果给到的信息与食材无关或者无法判断，返回None，其中第一个N大写。"}])
            elif taskName == 'assistant':
                # response = erniebot.ChatCompletion.create(model = "ernie-4.0", messages = [{"role" : "user", "content" : f"你现在是一个做题机器人，你必须严格地遵循这个规则：只返回象征着正确答案的数字标号，不要返回解释、思考过程等任何其他内容,也就是说你所返回的内容只能是一个单独的数字。如果你完全无法得出答案，返回None，其中第一个N大写。接下来我会给你一道题目。阅读这个选择题: {text}, 它有用数字编号的四个选项：0、{a},1、{b},2、{c},3、{d}。你必须仔细思考，厘清所有逻辑关系，也要根据现实中的实际综合思考。"}])
                response = {'result': '1'}
            elif taskName == 'food':
                response = erniebot.ChatCompletion.create(model = "ernie-4.0", messages = [{"role" : "user", "content" : f"你现在是一个做饭机器人，你需要同时使用这两种食材：{c}和{d}做一道菜。你要用我提供的两种食材做出下面两道菜的其中一道：菜肴一：{a}，菜肴二：{b}。如果你能做出的菜肴符合菜肴一，请返回数字1，如果符合菜肴二，请返回数字2，如果都不行请返回数字3。你可以非常简短地说一下理由。"}])

            if response != 'None':
                return response
            else:
                return None
            
        if default_value is None:
            if taskName == 'bmi':
                default_value = {'result': '1'}  # 默认正常
            elif taskName == 'ingredients' and ing_side == 'left':
                default_value = {'result': '7'}
            elif taskName == 'ingredients' and ing_side == 'right':
                default_value = {'result': '2'}
            elif taskName == 'assistant':
                default_value = {'result': '0'}  # 默认第一个选项
            elif taskName == 'food':
                default_value = {'result': '3'}

        try:
            # 使用线程池执行器
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                # 提交任务
                future = executor.submit(_ernie_call)
                
                # 等待结果，设置超时
                response = future.result(timeout=timeout)
                
                if response != 'None' and response is not None:
                    return response
                else:
                    return default_value
                    
        except concurrent.futures.TimeoutError:
            logger.info(f"ernieInteract timeout after {timeout}s, returning default value")
            return default_value
        except Exception as e:
            logger.info(f"ernieInteract error: {e}, returning default value")
            return default_value

    def yiyan_get_humattr(self, text):
        return self.hum_analysis.get_res_json(text)
    
    def yiyan_get_actions(self, text):
        return self.action_bot.get_res_json(text)
    
    def debug(self):
        # self.arm.arm_init()
        # self.set_xyz_relative(0, 100, 60, 0.5)
        while True:
            if self._stop_flag:
                return
            image = self.cap_front.read()
            res = self.cruise(image)
            det_front = self.front_det(image)
            error = res[0]
            angle = res[1]
            image = self.cap_side.read()
            det_task = self.task_det(image)
            # det_hum = self.mot_hum(image)
            
            logger.info("")
            logger.info("--------------")
            logger.info("error:{} angle{}".format(error, angle))
            logger.info("front:{}".format(det_front))
            det_task.sort(key=lambda x: (x[4])**2 + (x[5])**2)
            logger.info("task:{}".format(det_task))
            # logger.inf
            if len(det_task) > 0:
                for det in det_task:
                    
                    dis = self.det2pose(det[4:])
                    logger.info("det:{} dis:{}".format(det, dis))
                # logger.info("hum_det:{}".format(det_hum))
                # logger.info("left:{} right:{}".format(self.left_sensor.read(), self.right_sensor.read()))
                # self.delay(0.5)
            # self.det2pose(det_task[4:])
            # logger.info("hum_det:{}".format(det_hum))
            logger.info("left:{} right:{}".format(self.left_sensor.read(), self.right_sensor.read()))
            self.delay(1)

    def walk_lane_test(self):
        end_function = lambda: True
        self.lane_base(0.3, end_function, stop=self.STOP_PARAM)

    def close(self):
        self._stop_flag = False
        self._end_flag = True
        self.thread_key.join()
        self.cap_front.close()
        self.cap_side.close()

        print("car closed!!!")
        # self.grap_cam.close()

    def manage(self, programs_list:list, order_index=0):

        def all_task():
            time.sleep(4)
            for func in programs_list:
                func()
        
        def lane_test():
            self.lane_dis_offset(0.3, 30)

        programs_suffix = [all_task, lane_test, self.task.arm.reset, self.debug]
        programs = programs_list.copy()
        programs.extend(programs_suffix)
        # print(programs)
        # 选中的python脚本序号
        # 当前选中的序号
        win_num = 5
        win_order = 0
        # 把programs的函数名转字符串
        logger.info(order_index)
        programs_str = [str(i.__name__) for i in programs]
        logger.info(programs_str)
        dis_str = sellect_program(programs_str, order_index, win_order)
        self.display.show(dis_str)

        self.stop()
        run_flag = False
        stop_flag = False
        stop_count = 0
        while True:
            # self.button_all.event()
            btn = self.key.get_key()
            # 短按1=1,2=2,3=3,4=4
            # 长按1=5,2=6,3=7,4=8
            # logger.info(btn)
            # button_num = car.button_all.clicked()
            
            if btn != 0:
                # logger.info(btn)
                # 长按1按键，退出
                if btn == 5:
                    # run_flag = True
                    self._stop_flag = True
                    self._end_flag = True
                    break
                else:
                    if btn == 4:
                        # 序号减1
                        self.beep()
                        if order_index == 0:
                            order_index = len(programs)-1
                            win_order = win_num-1
                        else:
                            order_index -= 1
                            if win_order > 0:
                                win_order -= 1
                        # res = sllect_program(programs, num)
                        dis_str = sellect_program(programs_str, order_index, win_order)
                        self.display.show(dis_str)

                    elif btn == 2:
                        self.beep()
                        # 序号加1
                        if order_index == len(programs)-1:
                            order_index = 0
                            win_order = 0
                        else:
                            order_index += 1
                            if len(programs) < win_num:
                                win_num = len(programs)
                            if win_order != win_num-1:
                                win_order += 1
                        # res = sllect_program(programs, num)
                        dis_str = sellect_program(programs_str, order_index, win_order)
                        self.display.show(dis_str)

                    elif btn == 3:
                        # 确定执行
                        # 调用别的程序
                        dis_str = "\n{} running......\n".format(str(programs_str[order_index]))
                        self.display.show(dis_str)
                        self.beep()
                        self._stop_flag = False
                        programs[order_index]()
                        self._stop_flag = True
                        dis_str = sellect_program(programs_str, order_index, win_order)
                        self.stop()
                        self.beep()

                        # 自动跳转下一条
                        # if order_index == len(programs)-1:
                        #     order_index = 0
                        #     win_order = 0
                        # else:
                        #     order_index += 1
                        #     if len(programs) < win_num:
                        #         win_num = len(programs)
                        #     if win_order != win_num-1:
                        #         win_order += 1
                        # res = sllect_program(programs, num)
                        dis_str = sellect_program(programs_str, order_index, win_order)
                        self.display.show(dis_str)
                    logger.info(programs_str[order_index])
            else:
                self.delay(0.02)
                
            time.sleep(0.02)

        for i in range(2):
            self.beep()
            time.sleep(0.4)
        time.sleep(0.1)
        self.close()

if __name__ == "__main__":
    # kill_other_python()
    my_car = MyCar()
    time.sleep(0.4)
    # my_car.task.get_ingredients(1, arm_set=True)

    def start_det_loc():
        det1 = [15, 60, "cylinder3", 0, 0, 0, 0.47, 0.7]
        det2 = [14, 80, "cylinder2", 0, 0, 0, 0.69, 0.7]
        det3 = [13, 100, "cylinder1", 0,  0, 0, 0.77, 0.7]          
        dets = [det1, det2, det3]
        my_car.lane_det_location(0.2, dets)

    def lane_det_test():
        my_car.lane_det_dis2pt(0.2, 0.16)

    def move_test():
        my_car.set_vel_time(0.3, 0, -0.6, 1)

    def ocr_test():
        print(my_car.get_ocr())

    my_car.manage([start_det_loc, lane_det_test, move_test, ocr_test])
    # my_car.lane_time(0.3, 5)
    
    # my_car.lane_dis_offset(0.3, 1.2)
    # my_car.lane_sensor(0.3, 0.5)
    # my_car.debug() 

    # text = "犯人没有带着眼镜，穿着短袖"
    # criminal_attr = my_car.hum_analysis.get_res_json(text)
    # print(criminal_attr)
    # my_car.task.reset()
    # pt_tar = my_car.task.punish_crimall(arm_set=True)
    # hum_attr = my_car.get_hum_attr(pt_tar)
    # print(hum_attr)
    # res_bool = my_car.compare_humattr(criminal_attr, hum_attr)
    # print(res_bool)
    # pt_tar = [0, 1, 'pedestrian',  0, 0.02, 0.4, 0.22, 0.82]
    # for i in range(4):
    #     my_car.set_pos_offset([0.07, 0, 0])
    #     my_car.lane_det_location(0.1, pt_tar, det="mot", side=-1)
    # my_car.close()
    # text = my_car.get_ocr()
    # print(text)
    # pt_tar = my_car.task.pick_up_ball(arm_set=True)
    # my_car.lane_det_location(0.1, pt_tar)
    
    my_car.close()
    # my_car.debug()
    # while True:
    #     text = my_car.get_ocr()
    #     print(text)

    # my_car.task.reset()
    # my_car.lane_advance(0.3, dis_offset=0.01, value_h=500, sides=-1)
    # my_car.lane_task_location(0.3, 2)
    # my_car.lane_time(0.3, 5)
    # my_car.debug()
    
    # my_car.debug()

            
    # my_car.task.pick_up_block()
    # my_car.task.put_down_self_block()
    # my_car.lane_time(0.2, 2)
    # my_car.lane_advance(0.3, dis_offset=0.01, value_h=500, sides=-1)
    # my_car.lane_task_location(0.3, 2)
    # my_car.task.pick_up_block()
    # my_car.close()
    # logger.info(time.time())
    # my_car.lane_task_location(0.3, 2)


    # my_car.debug()
    # programs = [func1, func2, func3, func4, func5, func6]
    # my_car.manage(programs)
    # import sys
    # test_ord = 0
    # if len(sys.argv) >= 2:
    #     test_ord = int(sys.argv[1])
    # logger.info("test:", test_ord)
    # car_test(test_ord)
