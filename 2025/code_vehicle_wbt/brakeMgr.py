import time
import threading
import os
import numpy as np
from task_func import MyTask
from log_info import logger
from car_wrap import MyCar
from tools import CountRecord
import math
import sys, os

from cvTest import AnswerGet
from infer_cs import ClintInterface, Bbox
from vehicle import ArmBase

# 添加上本文件对应目录
sys.path.append(os.path.abspath(os.path.dirname(__file__))) 

class brakeMgr:
    def __init__(self, my_car):
        self.my_car = my_car
        self.task_det = ClintInterface('task')

    def targetBrake(self, speed, pts_tar, time_out=5, debug = False):
        startTime = time.time()
        target_ids = [pt[0] for pt in pts_tar]
        target_found = False
        
        if debug == False:
            first_img_side = self.my_car.cap_side.read()
            dets_ret = self.task_det(first_img_side)
            for det in dets_ret:
                det_id, obj_id, det_label, det_score, det_bbox = det[0], det[1], det[2], det[3], det[4:]
                    
                # 检查是否为目标物体（通过ID或标签匹配）
                is_target = (det_id in target_ids)
                    
                if is_target and det_score > 0.7:
                    # 准备停车
                    self.my_car.stop()
                    print(f"Target detected! ID: {det_id}, Label: {det_label}, Score: {det_score:.3f}")
                    target_found = True
                    return True

            while True:
                image = self.my_car.cap_front.read()
                error_y, error_angle = self.my_car.cruise(image)
                y_speed, angle_speed = self.my_car.lane_pid.get_out(-error_y, -error_angle)

                # 获取侧面摄像头图像
                img_side = self.my_car.cap_side.read()
                
                # 进行目标检测
                dets_ret = self.task_det(img_side)
                
                # 检查是否有匹配的目标且置信度>0.7
                for det in dets_ret:
                    det_id, obj_id, det_label, det_score, det_bbox = det[0], det[1], det[2], det[3], det[4:]
                    
                    # 检查是否为目标物体（通过ID或标签匹配）
                    is_target = (det_id in target_ids)

                    # 检查高度方向偏离限制（边界框中心y坐标）
                    bbox_center_y = det_bbox[1]  # 归一化坐标系中的y中心
                    # 获取图像高度
                    img_height = first_img_side.shape[0]
                    # 将归一化坐标转换为像素坐标
                    pixel_center_y = (bbox_center_y + 1) * img_height / 2
                    image_center_y = img_height / 2
                    height_deviation = abs(pixel_center_y - image_center_y)
                    
                    if is_target and det_score > 0.7 and height_deviation < 0.3 * img_height:
                        # 准备停车
                        # time.sleep(0.2)
                        self.my_car.stop()
                        print(f"Target detected! ID: {det_id}, Label: {det_label}, Score: {det_score:.3f}")
                        target_found = True

                if target_found:
                    return True
                
                if time.time() - startTime > time_out:
                    print("Timeout: No target detected within the specified time.")
                    return False
                
                # 继续前进
                self.my_car.set_velocity(speed, y_speed, angle_speed)

        else:
            while True:
                image = self.my_car.cap_front.read()
                error_y, error_angle = self.my_car.cruise(image)
                y_speed, angle_speed = self.my_car.lane_pid.get_out(-error_y, -error_angle)

                if time.time() - startTime > time_out:
                    print("debug Timeout")
                    return False
                
                # 继续前进
                self.my_car.set_velocity(speed, y_speed, angle_speed)

if __name__ == "__main__":
    my_car = MyCar()
    arm = ArmBase()
    arm.reset()
    arm.set(arm.horiz_mid, 0.16)

    brake_manager = brakeMgr(my_car)
    target = [[11, 30, 'tomato', 0, 0, 0.03, 0.25, 0.24]]

    brake_manager.targetBrake(0.2, target)
    my_car.close()