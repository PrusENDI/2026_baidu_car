import time
import threading
import os
import numpy as np
import cv2 as cv
from task_func import MyTask
from log_info import logger
from car_wrap import MyCar
from tools import CountRecord
import math
import sys, os

import difflib

from vehicle import CarBase, BluetoothPad, ScreenShow, Beep, ArmBase
from infer_cs import ClintInterface, Bbox

from brakeMgr import brakeMgr
# 添加上本文件对应目录
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

if __name__ == "__main__":
    my_car = MyCar()
    brake_manager = brakeMgr(my_car)
    my_car.task.arm.reset()

    my_car.task.arm.set(my_car.task.arm.horiz_mid, 0.12)
    # my_car.task.arm.switch_side(-1)

    for i in range(0,3):
        my_car.beep()
        time.sleep(0.2)

    target = [[0, 30, 'tomato', 0, 0, 0.03, 0.25, 0.24]]
    # brake_manager.targetBrake(0.2, target)
    # print(f"target No. {target[0][0]} detected, brake!")
    # time.sleep(0.2)
    my_car.lane_det_location(0.2, target, side=1, dis_out=1)

    my_car.close()