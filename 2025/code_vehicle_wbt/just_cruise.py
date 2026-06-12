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
# 添加上本文件对应目录
sys.path.append(os.path.abspath(os.path.dirname(__file__))) 

class JustCruise():
    def __init__(self):
        self.my_car = MyCar()
        self.my_car.STOP_PARAM = False
        self.my_car.task.reset()
        self.side = None

    def cruisePartOne(self):
            self.my_car.lane_dis_offset(0.3, 0.5)
            print(self.my_car.get_odometry())
            self.my_car.set_pose_offset([0.3,0,0], 1)
            print(self.my_car.get_odometry())
            det_side = self.my_car.lane_det_dis2pt(0.2, 0.19)

            # Get card side to turn
            self.side = self.my_car.get_card_side()
            print("side : ", self.side)
            self.my_car.set_pose_offset([0, self.side*0.1, math.pi/4*self.side], 1)
            # self.my_car.lane_dis_offset(0.2, 2.5)
            self.my_car.lane_dis_offset(0.2, 1.3)
            time.sleep(0.1)
            self.my_car.set_pose_offset([0.4, 0, 0], 1)
            time.sleep(0.1)
            self.my_car.set_pose_offset([0, self.side*0.1, math.pi/4*self.side], 1)

    def cruisePartTwo(self):
        self.my_car.beep()
        self.my_car.lane_dis_offset(0.2, 5.6)
        time.sleep(0.2)
        self.my_car.beep()
        # self.my_car.set_pose_offset([0.4, 0, 0], 1)

    def cruisePartThree(self):
        self.my_car.beep()
        self.my_car.lane_dis_offset(0.2, 12)
        # self.my_car.set_pose_offset([0.1, 0, 0], 1)
        # time.sleep(0.2)
        # self.my_car.beep()
        # self.my_car.set_pose_offset([0.4, 0, -3*(math.pi/4)], 1)
        # time.sleep(0.2)
        # self.my_car.beep()
        # self.my_car.set_pose_offset([0, 0.2, 0], 1)
        # self.my_car.lane_dis_offset(0.3, 3)


if __name__ == "__main__":
    just_cruise = JustCruise()
    
    '''
    from myBtn import myButton
    my_button = myButton()
    while True:
        key = my_button.retKey()
        if key != 0:
            break
    '''

    just_cruise.cruisePartOne()
    just_cruise.cruisePartTwo()
    just_cruise.cruisePartThree()

    just_cruise.my_car.close()
