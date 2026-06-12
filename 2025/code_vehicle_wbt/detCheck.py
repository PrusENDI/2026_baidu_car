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
import argparse

from vehicle import CarBase, BluetoothPad, ScreenShow, Beep, ArmBase
from ingredientsEncoder import ingEncoder
from foodDesEncoder import foodEncoder

from infer_cs import ClintInterface, Bbox
# 添加上本文件对应目录
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

parser = argparse.ArgumentParser(description="Task Detection Check")
parser.add_argument('-R', '--reset', action="store_true", help='Reset arm before detection check or not (DEFAULT NOT RESET)')
args = parser.parse_args()

if __name__ == "__main__":
    my_car = MyCar()
    my_car.STOP_PARAM = False
    directTask = ClintInterface('task')

    ing_encoder = ingEncoder()
    food_encoder = foodEncoder(my_car)

    if args.reset:
        my_car.task.arm.reset()
        my_car.task.arm.set(my_car.task.arm.horiz_mid, 0)
        # my_car.task.arm.switch_side(-1)
        # arm.set(arm.horiz_mid, 0.1)
    else:
        pass

    def showTaskDet():
        # 定义不同类别的颜色
        colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
        
        while True:
            if my_car._stop_flag:
                break
            img = my_car.cap_side.read()
            dets = directTask(img)

            print(f"food detection result : {food_encoder.orderedRec(dets)}")
            
            img_display = img.copy()
            
            for i, det in enumerate(dets):
                det_id, det_width, det_label, det_score, det_bbox = det[0], det[1], det[2], det[3], det[4:]
                x_c, y_c, w, h = det_bbox
                
                img_h, img_w = img.shape[:2]
                x1 = img_w * (1 + x_c) / 2 - img_w * w / 4
                x2 = x1 + img_w * w / 2
                y1 = img_h * (1 + y_c) / 2 - img_h * h / 4
                y2 = y1 + img_h * h / 2
                
                x1 = max(0, min(int(x1), img_w-1))
                y1 = max(0, min(int(y1), img_h-1))
                x2 = max(0, min(int(x2), img_w-1))
                y2 = max(0, min(int(y2), img_h-1))
                
                color = colors[i % len(colors)]
                
                cv.rectangle(img_display, (x1, y1), (x2, y2), color, 2)
                
                label_text = f"ID:{det_id} {det_label} {det_score:.2f}"
                
                (text_width, text_height), baseline = cv.getTextSize(label_text, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                
                cv.rectangle(img_display, (x1, y1-text_height-5), (x1+text_width, y1), color, -1)
                
                cv.putText(img_display, label_text, (x1, y1-5), 
                        cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            info_text = f"Objects: {len(dets)}"
            cv.putText(img_display, info_text, (10, 30), 
                    cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv.imshow("Task Detection", img_display)
            
            key = cv.waitKey(1) & 0xFF
            if key == 27 or key == ord('q'):
                break
            
            print(f"Detected {len(dets)} objects: {dets}")
            time.sleep(0.1)
        
        cv.destroyAllWindows()

    showTaskDet()
    my_car.close()