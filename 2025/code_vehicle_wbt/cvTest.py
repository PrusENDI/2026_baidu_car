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
from infer_cs import ClintInterface, Bbox
# 添加上本文件对应目录
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

class AnswerGet:
    '''
    The best height to get the answer is 0.1
    '''
    def __init__(self, my_car):
        self.ocr_rec = ClintInterface('ocr')
        self.my_car = my_car

    def draw_quarter_lines(self):
        img = self.my_car.cap_side.read()

        """在图像的四等分处画线"""
        img_h, img_w = img.shape[:2]
        
        # 计算四等分点
        quarter_w = img_w // 4
        quarter_h = img_h // 4
        
        # 创建副本以避免修改原图
        img_with_lines = img.copy()
        
        # 画垂直线（四等分宽度）
        for i in range(1, 4):  # 画3条垂直线
            x = quarter_w * i
            cv.line(img_with_lines, (x, 0), (x, img_h), (0, 255, 0), 2)
        
        # 画水平线（四等分高度）
        for i in range(1, 4):  # 画3条水平线
            y = quarter_h * i
            cv.line(img_with_lines, (0, y), (img_w, y), (0, 255, 0), 2)
        
        return img_with_lines

    def process_second_row_ocr(self):
        img = self.my_car.cap_side.read()

        """
        将图像四等分后，取第二行的四个方格依次送入OCR识别
        """
        img_h, img_w = img.shape[:2]
        
        # 计算四等分点
        quarter_w = img_w // 4
        quarter_h = img_h // 4
        
        # 第二行的y坐标范围 (从0开始计数，第二行是索引1)
        y_start = quarter_h * 1  # 第二行开始
        y_end = quarter_h * 2    # 第二行结束
        
        ocr_results = []
        
        # 从左到右处理第二行的四个方格
        for col in range(4):
            x_start = quarter_w * col
            x_end = quarter_w * (col + 1)
            
            # 裁切当前方格
            grid_img = img[y_start:y_end, x_start:x_end]
            ocr_result = self.ocr_rec(grid_img)
            ocr_results.append(ocr_result)
            # print(f"Grid [{col+1}]: {ocr_result}")
        
        return ocr_results

if __name__ == "__main__":
    my_car = MyCar()
    answer_get = AnswerGet(my_car)

    while True:
        img_display = answer_get.draw_quarter_lines()

        cv.imshow('Image with Quarter Lines', img_display)
        answers = answer_get.process_second_row_ocr()
        for i in range(4):
            print(f"Answer {i}: {answers[i]}")

        key = cv.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):
            break

    cv.destroyAllWindows()
    my_car.close()