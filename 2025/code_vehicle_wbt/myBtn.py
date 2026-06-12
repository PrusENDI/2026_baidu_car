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

class myButton:
    def __init__(self):
        self.key = Key4Btn(port_id=1)

    def retKey(self):
        btnValue = self.key.get_key()
        if btnValue != 0:
            print(f"按键值: {btnValue}")
                
            # 判断按键类型
            if btnValue <= 4:
                print(f"短按按键{btnValue}")
            elif btnValue <= 8:
                print(f"长按按键{btnValue-4}")
            elif btnValue <= 12:
                print(f"连续按按键{btnValue-8}")
        
        return btnValue

if __name__ == "__main__":
    my_button = myButton()
    my_button.retKey()