#!/usr/bin/python3
# -*- coding: utf-8 -*-
import os
import sys
# 添加上本文件对应目录，用于导入car_wrap_2026和car_task_function
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

import car_task_function as task_runtime
from car_wrap_2026 import MyCar
from car_task_function import (
    init,
    auto_lane_tracing,
    auto_seeding,
    target_shooting_detection,
    water_tower_task,
    target_shooting,
    crop_harvesting,
    sort_and_store,
    get_order,
    order_delivery
)
from manual_start import wait_for_start_key

def main():
    init()                                          # 初始化
    while True:
        task_runtime.my_car._stop_flag = False
        task_runtime.my_car._start_key_event.clear()
        task_runtime.my_car.beep()                  # 提示初始化完成，可以按按键1
        wait_for_start_key(task_runtime.my_car._start_key_event)
        #auto_lane_tracing(speed=0.3, dis_hold=99)  # 巡线测试，99m可以保证巡线到基地
        auto_seeding()                            # 播种任务
        animal_list = target_shooting_detection() # 识别虫害
        water_tower_task()                        # 灌溉任务
        #target_shooting()
        
        target_shooting(animal_list)              # 射击除害
        
        crop_harvesting()                          # 作物收集
        # sort_and_store(first_label)               # 作物储存
        order_list = get_order()                  # 订单获取
        # order_delivery(order_list)                # 订单配送（使用默认测试订单）
    

if __name__ == "__main__":
    main()
