#!/usr/bin/python3
# Orin 2026-07-17 使用明确的 Python 3 解释器；修改前 worktree 为 /usr/bin/python。
# -*- coding: utf-8 -*-
import os
import sys
# 添加上本文件对应目录，用于导入car_wrap_2026和car_task_function
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

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

def main():

    init()                                          # 初始化
    # Orin 2026-07-17 调试入口只运行播种任务，避免播种结束后自动进入其余任务。
    # 修改前 worktree 会继续依次运行虫害识别、灌溉、射击、收获、储存和订单配送。
    # auto_lane_tracing(speed=0.3, dis_hold=99)     # 可选独立巡线调试，默认不执行
    #auto_seeding()                                  # 播种任务
    # 修改前代码保留如下，需要恢复完整流程时逐项取消注释：
    # animal_list = target_shooting_detection()     # 识别虫害
    # water_tower_task()                            # 灌溉任务
    # target_shooting(animal_list)                  # 射击除害
    # crop_harvesting()                             # 作物收集
    # sort_and_store()                              # 作物储存
    # order_list = get_order()                      # 订单获取
    # order_delivery(order_list)                    # 完整流程：配送真实识别订单
    order_delivery()                               # 独立调试：配送函数内置订单


if __name__ == "__main__":
    main()
