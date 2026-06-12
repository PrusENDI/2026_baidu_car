#!/usr/bin/python
# -*- coding: utf-8 -*-
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

if __name__ == "__main__":
    # kill_other_python()
    my_car = MyCar()

    from brakeMgr import brakeMgr
    brake_manager = brakeMgr(my_car)

    my_car.STOP_PARAM = False
    my_car.task.reset()
    my_car.task.bmi_set(0)
    
    def hanoi_tower_func():
        my_car.lane_dis_offset(0.4, 0.5)
        # print(my_car.get_odometry())
        my_car.set_pose_offset([0.3, 0, 0], 1)
        # print(my_car.get_odometry())
        det_side = my_car.lane_det_dis2pt(0.2, 0.19)
        side = my_car.get_card_side()
        if side == -1:
            my_car.set_pose_offset([0, 0, math.pi / 4], 1)
            my_car.lane_dis_offset(0.3, 1.2)
            return
        # print(side)
        # 调整检测方向
        my_car.task.arm.switch_side(side * -1)
        
        try:
            # 调整车子朝向
            my_car.set_pose_offset([0, 0, math.pi / 4 * side], 1)

            # 第一个要抓取的圆柱
            cylinder_id = 1
            # 调整抓手位置，获取要抓取的圆柱信息
            pts = my_car.task.pick_up_cylinder(cylinder_id, True)
            # 走一段距离
            my_car.lane_dis_offset(0.3, 0.66)

            # 第二次感应到侧面位置
            # my_car.lane_sensor(0.2, value_h=0.3, sides=side*-1)
            my_car.lane_sensor(0.10, value_h=0.3, sides=side * -1, stop=True)     #orginal 0.2
            # return
            # 记录此时的位置
            pose_dict = {}
            pose_last = None

            for i in range(3):
                debugFlag = 3 - i

                # 根据给定信息定位目标
                index = my_car.lane_det_location(0.10, pts, side=side * -1)     #orginal 0.2
                my_car.beep()
                pose_dict[index] = my_car.get_odometry().copy()
                if i == 2:
                    pose_last = my_car.get_odometry().copy()
                # print(index)
                # pose_list.append([index, my_car.get_odometry().copy()])

                if i < 2:
                    my_car.set_pose_offset([0.08, 0, 0])
                    my_car.beep()

            # print(pose_dict)
            # 根据识别到的位置调整方向位置
            # angle = math.atan((pose_dict[2][1] - pose_dict[0][1]) / (pose_dict[2][0] - pose_dict[0][0]))
            # print(angle)
            # my_car.set_pose_offset([0, 0, -angle])
            # 重新定位最后一个圆柱
            # my_car.lane_det_location(0.2, pts, side=side*-1)
            angle_det = my_car.get_odometry()[2]
            # 计算目的地终点坐标
            pose_end = [0, 0, angle_det]
            pose_end[0] = pose_last[0] + 0.135 * math.cos(angle_det)
            pose_end[1] = pose_last[1] + 0.06 * math.sin(angle_det)     #orginal
            # print(det)
            # 调整到目的地
            # my_car.set_pose(det)
            for i in range(3):
                if i == 0:
                    det = pose_dict[i]
                    det[2] = angle_det
                    my_car.set_pose(det)
                    # my_car.lane_det_location(0.2, pts, side=side*-1)
                    my_car.task.pick_up_cylinder(i)
                    my_car.set_pose(pose_end)
                    my_car.task.put_down_cylinder(i)
                elif i == 1:
                    det = pose_dict[i]
                    det[2] = angle_det
                    my_car.set_pose(det)
                    # if side == -1:
                        # my_car.set_pose_offset([0.03, 0, 0])
                    # my_car.lane_det_location(0.2, pts, side=side*-1)
                    my_car.task.pick_up_cylinder(i)
                    my_car.set_pose(pose_end)
                    # if side == -1:
                        # my_car.set_pose_offset([0.03, 0, 0])
                    my_car.task.put_down_cylinder(i)
                # else:
                #     det = pose_dict[i]
                #     det[2] = angle_det
                #     my_car.set_pose(det)
                #     if side == -1:
                #         return
                #         # my_car.set_pose_offset([0.03, 0, 0])
                #     # my_car.lane_det_location(0.2, pts, side=side*-1)
                #     my_car.task.specialPickUp(i)
                #     # my_car.task.arm.grap(0)
                #     my_car.set_pose(pose_end)
                #     #my_car.task.arm.grap(0)
                #     my_car.task.put_down_cylinder(i)
            return

        except:
            my_car.lane_dis_offset(0.3, 0.08)
            return
    
    def bmi_cal():
        from numProcess import NumProcess
        num_process = NumProcess()

        my_car.task.arm.switch_side(1)
        
        my_car.set_pose_offset([0, -0.04, 0])     #add

        my_car.lane_dis_offset(0.35, 0.8)

        # 准备手臂位置
        pts = my_car.task.bmi_set(arm_set=True)

        time.sleep(0.2)
        my_car.lane_time(0.3, 3.3)
        time.sleep(0.2)
        my_car.lane_time(0, 1)
        my_car.set_pose_offset([0, -0.065, 0])

        # 巡航到bmi识别附件
        my_car.lane_sensor(0.3, value_h=0.30, sides=1)
        # 推开bmi识别标签
        # my_car.lane_dis_offset(0.3, 0.5)
        my_car.set_pose_offset([0.03, -0.03, 0])     #0.03  -0.03
        time.sleep(0.5)
        my_car.set_pose_offset([-0.13,0.23, 0])     #-0.13    0.23
        time.sleep(0.2)
        my_car.beep()
        my_car.beep()
        my_car.beep()
        my_car.set_pose_offset([0.1, -0.12, 0])
        time.sleep(0.2)
        my_car.set_pose_offset([0.12, 0, 0])
        # 调整bmi识别位置
        # my_car.lane_det_location(0.2, pts, side=1)
        # 识别相关文字
        text = my_car.get_ocr()
        time.sleep(0.3)
        print(text)

        '''
        repeat_times = 5
        for i in range(repeat_times):
            if i < repeat_times - 1:
                bmiValue = my_car.ernieInteract('bmi', text)
                print(f"Result: {bmiValue['result']}")
                try:
                    bmi_val = float(num_process.extract_first_number(bmiValue))
                except:
                    pass
                else:
                    if bmi_val < 18.5:
                        out = 1
                    elif 18.5 <= bmi_val <= 24:
                        out = 2
                    elif 24 < bmi_val <= 28:
                        out = 3
                    else:
                        out = 4
                    break
            
            elif i == repeat_times - 1:
                out = 2
        '''
        out = 3

        my_car.set_pose_offset([0.12, 0, 0])
        # my_car.beep()
        my_car.task.bmi_set(out)
        time.sleep(0.5)
        # 调整位置准备放置球
        # my_car.lane_dis_offset(0.21, 0.19)
        # my_car.set_pose_offset([0, 0.05, 0], 0.7)
        # my_car.task.put_down_ball()
    

    def camp_fun():
        angle_offset = -math.pi/2*0.82
        # dis_angle = -math.pi/2*0.3
        # dis = 1.
        dis_x = 1.36
        dis_y = -0.87
        # print(dis_x, dis_y)
        angle_now = my_car.get_odometry()[2]
        x_offset = dis_x*math.cos(angle_now) - dis_y*math.sin(angle_now)
        y_offset = dis_y*math.cos(angle_now) + dis_x*math.sin(angle_now)
        angle_tar = angle_now - math.pi*2 + angle_offset
        pose = my_car.get_odometry().copy()
        pose[0] = pose[0] + x_offset
        pose[1] = pose[1] + y_offset
        pose[2] = angle_tar
        # print(pose)
        # return
    
        my_car.lane_sensor(0.35, value_h=1, sides=-1)
        # time.sleep(25)
        # my_car.lane_sensor()
        my_car.lane_dis_offset(0.35, 0.5)
        my_car.set_vel_time(0.3, 0, -0.5, 1.8)

        # Original 2.95
        my_car.lane_dis_offset(0.35, 3.05)
        time.sleep(0.1)
        
        # my_car.set_pose(pose, vel=[0.2, 0.2, math.pi/3])
        my_car.set_pose_offset([0.2, 0, 0], 1)     #orginal 0.4

        # my_car.lane_dis_offset(0.3, 1.52)

        # my_car.
        # my_car.set_vel_time(0.3, 0, -0.1, 1)
        # my_car.move_advance([0.3, 0, 0], value_l=1, sides=-1)
        # my_car.lane_time(0, 1)
        # my_car.move_advance([0.3, 0, -0.2], value_l=1, sides=-1)
        # my_car.move_distance([0.3, 0, 0], 0.25)
        # my_car.move_advance([0.3, 0, 0], value_l=1, sides=-1)
        # my_car.move_advance([0.3, 0, 0], value_l=0.5, sides=-1)


    def send_fun():
        # my_car.move_advance([0.3, 0, 0], value_l=1, sides=-1)
        # my_car.move_distance([0.3, 0, -0.1], 0.25)
        # my_car.lane_sensor(0.3, value_l=1.1, sides=-1)
        # my_car.lane_dis_offset(0.3, 1.74)
        my_car.lane_time(0, 1)
        my_car.task.eject(1)
        my_car.set_pose_offset([0, -0.03, 0])     #add
        # my_car.set_pose_offset([0, 0, -math.pi/128])     #add
        

    # 获取食材
    def task_ingredients():
        from ingredientsIndex import IngIndex
        from ingredientsEncoder import ingEncoder

        ing_index = IngIndex()
        ing_encoder = ingEncoder()

        tar = my_car.task.get_ingredients(side=1,ocr_mode=True, arm_set=True)
        my_car.lane_sensor(0.3, value_h=0.2, sides=1)
        # my_car.lane_dis_offset(0.3, 0.17)
        my_car.lane_dis_offset(0.3, 0.175)
        # my_car.lane_det_location(0.2, tar, side=1)
        text_left = my_car.get_ocr()
        print(text_left)

        my_car.task.arm.switch_side(-1)
        time.sleep(0.2)
        # my_car.set_pose_offset([0, 0.05, 0])
        # time.sleep(0.2)
        text_right = my_car.get_ocr()
        print(text_right)

        my_car.set_pose_offset([0, -0.03, 0])

        # Use ernieBot to recognize ingredients from description
        from numProcess import NumProcess
        num_process = NumProcess()
        while True:
            if text_left == None:
                indexLeft = 2
                break

            ingredientLeft = my_car.ernieInteract('ingredients', text_left, ing_side='left')
            print(ingredientLeft['result'])

            try:
                indexLeft = int(num_process.extract_first_integer(ingredientLeft))
                print(f"indexLeft: {indexLeft}")
            except:
                pass
            else:
                break

        while True:
            if text_right == None:
                indexRight = 7
                break

            ingredientRight = my_car.ernieInteract('ingredients', text_right, ing_side='right')
            print(ingredientRight['result'])

            try:
                indexRight = int(num_process.extract_first_integer(ingredientRight))
                print(f"indexRight: {indexRight}")
            except:
                pass
            else:
                break

        # indexLeft = 2
        # indexRight = 7

        '''
        my_car.set_pose_offset([-0.18, 0, 0])
        # tar = my_car.task.pick_ingredients(1, 2, arm_set=True)
        # print(tar)
        tar = [[3, 30, 'tomato', 0, 0, 0.03, 0.25, 0.24]]
        my_car.lane_det_location(0.2, tar, side=1)
        my_car.beep()
        '''

        my_car.task.arm.set(1.5*my_car.task.arm.horiz_mid, 0.05)

        for i in range(20):
            dets = my_car.task_det(my_car.cap_side.read())
            grid_right = ing_encoder.encode(dets)
            target_right_position = ing_encoder.id2pos(indexRight, grid_right)

        print(f"Right target position: {target_right_position}")

        if target_right_position is None:
            print("No right ingredient detected, defaulting to (0, 1).")
            target_right_position = (0, 1)

        # Righ pick up
        my_car.set_pose_offset([-(target_right_position[1] - 1)*0.13, 0, 0])

        if target_right_position[0] == 0:
            rowRight = 2
        elif target_right_position[0] == 1:
            rowRight = 1

        my_car.task.pick_ingredients(1, rowRight, my_car = my_car)

        # Get back to the center
        my_car.set_pose_offset([(target_right_position[1] - 1)*0.13, 0, 0])

        # Switch to left
        my_car.task.arm.switch_side(1)
        time.sleep(0.2)
        my_car.task.arm.set(0.5*my_car.task.arm.horiz_mid, 0.05)

        for i in range(20):
            dets = my_car.task_det(my_car.cap_side.read())
            grid_left = ing_encoder.encode(dets)
            target_left_position = ing_encoder.id2pos(indexLeft, grid_left)

        print(f"Left target position: {target_left_position}")

        if target_left_position is None:
            print("No left ingredient detected, defaulting to (0, 1).")
            target_left_position = (0, 1)

        # Left pick up
        my_car.set_pose_offset([(target_left_position[1] - 1)*0.13, 0, 0])

        if target_left_position[0] == 0:
            rowLeft = 2
        elif target_left_position[0] == 1:
            rowLeft = 1

        my_car.task.pick_ingredients(2, rowLeft, my_car = my_car)

        import yaml
        try:
            with open('ingredients.yml', 'r', encoding='utf-8') as file:
                ingredients_data = yaml.safe_load(file)
            
            ingredient_left_label = None
            ingredient_right_label = None
            
            # 查找对应的食材标签
            for target in ingredients_data['targets']:
                if target[0] == indexLeft:  # target[0] 是 ID
                    ingredient_left_label = target[2]  # target[2] 是标签
                if target[0] == indexRight:
                    ingredient_right_label = target[2]
            
            print(f"Left ingredient: {ingredient_left_label}")
            print(f"Right ingredient: {ingredient_right_label}")
            
            return ingredient_left_label, ingredient_right_label
        
        except:
            print(f"Error reading ingredients.yml: {e}")
            return 'tomato','egg'     #orginal

    def task_answer(btnIn):
        my_car.lane_sensor(0.35, value_h=0.3, sides=1)
        my_car.task.arm.switch_side(1)
        # my_car.task.arm.set_hand_angle(48)
        my_car.move_distance([0.3, 0, 0], 0.22)
        my_car.set_pose_offset([0, -0.05, 0])
        my_car.stop()

        '''
        getAnswers = AnswerGet(my_car)

        my_car.task.arm.set(my_car.task.arm.horiz_mid, 0.1)
        answers = getAnswers.process_second_row_ocr()
        for answerNum in range(4):
            print(f"Answer{answerNum}: {answers[answerNum]}")

        tar = my_car.task.get_answer(arm_set=True)
        text = my_car.get_ocr()
        print(text)
        '''
        my_car.task.arm.set(my_car.task.arm.horiz_mid, 0.18)
        time.sleep(0.1)
        my_car.task.arm.set(1.5*my_car.task.arm.horiz_mid, 0.18)
        time.sleep(1)

        '''
        times = 5
        for i in range(times):
            if i < times - 1:
                correctAnswer = my_car.ernieInteract('assistant', text, a=answers[0], b=answers[1], c=answers[2], d=answers[3])
                print(correctAnswer['result'])
                try:
                    out = int(correctAnswer['result'])
                except:
                    pass
                else:
                    out = btnIn
                    break
            
            elif i == times - 1:
                out = btnIn
        '''

        out = btnIn
        
        print(f"Correct answer is {out}")
        pose_tar_offset = [0.08*out-0.07, 0, 0]
        time.sleep(1)
        my_car.set_pose_offset(pose_tar_offset)
        my_car.task.get_answer(my_car=my_car)

        '''
        tar = my_car.task.get_answer(arm_set=True)
        # my_car.lane_det_location(0.2, tar, side=1)
        text = my_car.get_ocr()
        print(text)
        answer = my_car.ernieInteract('assistant', text)
        print(answer['result'])
        out = int(answer['result'])
        pose_tar_offset = [0.08*out-0.07, 0, 0]
        my_car.set_pose_offset(pose_tar_offset)
        my_car.task.get_answer()
        # my_car.move_distance([0.3, 0, 0], 0.24)
        '''

    def task_fun2():
        # 巡航到投掷任务点2
        my_car.lane_sensor(0.3, value_h=0.5, sides=-1)     #orginal 0.3
        # 调整方向
        my_car.lane_time(0, 1)
        my_car.set_pose_offset([-0.2, 0, 0])      #-0.07
        my_car.task.eject(2)
        time.sleep(0.3)
        # my_car.set_pose_offset([-0.1, 0, 0])
        # my_car.task.arm.switch_side(-1)
        # my_car.move_distance([0.3, 0, 0], 0.24)
        # tar = my_car.task.get_answer(arm_set=True)

    def task_food(ingredient_01 = 'egg', ingredient_02 = 'tomato'):
        try:
            from numProcess import NumProcess
            num_process = NumProcess()

            from foodDesEncoder import foodEncoder
            food_encoder = foodEncoder(my_car)

            # my_car.lane_sensor(0.3, value_h=0.5, sides=-1)
            my_car.set_pose_offset([0.12, 0, 0])
            my_car.task.arm.set(my_car.task.arm.horiz_mid, 0.05)
            my_car.task.arm.switch_side(-1)
            time.sleep(1)
            
            #add
            my_car.task.arm.set_hand_angle(0)

            for i in range(10):
                current_img = my_car.cap_side.read()
                current_dets = my_car.task_det(current_img)
                result = food_encoder.orderedRec(dets=current_dets)
                print(f"Result: {result}")

            canDo = my_car.ernieInteract('food', timeout=20, a=result[0], b=result[1], c=ingredient_01, d=ingredient_02)
            print(f"Can do: {canDo['result']}")
            foodResult = int(num_process.extract_first_integer(canDo))
            print(f"Food result: {foodResult}")
            
            
            #add
            my_car.task.arm.set_hand_angle(135)

            if foodResult != 3:
                my_car.set_pose_offset([0.075, 0, 0])
                my_car.task.set_food(1, row=3-foodResult)
                my_car.set_pose_offset([0.07, 0, 0])      #yuan0.05
                my_car.task.set_food(2, row=3-foodResult)

                return
            else:
                my_car.set_pose_offset([0.41, 0, 0])
                
                
            #add
            my_car.task.arm.set_hand_angle(0)
            time.sleep(0.5)

            for i in range(10):
                current_img = my_car.cap_side.read()
                current_dets = my_car.task_det(current_img)
                result = food_encoder.orderedRec(dets=current_dets)
                print(f"Result: {result}")

            canDo = my_car.ernieInteract('food', timeout=20, a=result[0], b=result[1], c=ingredient_01, d=ingredient_02)
            print(f"Can do: {canDo['result']}")
            foodResult = int(num_process.extract_first_integer(canDo))
            print(f"Food result: {foodResult}")
            
            
            #add
            my_car.task.arm.set_hand_angle(135)

            if foodResult != 3:
                my_car.set_pose_offset([-0.16, 0, 0])     #-0.13
                my_car.task.set_food(1, row=3-foodResult)
                my_car.set_pose_offset([0.07, 0, 0])
                my_car.task.set_food(2, row=3-foodResult)

                return
            else:
                my_car.set_pose_offset([-0.16, 0, 0])
                my_car.task.set_food(1, row=1)
                my_car.set_pose_offset([0.07, 0, 0])
                my_car.task.set_food(2, row=1)

                return
            
            '''
            tar = my_car.task.set_food(arm_set=True)
            my_car.set_pose_offset([-0.1, 0, 0])
            if box == 1 or box == 2:
                my_car.lane_time(0, 1)
                my_car.lane_det_location(0.2, tar, side=-1)
                my_car.set_pose_offset([0.075, 0, 0])
                my_car.task.set_food(1, row=box)
                my_car.set_pose_offset([0.045, 0, 0])
                my_car.task.set_food(2, row=box)
            elif box == 3 or box == 4:
                my_car.lane_time(0, 1)
                my_car.lane_det_location(0.2, tar, side=-1)
                my_car.set_pose_offset([0, 0.05, 0])
                my_car.set_pose_offset([0.3, 0, 0])
                my_car.lane_time(0, 1)
                my_car.lane_det_location(0.2, tar, side=-1)
                my_car.set_pose_offset([-0.15, 0, 0])
                my_car.task.set_food(1, row = box - 2)
                my_car.set_pose_offset([0.045, 0, 0])
                my_car.task.set_food(2, row = box - 2)
            '''


            # my_car.lane_dis_offset(0.3, 0.17)
            # my_car.lane_det_location(0.2, tar, side=1)
            # my_car.task.pick_ingredients(1, 1)
        except:
            return

    def task_help():
        # my_car.task.help_peo()
        my_car.task.arm.switch_side(1)
        my_car.task.arm.set(my_car.task.arm.horiz_mid, 0.08)
        my_car.lane_dis_offset(0.2, 1.5)
        my_car.lane_sensor(0.3, value_h=0.5, sides=1)
        my_car.lane_dis_offset(0.2, 0.22)     #yuan0.2
        my_car.set_pose_offset([0, 0.18, 0])
        # my_car.task.help_peo()
        # my_car.task.arm.set_offset(0.06, 0)
        time.sleep(0.5)
        my_car.task.arm.switch_side(-1)
        my_car.set_pose_offset([0, -0.18, 0])
        my_car.set_pose_offset([-0.3, 0, 0])
        my_car.set_pose_offset([0, 0.4, 0])
        my_car.set_pose_offset([0, -0.4, 0])
        my_car.lane_dis_offset(0.4, 0.6)
        # my_car.set_pose_offset([-0.1, 0.15, 0])
        # my_car.set_pose_offset([0, -0.3, 0])
        # my_car.set_pose_offset([0.7, 0, 0], vel=[0.3, 0.3, 0])

        # my_car.set_pose_offset([0.1, -0.1, 0])

        # my_car.move_advance([0.2, 0, 0], value_h=0.5, sides=1, dis_out=0.05)
        # my_car.task.help_peo()

    def go_start():
        my_car.lane_sensor(0.3, value_l=0.4, sides=-1)
        my_car.set_pose_offset([0.85, 0, 0], 2.8)
        my_car.set_pose_offset([0.45, -0.09, -0.6], 2.5)
        # 前移
        # my_car.set_pose_offset([0.3, 0, 0], 2.5)
        # my_car.set_pose_offset([0.45, -0.09, -0.6], 2.5)
        # 离开道路到修整营地
        # my_car.set_pose_offset([0.15, -0.4, 0], 2)
        # 做任务
        # my_car.do_action_list(actions_map)

    def skip(skip_speed, skip_dis):
        my_car.lane_dis_offset(skip_speed, skip_dis)

    def mystery_task():
        my_car.lane_sensor(0.3, value_h=0.5, sides=1)
        my_car.set_pose_offset([0.23, 0, 0])
        my_car.task.arm.switch_side(1)
        my_car.task.arm.set(my_car.task.arm.horiz_mid, 0.1)
        my_car.task.arm.set(1.8*my_car.task.arm.horiz_mid, 0.1)
        my_car.task.arm.set_hand_angle(0)
        my_car.task.arm.grap(1)
        my_car.task.arm.set(1.8*my_car.task.arm.horiz_mid, 0.08)
        my_car.task.arm.set(1.8*my_car.task.arm.horiz_mid, 0.1)
        # my_car.task.arm.switch_side(-1)
        my_car.task.arm.switch_side(-2)
        my_car.task.arm.set(1.2*my_car.task.arm.horiz_mid, 0)
        my_car.task.arm.grap(0)
        my_car.task.arm.set(1.2*my_car.task.arm.horiz_mid, 0.1)

        my_car.lane_dis_offset(0.3, 0.95)     # orginal 0.9

        return

    def car_move():
        my_car.set_pose([0.20, 0,0], 1)

    my_car.beep()
    time.sleep(0.2)
    # functions = [hanoi_tower_func, bmi_cal, camp_fun, send_fun, task_ingredients, task_answer, task_fun2, task_food, task_help]
    # my_car.manage(functions, 9)
    from myBtn import myButton
    my_button = myButton()
    while True:
        key = my_button.retKey()
        if key != 0:
            break

    print(f"key: {int(key)}")
           
    # Tasks to be executed
    hanoi_tower_func()
    bmi_cal()
    camp_fun()
    mystery_task()
    send_fun()
    ing_1, ing_2 = task_ingredients()
    print(f"{ing_1},{ing_2}")
    skip(0.5, 0.2)
    task_answer(btnIn= int(key) - 1)
    task_fun2()
    task_food(ing_1, ing_2)
    # skip(0.4, 0.5)
    task_help()

    # my_car.task.bmi_set(2)
    
    # my_car.lane_time(0, 20)

    my_car.close()