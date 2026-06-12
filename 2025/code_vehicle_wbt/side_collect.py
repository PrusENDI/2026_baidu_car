import cv2
import threading
import time
import json
import subprocess
import os, sys
# 添加上两层目录
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))) 


from vehicle import CarBase, BluetoothPad, ScreenShow, Beep, ArmBase
from camera import Camera
from log_info import logger

'''
这个脚本用于收集侧面摄像头的采集
采集到的数据被保存在mySideCollection下
'''

class RemoteControlCar:
    def __init__(self, cap: Camera = None) -> None:
        # 获取当前存储目录
        path_dir = os.path.abspath(os.path.dirname(__file__))
        # 获取模型存储目录
        self.dir = os.path.join(path_dir, "mySideCollection")
        if not os.path.exists(self.dir):
            os.mkdir(self.dir)

        self.index = 0
        if cap is None:
            # 使用侧边摄像头 (camera index 2)
            self.cap = Camera(2, 640, 480)
        else:
            self.cap = cap

        self.car = CarBase()
        
        # 初始化机械臂
        self.arm = ArmBase()
        self.arm.reset()  # 机械臂复位
        
        # 定义机械臂高度选项
        self.arm_heights = [0.05] # 在这里设置采集时的机械臂高度，采集完整食材视角是0.05
        self.current_height_index = 0
        
        self.rings = Beep()
        self.beep()
        self.display = ScreenShow()

        self.blue_pad = BluetoothPad()

        self.state_base = [0.15, 0.15, 0.3]
        self.state_start = [0.3, 0.3, 0.5]
        self.speed_x = 0.15     # m/s
        self.speed_y = 0.15     # m/s
        self.ration_omage = 0.5
        self.car_state = [0.0, 0.0, 0.0]
        self.run_flag = False
        self.exit_flag = False
        self.json_data = []
        
        logger.info("remote control start!!")
        self.img_thread = threading.Thread(target=self.image_process, args=())
        self.img_thread.daemon = True
        self.img_thread.start()
        
        # 设置初始机械臂高度
        self.set_arm_height(0)
        
        self.car_process()
        
    def beep(self):
        self.rings.rings()
    
    def set_arm_height(self, height_index):
        """设置机械臂高度"""
        if 0 <= height_index < len(self.arm_heights):
            height = self.arm_heights[height_index]
            self.arm.set(self.arm.horiz_mid, height)
            self.arm.switch_side(1)
            self.current_height_index = height_index
            logger.info(f"Arm height set to: {height}m (index: {height_index})")
 
    def car_process(self):
        pad_exit_flag = False
        while not self.exit_flag:
            keys_val = self.blue_pad.read()
            # print(keys_val)
            if keys_val == [-1, -1, -1, -1, 0]:
                pad_exit_flag = False
                self.car_state = [0.0, 0.0, 0.0]
                logger.error("no bluepad")
                self.display.show("no bluepad\n")
                self.beep()
                time.sleep(1)
                
                continue
            else:
                if not pad_exit_flag:
                    height_info = f"Height: {self.arm_heights[self.current_height_index]}m"
                    self.display.show(f"press btn control\n 3 pressing record\n 4+2 stop\n 4+v del 30pic\n 4+o del all\n 1/2 arm height\n{height_info}\n")
                pad_exit_flag = True
            
            # 记录控制
            if (keys_val[4] & 1024) != 0:
                self.run_flag = True
            else:
                self.run_flag = False
                
            # 退出控制
            if keys_val[4] == 34816:
                self.close()
            # 删除最近图片
            elif keys_val[4] == 2052:
                self.del_last3s()
            # 删除所有图片
            elif keys_val[4] == 2304:
                self.restart()
            # 机械臂高度控制 - 按键1增加高度
            elif keys_val[4] == 1:
                next_index = (self.current_height_index + 1) % len(self.arm_heights)
                self.set_arm_height(next_index)
                height_info = f"Height: {self.arm_heights[self.current_height_index]}m"
                self.display.show(f"press btn control\n 3 pressing record\n 4+2 stop\n 4+v del 30pic\n 4+o del all\n 1/2 arm height\n{height_info}\n")
                self.beep()
            # 机械臂高度控制 - 按键2减少高度
            elif keys_val[4] == 2:
                next_index = (self.current_height_index - 1) % len(self.arm_heights)
                self.set_arm_height(next_index)
                height_info = f"Height: {self.arm_heights[self.current_height_index]}m"
                self.display.show(f"press btn control\n 3 pressing record\n 4+2 stop\n 4+v del 30pic\n 4+o del all\n 1/2 arm height\n{height_info}\n")
                self.beep()
                
            if self.run_flag:
                self.car_state[0] = self.state_base[0]
                self.car_state[1] = -1 * self.state_base[1] * keys_val[0]
                self.car_state[2] = -3.14 * self.state_base[2] * keys_val[2]
            else:
                self.car_state[0] = self.state_start[0] * keys_val[1]
                self.car_state[1] = -1 * self.state_start[1] * keys_val[0]
                self.car_state[2] = -3.14 * self.state_start[2] * keys_val[2]
            # print(*self.car_state)
            self.car.set_velocity(*self.car_state)
            time.sleep(0.05)

    def image_process(self):
        name_length = 4
        
        if os.path.exists(self.dir) is not True:
            os.mkdir(self.dir)
        json_name = "data.json"
        self.json_path = os.path.join(self.dir, json_name)
        
        while not self.exit_flag:
            if self.run_flag:
                data_dict = dict()
                # 获取图片
                image = self.cap.read()

                img_name = (name_length - len(str(self.index)))*'0' + str(self.index) +'.jpg'
                data_dict["img_path"] = img_name
                image_path = os.path.join(self.dir, img_name)
                cv2.imwrite(image_path, image)
                # 只记录车辆x,y速度，移除方位角
                data_dict["state"] = self.car_state[:2].copy()
                # 记录当前机械臂高度
                data_dict["arm_height"] = self.arm_heights[self.current_height_index]
                data_dict["arm_height_index"] = self.current_height_index
                
                self.json_data.append(data_dict)
                # print(img_name)
                logger.info("image:{} height:{}m".format(img_name, self.arm_heights[self.current_height_index]))
                self.index += 1
                if self.index%10 == 0:
                    self.save_json(self.json_data, self.json_path)
                if self.index%20 == 0:
                    height_info = f"Height: {self.arm_heights[self.current_height_index]}m"
                    self.display.show("image:{}\n{}\n".format(self.index, height_info))
                time.sleep(0.05)
            

    def del_last3s(self):
        self.beep()
        for i in range(30):
            # 尝试使用 pop() 方法移除最后一个元素
            try:
                data = self.json_data.pop()
                path = os.path.join(self.dir, data['img_path'])
                os.remove(path)
                self.index -= 1
                
            except IndexError:
                # 输出: 列表为空，无法使用 pop() 方法
                logger.info("image data zero now")
                return
        height_info = f"Height: {self.arm_heights[self.current_height_index]}m"
        self.display.show("image:{}\n{}\n".format(self.index, height_info))

    # 写一个网页，具有遥控控件的界面
    def controller_html(self):
        # 具体实现
        pass


    def restart(self):
        self.beep()
        time.sleep(0.4)
        self.beep()
        
        # 使用rm命令删除*.jpg
        # subprocess.run(["rm", "-rf", self.dir + "/*.jpg"])
        # 删除文件
        subprocess.run(["find", self.dir, "-name", "*.jpg", "-delete"])
        self.json_data = []
        self.index = 0
        height_info = f"Height: {self.arm_heights[self.current_height_index]}m"
        self.display.show("image:{}\n{}\n".format(self.index, height_info))

    @staticmethod
    def save_json(json_data, path):
        with open(path, 'w') as fp:
            json.dump(json_data, fp)

    def get_state(self):
        return self.car_state

    def close(self):
        self.save_json(self.json_data, self.json_path)
        height_info = f"Height: {self.arm_heights[self.current_height_index]}m"
        self.display.show("control end!\nimage:{}\n{}\n".format(self.index, height_info))
        self.exit_flag = True
        self.img_thread.join()
        self.cap.close()

        for i in range(3):
            self.beep()
            time.sleep(0.4)


if __name__ == "__main__":
    remote_car = RemoteControlCar()