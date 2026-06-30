#!/usr/bin/python3
# -*- coding: utf-8 -*-

import threading
from multiprocessing import Process
import time

import cv2
import platform
import os, sys
import glob


from .log_wrap import logger


def _project_camera_sort_key(path):
    name = os.path.basename(path)
    try:
        return int(name.replace("cam", ""))
    except ValueError:
        return 9999


def list_project_camera_paths():
    return sorted(
        [
            path
            for path in glob.glob("/dev/cam*")
            if os.path.basename(path)[3:].isdigit()
        ],
        key=_project_camera_sort_key,
    )


def resolve_camera_path(index):
    requested = "/dev/cam" + str(index)
    if os.path.exists(requested):
        return requested

    try:
        number = int(index)
    except (TypeError, ValueError):
        return requested

    cameras = list_project_camera_paths()
    if number >= 1 and len(cameras) >= number:
        fallback = cameras[number - 1]
        logger.info("摄像头%s不存在，自动使用%s" % (requested, fallback))
        return fallback
    return requested

class Camera:
    def __init__(self, index=1, width=640, height=480):
        # if src ==0:
        #     self.src = "/dev/video0"
        # elif src == 1:
        #     self.src = "/dev/video1"

        self.width = width
        self.height = height
        self.index = index
        
        # self.src =src
        self.cap = None
        self.frame = None
        # 暂停标志
        self.pause_flag = False
        self.stop_flag = False

        self.init()
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        # thread是否运行标志
        self.flag_thread = False
        self.start_back_thread()
        # self.start()

    def init(self):
        while True:
            try:
                if 'Windows' in platform.platform():
                    self.src = self.index
                    self.cap = cv2.VideoCapture(self.src, cv2.CAP_DSHOW)
                else:
                    self.src = resolve_camera_path(self.index)
                    # 如果self.src不存在，则报错
                    if os.path.exists(self.src) == False:
                        logger.error("摄像头{}不存在".format(self.src))
                        time.sleep(1)
                        continue
                    self.cap = cv2.VideoCapture(self.src)
                break
            except Exception as e:
                # print(e)
                logger.error("init:摄像头打开错误!")
                self.cap.release()
                # self.video_detect()
    
    def start_back_thread(self):
        # 如果未开启线程，开启线程
        if not self.flag_thread:
            self.cap_thread = threading.Thread(target=self.update, args=())
            self.cap_thread.daemon = True
            self.cap_thread.start()
            self.flag_thread = True
        time.sleep(0.5)
            
    def update(self):
        while True:
            if self.stop_flag:
                break
            if self.pause_flag:
                continue
            try:
                ret, frame = self.cap.read()
                if ret:
                    self.frame = frame
                else:
                    logger.error("read:读取图像错误!!!!")
                    self.cap.release()
                    self.init()
                    self.set_size(self.width, self.height)
                    time.sleep(1)
            except Exception as e:
                # print(e)
                logger.error("exception:摄像头错误!!")
                self.cap.release()
                self.init()
                self.set_size(self.width, self.height)

    def set_size(self, width, height):
        self.width = width
        self.height = height
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

    def read(self):
        while self.frame is None:
            time.sleep(0.1)
        return self.frame

    def close(self):
        self.stop_flag = True
        # 等待进程结束
        self.cap_thread.join()
        logger.info("{} close".format(self.src))
        self.cap.release()


def main():
    camera = Camera(1, 640, 480)
    # logger.info("camera test")
    # start_time = time.time()
    while True:
        try:
            img = camera.read()
            # print(img.shape)
            cv2.imshow("img", img)
            key = cv2.waitKey(1)
            # cost_time = time.time() - start_time
            # start_time = time.time()
            # print("fps:", 1 / cost_time)
            if key == ord('q'):
                time.sleep(0.1)
                break
        except Exception as e:
            logger.error(e)
    camera.close()
    logger.info("over")
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
