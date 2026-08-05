# --*-- coding: utf-8 --*--
# infer_back_end.py

import zmq
import json
import cv2
import yaml
import numpy as np
from threading import Thread
import time
import os
import sys

# #region debug-point A-E:startup-timing
import urllib.request

def _debug_report(hypothesis_id, location, msg, data=None):
    try:
        meminfo = {}
        with open("/proc/meminfo", encoding="utf-8") as mem_file:
            for line in mem_file:
                key, value = line.split(":", 1)
                meminfo[key] = int(value.strip().split()[0])
        payload = {
            "sessionId": "dual-model-startup",
            "runId": "post-fix",
            "hypothesisId": hypothesis_id,
            "location": location,
            "msg": "[DEBUG] " + msg,
            "data": {
                **(data or {}),
                "pid": os.getpid(),
                "rssKb": int(open(f"/proc/{os.getpid()}/statm").read().split()[1]) * os.sysconf("SC_PAGE_SIZE") // 1024,
                "memAvailableKb": meminfo.get("MemAvailable"),
                "swapFreeKb": meminfo.get("SwapFree"),
            },
            "ts": int(time.time() * 1000),
        }
        urllib.request.urlopen(
            urllib.request.Request(
                "http://127.0.0.1:7777/event",
                data=json.dumps(payload).encode(),
                headers={"Content-Type": "application/json"},
            ),
            timeout=0.2,
        ).read()
    except Exception:
        pass
# #endregion

# 添加上两层目录
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
# 添加项目根目录到Python路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")))

# 导入infer_front中的函数
# #region debug-point D:module-import
_debug_report("D", "infer_back_end.py:imports", "import infer_front started")
from smartcar.paddlebaidu.infer_cs.base.infer_front import get_yaml, get_path_relative
_debug_report("D", "infer_back_end.py:imports", "import infer_front finished")
_debug_report("D", "infer_back_end.py:imports", "import paddle_jetson started")
from smartcar.paddlebaidu.paddle_jetson import YoloeInfer, LaneInfer, OCRReco
_debug_report("D", "infer_back_end.py:imports", "import paddle_jetson finished")
# #endregion
# from smartcar.whalesbot.tools.tools_class import get_yaml

class InferServer:
    def __init__(self):
        # 导入推理客户端的配置
        # configs = ClintInterface.configs
        configs = get_yaml('config_car.yml')['infer_cfg']
        startup_started = time.perf_counter()
        _debug_report("E", "infer_back_end.py:InferServer", "backend initialization started", {"services": [conf["name"] for conf in configs]})
        
        self.flag_infer_initok = False
    
        self.flag_end = False
        # 开启对应的线程和服务
        self.threads_list = []
        self.server_dict = {}
        
        # self.lane_server = self.get_server(5001)
        for conf in configs:
            print(conf)
            # 创建获取zmq服务
            server = self.get_server(conf['port'])
            self.server_dict[conf['name']] = server
            # 创建线程
            # thread_tmp = Thread(target=eval('self.'+conf['name']+'_process'))
            # 带参数线程，此处参数为各种推理模型
            thread_tmp = Thread(target=self.process_demo, args=(conf['name'],))
            # thread_tmp = Thread(target=self.lane_process)
            thread_tmp.daemon = True
            thread_tmp.start()
            # 添加进程
            self.threads_list.append(thread_tmp)
        
        from smartcar.paddlebaidu.paddle_jetson import YoloeInfer, LaneInfer, OCRReco # , HummanAtrr, MotHuman

        InferFactory = {
            "YoloeInfer": YoloeInfer,
            "LaneInfer": LaneInfer,
            "OCRReco": OCRReco,
            # "HummanAtrr": HummanAtrr,
            # "MotHuman": MotHuman
        }
        # 创建推理模型
        self.infer_dict = {}

        for conf in configs:
            model_started = time.perf_counter()
            _debug_report("A", "infer_back_end.py:model-create", "model creation started", {"name": conf["name"], "modelDir": conf.get("model_dir"), "inferType": conf["infer_type"]})
            InferType = InferFactory[conf['infer_type']]
            if InferType == OCRReco :
                if 'det_model_dir'in conf and 'rec_model_dir'  in conf:
                    infer = InferType(conf['det_model_dir'], conf['rec_model_dir'],run_mode= conf['run_mode'])
                else:
                    raise InferType()
            else:
                if 'model_dir' in conf:
                    infer = InferType(conf['model_dir'], run_mode= conf['run_mode'])
                else:
                    infer = InferType(run_mode= conf['run_mode'])
            self.infer_dict[conf['name']] = infer
            _debug_report("A", "infer_back_end.py:model-create", "model creation finished", {"name": conf["name"], "elapsedMs": round((time.perf_counter() - model_started) * 1000, 2)})

        # 创建推理模型
        # self.lane_infer = LaneInfer()
        # self.front_infer = YoloInfer("front_model2") # "trt_fp32")
        # self.task_infer = YoloInfer("task_model3") # "trt_fp32")
        # self.ocr_infer = OCRReco()
        # self.humattr_infer = HummanAtrr()
        # self.mot_infer = MotHuman()
        
        # 新建一个空白图片，用于预先图片推理
        img = np.zeros((240, 240, 3), np.uint8)
        # 预加载推理几张图片，刚开始推理时速度慢，会有卡顿
        for i in range(3):
            for conf in configs:
                warmup_started = time.perf_counter()
                _debug_report("B", "infer_back_end.py:warmup", "warmup started", {"name": conf["name"], "round": i + 1})
                infer_tmp = self.infer_dict[conf['name']]
                infer_tmp(img)
                _debug_report("B", "infer_back_end.py:warmup", "warmup finished", {"name": conf["name"], "round": i + 1, "elapsedMs": round((time.perf_counter() - warmup_started) * 1000, 2)})
        print("infer init ok")

        self.flag_infer_initok = True
        _debug_report("C", "infer_back_end.py:InferServer", "backend initialization finished", {"elapsedMs": round((time.perf_counter() - startup_started) * 1000, 2)})


    def get_server(self, port):
        context = zmq.Context()
        socket = context.socket(zmq.REP)
        socket.bind(f"tcp://127.0.0.1:{port}")
        return socket
    
    def process_demo(self, name):
        
        print(time.strftime("%Y-%m-%d %H:%M:%S"), "{} process start".format(name))
        server:zmq.Socket = self.server_dict[name]
        # lambda定义推理函数，含有归一化处理参数为True, 此处定义方便后续调用
        func = lambda x: self.infer_dict[name](x, True)

        while True:
            if self.flag_end:
                return
            response = server.recv()

            head = response[:5]
            res = []
            if head == b"ATATA":
                if self.flag_infer_initok:
                    res = True
                else:
                    res = False
            elif head == b"image":
                # 把bytes转为jpg格式
                img = cv2.imdecode(np.frombuffer(response[5:], dtype=np.uint8), 1)
                if self.flag_infer_initok:
                    # res = self.lane_infer(img).tolist()
                    # lambda函数
                    res = func(img)
                    
            json_data = json.dumps(res)
            json_data = bytes(json_data, encoding='utf-8')
            server.send(json_data)

    def close(self):
        print("closing...")
        self.flag_end = True
        for thread in self.threads_list:
            # 等待结束
            thread.join()
            # 关闭
            thread.close()

def main():
    print("infer_back_end.py 程序开始运行")
    try:
        infer_back = InferServer()
    except Exception as exc:
        _debug_report("D", "infer_back_end.py:main", "backend initialization failed", {"errorType": type(exc).__name__, "error": str(exc)})
        raise

    while True:
        try:
            time.sleep(1)
        except Exception as e:
            print(e)
            break
    time.sleep(0.1)
    infer_back.close()

if __name__ == "__main__":
    main()
