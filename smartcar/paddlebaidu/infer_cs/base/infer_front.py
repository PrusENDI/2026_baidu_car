#!/usr/bin/python
# -*- coding: utf-8 -*-
import zmq
import cv2
import numpy as np
import json
import subprocess
import psutil
import yaml

import time, os, sys, socket
# 添加上两层目录
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..","..", ".."))) 
from smartcar.whalesbot.tools.log_wrap import logger
# from smartcar.whalesbot.tools.tools_class import get_yaml

def get_yaml(path):
    root_path = get_path_relative("..", "..", "..", "..")
    config_path = os.path.join(root_path, "config_car.yml")
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.load(f, Loader=yaml.FullLoader)
    except Exception as e:
        print('{} not found'.format(config_path))
        print(e)
        return None

def get_path_relative(*args):
    local_dir = os.path.abspath(os.path.dirname(__file__))
    return os.path.join(local_dir, *args)


def get_zmp_client(port):
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    res = socket.connect(f"tcp://127.0.0.1:{port}")
    # print(res)
    return socket

def get_python_processes():
    
    # print("----------")
    python_processes = []
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            if 'python' in proc.info['name'].lower() and len(proc.info['cmdline']) > 1 and len(proc.info['cmdline'][1]) < 100:
                info = [proc.info['pid'], proc.info['cmdline'][1]]
                python_processes.append(info)
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
    return python_processes


def command_contains_script(cmdline, script_name):
    """判断命令行是否真正执行目标脚本，而非仅在字符串中提到它。"""
    if not cmdline:
        return False
    for index, arg in enumerate(cmdline[1:], start=1):
        # python -c "...infer_back_end.py..." 不是后端脚本进程，不能按文件名误认。
        if "-c" in cmdline[1:index]:
            continue
        if os.path.basename(str(arg)) == script_name:
            return True
    return False


def is_process_in_project(project_root, process_cwd):
    """判断进程工作目录是否属于当前项目，避免误认其他 worktree。"""
    if not process_cwd:
        return False
    root = os.path.realpath(os.path.abspath(project_root))
    cwd = os.path.realpath(os.path.abspath(process_cwd))
    try:
        return os.path.commonpath([root, cwd]) == root
    except ValueError:
        return False


def is_port_listening(port, host="127.0.0.1", timeout=0.2):
    """用 TCP connect 检查推理后端端口是否确实可连接。"""
    try:
        with socket.create_connection((host, int(port)), timeout=timeout):
            return True
    except (OSError, ValueError, TypeError):
        return False
    # for process in python_processes:
    #     print(f"PID: {process['pid']}, Name: {process['name']}, Cmdline: {process['cmdline']}")
    # print("    ")

class Bbox:
    def __init__(self, box=None, rect=None, size=[640, 480]) -> None:
        self.size = np.array(size) / 2
        self.size_concat = np.concatenate((self.size, self.size))

        if box is not None:
            box_np = np.array(box)
            # 如果所有值的绝对值都小于1，表示归一化
            # np.abs(box_np, out=box_np)
            if (np.abs(box_np) < 2).all():
                self.box_normalise = box_np
                self.box = self.denormalise(box_np, self.size)
                # print(self.box)
            else:
                self.box = box_np
                self.box_normalise = self.normalise(box_np, self.size)
            self.rect = self.box_to_rect(self.box, self.size)
        elif rect is not None:
            self.rect = np.array(rect)
            self.box = self.rect_to_box(self.rect, self.size)
            self.box_normalise = self.normalise(self.box, self.size)
    
    def get_rect(self):
        return self.rect
    
    def get_box(self):
        return self.box

    @staticmethod
    def normalise(box, size):
        return box / np.concatenate((size, size))
    
    @staticmethod
    # 去归一化
    def denormalise(box_nor, size):
        return (box_nor * np.concatenate((size, size))).astype(np.int32)

    @staticmethod
    def rect_to_box(rect, size):
        pt_tl = rect[:2]
        pt_br = rect[2:]
        pt_center = (pt_tl + pt_br) / 2 - size
        box_wd = pt_br - pt_tl
        return np.concatenate((pt_center, box_wd)).astype(np.int32)

    @staticmethod
    def box_to_rect(box, size):
        pt_center = box[:2]
        box_wd = box[2:]
        pt_tl = (size + pt_center - box_wd / 2).astype(np.int32)
        pt_br = (size + pt_center + box_wd / 2).astype(np.int32)
        # print(pt_tl, pt_br)
        rect = np.concatenate((pt_tl, pt_br))
        # 限制最大最小值
        max_size = np.concatenate((size, size))*2
        # print(max_size)
        np.clip(rect, 0, max_size, out=rect)
        return rect

class ClintInterface:
    # configs = [
    #         {'name':'lane', 'infer_type': 'LaneInfer', 'params': [], 'port':5001, 'img_size':[128, 128]},
    #         {'name':'task', 'infer_type': 'YoloeInfer', 'params': ['task_model3'], 'port':5002, 'img_size':[416, 416]},
    #         {'name':'front', 'infer_type':'YoloeInfer', 'params': ['front_model2'], 'port':5003, 'img_size':[416, 416]},
    #         {'name':'ocr', 'infer_type':'OCRReco', 'params': [], 'port':5004,'img_size':None},
    #         {'name':'humattr', 'infer_type':'HummanAtrr', 'params': [], 'port':5005, 'img_size':None},
    #         {'name':'mot', 'infer_type':'MotHuman', 'params': [], 'port':5006, 'img_size':None}
    #         ]
    
    def __init__(self, name):
        root_cfg = get_yaml('config_car.yml')
        backend_cfg = root_cfg.get('infer_backend', {})
        self.backend_mode = os.environ.get(
            'SMARTCAR_INFER_BACKEND', backend_cfg.get('mode', 'worktree')
        ).strip()
        if self.backend_mode == 'worktree':
            self.configs = root_cfg['infer_cfg']
            self.auto_start_backend = True
            self.connect_timeout = None
        elif self.backend_mode == 'external_7_17':
            external_cfg = backend_cfg.get('external_7_17', {})
            self.configs = external_cfg.get('services', [])
            self.auto_start_backend = False
            try:
                self.connect_timeout = float(
                    external_cfg.get('connect_timeout', 15.0)
                )
                request_timeout = float(
                    external_cfg.get('request_timeout', 15.0)
                )
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    'external_7_17 的 connect_timeout/request_timeout 必须是数字'
                ) from exc
            if self.connect_timeout <= 0 or request_timeout <= 0:
                raise ValueError(
                    'external_7_17 的 connect_timeout/request_timeout 必须大于 0'
                )
            self.request_timeout_ms = int(request_timeout * 1000)
        else:
            raise ValueError(
                '未知推理后端模式: '
                f'{self.backend_mode!r}; 只能是 worktree 或 external_7_17'
            )

        logger.info("{}连接服务器...".format(name))
        model_cfg = self.get_config(name)
        if model_cfg is None:
            raise ValueError(
                f'推理服务未配置 mode={self.backend_mode}, name={name!r}'
            )
        self.img_size = model_cfg['img_size']
        self.infer_port = model_cfg['port']
        self.client_timeout_ms = (
            1000 if self.backend_mode == 'external_7_17' else None
        )
        self.client = self.get_zmp_client(
            self.infer_port, timeout_ms=self.client_timeout_ms
        )
        
        infer_back_end_file = "infer_back_end.py"
        # 检查后台程序是否运行, 如果未开启, 则开启
        if self.auto_start_backend:
            self.check_back_python(infer_back_end_file, port=self.infer_port)
        else:
            logger.info(
                f'使用外部 7_17 推理后端 name={name}, port={self.infer_port}; '
                '不会扫描或启动本工作树后端'
            )

        flag = False
        deadline = (
            None
            if self.connect_timeout is None
            else time.monotonic() + self.connect_timeout
        )
        last_error = None
        while True:
            try:
                if self.get_state():
                    if flag:
                        logger.info("")
                    break
                last_error = None
            except zmq.ZMQError as exc:
                last_error = exc
                if not self.auto_start_backend:
                    self.client.close(linger=0)
                    self.client = self.get_zmp_client(
                        self.infer_port, timeout_ms=self.client_timeout_ms
                    )
                else:
                    raise
            if deadline is not None and time.monotonic() >= deadline:
                self.client.close(linger=0)
                detail = (
                    f'{type(last_error).__name__}: {last_error}'
                    if last_error is not None
                    else '后端尚未完成模型初始化'
                )
                raise RuntimeError(
                    '无法连接外部 7_17 推理后端，请先从 7_17 目录启动 '
                    f'infer_back_end.py: name={name}, port={self.infer_port}, '
                    f'timeout={self.connect_timeout:.1f}s, detail={detail}'
                )
            # 输出一个提示信息，不换行
            print('.', end='', flush=True)
            # logger.info(".")
            time.sleep(1)
            flag = True
        if not self.auto_start_backend:
            # 就绪探测使用短超时；正式图片推理允许使用独立的较长超时。
            self.client.close(linger=0)
            self.client_timeout_ms = self.request_timeout_ms
            self.client = self.get_zmp_client(
                self.infer_port, timeout_ms=self.client_timeout_ms
            )
        # print(self.client)
        # print("连接服务器成功")
        logger.info("{}连接服务器成功".format(name))
    
    def check_back_python(self, file_name):
        dir_file = os.path.abspath(os.path.dirname(__file__))
        file_path = os.path.join(dir_file, file_name)
        # print(file_path)
        if not os.path.exists(file_path):
            raise Exception("后台脚本文件不存在")
        # 获取正在运行的python脚本
        py_lists = get_python_processes()
        for py_iter in py_lists:
            # 检测是否存在后台运行的脚本
            print(py_iter)
            if file_name in py_iter[1]:
                print(f"{file_name}运行{py_iter}")
                return
        else:
            # 开启后台脚本，后台运行, 忽略输入输出
            # 使用subprocess调用脚本
            logger.info("开启{}脚本, 后台运行中, 请等待".format(file_name))
            cmd_str = 'python3 ' + file_path + ' &'
            # shell=True告诉subprocess模块在运行命令时使用系统的默认shell。这使得可以像在命令行中一样执行命令，包括使用通配符和其他shell特性
            subprocess.Popen(cmd_str, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            # print("已启动")
            time.sleep(1)
            # 这里的> /dev/null 2>&1将标准输出和标准错误都重定向到/dev/null，实现与之前subprocess.Popen相同的效果
            # os.system(cmd_str + " > /dev/null 2>&1")
        

    # 修改后的实现放在原方法之后，保留上方旧实现作为审计记录；类定义中后出现的
    # 同名方法会覆盖旧实现，但不会删除现场曾使用过的代码。
    def check_back_python(self, file_name, port=None):
        """按当前项目目录和端口确认后端，必要时启动并输出诊断信息。"""
        dir_file = os.path.abspath(os.path.dirname(__file__))
        file_path = os.path.join(dir_file, file_name)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"推理后端脚本不存在: {file_path}")

        project_root = os.path.abspath(os.path.join(dir_file, "..", "..", "..", ".."))
        print(f"[infer-backend] project_root={project_root}")
        print(f"[infer-backend] script={file_path}")
        print(f"[infer-backend] required_port={port}")

        same_project_process = False
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                cmdline = proc.info.get('cmdline') or []
                if not command_contains_script(cmdline, file_name):
                    continue
                process_cwd = proc.cwd()
                in_project = is_process_in_project(project_root, process_cwd)
                print(
                    f"[infer-backend] candidate pid={proc.pid} cwd={process_cwd!r} "
                    f"same_project={in_project} cmdline={cmdline!r}"
                )
                if not in_project:
                    print(
                        f"[infer-backend][WARN] 忽略其他项目目录的 {file_name}: "
                        f"pid={proc.pid}, cwd={process_cwd!r}"
                    )
                    continue
                same_project_process = True
                if port is None or is_port_listening(port):
                    print(f"[infer-backend] 当前项目后端已运行: pid={proc.pid}, port={port}")
                    return True
                print(
                    f"[infer-backend][WARN] 当前项目进程存在但端口 {port} 尚未监听，"
                    "不重复启动；继续等待其完成初始化。"
                )
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess) as exc:
                print(f"[infer-backend][WARN] 读取候选进程失败: {exc}")

        if same_project_process:
            return False

        print(
            f"[infer-backend] 未找到当前项目中可用的 {file_name}，"
            "准备启动；其他目录同名进程不会复用。"
        )
        try:
            # 修改前使用 shell=True 且将 stdout/stderr 丢到 DEVNULL，会隐藏后端 traceback。
            process = subprocess.Popen(
                [sys.executable, file_path],
                cwd=project_root,
                stdout=None,
                stderr=None,
                start_new_session=True,
            )
        except Exception as exc:
            print(f"[infer-backend][ERROR] 启动 {file_name} 失败: {exc!r}")
            raise

        print(f"[infer-backend] 已启动 pid={process.pid}, cwd={project_root}")
        time.sleep(1)
        if port is not None and not is_port_listening(port):
            print(
                f"[infer-backend][ERROR] 后端进程已启动但端口 {port} 尚未监听；"
                "请检查后端 traceback、模型加载和串口占用。"
            )
            return False
        if port is not None:
            print(f"[infer-backend] 端口 {port} 已开始监听")
        return True

    def get_config(self, name):
        for conf in self.configs:
            if conf['name'] == name:
                return conf
            
    @staticmethod
    def get_zmp_client(port, timeout_ms=None):
        context = zmq.Context()
        socket = context.socket(zmq.REQ)
        socket.setsockopt(zmq.LINGER, 0)
        if timeout_ms is not None:
            socket.setsockopt(zmq.SNDTIMEO, int(timeout_ms))
            socket.setsockopt(zmq.RCVTIMEO, int(timeout_ms))
        socket.connect(f"tcp://127.0.0.1:{port}")
        return socket

    def __call__(self, *args, **kwds):
        return self.get_infer(*args, **kwds)

    def get_state(self):
        data = bytes('ATATA', encoding='utf-8')
        self.client.send(data)
        response = self.client.recv()
        response = json.loads(response)
        return response

    def get_infer(self, img):
        if self.img_size is not None:
            img = cv2.resize(img, self.img_size)
        img = cv2.imencode('.jpg', img)[1].tobytes()
        data = bytes('image', encoding='utf-8') + img
        try:
            self.client.send(data)
            response = self.client.recv()
        except zmq.ZMQError as exc:
            if not self.auto_start_backend:
                # REQ 套接字超时后不能直接发送下一条请求，必须重新连接。
                self.client.close(linger=0)
                self.client = self.get_zmp_client(
                    self.infer_port, timeout_ms=self.client_timeout_ms
                )
                raise RuntimeError(
                    '外部 7_17 推理请求失败，连接已重置: '
                    f'port={self.infer_port}, error={exc}'
                ) from exc
            raise
        response = json.loads(response)
        return response

def main_client():
    from camera import Camera
    cap = Camera(1, 640, 480)
    # cap.set_size(640, 480)
    # cap.start_back_thread()
    # infer_client = ClintInterface('lane')
    infer_client = ClintInterface("ocr")
    # infer_client = ClintInterface('task')
    # infer_client = ClintInterface('mot')
    # infer_client = ClintInterface('front')
    # infer_client = infer_clint
    # while True:
    #     print(infer_client.get_state())
    #     time.sleep(1)
    # infer_client = TaskDetectClient()
    last_time = time.time()
    while True:
        img = cap.read()
        # img = cv2.resize(img, (128, 128))
        dets_ret = infer_client.get_infer(img[300:, 200:460])
        # dets_ret = infer_client.get_infer(img)
        # dets_ret = infer_client.get_infer(img)
        print(dets_ret)
        # for det in dets_ret:
        #     cls_id, obj_id, label, score, bbox = det[0], det[1], det[2], det[3], det[4:]
        #     rect = Bbox(box=bbox, size=[640, 480]).get_rect()
        #     print(rect)
        #     cv2.rectangle(img, rect[0:2], rect[2:4], (255, 0, 0), 2)
        
        # response = task_det_client.get_infer(img)
        cv2.imshow("img", img)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        # print(response)
        fps = 1 / (time.time() - last_time)
        last_time = time.time()
        print("fps:", fps)
    cap.close()
    cv2.destroyAllWindows()

def stop_process(py_str):
    py_lists = get_python_processes()
    print(py_lists)
    for py_procees in py_lists:
        pid_id, py_name = py_procees[0], py_procees[1]
        # print(pid_id, py_name)
        if py_str in py_procees[1]:
            psutil.Process(pid_id).terminate()
            print("stop", py_name)
            return


if __name__ == '__main__':
    import argparse
    args = argparse.ArgumentParser()
    args.add_argument('--op', type=str, default="infer")
    args = args.parse_args()
    print(args)
    if args.op == "infer":
        main_client()
    if args.op == "stop":
        stop_process("infer_back_end.py")
