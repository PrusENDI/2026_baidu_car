"""Common tools with lazy loading for hardware-dependent helpers."""

from importlib import import_module

# 修改前（保留原始代码）：tools 包初始化时会直接导入 collect_control；
# collect_control.py 顶层又导入 whalesbot.vehicle，导致仅导入 log_wrap
# 也打开唯一的车辆串口。
# from .tools_class import *
# from .camera import Camera
# from .streamer import Streamer
# from .log_wrap import logger
# from .collect_control import CollectControlCar

# 这些模块不创建车辆串口，可继续作为常用工具导出；硬件相关的
# CollectControlCar 改为首次访问时加载。
from .tools_class import *
from .camera import Camera
from .streamer import Streamer
from .log_wrap import logger

__all__ = [
    'Camera', 'Streamer', 'logger', 'CollectControlCar',
]


def __getattr__(name):
    if name == 'CollectControlCar':
        module = import_module('.collect_control', __name__)
        value = module.CollectControlCar
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
