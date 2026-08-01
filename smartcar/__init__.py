"""Smartcar package public API with lazy imports.

The previous implementation imported ``whalesbot.vehicle`` at package import
time.  That transitively constructed ``SerialWrap`` and opened the only
vehicle serial port even when only the inference client was requested.  The
old imports are preserved below as comments for auditability; exports are now
resolved on first attribute access.
"""

from importlib import import_module

# 修改前（保留原始代码）：这些顶层导入会让 infer_back_end 导入 smartcar 时
# 立即加载 vehicle/SerialWrap，和 car_start 抢占同一条 /dev/ttyUSB* 串口。
# from .whalesbot.tools import Camera, Streamer, logger, CountRecord, get_yaml, IndexWrap, PID, CollectControlCar
# from .whalesbot.vehicle import (ArmController, ScreenShow, Key4Btn, Infrared,
#     LedLight, MecanumDriver, Beep, Motors, Motor4, AnalogInput, Battry,
#     BoardKey, NixieTube, ServoBus, ServoPwm, BluetoothPad, MotorConvert,
#     WheelWrap, MotorWrap, PoutD, StepperWrap)
# from .paddlebaidu.infer_cs import ClintInterface, Bbox
# from .paddlebaidu.ernie_bot import ErnieBotWrap, HumAttrPrompt, ActionPrompt, ImagePrompt

_LAZY_EXPORTS = {
    'Camera': ('.whalesbot.tools', 'Camera'), 'Streamer': ('.whalesbot.tools', 'Streamer'),
    'logger': ('.whalesbot.tools', 'logger'), 'CountRecord': ('.whalesbot.tools', 'CountRecord'),
    'get_yaml': ('.whalesbot.tools', 'get_yaml'), 'IndexWrap': ('.whalesbot.tools', 'IndexWrap'),
    'PID': ('.whalesbot.tools', 'PID'), 'CollectControlCar': ('.whalesbot.tools', 'CollectControlCar'),
    'ArmController': ('.whalesbot.vehicle', 'ArmController'), 'ScreenShow': ('.whalesbot.vehicle', 'ScreenShow'),
    'Key4Btn': ('.whalesbot.vehicle', 'Key4Btn'), 'Infrared': ('.whalesbot.vehicle', 'Infrared'),
    'LedLight': ('.whalesbot.vehicle', 'LedLight'), 'MecanumDriver': ('.whalesbot.vehicle', 'MecanumDriver'),
    'Beep': ('.whalesbot.vehicle', 'Beep'), 'Motors': ('.whalesbot.vehicle', 'Motors'),
    'Motor4': ('.whalesbot.vehicle', 'Motor4'), 'AnalogInput': ('.whalesbot.vehicle', 'AnalogInput'),
    'Battry': ('.whalesbot.vehicle', 'Battry'), 'BoardKey': ('.whalesbot.vehicle', 'BoardKey'),
    'NixieTube': ('.whalesbot.vehicle', 'NixieTube'), 'ServoBus': ('.whalesbot.vehicle', 'ServoBus'),
    'ServoPwm': ('.whalesbot.vehicle', 'ServoPwm'), 'BluetoothPad': ('.whalesbot.vehicle', 'BluetoothPad'),
    'MotorConvert': ('.whalesbot.vehicle', 'MotorConvert'), 'WheelWrap': ('.whalesbot.vehicle', 'WheelWrap'),
    'MotorWrap': ('.whalesbot.vehicle', 'MotorWrap'), 'PoutD': ('.whalesbot.vehicle', 'PoutD'),
    'StepperWrap': ('.whalesbot.vehicle', 'StepperWrap'), 'ClintInterface': ('.paddlebaidu.infer_cs', 'ClintInterface'),
    'Bbox': ('.paddlebaidu.infer_cs', 'Bbox'), 'ErnieBotWrap': ('.paddlebaidu.ernie_bot', 'ErnieBotWrap'),
    'HumAttrPrompt': ('.paddlebaidu.ernie_bot', 'HumAttrPrompt'), 'ActionPrompt': ('.paddlebaidu.ernie_bot', 'ActionPrompt'),
    'ImagePrompt': ('.paddlebaidu.ernie_bot', 'ImagePrompt'),
}
__all__ = list(_LAZY_EXPORTS)


def __getattr__(name):
    """兼容旧 API，同时把车辆和串口初始化推迟到真正使用时。"""
    try:
        module_name, attr_name = _LAZY_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(import_module(module_name, __name__), attr_name)
    globals()[name] = value
    return value
