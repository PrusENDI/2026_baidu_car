"""Smartcar包的主模块

该模块提供了智能车相关的核心功能，包括：
- 摄像头控制
- 机械臂控制
- 车辆驱动
- 目标检测
- 自然语言处理
"""

from importlib import import_module


__all__ = [
    # 摄像头和工具
    'Camera', 'Streamer', 'logger', 'CountRecord', 'get_yaml', 'IndexWrap', 'PID','CollectControlCar',
    # 车辆控制
    'ArmController', 'ScreenShow', 'Key4Btn', 'Infrared', 'LedLight', 'MecanumDriver', 'Beep',
    'Motors', 'Motor4', 'AnalogInput', 'Battry', 'BoardKey', 'NixieTube', 'ServoBus',
    'ServoPwm', 'BluetoothPad', 'MotorConvert', 'WheelWrap', 'MotorWrap', 'PoutD', 'StepperWrap',
    # 目标检测和NLP
    'ClintInterface', 'Bbox', 'ErnieBotWrap', 'HumAttrPrompt', 'ActionPrompt', 'ImagePrompt'
]

_TOOL_EXPORTS = {
    'Camera', 'Streamer', 'logger', 'CountRecord', 'get_yaml', 'IndexWrap', 'PID',
    'CollectControlCar',
}
_VEHICLE_EXPORTS = {
    'ArmController', 'ScreenShow', 'Key4Btn', 'Infrared', 'LedLight', 'MecanumDriver', 'Beep',
    'Motors', 'Motor4', 'AnalogInput', 'Battry', 'BoardKey', 'NixieTube', 'ServoBus',
    'ServoPwm', 'BluetoothPad', 'MotorConvert', 'WheelWrap', 'MotorWrap', 'PoutD', 'StepperWrap',
}
_INFER_EXPORTS = {'ClintInterface', 'Bbox'}
_ERNIE_EXPORTS = {'ErnieBotWrap', 'HumAttrPrompt', 'ActionPrompt', 'ImagePrompt'}
_EXPORTS = {
    **{name: '.whalesbot.tools' for name in _TOOL_EXPORTS},
    **{name: '.whalesbot.vehicle' for name in _VEHICLE_EXPORTS},
    **{name: '.paddlebaidu.infer_cs' for name in _INFER_EXPORTS},
    **{name: '.paddlebaidu.ernie_bot' for name in _ERNIE_EXPORTS},
}


def __getattr__(name):
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError("module {!r} has no attribute {!r}".format(__name__, name))
    value = getattr(import_module(module_name, __name__), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
