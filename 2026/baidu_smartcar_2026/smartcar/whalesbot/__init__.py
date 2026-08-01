"""Whalesbot components, exported lazily to keep inference imports isolated."""

from importlib import import_module

# 修改前（保留原始代码）：
# from .tools import *
# from .vehicle import *
# 这会在导入 smartcar.whalesbot.tools 时无条件加载 vehicle 和串口驱动。

_TOOL_EXPORTS = ('Camera', 'Streamer', 'logger', 'CountRecord', 'get_yaml', 'IndexWrap', 'PID', 'CollectControlCar')
_VEHICLE_EXPORTS = ('Infrared', 'Motors', 'Motor4', 'AnalogInput', 'Battry', 'Key4Btn', 'BoardKey', 'Beep', 'NixieTube', 'ScreenShow', 'ServoBus', 'ServoPwm', 'BluetoothPad', 'LedLight', 'MotorConvert', 'WheelWrap', 'MotorWrap', 'PoutD', 'StepperWrap', 'ArmController', 'MecanumDriver')
__all__ = list(_TOOL_EXPORTS + _VEHICLE_EXPORTS)


def __getattr__(name):
    if name in _TOOL_EXPORTS:
        module = import_module('.tools', __name__)
    elif name in _VEHICLE_EXPORTS:
        module = import_module('.vehicle', __name__)
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(module, name)
    globals()[name] = value
    return value
