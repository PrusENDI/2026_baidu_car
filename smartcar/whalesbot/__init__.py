from importlib import import_module

from . import tools as _tools


_VEHICLE_EXPORTS = [
    'Infrared', 'Motors', 'Motor4', 'AnalogInput', 'Battry', 'Key4Btn', 'StepperWrap',
    'BoardKey', 'Beep', 'NixieTube', 'ScreenShow', 'ServoBus', 'ServoPwm', 'PoutD',
    'BluetoothPad', 'LedLight', 'MotorConvert', 'WheelWrap', 'MotorWrap',
    'ArmController', 'MecanumDriver',
]
__all__ = list(_tools.__all__) + _VEHICLE_EXPORTS
_EXPORTS = {
    **{name: '.tools' for name in _tools.__all__},
    **{name: '.vehicle' for name in _VEHICLE_EXPORTS},
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
