from importlib import import_module

from . import tools_class as _tools_class
from .tools_class import *
from .camera import Camera
from .streamer import Streamer
from .log_wrap import logger


__all__ = [
    name for name in vars(_tools_class)
    if not name.startswith('_')
] + ['Camera', 'Streamer', 'logger', 'CollectControlCar']


def __getattr__(name):
    if name != 'CollectControlCar':
        raise AttributeError("module {!r} has no attribute {!r}".format(__name__, name))
    value = import_module('.collect_control', __name__).CollectControlCar
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
