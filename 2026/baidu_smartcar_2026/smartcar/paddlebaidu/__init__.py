"""Paddle inference components with lazy, dependency-isolated exports."""

from importlib import import_module

# 修改前（保留原始代码）：
# from .ernie_bot import ErnieBotWrap, HumAttrPrompt, ActionPrompt, ImagePrompt, OrderPrompt
# from .paddle_jetson import YoloeInfer, OCRReco, LaneInfer
# from .infer_cs import ClintInterface, Bbox
# 顶层导入会让任意推理子模块都加载重型组件，增加初始化失败和串口副作用。

_LAZY_EXPORTS = {
    'ErnieBotWrap': ('.ernie_bot', 'ErnieBotWrap'), 'HumAttrPrompt': ('.ernie_bot', 'HumAttrPrompt'),
    'ActionPrompt': ('.ernie_bot', 'ActionPrompt'), 'ImagePrompt': ('.ernie_bot', 'ImagePrompt'),
    'OrderPrompt': ('.ernie_bot', 'OrderPrompt'), 'YoloeInfer': ('.paddle_jetson', 'YoloeInfer'),
    'OCRReco': ('.paddle_jetson', 'OCRReco'), 'LaneInfer': ('.paddle_jetson', 'LaneInfer'),
    'ClintInterface': ('.infer_cs', 'ClintInterface'), 'Bbox': ('.infer_cs', 'Bbox'),
}
__all__ = list(_LAZY_EXPORTS)


def __getattr__(name):
    try:
        module_name, attr_name = _LAZY_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(import_module(module_name, __name__), attr_name)
    globals()[name] = value
    return value
