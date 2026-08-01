"""回归测试：推理服务导入不能隐式初始化车辆串口。"""

import subprocess
import sys


def test_inference_import_does_not_load_vehicle_driver():
    """只导入推理前端时，车辆驱动和 serial_wrap 不应进入 sys.modules。"""
    code = (
        "import sys; "
        "import smartcar.paddlebaidu.infer_cs.base.infer_front; "
        "assert 'smartcar.whalesbot.vehicle' not in sys.modules; "
        "assert 'smartcar.whalesbot.vehicle.base.serial_wrap' not in sys.modules"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr or result.stdout
