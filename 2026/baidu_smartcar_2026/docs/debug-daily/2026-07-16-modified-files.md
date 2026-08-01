# 2026-07-16 修改文件内容归档

本归档保存今天已提交的实现文件在当前 `HEAD` 的完整内容，便于离线审阅。范围为当天提交所涉及的 `.gitignore`、录制设计、实现与单元测试；不包含拉取的运行日志或本日报自身，避免重复归档。

## `.gitignore`

```text
# Ignore the access_token.yaml file
**/access_token.yaml
.gitee

# Ignore
**/dataset/
**/test.py
logs/orin-status/



# Byte-compiled / optimized / DLL files
**/__pycache__/
*.py[cod]
*$py.class
.vscode/
# C extensions
*.so

# Distribution / packaging
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
share/python-wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# PyInstaller
#  Usually these files are written by a python script from a template
#  before PyInstaller builds the exe, so as to inject date/other infos into it.
*.manifest
*.spec

# Installer logs
pip-log.txt
pip-delete-this-directory.txt

# Unit test / coverage reports
htmlcov/
.tox/
.nox/
.coverage
.coverage.*
.cache
nosetests.xml
coverage.xml
*.cover
*.py,cover
.hypothesis/
.pytest_cache/
cover/

# Translations
*.mo
*.pot

# Django stuff:
*.log
local_settings.py
db.sqlite3
db.sqlite3-journal

# Flask stuff:
instance/
.webassets-cache

# Scrapy stuff:
.scrapy

# Sphinx documentation
docs/_build/

# PyBuilder
.pybuilder/
target/

# Jupyter Notebook
.ipynb_checkpoints

# IPython
profile_default/
ipython_config.py

# pyenv
#   For a library or package, you might want to ignore these files since the code is
#   intended to run in multiple environments; otherwise, check them in:
# .python-version

# pipenv
#   According to pypa/pipenv#598, it is recommended to include Pipfile.lock in version control.
#   However, in case of collaboration, if having platform-specific dependencies or dependencies
#   having no cross-platform support, pipenv may install dependencies that don't work, or not
#   install all needed dependencies.
#Pipfile.lock

# PEP 582; used by e.g. github.com/David-OConnor/pyflow
__pypackages__/

# Celery stuff
celerybeat-schedule
celerybeat.pid

# SageMath parsed files
*.sage.py

# Environments
.env
.venv
env/
venv/
ENV/
env.bak/
venv.bak/

# Spyder project settings
.spyderproject
.spyproject

# Rope project settings
.ropeproject

# mkdocs documentation
/site

# mypy
.mypy_cache/
.dmypy.json
dmypy.json

# Pyre type checker
.pyre/

# pytype static type analyzer
.pytype/

# Cython debug symbols
cython_debug/
```

## `docs/superpowers/specs/2026-07-16-lane-test-telemetry-design.md`

```markdown
# 循迹实车测试遥测与复盘设计

## 目标

为 `tools/safe_lane_test.py` 增加**可选**的测试产物记录能力，使每一次实车循迹测试都能保存可复盘的车载标注视频、逐帧 CSV 和运行元数据。该能力用于基于证据诊断稳定左压线和锐角不能通过的问题；本次不自动修改控制参数、模型权重、官方入口或车辆配置。

## 范围与约束

- Windows 本地工作区是唯一编辑端；Orin 仅同步、运行和产生测试数据。
- 仅修改独立工具及其测试，不修改 `car_start_2026.py`、`car_wrap_2026.py`、`config_car.yml` 或模型权重。
- 未指定记录参数时，现有安全测试的行为保持不变。
- 现有安全边界保持有效：速度不高于 0.10、时长不超过 300 秒、首帧和循环推理超时、无效输出拒绝、异常/Ctrl+C/到时均停车。
- 实车运行由用户执行；Codex 仅同步工具、读取明确指定的产物目录并分析。

## 命令行接口

`safe_lane_test.py` 新增可选参数：

```text
--record-dir <path>
```

指定该参数时，脚本在此目录下创建本次测试的唯一时间戳子目录。未指定时，不创建录像、CSV 或元数据文件。记录目录计划位于 Orin 运行副本的：

```text
logs/lane-tests/<UTC 时间戳>/
```

用户使用示例：

```bash
python3 -u tools/safe_lane_test.py \
  --speed 0.05 --duration 10 --startup-timeout 30 \
  --record-dir logs/lane-tests
```

以上命令仍须在每次实车运行前向用户明确说明速度、时长与停车保护。

## 结构与数据流

记录器是 `safe_lane_test.py` 内的独立、小型组件，不使用线程、网络或额外推理进程。它接收同一控制循环已得到的数据，不改变模型输出或 PID/底盘命令。

```text
cam1 frame -> lane inference -> y/angle PID -> set_velocity
     |              |                |
     +--------------+----------------+
                    |
                    v
               TestRecorder
              /      |       \
       annotated.mp4 frames.csv metadata.json
```

记录器应在主循环首次向底盘下发运动命令前完成初始化。初始化失败时，记录具体错误、退出测试，并沿用 `finally` 的停车与资源清理流程。每一帧先完成模型验证和 PID 计算，再同时写 CSV 和标注帧；最后才调用 `set_velocity`。这样每条记录准确表示随后下发的命令。

## 产物格式

### 标注视频

`annotated.mp4` 使用 `cam1` 原始 320×240 BGR 图像，叠加：

- 运行时长与帧号；
- 模型横向误差 `error_y` 和角度误差 `error_angle`；
- PID 输出 `y_cmd` 和 `yaw_cmd`；
- 下发速度 `vx`、`vy`、`yaw`；
- 瞬时循环帧率。

视频不是控制输入，写入失败不会允许车辆继续在没有可审计记录的状态下运行。

### 逐帧 CSV

`frames.csv` 采用 UTF-8、首行为固定字段名，一行对应一帧成功验证、即将下发的控制数据。字段为：

```text
elapsed_s,frame_index,infer_ms,loop_fps,error_y,error_angle,
y_pid,yaw_pid,vx,vy,yaw,video_written
```

`elapsed_s` 使用单调时钟相对本次循环起点的秒数；`infer_ms` 覆盖 JPEG 编码、ZMQ 请求和模型响应的总耗时；`loop_fps` 是相邻循环的瞬时帧率；`video_written` 指示该帧是否已写入视频。

### 元数据 JSON

`metadata.json` 包含：格式版本、开始和结束 UTC 时间、程序参数、模型服务路径、PID 参数、视频尺寸/编码、帧数、已写视频帧数、最终停止原因和文件名。它不包含密钥、SSH 地址或其他敏感网络信息。

## 故障处理与资源清理

- 录像/CSV/元数据初始化失败：在创建 `Driver` 或下发任何运动命令前退出；最终输出 `SAFE_STOP`。
- 单帧记录写入失败：将停止原因设为明确的记录错误，退出控制循环；`finally` 先调用 `car.stop()`，再关闭视频、CSV、相机、流媒体、ZMQ 和推理子进程。
- 既有推理、相机、输出校验和 Ctrl+C 错误路径不改变，均以停车优先。
- `metadata.json` 在 `finally` 中写入最终停止原因；即使视频编码器关闭失败，也应尽力写入可诊断的 JSON。

## 分析闭环

测试后只拉取本次 `logs/lane-tests/<时间戳>/` 到本地。分析时将 CSV、标注车载视频和可选的外部手机视频结合：

1. 稳定左压线：比较 `error_y` 均值、PID `y_pid` 均值和外部视频中的实际偏移。
   - 输出接近零但实际偏左：优先排查模型零点/相机几何，后续考虑显式横向零点补偿。
   - 输出持续要求反向纠偏但车不回正：排查 PID 增益、底盘响应和速度映射。
   - 输出方向与车身动作相反：先修正坐标符号，禁止用增益掩盖问题。
2. 锐角：截取锐角前后各两秒，检查误差变化、控制是否达到 PID 上限、推理时延/帧率和车道线可见性。
   - 控制饱和：再设计针对性减速或锐角增益策略。
   - 车道不可见或模型输出跳变：补充训练数据/调整相机视野。
   - 响应不足但视觉稳定：再调整角度 PID。

本次只提供观测和证据，任何补偿、PID 或锐角策略必须在获得首份记录后另行设计、测试和确认。

## 验证

本地自动化测试覆盖：

- 未给 `--record-dir` 时不创建产物且保持现有行为；
- 时间戳目录与 CSV 固定字段；
- 每帧记录的模型、PID 与将下发命令一致；
- 视频/CSV 初始化失败会在运动前退出；
- 单帧写失败进入停车路径；
- 元数据记录正常、异常、Ctrl+C 与到时四类停止原因；
- 原有速度/时长/输出校验测试持续通过。

在 Orin 上仅做语法检查、`--preflight-only` 和记录目录创建验证；实车测试由用户自行执行。测试结束后，检查不存在 `safe_lane_test.py`、`lane_only_infer_server.py` 残留，也不存在 5001 端口监听。
```

## `tools/safe_lane_test.py`

```python
"""Bounded lane-following test with a standalone, timeout-protected lane service."""
import argparse
import csv
import datetime
import importlib.util
import json
import math
import subprocess
import sys
import time
import types
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SERVER = Path(__file__).with_name("lane_only_infer_server.py")

CSV_FIELDS = [
    "elapsed_s", "frame_index", "infer_ms", "loop_fps", "error_y", "error_angle",
    "y_pid", "yaw_pid", "vx", "vy", "yaw", "video_written",
]


def utc_session_name(now=None):
    now = now or datetime.datetime.now(datetime.timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=datetime.timezone.utc)
    else:
        now = now.astimezone(datetime.timezone.utc)
    return now.strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]


def create_record_session(base_dir, session_name):
    session_dir = Path(base_dir) / session_name
    session_dir.mkdir(parents=True, exist_ok=False)
    return session_dir


def open_video_writer(path, size):
    import cv2

    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 20.0, size)
    if not writer.isOpened():
        writer.release()
        raise RuntimeError("recording_video_open_failed")
    return writer


def _utc_timestamp():
    return datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z")


class TestRecorder:
    def __init__(self, session_dir, writer, csv_handle, csv_writer, frame_size, parameters):
        self.session_dir = session_dir
        self.writer = writer
        self.csv_handle = csv_handle
        self.csv_writer = csv_writer
        self.frame_size = frame_size
        self.parameters = parameters
        self.started_at_utc = _utc_timestamp()
        self.frame_count = 0
        self.video_frame_count = 0
        self.closed = False

    @classmethod
    def create(cls, base_dir, session_name, frame_size, parameters):
        session_dir = create_record_session(base_dir, session_name)
        writer = None
        csv_handle = None
        try:
            writer = open_video_writer(session_dir / "annotated.mp4", frame_size)
            csv_handle = (session_dir / "frames.csv").open("w", encoding="utf-8", newline="")
            csv_writer = csv.DictWriter(csv_handle, fieldnames=CSV_FIELDS)
            csv_writer.writeheader()
            csv_handle.flush()
            return cls(session_dir, writer, csv_handle, csv_writer, frame_size, parameters)
        except Exception:
            if csv_handle is not None:
                try:
                    csv_handle.close()
                except Exception:
                    pass
            if writer is not None:
                try:
                    writer.release()
                except Exception:
                    pass
            try:
                session_dir.rmdir()
            except OSError:
                pass
            raise

    def record(self, frame, elapsed_s, frame_index, infer_ms, loop_fps, error_y, error_angle,
               y_pid, yaw_pid, vx, vy, yaw):
        if self.closed:
            raise RuntimeError("recorder_closed")
        import cv2

        annotated = frame.copy()
        lines = [
            f"t={elapsed_s:.3f}s frame={frame_index} infer={infer_ms:.1f}ms fps={loop_fps:.1f}",
            f"error_y={error_y:.3f} error_angle={error_angle:.3f}",
            f"y_pid={y_pid:.3f} yaw_pid={yaw_pid:.3f}",
            f"vx={vx:.3f} vy={vy:.3f} yaw={yaw:.3f}",
        ]
        for line_index, line in enumerate(lines):
            cv2.putText(annotated, line, (8, 24 + line_index * 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        self.writer.write(annotated)
        self.video_frame_count += 1
        self.csv_writer.writerow({
            "elapsed_s": elapsed_s,
            "frame_index": frame_index,
            "infer_ms": infer_ms,
            "loop_fps": loop_fps,
            "error_y": error_y,
            "error_angle": error_angle,
            "y_pid": y_pid,
            "yaw_pid": yaw_pid,
            "vx": vx,
            "vy": vy,
            "yaw": yaw,
            "video_written": "true",
        })
        self.csv_handle.flush()
        self.frame_count += 1

    def close(self, stop_reason):
        if self.closed:
            return
        errors = []
        try:
            self.writer.release()
        except Exception as exc:
            errors.append(exc)
        try:
            self.csv_handle.close()
        except Exception as exc:
            errors.append(exc)
        metadata = {
            "format_version": 1,
            "started_at_utc": self.started_at_utc,
            "ended_at_utc": _utc_timestamp(),
            "parameters": self.parameters,
            "frame_size": list(self.frame_size),
            "video_file": "annotated.mp4",
            "csv_file": "frames.csv",
            "frame_count": self.frame_count,
            "video_frame_count": self.video_frame_count,
            "stop_reason": stop_reason,
        }
        temporary = self.session_dir / "metadata.json.tmp"
        try:
            temporary.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
            temporary.replace(self.session_dir / "metadata.json")
        except Exception as exc:
            errors.append(exc)
        else:
            self.closed = True
        if errors:
            first_error = errors[0]
            first_error.recorder_cleanup_errors = tuple(
                f"{type(error).__name__}: {error}" for error in errors[1:]
            )
            raise first_error


def validate_limits(speed, duration):
    if not 0 < speed <= 0.10 or not 0 < duration <= 300:
        return "speed 0..0.10; duration 0..300"
    return None


def validate_output(output, limit):
    if not isinstance(output, (list, tuple)) or len(output) != 2:
        return "invalid_model_output"
    if not all(isinstance(value, (int, float)) and math.isfinite(value) for value in output):
        return "invalid_model_output"
    if max(abs(output[0]), abs(output[1])) > limit:
        return "model_output_limit"
    return None


def wait_ready(client, timeout, sleep=time.sleep):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if client.ready(timeout_ms=250):
            return True
        sleep(0.1)
    return False


def preflight_infer(client, frame):
    return client.infer(frame, timeout_ms=5000)


class RecordingWriteError(Exception):
    def __init__(self, cause):
        super().__init__(str(cause))
        self.cause = cause


def run_lane_loop(duration, speed, output_limit, cap, stream, lane, car, py, pa,
                  monotonic=time.monotonic, recorder=None):
    start = monotonic()
    previous_loop_start = start
    frame_index = 0
    loop_start = monotonic()
    while loop_start - start < duration:
        frame = cap.read()
        if frame is None:
            return "camera_failure"
        stream.update_frame(frame, "cam1")
        infer_start = monotonic()
        timeout_ms = 5000 if frame_index == 0 else 1000
        output = lane.infer(frame, timeout_ms=timeout_ms)
        infer_ms = (monotonic() - infer_start) * 1000.0
        if output is None:
            return "inference_timeout"
        error = validate_output(output, output_limit)
        if error:
            return error
        error_y, error_angle = output
        y_cmd = py(-error_y)
        yaw_cmd = pa(-error_angle)
        elapsed_s = loop_start - start
        loop_elapsed = loop_start - previous_loop_start
        loop_fps = 1.0 / loop_elapsed if loop_elapsed > 0 else 0.0
        if recorder is not None:
            try:
                recorder.record(
                    frame, elapsed_s, frame_index, infer_ms, loop_fps,
                    error_y, error_angle, y_cmd, yaw_cmd, speed, y_cmd, yaw_cmd,
                )
            except Exception as exc:
                raise RecordingWriteError(exc) from exc
        car.set_velocity(speed, y_cmd, yaw_cmd)
        frame_index += 1
        previous_loop_start = loop_start
        loop_start = monotonic()
    return "duration_elapsed"


class LaneClient:
    def __init__(self, port=5001):
        import zmq

        self.zmq = zmq
        self.port = port
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.connect(f"tcp://127.0.0.1:{port}")

    def _request(self, request, timeout_ms):
        self.socket.setsockopt(self.zmq.SNDTIMEO, timeout_ms)
        self.socket.setsockopt(self.zmq.RCVTIMEO, timeout_ms)
        try:
            self.socket.send(request)
            return json.loads(self.socket.recv().decode("utf-8"))
        except self.zmq.Again:
            self.socket.close(0)
            self.socket = self.context.socket(self.zmq.REQ)
            self.socket.connect(f"tcp://127.0.0.1:{self.port}")
            return None

    def ready(self, timeout_ms=250):
        return self._request(b"ATATA", timeout_ms) is True

    def infer(self, frame, timeout_ms=1000):
        import cv2

        encoded = cv2.imencode(".jpg", frame)[1].tobytes()
        return self._request(b"image" + encoded, timeout_ms)

    def close(self):
        self.socket.close(0)
        self.context.term()


def package(name, path):
    mod = types.ModuleType(name)
    mod.__path__ = [str(path)]
    mod.__package__ = name
    sys.modules[name] = mod


def load(name, path, package_name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = package_name
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def hardware_components():
    for name, path in {
        "smartcar": ROOT / "smartcar",
        "smartcar.whalesbot": ROOT / "smartcar/whalesbot",
        "smartcar.whalesbot.tools": ROOT / "smartcar/whalesbot/tools",
        "smartcar.whalesbot.vehicle": ROOT / "smartcar/whalesbot/vehicle",
        "smartcar.whalesbot.vehicle.driver": ROOT / "smartcar/whalesbot/vehicle/driver",
    }.items():
        package(name, path)
    tools = ROOT / "smartcar/whalesbot/tools"
    log = load("smartcar.whalesbot.tools.log_wrap", tools / "log_wrap.py", "smartcar.whalesbot.tools")
    util = load("smartcar.whalesbot.tools.tools_class", tools / "tools_class.py", "smartcar.whalesbot.tools")
    sys.modules["smartcar.whalesbot.tools"].logger = log.logger
    sys.modules["smartcar.whalesbot.tools"].PID = util.PID
    sys.modules["smartcar.whalesbot.tools"].CountRecord = util.CountRecord
    camera = load("smartcar.whalesbot.tools.camera", tools / "camera.py", "smartcar.whalesbot.tools")
    streamer = load("smartcar.whalesbot.tools.streamer", tools / "streamer.py", "smartcar.whalesbot.tools")
    mecanum = load("smartcar.whalesbot.vehicle.driver.mecanum", ROOT / "smartcar/whalesbot/vehicle/driver/mecanum.py", "smartcar.whalesbot.vehicle.driver")
    return camera.Camera, streamer.Streamer, mecanum.MecanumDriver, util.PID


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--speed", type=float, default=0.08)
    parser.add_argument("--duration", type=float, default=15.0)
    parser.add_argument("--output-limit", type=float, default=1.2)
    parser.add_argument("--startup-timeout", type=float, default=30.0)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--record-dir", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    limit_error = validate_limits(args.speed, args.duration)
    if limit_error:
        raise SystemExit(limit_error)

    process = subprocess.Popen([sys.executable, "-u", str(SERVER)], stdout=sys.stdout, stderr=sys.stderr)
    lane = LaneClient()
    cap = car = stream = recorder = None
    reason = "startup_failure"
    try:
        if not wait_ready(lane, args.startup_timeout):
            reason = "lane_service_timeout"
            return
        Camera, Streamer, Driver, PID = hardware_components()
        cap = Camera(1, 320, 240)
        frame = cap.read()
        if frame is None:
            reason = "camera_failure"
            return
        output = preflight_infer(lane, frame)
        if output is None:
            reason = "preflight_inference_timeout"
            return
        reason = validate_output(output, args.output_limit)
        if reason:
            return
        print(f"LANE_PREFLIGHT output={output}")
        if args.preflight_only:
            reason = "preflight_complete"
            return
        if args.record_dir is not None:
            try:
                recorder = TestRecorder.create(
                    args.record_dir,
                    utc_session_name(),
                    (320, 240),
                    {
                        "speed": args.speed,
                        "duration": args.duration,
                        "output_limit": args.output_limit,
                    },
                )
            except Exception as exc:
                reason = "recording_init_failure:" + type(exc).__name__
                print(reason)
                return
        car = Driver()
        car.stop()
        stream = Streamer()
        py = PID(5, 0.1, 0, setpoint=0, output_limits=(-0.7, 0.7))
        pa = PID(3, 0, 0, setpoint=0, output_limits=(-1.5, 1.5))
        reason = run_lane_loop(
            args.duration, args.speed, args.output_limit,
            cap, stream, lane, car, py, pa,
            recorder=recorder,
        )
    except KeyboardInterrupt:
        reason = "keyboard_interrupt"
    except RecordingWriteError as exc:
        reason = "recording_write_failure:" + type(exc.cause).__name__
        print(reason)
    except Exception as exc:
        reason = "runtime_error:" + type(exc).__name__
        print(reason)
    finally:
        if car is not None:
            try:
                car.stop()
            except Exception as exc:
                print("car_stop_failure:" + type(exc).__name__)
        if recorder is not None:
            try:
                recorder.close(reason)
            except Exception as exc:
                print("recording_close_failure:" + type(exc).__name__)
        if cap is not None:
            cap.close()
        if stream is not None:
            stream.stop()
        lane.close()
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
        print("SAFE_STOP reason=" + reason)


if __name__ == "__main__":
    main()
```

## `tests/test_safe_lane_test.py`

```python
import importlib.util
import json
from contextlib import ExitStack
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


path = Path(__file__).parents[1] / "tools" / "safe_lane_test.py"
spec = importlib.util.spec_from_file_location("safe_lane_test", path)
safe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(safe)


class SafetyTests(unittest.TestCase):
    def test_record_dir_argument_defaults_to_none(self):
        self.assertIsNone(safe.parse_args([]).record_dir)

    def test_argument_limits_allow_300_seconds_but_reject_longer(self):
        self.assertIsNone(safe.validate_limits(0.10, 300.0))
        self.assertEqual(
            "speed 0..0.10; duration 0..300",
            safe.validate_limits(0.10, 300.1),
        )
        self.assertEqual(
            "speed 0..0.10; duration 0..300",
            safe.validate_limits(0.11, 15.0),
        )

    def test_rejects_non_finite_or_excessive_model_output(self):
        self.assertEqual("invalid_model_output", safe.validate_output([float("nan"), 0.0], 1.0))
        self.assertEqual("model_output_limit", safe.validate_output([1.1, 0.0], 1.0))
        self.assertIsNone(safe.validate_output([0.2, -0.3], 1.0))

    def test_rejects_wrong_length_lane_output(self):
        self.assertEqual("invalid_model_output", safe.validate_output([0.1], 1.0))
        self.assertEqual("invalid_model_output", safe.validate_output([0.1, 0.2, 0.3], 1.0))

    def test_wait_ready_times_out_without_server(self):
        class NeverReady:
            def ready(self, timeout_ms):
                return False

        self.assertFalse(safe.wait_ready(NeverReady(), timeout=0.0, sleep=lambda _: None))

    def test_cold_start_preflight_allows_five_seconds(self):
        class Client:
            timeout_ms = None

            def infer(self, frame, timeout_ms):
                self.timeout_ms = timeout_ms
                return [0.1, -0.2]

        client = Client()
        self.assertEqual([0.1, -0.2], safe.preflight_infer(client, object()))
        self.assertEqual(5000, client.timeout_ms)

    def test_loop_uses_cold_start_timeout_then_normal_timeout(self):
        class Cap:
            def read(self):
                return object()

        class Stream:
            def update_frame(self, frame, name):
                pass

        class Lane:
            def __init__(self):
                self.timeouts = []

            def infer(self, frame, timeout_ms):
                self.timeouts.append(timeout_ms)
                return [0.1, -0.2]

        class Car:
            def set_velocity(self, speed, y, angle):
                pass

        lane = Lane()
        ticks = iter([0.0, 0.0, 0.0, 0.01, 0.02, 0.02, 0.03, 1.0])
        reason = safe.run_lane_loop(
            duration=0.1, speed=0.05, output_limit=1.0,
            cap=Cap(), stream=Stream(), lane=lane, car=Car(),
            py=lambda value: value, pa=lambda value: value,
            monotonic=lambda: next(ticks),
        )

        self.assertEqual("duration_elapsed", reason)
        self.assertEqual([5000, 1000], lane.timeouts)

    def test_initial_loop_inference_timeout_stops_before_velocity_command(self):
        expected_timeout = 5000

        class Cap:
            def read(self):
                return object()

        class Stream:
            def update_frame(self, frame, name):
                pass

        class Lane:
            def infer(self, frame, timeout_ms):
                if timeout_ms != expected_timeout:
                    raise AssertionError("expected cold-start timeout")
                return None

        class Car:
            def __init__(self):
                self.calls = []

            def set_velocity(self, speed, y, angle):
                self.calls.append((speed, y, angle))

        car = Car()
        ticks = iter([0.0, 0.0, 0.0, 0.01])
        reason = safe.run_lane_loop(
            duration=0.1, speed=0.05, output_limit=1.0,
            cap=Cap(), stream=Stream(), lane=Lane(), car=car,
            py=lambda value: value, pa=lambda value: value,
            monotonic=lambda: next(ticks),
        )

        self.assertEqual("inference_timeout", reason)
        self.assertEqual([], car.calls)

    def test_recorder_none_keeps_normal_duration_elapsed_behavior(self):
        class Cap:
            def read(self):
                return object()

        class Stream:
            def update_frame(self, frame, name):
                pass

        class Lane:
            def infer(self, frame, timeout_ms):
                return [0.1, -0.2]

        class Car:
            calls = 0

            def set_velocity(self, speed, y, angle):
                self.calls += 1

        ticks = iter([0.0, 0.0, 0.0, 0.0, 1.0])
        car = Car()
        reason = safe.run_lane_loop(
            duration=0.1,
            speed=0.05,
            output_limit=1.0,
            cap=Cap(),
            stream=Stream(),
            lane=Lane(),
            car=car,
            py=lambda value: value,
            pa=lambda value: value,
            monotonic=lambda: next(ticks),
            recorder=None,
        )
        self.assertEqual("duration_elapsed", reason)
        self.assertEqual(1, car.calls)


class LaneLoopRecordingTests(unittest.TestCase):
    def test_records_exact_chassis_commands_before_setting_velocity(self):
        events = []

        class Cap:
            def read(self):
                return "frame"

        class Stream:
            def update_frame(self, frame, name):
                events.append(("stream", frame, name))

        class Lane:
            def infer(self, frame, timeout_ms):
                events.append(("infer", frame))
                return [0.25, -0.25]

        class Recorder:
            def __init__(self):
                self.calls = []

            def record(self, *args):
                self.calls.append(args)
                events.append(("record", args))

        class Car:
            def __init__(self):
                self.calls = []

            def set_velocity(self, speed, vy, yaw):
                self.calls.append((speed, vy, yaw))
                events.append(("command", speed, vy, yaw))

        recorder = Recorder()
        car = Car()
        ticks = iter([0.0, 0.0, 0.0, 0.01, 1.0])

        reason = safe.run_lane_loop(
            duration=0.1,
            speed=0.05,
            output_limit=1.0,
            cap=Cap(),
            stream=Stream(),
            lane=Lane(),
            car=car,
            py=lambda value: value * 2,
            pa=lambda value: value * 3,
            monotonic=lambda: next(ticks),
            recorder=recorder,
        )

        self.assertEqual("duration_elapsed", reason)
        self.assertLess(
            next(index for index, event in enumerate(events) if event[0] == "record"),
            next(index for index, event in enumerate(events) if event[0] == "command"),
        )
        self.assertEqual((0.05, -0.5, 0.75), car.calls[0])
        recorded = recorder.calls[0]
        self.assertEqual(
            ("frame", 0.0, 0, 10.0, 0.0, 0.25, -0.25,
             -0.5, 0.75, 0.05, -0.5, 0.75),
            recorded,
        )


class CliRecordingLifecycleTests(unittest.TestCase):
    def _run_main(self, argv, *, recorder_create=None, loop=None, preflight_output=(0.0, 0.0)):
        events = []

        class Process:
            def terminate(self):
                events.append("process.terminate")

            def wait(self, timeout):
                events.append("process.wait")

        class Lane:
            def close(self):
                events.append("lane.close")

        class Cap:
            def read(self):
                return "frame"

            def close(self):
                events.append("cap.close")

        class Stream:
            def stop(self):
                events.append("stream.stop")

        class Driver:
            def __init__(self):
                events.append("driver.init")

            def stop(self):
                events.append("car.stop")

        messages = []
        patches = [
            patch.object(safe.subprocess, "Popen", return_value=Process()),
            patch.object(safe, "LaneClient", return_value=Lane()),
            patch.object(safe, "wait_ready", return_value=True),
            patch.object(safe, "hardware_components", return_value=(lambda *_: Cap(), lambda: Stream(), Driver, lambda *_, **__: object())),
            patch.object(safe, "preflight_infer", return_value=preflight_output),
            patch("builtins.print", side_effect=messages.append),
        ]
        if recorder_create is not None:
            patches.append(patch.object(safe.TestRecorder, "create", side_effect=recorder_create))
        if loop is not None:
            patches.append(patch.object(safe, "run_lane_loop", side_effect=loop))
        with ExitStack() as stack:
            for active_patch in patches:
                stack.enter_context(active_patch)
            safe.main(argv)
        return events, messages

    def test_preflight_only_record_dir_never_creates_recorder_or_driver(self):
        events, messages = self._run_main(
            ["--preflight-only", "--record-dir", "records"],
            recorder_create=AssertionError("recorder must not be created"),
        )

        self.assertNotIn("driver.init", events)
        self.assertIn("SAFE_STOP reason=preflight_complete", messages)

    def test_preflight_timeout_has_explicit_reason(self):
        events, messages = self._run_main(
            [], loop=AssertionError("loop must not run"), preflight_output=None,
        )

        self.assertNotIn("driver.init", events)
        self.assertEqual(["SAFE_STOP reason=preflight_inference_timeout"], messages)

    def test_recording_init_failure_prevents_driver_and_safely_stops(self):
        events, messages = self._run_main(
            ["--record-dir", "records"],
            recorder_create=OSError("disk unavailable"),
        )

        self.assertNotIn("driver.init", events)
        self.assertIn("SAFE_STOP reason=recording_init_failure:OSError", messages)

    def test_loop_recorder_failure_stops_car_before_closing_recorder(self):
        class Recorder:
            def close(self, reason):
                events.append(("recorder.close", reason))

        events = []
        recorder = Recorder()

        class Process:
            def terminate(self):
                events.append("process.terminate")

            def wait(self, timeout):
                events.append("process.wait")

        class Lane:
            def close(self):
                events.append("lane.close")

        class Cap:
            def read(self):
                return "frame"

            def close(self):
                events.append("cap.close")

        class Stream:
            def stop(self):
                events.append("stream.stop")

        class Driver:
            def stop(self):
                events.append("car.stop")

        messages = []
        with patch.object(safe.subprocess, "Popen", return_value=Process()), \
             patch.object(safe, "LaneClient", return_value=Lane()), \
             patch.object(safe, "wait_ready", return_value=True), \
             patch.object(safe, "hardware_components", return_value=(lambda *_: Cap(), lambda: Stream(), Driver, lambda *_, **__: object())), \
             patch.object(safe, "preflight_infer", return_value=[0.0, 0.0]), \
             patch.object(safe.TestRecorder, "create", return_value=recorder), \
             patch.object(safe, "run_lane_loop", side_effect=safe.RecordingWriteError(RuntimeError("write failed"))), \
             patch("builtins.print", side_effect=messages.append):
            safe.main(["--record-dir", "records"])

        self.assertIn("SAFE_STOP reason=recording_write_failure:RuntimeError", messages)
        self.assertLess(events.index("car.stop"), events.index(("recorder.close", "recording_write_failure:RuntimeError")))

    def test_final_stop_failure_does_not_skip_remaining_cleanup(self):
        events = []

        class Recorder:
            def close(self, reason):
                events.append(("recorder.close", reason))

        class Process:
            def terminate(self):
                events.append("process.terminate")

            def wait(self, timeout):
                events.append("process.wait")

        class Lane:
            def close(self):
                events.append("lane.close")

        class Cap:
            def read(self):
                return "frame"

            def close(self):
                events.append("cap.close")

        class Stream:
            def stop(self):
                events.append("stream.stop")

        class Driver:
            def __init__(self):
                self.stop_calls = 0

            def stop(self):
                self.stop_calls += 1
                events.append("car.stop")
                if self.stop_calls == 2:
                    raise RuntimeError("stop failed")

        messages = []
        with patch.object(safe.subprocess, "Popen", return_value=Process()), \
             patch.object(safe, "LaneClient", return_value=Lane()), \
             patch.object(safe, "wait_ready", return_value=True), \
             patch.object(safe, "hardware_components", return_value=(lambda *_: Cap(), lambda: Stream(), Driver, lambda *_, **__: object())), \
             patch.object(safe, "preflight_infer", return_value=[0.0, 0.0]), \
             patch.object(safe.TestRecorder, "create", return_value=Recorder()), \
             patch.object(safe, "run_lane_loop", return_value="duration_elapsed"), \
             patch("builtins.print", side_effect=messages.append):
            safe.main(["--record-dir", "records"])

        final_stop = len(events) - 1 - events[::-1].index("car.stop")
        self.assertEqual(
            [
                "car.stop",
                ("recorder.close", "duration_elapsed"),
                "cap.close",
                "stream.stop",
                "lane.close",
                "process.terminate",
                "process.wait",
            ],
            events[final_stop:],
        )
        self.assertIn("SAFE_STOP reason=duration_elapsed", messages)


class RecorderContractTests(unittest.TestCase):
    def test_session_name_is_utc_timestamp_with_random_hex_suffix(self):
        now = safe.datetime.datetime(2026, 7, 16, 9, 8, 7, tzinfo=safe.datetime.timezone.utc)

        name = safe.utc_session_name(now)

        self.assertRegex(name, r"^20260716T090807Z-[0-9a-f]{8}$")

    def test_create_record_session_creates_exclusive_child_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            session = safe.create_record_session(Path(directory), "session-1")

            self.assertEqual(Path(directory) / "session-1", session)
            self.assertTrue(session.is_dir())
            with self.assertRaises(FileExistsError):
                safe.create_record_session(Path(directory), "session-1")

    def test_open_video_writer_uses_mp4v_20fps_and_rejects_closed_writer(self):
        calls = []

        class Writer:
            def isOpened(self):
                return False

            def release(self):
                pass

        cv2 = SimpleNamespace(
            VideoWriter_fourcc=lambda *codec: calls.append(codec) or 123,
            VideoWriter=lambda path, fourcc, fps, size: calls.append((path, fourcc, fps, size)) or Writer(),
        )
        with patch.dict(sys.modules, {"cv2": cv2}):
            with self.assertRaisesRegex(RuntimeError, "^recording_video_open_failed$"):
                safe.open_video_writer("out.mp4", (320, 240))

        self.assertEqual(("m", "p", "4", "v"), calls[0])
        self.assertEqual(("out.mp4", 123, 20.0, (320, 240)), calls[1])

    def test_recorder_writes_annotated_video_csv_and_atomic_metadata(self):
        class Writer:
            def __init__(self):
                self.frames = []
                self.released = False

            def write(self, frame):
                self.frames.append(frame)

            def release(self):
                self.released = True

        writer = Writer()
        cv2 = SimpleNamespace(FONT_HERSHEY_SIMPLEX=0, putText=lambda *args: args[0])
        frame = SimpleNamespace(copy=lambda: "annotated-frame")
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(safe, "open_video_writer", return_value=writer), \
             patch.dict(sys.modules, {"cv2": cv2}):
            recorder = safe.TestRecorder.create(
                directory, "session-1", (320, 240), {"speed": 0.08},
            )
            recorder.record(frame, 1.25, 3, 12.5, 25.0, 0.1, -0.2, 0.3, -0.4, 0.08, 0.0, -0.1)
            recorder.close("duration_elapsed")
            recorder.close("ignored")

            session = Path(directory) / "session-1"
            self.assertEqual(["annotated-frame"], writer.frames)
            self.assertTrue(writer.released)
            self.assertEqual(
                ",".join(safe.CSV_FIELDS),
                (session / "frames.csv").read_text(encoding="utf-8").splitlines()[0],
            )
            row = (session / "frames.csv").read_text(encoding="utf-8").splitlines()[1].split(",")
            self.assertEqual("true", row[-1])
            self.assertEqual("3", row[1])
            self.assertFalse((session / "metadata.json.tmp").exists())
            metadata = json.loads((session / "metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(1, metadata["format_version"])
            self.assertEqual({"speed": 0.08}, metadata["parameters"])
            self.assertEqual([320, 240], metadata["frame_size"])
            self.assertEqual("annotated.mp4", metadata["video_file"])
            self.assertEqual("frames.csv", metadata["csv_file"])
            self.assertEqual(1, metadata["frame_count"])
            self.assertEqual(1, metadata["video_frame_count"])
            self.assertEqual("duration_elapsed", metadata["stop_reason"])
            self.assertTrue(metadata["started_at_utc"])
            self.assertTrue(metadata["ended_at_utc"])

    def test_create_removes_empty_session_when_video_writer_initialization_fails(self):
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(safe, "open_video_writer", side_effect=RuntimeError("writer failed")):
            with self.assertRaisesRegex(RuntimeError, "writer failed"):
                safe.TestRecorder.create(directory, "session-1", (320, 240), {})

            self.assertFalse((Path(directory) / "session-1").exists())

    def test_create_releases_writer_when_csv_initialization_fails(self):
        class Writer:
            released = False

            def release(self):
                self.released = True

        writer = Writer()
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(safe, "open_video_writer", return_value=writer), \
             patch.object(Path, "open", side_effect=OSError("csv failed")):
            with self.assertRaisesRegex(OSError, "csv failed"):
                safe.TestRecorder.create(directory, "session-1", (320, 240), {})

        self.assertTrue(writer.released)

    def test_create_closes_csv_and_releases_writer_when_header_write_fails(self):
        class Writer:
            released = False

            def release(self):
                self.released = True

        class CsvHandle:
            closed = False

            def close(self):
                self.closed = True

        class HeaderWriter:
            def __init__(self, handle, fieldnames):
                pass

            def writeheader(self):
                raise OSError("header failed")

        writer = Writer()
        csv_handle = CsvHandle()
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(safe, "open_video_writer", return_value=writer), \
             patch.object(Path, "open", return_value=csv_handle), \
             patch.object(safe.csv, "DictWriter", HeaderWriter):
            with self.assertRaisesRegex(OSError, "header failed"):
                safe.TestRecorder.create(directory, "session-1", (320, 240), {})

        self.assertTrue(writer.released)
        self.assertTrue(csv_handle.closed)

    def test_close_publishes_metadata_when_release_and_csv_close_fail(self):
        class Writer:
            def release(self):
                raise RuntimeError("release failed")

        class CsvHandle:
            def close(self):
                raise RuntimeError("csv close failed")

        with tempfile.TemporaryDirectory() as directory:
            session = Path(directory) / "session-1"
            session.mkdir()
            recorder = safe.TestRecorder(session, Writer(), CsvHandle(), object(), (320, 240), {})

            with self.assertRaisesRegex(RuntimeError, "^release failed$") as raised:
                recorder.close("safe_stop")

            metadata = json.loads((session / "metadata.json").read_text(encoding="utf-8"))
            self.assertEqual("safe_stop", metadata["stop_reason"])
            self.assertEqual(
                ("RuntimeError: csv close failed",),
                raised.exception.recorder_cleanup_errors,
            )


if __name__ == "__main__":
    unittest.main()
```


