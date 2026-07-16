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
            if len(errors) == 1:
                raise errors[0]
            raise ExceptionGroup("recorder_close_failed", errors)


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


def run_lane_loop(duration, speed, output_limit, cap, stream, lane, car, py, pa,
                  monotonic=time.monotonic):
    start = monotonic()
    while monotonic() - start < duration:
        frame = cap.read()
        if frame is None:
            return "camera_failure"
        stream.update_frame(frame, "cam1")
        output = lane.infer(frame)
        error = validate_output(output, output_limit)
        if error:
            return error
        error_y, error_angle = output
        car.set_velocity(speed, py(-error_y), pa(-error_angle))
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--speed", type=float, default=0.08)
    parser.add_argument("--duration", type=float, default=15.0)
    parser.add_argument("--output-limit", type=float, default=1.2)
    parser.add_argument("--startup-timeout", type=float, default=30.0)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    limit_error = validate_limits(args.speed, args.duration)
    if limit_error:
        raise SystemExit(limit_error)

    process = subprocess.Popen([sys.executable, "-u", str(SERVER)], stdout=sys.stdout, stderr=sys.stderr)
    lane = LaneClient()
    cap = car = stream = None
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
        reason = validate_output(output, args.output_limit)
        if reason:
            return
        print(f"LANE_PREFLIGHT output={output}")
        if args.preflight_only:
            reason = "preflight_complete"
            return
        car = Driver()
        car.stop()
        stream = Streamer()
        py = PID(5, 0.1, 0, setpoint=0, output_limits=(-0.7, 0.7))
        pa = PID(3, 0, 0, setpoint=0, output_limits=(-1.5, 1.5))
        reason = run_lane_loop(
            args.duration, args.speed, args.output_limit,
            cap, stream, lane, car, py, pa,
        )
    except KeyboardInterrupt:
        reason = "keyboard_interrupt"
    except Exception as exc:
        reason = "runtime_error:" + type(exc).__name__
        print(reason)
    finally:
        if car is not None:
            car.stop()
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
