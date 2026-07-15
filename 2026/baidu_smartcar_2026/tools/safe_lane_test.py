"""Bounded lane-following test with a standalone, timeout-protected lane service."""
import argparse
import importlib.util
import json
import math
import subprocess
import sys
import time
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SERVER = Path(__file__).with_name("lane_only_infer_server.py")


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
    if not 0 < args.speed <= 0.10 or not 0 < args.duration <= 15:
        raise SystemExit("speed 0..0.10; duration 0..15")

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
