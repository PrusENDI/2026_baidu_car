"""SSH-controlled, low-speed OpenCV lane test for ``collect_data.py`` only."""

import json
import queue
import signal
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2

from .opencv_lane import LaneAnalyzerConfig, OpenCVLaneAnalyzer
from .pid_control import CvLanePidConfig, CvLanePidController


class CvTestSessionWriter:
    """Save the exact 128x128 CNN image and its CV teacher errors."""

    def __init__(self, output_root: Path, controller_config: CvLanePidConfig) -> None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        self.session_dir = Path(output_root) / f"cv_low_speed_{stamp}"
        self.session_dir.mkdir(parents=True, exist_ok=False)
        self.controller_config = controller_config
        self.records = []
        self.closed = False
        self._write_metadata()

    def append(self, cnn_image, analysis, command) -> None:
        if self.closed:
            raise RuntimeError("CV test session is already closed")
        image_name = f"{len(self.records):06d}.jpg"
        image_path = self.session_dir / image_name
        if not cv2.imwrite(str(image_path), cnn_image):
            raise RuntimeError(f"could not save {image_path}")
        self.records.append({
            "img_path": image_name,
            # Training still reads state[1:3]. This isolated test session is
            # deliberately marked unusable until real-car review passes.
            "state": [command.forward_speed, command.error_y,
                      command.error_angle],
            "control": [command.forward_speed, command.lateral_speed,
                        command.angular_speed],
            "teacher": "opencv_stateless",
            "cv": analysis.to_record(),
            "timestamp": time.time(),
        })
        if len(self.records) % 10 == 0:
            self._write_data()

    def close(self) -> None:
        if not self.closed:
            self._write_data()
            self.closed = True

    def _write_data(self) -> None:
        target = self.session_dir / "data.json"
        temporary = self.session_dir / "data.json.tmp"
        temporary.write_text(
            json.dumps(self.records, ensure_ascii=False), encoding="utf-8")
        temporary.replace(target)

    def _write_metadata(self) -> None:
        metadata = {
            "purpose": "OpenCV PID low-speed real-car test",
            "usable_for_training": False,
            "saved_image_size": [128, 128],
            "state_fields": ["forward_speed", "error_y", "error_angle"],
            "control_fields": ["forward_speed", "lateral_speed",
                               "angular_speed"],
            "controller": vars(self.controller_config),
        }
        (self.session_dir / "session.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")


class OpenCVLaneSshTest:
    """Run CV lane control after explicit commands received over SSH stdin."""

    LOOP_SECONDS = 0.05
    MAX_FRAME_AGE_SECONDS = 0.25

    def __init__(self, camera, car, output_root="dataset/cv_lane_tests",
                 frame_callback=None) -> None:
        self.camera = camera
        self.car = car
        self.output_root = Path(output_root)
        self.frame_callback = frame_callback
        self.analyzer = OpenCVLaneAnalyzer(LaneAnalyzerConfig(
            threshold=175,
            segmentation="dark",
            work_size=(320, 240),
            cnn_size=(128, 128),
            roi_top_ratio=0.30,
            roi_bottom_ratio=0.20,
        ))
        self.controller = CvLanePidController(CvLanePidConfig())
        self.commands = queue.Queue()
        self.running = False
        self.exiting = False
        self.writer: Optional[CvTestSessionWriter] = None

    def run(self) -> None:
        self._stop_vehicle()
        previous_handlers = self._install_stop_signal_handlers()
        command_thread = threading.Thread(target=self._read_commands, daemon=True)
        command_thread.start()
        print("CV low-speed test ready. Commands: start, stop, status, quit", flush=True)
        try:
            while not self.exiting:
                self._consume_commands()
                if not self.running:
                    time.sleep(self.LOOP_SECONDS)
                    continue
                self._control_once()
                time.sleep(self.LOOP_SECONDS)
        except KeyboardInterrupt:
            print("\nCtrl+C received; stopping vehicle.", flush=True)
        finally:
            self.exiting = True
            self._disarm("session ended")
            self._restore_signal_handlers(previous_handlers)

    def _control_once(self) -> None:
        try:
            if hasattr(self.camera, "frame_timestamp"):
                frame_timestamp = self.camera.frame_timestamp
                if (frame_timestamp is None or
                        time.monotonic() - frame_timestamp >
                        self.MAX_FRAME_AGE_SECONDS):
                    self._disarm("camera frame is stale")
                    return
            image = self.camera.read().copy()
            analysis = self.analyzer.process(image)
            command = self.controller.compute(analysis)
            if not command.valid:
                self._disarm(f"OpenCV invalid: {command.reason}")
                return
            self.car.set_velocity(
                command.forward_speed,
                command.lateral_speed,
                command.angular_speed,
            )
            cnn_image = self.analyzer.make_cnn_image(image)
            self.writer.append(cnn_image, analysis, command)
            if self.frame_callback is not None:
                self.frame_callback(self.analyzer.draw_debug(image, analysis))
        except Exception as exc:
            self._disarm(f"control exception: {exc}")

    def _consume_commands(self) -> None:
        while True:
            try:
                command = self.commands.get_nowait()
            except queue.Empty:
                return
            if command == "start":
                if self.running:
                    print("CV control is already running.", flush=True)
                    continue
                self.controller.reset()
                self.writer = CvTestSessionWriter(
                    self.output_root, self.controller.config)
                self.running = True
                print(f"STARTED: {self.writer.session_dir}", flush=True)
            elif command == "stop":
                self._disarm("SSH stop command")
            elif command == "status":
                state = "RUNNING" if self.running else "STOPPED"
                session = self.writer.session_dir if self.writer else "none"
                print(f"{state}; session={session}", flush=True)
            elif command in {"quit", "exit"}:
                self._disarm("SSH quit command")
                self.exiting = True
            elif command:
                print("Unknown command. Use: start, stop, status, quit", flush=True)

    def _disarm(self, reason: str) -> None:
        was_running = self.running
        self.running = False
        self._stop_vehicle()
        self.controller.reset()
        if self.writer is not None:
            self.writer.close()
            session = self.writer.session_dir
            count = len(self.writer.records)
            self.writer = None
            print(f"STOPPED: {reason}; saved={count}; session={session}", flush=True)
        elif was_running:
            print(f"STOPPED: {reason}", flush=True)

    def _stop_vehicle(self) -> None:
        self.car.set_velocity(0.0, 0.0, 0.0)

    def _read_commands(self) -> None:
        while not self.exiting:
            try:
                command = input("cv> ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                self.commands.put("quit")
                return
            self.commands.put(command)

    @staticmethod
    def _raise_stop_signal(signum, frame) -> None:
        del signum, frame
        raise KeyboardInterrupt

    def _install_stop_signal_handlers(self):
        handlers = {}
        for name in ("SIGHUP", "SIGTERM"):
            sig = getattr(signal, name, None)
            if sig is not None:
                handlers[sig] = signal.getsignal(sig)
                signal.signal(sig, self._raise_stop_signal)
        return handlers

    @staticmethod
    def _restore_signal_handlers(handlers) -> None:
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
