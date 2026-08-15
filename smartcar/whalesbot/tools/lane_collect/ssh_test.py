"""SSH-controlled, low-speed OpenCV lane test for ``collect_data.py`` only."""

import json
import queue
import signal
import threading
import time
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2

from .opencv_lane import LaneAnalyzerConfig, OpenCVLaneAnalyzer, StandardLaneReference
from .pid_control import CvLanePidConfig, CvLanePidController
from .turn_state import CrossStraightStateMachine


class CvTestSessionWriter:
    """Save the exact 128x128 CNN image and its applied vehicle command."""

    def __init__(self, output_root: Path, controller_config: CvLanePidConfig) -> None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        self.session_dir = Path(output_root) / f"cv_low_speed_{stamp}"
        self.session_dir.mkdir(parents=True, exist_ok=False)
        self.controller_config = controller_config
        self.records = []
        self.closed = False
        self._write_metadata()

    def append(self, cnn_image, analysis, command, *, held=False,
               command_source="opencv") -> None:
        if self.closed:
            raise RuntimeError("CV test session is already closed")
        image_name = f"{len(self.records):06d}.jpg"
        image_path = self.session_dir / image_name
        if not cv2.imwrite(str(image_path), cnn_image):
            raise RuntimeError(f"could not save {image_path}")
        self.records.append({
            "img_path": image_name,
            # Keep the same label semantics as manual collection: state is
            # the command actually sent to car.set_velocity(x, y, z).
            "state": [command.forward_speed, command.lateral_speed,
                      command.angular_speed],
            "control": [command.forward_speed, command.lateral_speed,
                        command.angular_speed],
            # Keep the legacy manual-collection label contract: state[1:3]
            # are the vehicle commands sent to set_velocity().  The teacher
            # is intentionally named as a PID command teacher because its
            # output is not a stateless geometric error.
            "teacher": "opencv_pid_command",
            "label_semantics": "vehicle_command_compatible_with_manual",
            "held": bool(held),
            "command_source": str(command_source),
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
            "usable_for_training": True,
            "saved_image_size": [128, 128],
            "state_fields": ["forward_speed", "lateral_speed",
                             "angular_speed"],
            "control_fields": ["forward_speed", "lateral_speed",
                               "angular_speed"],
            "label_semantics": "vehicle_command_compatible_with_manual",
            "teacher": "opencv_pid_command",
            "controller": vars(self.controller_config),
        }
        (self.session_dir / "session.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")


class OpenCVLaneSshTest:
    """Run CV lane control after explicit commands received over SSH stdin."""

    LOOP_SECONDS = 0.05
    MAX_FRAME_AGE_SECONDS = 0.25
    MAX_INVALID_HOLD_FRAMES = 5

    def __init__(self, camera, car, output_root="dataset/cv_lane_tests",
                 frame_callback=None, standard_root="standard") -> None:
        self.camera = camera
        self.car = car
        self.output_root = Path(output_root)
        self.frame_callback = frame_callback
        standard_root = Path(standard_root)
        reference = StandardLaneReference.from_files(
            standard_root / "standard_lane.json",
            standard_root / "perspective.json",
        )
        self.analyzer = OpenCVLaneAnalyzer(LaneAnalyzerConfig(
            threshold=175,
            segmentation="dark",
            work_size=(320, 240),
            cnn_size=(128, 128),
            roi_top_ratio=0.30,
            roi_bottom_ratio=0.20,
        ), reference=reference)
        self.controller = CvLanePidController(CvLanePidConfig())
        self.cross_state = CrossStraightStateMachine()
        self.commands = queue.Queue()
        self.running = False
        self.exiting = False
        self.writer: Optional[CvTestSessionWriter] = None
        self.last_valid_command = None
        self.invalid_hold_frames = 0

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
            decision = self.cross_state.update(analysis)
            analysis = decision.result
            command = self.controller.compute(analysis)
            command_source = decision.source
            held = False
            if command.valid and decision.speed_scale != 1.0:
                command = replace(
                    command,
                    forward_speed=command.forward_speed * decision.speed_scale,
                    reason=decision.source,
                )
            if not command.valid:
                self.invalid_hold_frames += 1
                if (self.last_valid_command is not None and
                        self.invalid_hold_frames <= self.MAX_INVALID_HOLD_FRAMES):
                    command = replace(
                        self.last_valid_command,
                        reason="short_invalid_hold",
                    )
                    command_source = "short_invalid_hold"
                    held = True
                else:
                    self._disarm(f"OpenCV invalid: {command.reason}")
                    return
            else:
                self.invalid_hold_frames = 0
                self.last_valid_command = command
            self.car.set_velocity(
                command.forward_speed,
                command.lateral_speed,
                command.angular_speed,
            )
            cnn_image = self.analyzer.make_cnn_image(image)
            self.writer.append(
                cnn_image, analysis, command, held=held,
                command_source=command_source,
            )
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
                self.cross_state.reset()
                self.last_valid_command = None
                self.invalid_hold_frames = 0
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
        self.last_valid_command = None
        self.invalid_hold_frames = 0
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
