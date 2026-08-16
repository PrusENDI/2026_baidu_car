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

from .calibration import ErrorMapping
from .opencv_lane import LaneAnalyzerConfig, OpenCVLaneAnalyzer, StandardLaneReference
from .pid_control import CvLanePidConfig, CvLanePidController
from .temporal_filter import PreviewTimingFilter
from .turn_state import CrossStraightStateMachine


class CvTestSessionWriter:
    """Save each raw frame, PID-before label, and applied command."""

    def __init__(self, output_root: Path, controller_config: CvLanePidConfig) -> None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        self.session_dir = Path(output_root) / f"cv_low_speed_{stamp}"
        self.session_dir.mkdir(parents=True, exist_ok=False)
        self.controller_config = controller_config
        self.teacher = ("opencv_ipm_pure_pursuit"
                        if controller_config.steering_mode == "pure_pursuit"
                        else "opencv_ipm_error")
        self.records = []
        self.closed = False
        self._write_metadata()

    def append(self, cnn_image, analysis, command, *, held=False,
               command_source="opencv", odometry_distance_m=None,
               frame_distance_m=None, distance_source="unavailable") -> None:
        if self.closed:
            raise RuntimeError("CV test session is already closed")
        image_name = f"{len(self.records):06d}.jpg"
        image_path = self.session_dir / image_name
        if not cv2.imwrite(str(image_path), cnn_image):
            raise RuntimeError(f"could not save {image_path}")
        self.records.append({
            "img_path": image_name,
            # Match the existing CNN contract: state[1:3] is consumed as
            # error_y/error_angle and then passed through lane_pid.
            "state": [command.forward_speed, command.error_y,
                      command.error_angle],
            # Keep the actual mecanum command separately for replay and
            # closed-loop diagnostics.
            "control": [command.forward_speed, command.lateral_speed,
                        command.angular_speed],
            "teacher": self.teacher,
            "label_semantics": "cnn_pid_input_error",
            "held": bool(held),
            "command_source": str(command_source),
            "control_reason": str(command.reason),
            "steering_demand": float(command.steering_demand),
            "target_forward_speed": float(command.target_forward_speed),
            # Encoder-based distance measured by the chassis odometry.  The
            # per-frame delta is relative to the previous saved frame, while
            # odometry_distance_m is relative to the beginning of this
            # collection session.
            "odometry_distance_m": odometry_distance_m,
            "frame_distance_m": frame_distance_m,
            "distance_source": str(distance_source),
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
            "saved_image_size": [320, 240],
            "state_fields": ["forward_speed", "error_y", "error_angle"],
            "control_fields": ["forward_speed", "lateral_speed",
                               "angular_speed"],
            "label_semantics": "cnn_pid_input_error",
            "teacher": self.teacher,
            "distance_fields": {
                "odometry_distance_m": "session-relative chassis odometry",
                "frame_distance_m": "delta since previous saved frame",
                "distance_source": "encoder_odometry when available",
            },
            "controller": vars(self.controller_config),
            "speed_control": {
                "source": ("absolute pure-pursuit curvature"
                           if self.controller_config.steering_mode ==
                           "pure_pursuit" else
                           "absolute PID-before heading error"),
                "entry_behavior": "fast deceleration",
                "exit_behavior": "slow acceleration",
            },
            "launch_guard": {
                "distance_m": float(self.controller_config.initial_straight_distance_m),
                "source": "encoder_odometry",
                "behavior": "forward-only before first bend",
            },
        }
        (self.session_dir / "session.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")


class OpenCVLaneSshTest:
    """Run CV lane control after explicit commands received over SSH stdin."""

    LOOP_SECONDS = 0.05
    MAX_FRAME_AGE_SECONDS = 0.25
    MAX_INVALID_HOLD_FRAMES = 10
    # Temporarily bypass the route-specific crossing controller while
    # validating ordinary and right-angle bends at the higher test speed.
    CROSS_STATE_ENABLED = False
    # Implemented and available for offline replay, but kept off on the real
    # car until the first-turn timing error and +9-frame best lag pass.
    PREVIEW_TIMING_ENABLED = False

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
            threshold=None,
            adaptive_threshold_min=150,
            adaptive_threshold_max=165,
            segmentation="dark",
            work_size=(320, 240),
            cnn_size=(128, 128),
            roi_top_ratio=0.30,
            roi_bottom_ratio=0.20,
            error_mapping=ErrorMapping(
                lateral_scale=-0.10,
                heading_scale=-0.40,
            ),
        ), reference=reference)
        self.controller = CvLanePidController(CvLanePidConfig(
            steering_mode="pure_pursuit"))
        self.temporal_filter = PreviewTimingFilter(
            error_mapping=self.analyzer.config.error_mapping)
        self.cross_state = CrossStraightStateMachine()
        self.commands = queue.Queue()
        self.running = False
        self.exiting = False
        self.writer: Optional[CvTestSessionWriter] = None
        self.last_valid_command = None
        self.invalid_hold_frames = 0
        self.distance_origin_m = None
        self.last_distance_m = None

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
            if self.PREVIEW_TIMING_ENABLED:
                analysis = self.temporal_filter.update(analysis)
            distance_m, distance_source = self._read_distance()
            if distance_m is not None:
                if self.distance_origin_m is None:
                    self.distance_origin_m = distance_m
                distance_m -= self.distance_origin_m
            decision = None
            if self.CROSS_STATE_ENABLED:
                decision = self.cross_state.update(
                    analysis, distance_m=distance_m)
                analysis = decision.result
            command = self.controller.compute(analysis, distance_m=distance_m)
            command_source = decision.source if decision is not None else "standard"
            held = False
            if (command.valid and decision is not None and
                    decision.speed_scale != 1.0):
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
            # The first 0.15 m is a launch-only protection.  It is based on
            # chassis odometry, never on image/frame number, and is not reused
            # for subsequent bends.
            launch_distance = max(
                float(self.controller.config.initial_straight_distance_m), 0.0)
            launch_guard = (distance_m is not None and
                            distance_m < launch_distance)
            if launch_guard:
                command = replace(
                    command, lateral_speed=0.0, angular_speed=0.0,
                    reason="initial_straight_guard")
                command_source = "initial_straight_guard"
            self.car.set_velocity(
                command.forward_speed,
                command.lateral_speed,
                command.angular_speed,
            )
            frame_distance_m = None
            if distance_m is not None:
                if self.last_distance_m is not None:
                    frame_distance_m = distance_m - self.last_distance_m
                self.last_distance_m = distance_m
            # Manual lane collection stores the raw Camera(1) frame at
            # 320x240.  Keep CV collection in the same geometry; CNN
            # inference can resize this source image separately when needed.
            saved_image = image
            self.writer.append(
                saved_image, analysis, command, held=held,
                command_source=command_source,
                odometry_distance_m=distance_m,
                frame_distance_m=frame_distance_m,
                distance_source=distance_source,
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
                self.temporal_filter.reset()
                if self.CROSS_STATE_ENABLED:
                    self.cross_state.reset()
                self.last_valid_command = None
                self.invalid_hold_frames = 0
                self.writer = CvTestSessionWriter(
                    self.output_root, self.controller.config)
                self.distance_origin_m = None
                self.last_distance_m = None
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
        self.temporal_filter.reset()
        self.last_valid_command = None
        self.invalid_hold_frames = 0
        self.distance_origin_m = None
        self.last_distance_m = None
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

    def _read_distance(self):
        """Return chassis odometry distance, without breaking control if absent."""
        getter = getattr(self.car, "get_distance", None)
        if not callable(getter):
            return None, "unavailable"
        try:
            value = float(getter())
        except (TypeError, ValueError, RuntimeError, OSError):
            return None, "unavailable"
        return value, "encoder_odometry"

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
