#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""Standalone camera streaming debug tool."""

import argparse
import glob
import os
import sys
import threading
import time
from dataclasses import dataclass


@dataclass(frozen=True)
class CameraSpec:
    name: str
    index: str
    width: int
    height: int


def parse_size(value):
    text = value.lower().replace("*", "x")
    if "x" not in text:
        raise argparse.ArgumentTypeError("size must look like WIDTHxHEIGHT")
    width_text, height_text = text.split("x", 1)
    width = int(width_text)
    height = int(height_text)
    if width <= 0 or height <= 0:
        raise argparse.ArgumentTypeError("width and height must be positive")
    return width, height


def parse_angle_list(value):
    angles = []
    for item in value.split(","):
        item = item.strip()
        if item:
            angles.append(int(item))
    if not angles:
        raise argparse.ArgumentTypeError("angle list cannot be empty")
    return angles


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Debug camera input with the existing MJPEG streamer."
    )
    parser.add_argument("--cam1", default="1", help="cam1 project index. Default: 1")
    parser.add_argument("--cam2", default="2", help="cam2 project index. Default: 2")
    parser.add_argument(
        "--cam1-source",
        default=None,
        help="explicit cam1 device path, for example /dev/video0",
    )
    parser.add_argument(
        "--cam2-source",
        default=None,
        help="explicit cam2 device path, for example /dev/video1",
    )
    parser.add_argument(
        "--cam1-size",
        type=parse_size,
        default=(320, 240),
        help="cam1 size as WIDTHxHEIGHT. Default: 320x240",
    )
    parser.add_argument(
        "--cam2-size",
        type=parse_size,
        default=(640, 480),
        help="cam2 size as WIDTHxHEIGHT. Default: 640x480",
    )
    parser.add_argument(
        "--single",
        choices=("cam1", "cam2"),
        default=None,
        help="stream only one camera",
    )
    parser.add_argument("--port", type=int, default=5000, help="stream port")
    parser.add_argument("--fps", type=int, default=30, help="stream fps")
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="print /dev/cam* and /dev/video* devices and exit",
    )
    parser.add_argument(
        "--loop-delay",
        type=float,
        default=0.033,
        help="camera read loop delay in seconds",
    )
    parser.add_argument(
        "--camera-servo",
        action="store_true",
        help="enable ServoPwm(1, 180) camera angle toggle from collect_control.py",
    )
    parser.add_argument(
        "--camera-servo-port",
        type=int,
        default=1,
        help="PWM servo port for camera angle toggle. Default: 1",
    )
    parser.add_argument(
        "--camera-servo-angles",
        type=parse_angle_list,
        default=[-42, 165],
        help="comma-separated toggle angles. Default: -42,165",
    )
    parser.add_argument(
        "--camera-servo-speed",
        type=int,
        default=100,
        help="PWM servo speed. Default: 100",
    )
    return parser.parse_args(argv)


def _project_camera_sort_key(path):
    name = os.path.basename(path)
    try:
        return int(name.replace("cam", ""))
    except ValueError:
        return 9999


def list_project_camera_devices(device_lister=None):
    if device_lister is None:
        device_lister = lambda: glob.glob("/dev/cam*")
    return sorted(
        [
            path
            for path in device_lister()
            if os.path.basename(path).startswith("cam")
            and os.path.basename(path)[3:].isdigit()
        ],
        key=_project_camera_sort_key,
    )


def auto_camera_index(position, device_lister=None):
    cameras = list_project_camera_devices(device_lister)
    if position < 1 or len(cameras) < position:
        raise RuntimeError(
            "auto camera {} requested, but only found: {}".format(
                position, ", ".join(cameras) or "none"
            )
        )
    return os.path.basename(cameras[position - 1]).replace("cam", "")


def resolve_camera_spec_index(value, position, device_lister=None):
    if str(value).lower() == "auto":
        return auto_camera_index(position, device_lister)
    return str(value)


def build_camera_specs(args, device_lister=None):
    cam1_width, cam1_height = args.cam1_size
    cam2_width, cam2_height = args.cam2_size
    cam1_index = args.cam1_source or resolve_camera_spec_index(args.cam1, 1, device_lister)
    cam2_index = args.cam2_source or resolve_camera_spec_index(args.cam2, 2, device_lister)
    specs = [
        CameraSpec("cam1", cam1_index, cam1_width, cam1_height),
        CameraSpec("cam2", cam2_index, cam2_width, cam2_height),
    ]
    if args.single:
        specs = [spec for spec in specs if spec.name == args.single]
    return specs


def resolve_camera_source(source):
    if source.startswith("/dev/"):
        return source
    return "/dev/cam{}".format(source)


def list_camera_devices():
    devices = sorted(set(glob.glob("/dev/cam*") + glob.glob("/dev/video*")))
    if not devices:
        print("No /dev/cam* or /dev/video* devices found.")
        return []
    for device in devices:
        target = os.path.realpath(device)
        if target != device:
            print(f"{device} -> {target}")
        else:
            print(device)
    return devices


class DirectCamera:
    def __init__(self, source, width, height):
        import cv2

        self.source = resolve_camera_source(str(source))
        self.width = width
        self.height = height
        self.cap = cv2.VideoCapture(self.source)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        if not self.cap.isOpened():
            raise RuntimeError(f"camera source cannot be opened: {self.source}")
        print(f"Opened {self.source} at {self.width}x{self.height}")

    def read(self):
        ok, frame = self.cap.read()
        if not ok:
            print(f"read failed: {self.source}")
            return None
        return frame

    def close(self):
        self.cap.release()


class CameraDebugRunner:
    def __init__(
        self,
        camera_factory,
        streamer_factory,
        specs,
        loop_delay=0.033,
        servo_controller=None,
    ):
        self.camera_factory = camera_factory
        self.streamer_factory = streamer_factory
        self.specs = list(specs)
        self.loop_delay = loop_delay
        self.servo_controller = servo_controller
        self.cameras = {}
        self.streamer = None
        self.running = False
        self.input_thread = None

    def start(self):
        self.streamer = self.streamer_factory()
        for spec in self.specs:
            self.cameras[spec.name] = self.camera_factory(
                spec.index, spec.width, spec.height
            )
        if self.servo_controller is not None:
            angle = self.servo_controller.initialize()
            print(f"Camera servo initialized to {angle}. Type 't' + Enter to toggle.")
        self.running = True
        self.start_input_thread()
        print("Camera debug stream started.")
        print("Open http://<Orin IP>:{} in a browser.".format(self.streamer.port))
        print("Streaming: {}".format(", ".join(spec.name for spec in self.specs)))
        print("Type 'q' + Enter to quit.")

    def start_input_thread(self):
        self.input_thread = threading.Thread(target=self.read_commands, args=())
        self.input_thread.daemon = True
        self.input_thread.start()

    def read_commands(self):
        while self.running:
            line = sys.stdin.readline()
            if not line:
                time.sleep(0.1)
                continue
            command = line.strip().lower()
            if command in ("q", "quit", "exit"):
                self.running = False
                break
            if command in ("t", "toggle"):
                if self.servo_controller is None:
                    print("camera servo is disabled")
                    continue
                angle = self.servo_controller.toggle()
                print(f"Camera servo angle {angle}")

    def run_forever(self):
        self.start()
        try:
            while self.running:
                for name, camera in self.cameras.items():
                    frame = camera.read()
                    if frame is not None:
                        self.streamer.update_frame(frame, name)
                time.sleep(self.loop_delay)
        finally:
            self.close()

    def close(self):
        self.running = False
        for camera in self.cameras.values():
            try:
                camera.close()
            except Exception as exc:
                print(f"camera close error: {exc}")
        if self.streamer is not None:
            try:
                self.streamer.stop()
            except Exception as exc:
                print(f"streamer stop error: {exc}")


class CameraServoController:
    def __init__(self, servo_factory, angles=None, speed=100):
        self.servo_factory = servo_factory
        self.angles = list([-42, 165] if angles is None else angles)
        self.speed = int(speed)
        self.index = 0
        self.servo = None

    def initialize(self):
        self.servo = self.servo_factory()
        self.index = 0
        return self.set_current_angle()

    def toggle(self):
        if self.servo is None:
            self.initialize()
        else:
            self.index = (self.index + 1) % len(self.angles)
            self.set_current_angle()
        return self.angles[self.index]

    def set_current_angle(self):
        angle = self.angles[self.index]
        self.servo.set_angle(angle, self.speed)
        return angle


def main(argv=None):
    args = parse_args(argv)
    if args.list_devices:
        list_camera_devices()
        return

    from smartcar.whalesbot.tools.streamer import Streamer
    from smartcar.whalesbot.vehicle import ServoPwm

    specs = build_camera_specs(args)
    servo_controller = None
    if args.camera_servo:
        servo_controller = CameraServoController(
            servo_factory=lambda: ServoPwm(args.camera_servo_port, 180),
            angles=args.camera_servo_angles,
            speed=args.camera_servo_speed,
        )
    runner = CameraDebugRunner(
        camera_factory=lambda source, width, height: DirectCamera(source, width, height),
        streamer_factory=lambda: Streamer(port=args.port, fps=args.fps),
        specs=specs,
        loop_delay=args.loop_delay,
        servo_controller=servo_controller,
    )
    try:
        runner.run_forever()
    except KeyboardInterrupt:
        print("\nCamera debug stopped.")


if __name__ == "__main__":
    main()
