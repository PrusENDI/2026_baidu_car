#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""Start collect data without importing smartcar package root.

This launcher keeps the original collect_data.py and collect_control.py intact,
but avoids smartcar/__init__.py so Paddle/sklearn are not loaded before the
camera and hardware startup path.
"""

import argparse
import importlib.util
import os
import pathlib
import socket
import sys
import types


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


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Safe launcher for collect_data.py without smartcar root imports."
    )
    parser.add_argument("--cam1", default="1", help="project camera index for cam1")
    parser.add_argument("--cam2", default="2", help="project camera index for cam2")
    parser.add_argument("--cam1-size", type=parse_size, default=(320, 240))
    parser.add_argument("--cam2-size", type=parse_size, default=(640, 480))
    parser.add_argument("--dir1", default="dataset/image_set_lane")
    parser.add_argument("--dir2", default="dataset/image_set_object")
    parser.add_argument("--port", type=int, default=5000, help="streamer port to check")
    parser.add_argument("--skip-preflight", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args(argv)


def project_root_from_file():
    return pathlib.Path(__file__).resolve().parents[3]


def install_lightweight_packages(project_root):
    project_root = pathlib.Path(project_root).resolve()
    packages = {
        "smartcar": project_root / "smartcar",
        "smartcar.whalesbot": project_root / "smartcar" / "whalesbot",
        "smartcar.whalesbot.tools": project_root / "smartcar" / "whalesbot" / "tools",
    }
    for name, path in packages.items():
        module = sys.modules.get(name)
        if module is None:
            module = types.ModuleType(name)
            sys.modules[name] = module
        module.__path__ = [str(path)]
        module.__package__ = name


def load_module(module_name, file_path, package=None):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    if package is not None:
        module.__package__ = package
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def install_tools_logger(project_root):
    tools_dir = pathlib.Path(project_root) / "smartcar" / "whalesbot" / "tools"
    tools_package = sys.modules["smartcar.whalesbot.tools"]
    log_module = load_module(
        "smartcar.whalesbot.tools.log_wrap",
        tools_dir / "log_wrap.py",
        package="smartcar.whalesbot.tools",
    )
    tools_class_module = load_module(
        "smartcar.whalesbot.tools.tools_class",
        tools_dir / "tools_class.py",
        package="smartcar.whalesbot.tools",
    )
    tools_package.logger = log_module.logger
    tools_package.PID = tools_class_module.PID
    tools_package.CountRecord = tools_class_module.CountRecord
    tools_package.get_yaml = tools_class_module.get_yaml
    tools_package.IndexWrap = tools_class_module.IndexWrap
    tools_package.limit_val = tools_class_module.limit_val
    return log_module.logger


def camera_source(index):
    text = str(index)
    if text.startswith("/dev/"):
        return text
    return "/dev/cam{}".format(text)


def is_port_listening(port, host="127.0.0.1", timeout=0.2):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(timeout)
    try:
        return sock.connect_ex((host, int(port))) == 0
    finally:
        sock.close()


def preflight(cam_sources, path_exists=os.path.exists, port_checker=is_port_listening, ports=None):
    errors = []
    for source in cam_sources:
        if not path_exists(source):
            errors.append("missing camera device: {}".format(source))
    for port in ports or []:
        if port_checker(port):
            errors.append("streamer port already in use: {}".format(port))
    return errors


def main(argv=None):
    args = parse_args(argv)
    project_root = project_root_from_file()
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    install_lightweight_packages(project_root)
    install_tools_logger(project_root)
    tools_dir = project_root / "smartcar" / "whalesbot" / "tools"
    camera_module = load_module(
        "smartcar.whalesbot.tools.camera",
        tools_dir / "camera.py",
        package="smartcar.whalesbot.tools",
    )

    cam_sources = [
        camera_module.resolve_camera_path(args.cam1),
        camera_module.resolve_camera_path(args.cam2),
    ]
    if not args.skip_preflight:
        errors = preflight(cam_sources, ports=[args.port])
        if errors:
            for error in errors:
                print("preflight error:", error)
            raise SystemExit(2)
    if args.preflight_only:
        print("preflight ok")
        return

    collect_module = load_module(
        "smartcar.whalesbot.tools.collect_control",
        tools_dir / "collect_control.py",
        package="smartcar.whalesbot.tools",
    )

    cam1 = camera_module.Camera(args.cam1, *args.cam1_size)
    cam2 = camera_module.Camera(args.cam2, *args.cam2_size)
    collect_module.CollectControlCar(
        cap1=cam1,
        cap2=cam2,
        dir1=args.dir1,
        dir2=args.dir2,
    )


if __name__ == "__main__":
    main()
