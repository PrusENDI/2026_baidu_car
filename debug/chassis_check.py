#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Small, guarded chassis checks for Jetson Orin hardware bring-up."""

import argparse
import math
import time


def build_parser():
    parser = argparse.ArgumentParser(
        description="Run small guarded Mecanum chassis checks. Commands are dry-run unless --apply is set."
    )
    parser.add_argument("--apply", action="store_true", help="Actually initialize hardware and execute the command.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("status", help="Initialize chassis and print odometry when --apply is set.")
    subparsers.add_parser("stop", help="Send a stop command when --apply is set.")
    subparsers.add_parser("reset-odometry", help="Reset odometry to zero when --apply is set.")

    move = subparsers.add_parser("move", help="Move at low speed for a short duration.")
    move.add_argument("--x", type=float, default=0.0, help="Forward/back velocity in m/s.")
    move.add_argument("--y", type=float, default=0.0, help="Left/right velocity in m/s.")
    move.add_argument("--z", type=float, default=0.0, help="Yaw velocity in rad/s.")
    move.add_argument("--seconds", type=float, default=0.3, help="Duration, max 2.0 seconds.")

    return parser


def clamp_motion(args):
    args.x = max(min(args.x, 0.15), -0.15)
    args.y = max(min(args.y, 0.15), -0.15)
    args.z = max(min(args.z, 0.5), -0.5)
    args.seconds = max(min(args.seconds, 2.0), 0.0)


def describe(args):
    if args.command == "move":
        clamp_motion(args)
        return f"move x={args.x:.3f} y={args.y:.3f} z={args.z:.3f} seconds={args.seconds:.2f}"
    return args.command


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    planned = describe(args)

    if not args.apply:
        print(f"DRY-RUN chassis command: {planned}")
        print("Add --apply to initialize hardware and execute it on the Orin.")
        return 0

    from smartcar.whalesbot.vehicle import MecanumDriver

    car = MecanumDriver()
    try:
        if args.command == "status":
            print(f"odometry={car.get_odometry().tolist()} distance={car.get_distance():.4f}")
        elif args.command == "stop":
            car.stop()
            print("chassis stopped")
        elif args.command == "reset-odometry":
            car.reset_position()
            print(f"odometry reset: {car.get_odometry().tolist()}")
        elif args.command == "move":
            print(f"executing chassis command: {planned}")
            car.set_velocity_for_duration(args.x, args.y, args.z, args.seconds)
            time.sleep(0.1)
            print(f"odometry={car.get_odometry().tolist()} distance={car.get_distance():.4f}")
        else:
            parser.error(f"unsupported command: {args.command}")
    finally:
        car.stop()
        close = getattr(car, "close", None)
        if callable(close):
            close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
