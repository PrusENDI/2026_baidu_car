#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Small, guarded arm checks for Jetson Orin hardware bring-up."""

import argparse
import time


ARM_SIDES = ("LEFT", "MID", "RIGHT")
HAND_POSES = ("UP", "MID", "DOWN")


def build_parser():
    parser = argparse.ArgumentParser(
        description="Run small guarded arm checks. Commands are dry-run unless --apply is set."
    )
    parser.add_argument("--apply", action="store_true", help="Actually initialize hardware and execute the command.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("status", help="Initialize arm and print known positions when --apply is set.")
    subparsers.add_parser("reset", help="Reset arm position when --apply is set.")

    x_move = subparsers.add_parser("x", help="Move arm X axis to an absolute position.")
    x_move.add_argument("position", type=float, help="Target X position in meters, clamped to 0.00..0.315.")

    y_move = subparsers.add_parser("y", help="Move arm Y axis to an absolute position.")
    y_move.add_argument("position", type=float, help="Target Y position in meters, clamped to 0.00..0.20.")

    arm = subparsers.add_parser("arm", help="Set arm side servo.")
    arm.add_argument("side", choices=ARM_SIDES)

    hand = subparsers.add_parser("hand", help="Set hand servo.")
    hand.add_argument("pose", choices=HAND_POSES)

    grasp = subparsers.add_parser("grasp", help="Turn suction on or off.")
    grasp.add_argument("state", choices=("on", "off"))

    return parser


def describe(args):
    if args.command == "x":
        args.position = max(min(args.position, 0.315), 0.0)
        return f"x position={args.position:.4f}"
    if args.command == "y":
        args.position = max(min(args.position, 0.20), 0.0)
        return f"y position={args.position:.4f}"
    if args.command == "arm":
        return f"arm side={args.side}"
    if args.command == "hand":
        return f"hand pose={args.pose}"
    if args.command == "grasp":
        return f"grasp state={args.state}"
    return args.command


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    planned = describe(args)

    if not args.apply:
        print(f"DRY-RUN arm command: {planned}")
        print("Add --apply to initialize hardware and execute it on the Orin.")
        return 0

    from smartcar.whalesbot.vehicle import ArmController

    arm = ArmController()
    if args.command == "status":
        print(f"x={arm.x_get_position():.4f} y={arm.y_get_position():.4f} side={arm.side}")
    elif args.command == "reset":
        print("executing arm reset")
        arm.reset_position()
        print(f"x={arm.x_get_position():.4f} y={arm.y_get_position():.4f} side={arm.side}")
    elif args.command == "x":
        print(f"executing arm command: {planned}")
        arm.move_x_position(args.position)
    elif args.command == "y":
        print(f"executing arm command: {planned}")
        arm.move_y_position(args.position)
    elif args.command == "arm":
        print(f"executing arm command: {planned}")
        arm.set_arm_angle(args.side)
        time.sleep(0.5)
    elif args.command == "hand":
        print(f"executing arm command: {planned}")
        arm.set_hand_angle(args.pose)
        time.sleep(0.5)
    elif args.command == "grasp":
        print(f"executing arm command: {planned}")
        arm.grasp(args.state == "on")
        time.sleep(0.3)
    else:
        parser.error(f"unsupported command: {args.command}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
