#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""Bluetooth gamepad control for chassis-only testing."""

import argparse
import math
import time


DISCONNECTED_PAD_VALUE = [-1, -1, -1, -1, 0]
EXIT_BUTTON_MASK = (1 << 14) | (1 << 15)
DEFAULT_X_SCALE = 0.15
DEFAULT_Y_SCALE = 0.15
DEFAULT_YAW_SCALE = 0.25
DEFAULT_LOOP_DELAY = 0.05


def is_disconnected_pad(keys_val):
    return list(keys_val) == DISCONNECTED_PAD_VALUE


def should_exit(keys_val):
    return len(keys_val) >= 5 and keys_val[4] == EXIT_BUTTON_MASK


def pad_to_velocity(
    keys_val,
    x_scale=DEFAULT_X_SCALE,
    y_scale=DEFAULT_Y_SCALE,
    yaw_scale=DEFAULT_YAW_SCALE,
):
    """Map BluetoothPad.read() values to MecanumDriver.set_velocity values."""
    return (
        x_scale * keys_val[1],
        -y_scale * keys_val[0],
        -math.pi * yaw_scale * keys_val[2],
    )


class ChassisPadController:
    def __init__(
        self,
        car,
        blue_pad,
        x_scale=DEFAULT_X_SCALE,
        y_scale=DEFAULT_Y_SCALE,
        yaw_scale=DEFAULT_YAW_SCALE,
        loop_delay=DEFAULT_LOOP_DELAY,
    ):
        self.car = car
        self.blue_pad = blue_pad
        self.x_scale = x_scale
        self.y_scale = y_scale
        self.yaw_scale = yaw_scale
        self.loop_delay = loop_delay

    def stop(self):
        self.car.set_velocity(0.0, 0.0, 0.0)

    def run(self):
        print("Chassis pad control started. Press L1+L2 or Ctrl+C to exit.")
        try:
            while True:
                keys_val = self.blue_pad.read()

                if is_disconnected_pad(keys_val):
                    self.stop()
                    print("Bluetooth pad not detected; waiting...")
                    time.sleep(1.0)
                    continue

                if should_exit(keys_val):
                    print("Exit buttons detected.")
                    break

                velocity = pad_to_velocity(
                    keys_val,
                    x_scale=self.x_scale,
                    y_scale=self.y_scale,
                    yaw_scale=self.yaw_scale,
                )
                self.car.set_velocity(*velocity)
                print(
                    "pad=({:+.2f}, {:+.2f}, {:+.2f}, {:+.2f}, {}) "
                    "vel=({:+.3f}, {:+.3f}, {:+.3f})".format(
                        keys_val[0],
                        keys_val[1],
                        keys_val[2],
                        keys_val[3],
                        keys_val[4],
                        velocity[0],
                        velocity[1],
                        velocity[2],
                    )
                )
                time.sleep(self.loop_delay)
        finally:
            self.stop()
            print("Chassis stopped.")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Control only the mecanum chassis with the Bluetooth gamepad."
    )
    parser.add_argument(
        "--x-scale",
        type=float,
        default=DEFAULT_X_SCALE,
        help="Forward/backward speed scale. Default: 0.15",
    )
    parser.add_argument(
        "--y-scale",
        type=float,
        default=DEFAULT_Y_SCALE,
        help="Lateral speed scale. Default: 0.15",
    )
    parser.add_argument(
        "--yaw-scale",
        type=float,
        default=DEFAULT_YAW_SCALE,
        help="Yaw speed scale before multiplying by pi. Default: 0.25",
    )
    parser.add_argument(
        "--loop-delay",
        type=float,
        default=DEFAULT_LOOP_DELAY,
        help="Control loop delay in seconds. Default: 0.05",
    )
    return parser.parse_args()


def main():
    from smartcar.whalesbot.vehicle import BluetoothPad, MecanumDriver

    args = parse_args()
    controller = ChassisPadController(
        car=MecanumDriver(),
        blue_pad=BluetoothPad(),
        x_scale=args.x_scale,
        y_scale=args.y_scale,
        yaw_scale=args.yaw_scale,
        loop_delay=args.loop_delay,
    )
    controller.run()


if __name__ == "__main__":
    main()
