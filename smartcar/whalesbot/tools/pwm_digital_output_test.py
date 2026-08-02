#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Manually toggle an MC602 PWM connector as a digital output."""

import argparse


# MC602 firmware maps the board PWM connectors D1-D7 to internal P7-P13.
# This diagnostic intentionally exposes only D3-D7, matching the available
# connectors for the current hardware test.
PWM_DIGITAL_PORTS = {
    "D3": 9,
    "D4": 10,
    "D5": 11,
    "D6": 12,
    "D7": 13,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Toggle MC602 PWM connector D3-D7 between low and high."
    )
    parser.add_argument(
        "--port",
        type=str.upper,
        choices=tuple(PWM_DIGITAL_PORTS),
        default="D4",
        help="physical PWM connector to test (default: D4)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    internal_port = PWM_DIGITAL_PORTS[args.port]

    # Importing the hardware stack searches for and connects to the controller,
    # so keep it out of module import time.
    from smartcar.whalesbot.vehicle import PoutD

    output = PoutD(internal_port)
    state = 0

    print(
        f"Testing PWM connector {args.port} as digital output "
        f"(internal P{internal_port})."
    )
    print("Measure the signal pin relative to GND. Do not connect a load yet.")

    output.set(0)
    print(f"{args.port}: LOW")

    try:
        while True:
            command = input("Press Enter to toggle, or type q then Enter to quit: ")
            if command.strip().lower() in {"q", "quit", "exit"}:
                break
            if command.strip():
                print("Unknown command. Press Enter to toggle or type q to quit.")
                continue

            state = 1 - state
            output.set(state)
            print(f"{args.port}: {'HIGH' if state else 'LOW'}")
    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        output.set(0)
        print(f"{args.port}: LOW (safe shutdown)")


if __name__ == "__main__":
    main()
