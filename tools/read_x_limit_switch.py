#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""Read a COM+NC X limit switch from an MC602 dedicated AI port."""

import argparse
from datetime import datetime
import math
from pathlib import Path
import sys
import time


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_port(value):
    text = value.strip().lower()
    if text.startswith("ai"):
        text = text[2:]
    try:
        port = int(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"port must look like 1 or AI1, got {value!r}"
        ) from exc
    if not 1 <= port <= 3:
        raise argparse.ArgumentTypeError("dedicated analog port must be AI1, AI2, or AI3")
    return port


def positive_float(value):
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("value must be a finite number greater than 0")
    return number


def positive_int(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("value must be an integer greater than 0")
    return number


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Read an MC602 dedicated-AI AnalogInput2-connected COM+NC limit switch. "
            "This script never creates or commands a motor."
        )
    )
    parser.add_argument(
        "--port",
        required=True,
        type=parse_port,
        help="MC602 dedicated analog port: AI1, AI2, or AI3",
    )
    parser.add_argument(
        "--interval",
        type=positive_float,
        default=0.1,
        help="sample interval in seconds (default: 0.1)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        help="trigger threshold; omit during initial raw-value collection",
    )
    parser.add_argument(
        "--active",
        choices=("above", "below"),
        help="whether values above or below the threshold mean triggered",
    )
    parser.add_argument(
        "--stable-samples",
        type=positive_int,
        default=5,
        help="consecutive triggered samples required for stable=true (default: 5)",
    )
    return parser


def parse_args(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if (args.threshold is None) != (args.active is None):
        parser.error("--threshold and --active must be provided together")
    if args.threshold is not None and not math.isfinite(args.threshold):
        parser.error("--threshold must be finite")
    return args


def create_sensor(port):
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    from smartcar.whalesbot.vehicle.base import controller_wrap

    if controller_wrap.ctl_id != 1:
        device_name = getattr(controller_wrap.serial_wrap.dev, "name", "unknown")
        raise RuntimeError(
            f"MC602 required, detected controller={device_name!r}"
        )
    return controller_wrap.AnalogInput2(port)


def read_fresh(sensor):
    mc602_sensor = sensor.sensor_2
    mc602_sensor.last_data = None
    raw = float(sensor.read())
    if mc602_sensor.last_data is None:
        raise RuntimeError("MC602 returned no fresh AnalogInput sample")
    if not math.isfinite(raw):
        raise RuntimeError(f"non-finite AnalogInput value: {raw!r}")
    return raw


def classify(raw, threshold, active):
    triggered = raw >= threshold if active == "above" else raw <= threshold
    return ("TRIGGERED" if triggered else "RELEASED"), triggered


def run(args, sensor):
    print(
        "[X_LIMIT_READER] READ-ONLY sensor diagnostic; "
        "no motor or servo commands are sent.",
        flush=True,
    )
    print(
        f"[X_LIMIT_READER] wiring=COM+NC port=AI{args.port} "
        f"interval={args.interval}s threshold={args.threshold} "
        f"active={args.active} stable_samples={args.stable_samples}",
        flush=True,
    )
    if args.threshold is None:
        print(
            "[X_LIMIT_READER] state remains UNCONFIGURED until both "
            "--threshold and --active are supplied.",
            flush=True,
        )

    consecutive = 0
    try:
        while True:
            timestamp = datetime.now().isoformat(timespec="milliseconds")
            try:
                raw = read_fresh(sensor)
                if args.threshold is None:
                    state = "UNCONFIGURED"
                    consecutive = 0
                    stable = False
                else:
                    state, triggered = classify(raw, args.threshold, args.active)
                    consecutive = consecutive + 1 if triggered else 0
                    stable = consecutive >= args.stable_samples
                print(
                    f"{timestamp} port=AI{args.port} raw={raw:.1f} "
                    f"state={state} consecutive={consecutive} stable={stable}",
                    flush=True,
                )
            except Exception as exc:
                consecutive = 0
                print(
                    f"{timestamp} port=AI{args.port} raw=None state=ERROR "
                    f"consecutive=0 stable=False "
                    f"error={type(exc).__name__}:{exc}",
                    file=sys.stderr,
                    flush=True,
                )
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\n[X_LIMIT_READER] stopped by user; no motor was commanded.", flush=True)
    return 0


def main(argv=None):
    args = parse_args(argv)
    try:
        sensor = create_sensor(args.port)
    except Exception as exc:
        print(
            f"[X_LIMIT_READER] startup error: {type(exc).__name__}: {exc}",
            file=sys.stderr,
            flush=True,
        )
        return 2
    return run(args, sensor)


if __name__ == "__main__":
    raise SystemExit(main())
