#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""Interactive PWM and RS485 bus servo calibration tool."""

import argparse
import time


DEFAULT_SPEED = 100
DEFAULT_HOME_PWM = {1: -42, 2: -37}
DEFAULT_HOME_BUS = {2: 0}


def parse_home_list(value):
    result = {}
    if not value:
        return result
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"expected PORT:ANGLE, got {item!r}")
        port_text, angle_text = item.split(":", 1)
        result[int(port_text)] = int(angle_text)
    return result


class ServoCalibrationSession:
    def __init__(
        self,
        pwm_factory,
        bus_factory,
        home_pwm=None,
        home_bus=None,
        default_speed=DEFAULT_SPEED,
    ):
        self.pwm_factory = pwm_factory
        self.bus_factory = bus_factory
        self.home_pwm = dict(DEFAULT_HOME_PWM if home_pwm is None else home_pwm)
        self.home_bus = dict(DEFAULT_HOME_BUS if home_bus is None else home_bus)
        self.default_speed = int(default_speed)
        self.speed = int(default_speed)
        self.current_kind = None
        self.current_port = None
        self.current_angle = None
        self._pwm = {}
        self._bus = {}

    def handle_line(self, line):
        parts = line.strip().split()
        if not parts:
            return ""
        cmd = parts[0].lower()
        args = parts[1:]

        if cmd in ("quit", "exit", "q"):
            return "quit"
        if cmd == "help":
            return self.help_text()
        if cmd in ("pwm", "bus"):
            return self.select(cmd, args)
        if cmd == "speed":
            return self.set_speed(args)
        if cmd == "angle":
            return self.set_angle_arg(args)
        if cmd == "step":
            return self.step(args)
        if cmd == "center":
            return self.center()
        if cmd == "home-all":
            return self.home_all()
        if cmd == "sweep":
            return self.sweep(args)
        return f"unknown command: {cmd}"

    def select(self, kind, args):
        if len(args) != 1:
            return f"usage: {kind} PORT"
        self.current_kind = kind
        self.current_port = int(args[0])
        self.current_angle = self.home_angle(kind, self.current_port)
        self.servo(kind, self.current_port)
        return f"selected {kind} {self.current_port}"

    def set_speed(self, args):
        if len(args) != 1:
            return "usage: speed VALUE"
        self.speed = int(args[0])
        if self.current_kind == "bus" and self.current_port is not None:
            self.servo("bus", self.current_port).set_speed(self.speed)
            return f"bus {self.current_port} speed {self.speed}"
        return f"default speed {self.speed}"

    def set_angle_arg(self, args):
        if len(args) != 1:
            return "usage: angle VALUE"
        return self.set_current_angle(int(args[0]))

    def step(self, args):
        if len(args) != 1:
            return "usage: step DELTA"
        if self.current_angle is None:
            self.current_angle = 0
        return self.set_current_angle(self.current_angle + int(args[0]))

    def center(self):
        self.require_current()
        return self.set_current_angle(
            self.home_angle(self.current_kind, self.current_port)
        )

    def home_all(self):
        count = 0
        for port, angle in sorted(self.home_pwm.items()):
            self.servo("pwm", port).set_angle(angle, self.speed)
            count += 1
        for port, angle in sorted(self.home_bus.items()):
            self.servo("bus", port).set_angle(angle, self.speed)
            count += 1
        return f"homed {count} servos"

    def sweep(self, args):
        if len(args) != 3:
            return "usage: sweep START END STEP"
        self.require_current()
        start, end, step = [int(arg) for arg in args]
        if step == 0:
            return "sweep step cannot be 0"

        direction = 1 if end >= start else -1
        step = abs(step) * direction
        angle = start
        while (direction > 0 and angle <= end) or (direction < 0 and angle >= end):
            self.set_current_angle(angle)
            time.sleep(0.2)
            angle += step
        return f"swept {start} to {end}"

    def set_current_angle(self, angle):
        self.require_current()
        self.current_angle = int(angle)
        self.servo(self.current_kind, self.current_port).set_angle(
            self.current_angle, self.speed
        )
        return f"{self.current_kind} {self.current_port} angle {self.current_angle} speed {self.speed}"

    def home_angle(self, kind, port):
        homes = self.home_pwm if kind == "pwm" else self.home_bus
        return homes.get(port, 90 if kind == "pwm" else 0)

    def servo(self, kind, port):
        store = self._pwm if kind == "pwm" else self._bus
        factory = self.pwm_factory if kind == "pwm" else self.bus_factory
        if port not in store:
            store[port] = factory(port)
        return store[port]

    def require_current(self):
        if self.current_kind is None or self.current_port is None:
            raise ValueError("select a servo first with 'pwm PORT' or 'bus PORT'")

    def help_text(self):
        return "\n".join(
            [
                "commands:",
                "  pwm PORT",
                "  bus PORT",
                "  angle VALUE",
                "  speed VALUE",
                "  step DELTA",
                "  center",
                "  home-all",
                "  sweep START END STEP",
                "  quit",
            ]
        )


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Interactive PWM and RS485 bus servo calibration."
    )
    parser.add_argument(
        "--home-pwm",
        default="1:-42,2:-37",
        help="PWM home targets as PORT:ANGLE pairs. Default: 1:-42,2:-37",
    )
    parser.add_argument(
        "--home-bus",
        default="2:0",
        help="Bus servo home targets as PORT:ANGLE pairs. Default: 2:0",
    )
    parser.add_argument(
        "--speed",
        type=int,
        default=DEFAULT_SPEED,
        help="Default servo speed. Default: 100",
    )
    return parser


def main():
    from smartcar.whalesbot.vehicle import ServoBus, ServoPwm

    args = build_arg_parser().parse_args()
    session = ServoCalibrationSession(
        pwm_factory=lambda port: ServoPwm(port, 180),
        bus_factory=lambda port: ServoBus(port),
        home_pwm=parse_home_list(args.home_pwm),
        home_bus=parse_home_list(args.home_bus),
        default_speed=args.speed,
    )

    print("Servo calibration control. Type 'help' for commands.")
    while True:
        try:
            result = session.handle_line(input("servo> "))
        except (KeyboardInterrupt, EOFError):
            print("\nquit")
            break
        except Exception as exc:
            print(f"error: {exc}")
            continue

        if result == "quit":
            break
        if result:
            print(result)


if __name__ == "__main__":
    main()
