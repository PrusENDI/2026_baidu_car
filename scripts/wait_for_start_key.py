#!/usr/bin/env python3
import argparse
import time


START_KEY_VALUE = 12
START_KEY_PORT = 5


def wait_for_start(key, display, sleep=time.sleep, poll_interval=0.1):
    display.show("wait to start")
    while True:
        try:
            value = int(key.get_key())
        except (TypeError, ValueError):
            value = 0
        if value == START_KEY_VALUE:
            display.show("started!!!")
            sleep(0.3)
            return
        sleep(poll_interval)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--show-loading", action="store_true")
    return parser.parse_args()


def main():
    from smartcar.whalesbot.vehicle import Key4Btn, ScreenShow

    args = parse_args()
    display = ScreenShow()
    if args.show_loading:
        display.show("loading inference")
        return 0
    wait_for_start(Key4Btn(START_KEY_PORT), display)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
