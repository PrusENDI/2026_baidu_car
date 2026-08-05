#!/usr/bin/env python3
import argparse
import json
import time

import zmq


SERVICE_PORTS = (5001, 5002, 5005)


def probe_service(port, timeout_ms=1000):
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.setsockopt(zmq.LINGER, 0)
    socket.setsockopt(zmq.SNDTIMEO, timeout_ms)
    socket.setsockopt(zmq.RCVTIMEO, timeout_ms)
    socket.connect(f"tcp://127.0.0.1:{port}")
    try:
        socket.send(b"ATATA")
        return json.loads(socket.recv().decode("utf-8")) is True
    except (zmq.ZMQError, UnicodeDecodeError, json.JSONDecodeError):
        return False
    finally:
        socket.close(0)
        context.term()


def all_services_ready(probe=probe_service, timeout_ms=1000):
    results = [probe(port, timeout_ms) for port in SERVICE_PORTS]
    return all(results)


def wait_until_ready(ready, sleep=time.sleep, interval=1.0):
    while not ready():
        sleep(interval)
    return True


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wait", action="store_true")
    parser.add_argument("--timeout-ms", type=int, default=1000)
    parser.add_argument("--interval", type=float, default=1.0)
    return parser.parse_args()


def main():
    args = parse_args()
    ready = lambda: all_services_ready(timeout_ms=args.timeout_ms)
    if args.wait:
        wait_until_ready(ready=ready, interval=args.interval)
        return 0
    return 0 if ready() else 1


if __name__ == "__main__":
    raise SystemExit(main())
