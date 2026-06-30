#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Quick USB camera burst capture for weak-class dataset collection."""

from __future__ import annotations

import argparse
import csv
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


TARGET_CLASSES = (
    "ball_blue",
    "ball_yellow",
    "water_l1",
    "water_l2",
    "water_l3",
    "negative_blue",
    "negative_yellow",
    "negative_water",
)


@dataclass(frozen=True)
class CaptureSession:
    camera: int
    size: tuple[int, int]
    class_name: str
    session: str
    output_dir: Path
    count: int
    fps: float
    preview: bool
    warmup: int
    dry_run: bool


def parse_size(value: str) -> tuple[int, int]:
    text = value.lower().replace("*", "x")
    if "x" not in text:
        raise argparse.ArgumentTypeError("size must look like WIDTHxHEIGHT")
    width_text, height_text = text.split("x", 1)
    try:
        width = int(width_text)
        height = int(height_text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("width and height must be integers") from exc
    if width <= 0 or height <= 0:
        raise argparse.ArgumentTypeError("width and height must be positive")
    return width, height


def default_session_name() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Capture quick training images from a USB or car-mounted camera."
    )
    parser.add_argument(
        "--class-name",
        required=True,
        choices=TARGET_CLASSES,
        help="folder/label hint for this capture batch",
    )
    parser.add_argument("--camera", type=int, default=0, help="OpenCV camera index")
    parser.add_argument(
        "--size",
        type=parse_size,
        default=(640, 480),
        help="capture size as WIDTHxHEIGHT. Default: 640x480",
    )
    parser.add_argument(
        "--session",
        default=None,
        help="short scene name, for example car_side_near or desk_lamp_far",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("dataset") / "quick_capture",
        help="base output directory",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=0,
        help="auto-capture this many images. 0 means manual mode",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=1.0,
        help="auto-capture rate when --count is greater than 0",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=10,
        help="discard this many frames before saving",
    )
    parser.add_argument(
        "--no-preview",
        action="store_true",
        help="disable preview window; useful over SSH or headless runs",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the planned output directory and exit without opening camera",
    )
    return parser.parse_args(argv)


def build_session(args) -> CaptureSession:
    session_name = args.session or default_session_name()
    output_dir = args.out / args.class_name / session_name
    return CaptureSession(
        camera=args.camera,
        size=args.size,
        class_name=args.class_name,
        session=session_name,
        output_dir=output_dir,
        count=max(0, args.count),
        fps=args.fps,
        preview=not args.no_preview,
        warmup=max(0, args.warmup),
        dry_run=args.dry_run,
    )


def capture_interval(args_or_session) -> float:
    fps = float(args_or_session.fps)
    if fps <= 0:
        raise ValueError("fps must be positive")
    return 1.0 / fps


def build_image_path(output_dir: Path, class_name: str, session: str, seq: int) -> Path:
    return output_dir / f"{class_name}_{session}_{seq:06d}.jpg"


def write_manifest_row(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "file_name",
                "class_name",
                "session",
                "camera",
                "width",
                "height",
                "captured_at",
            ],
        )
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def next_sequence(output_dir: Path) -> int:
    if not output_dir.exists():
        return 1
    max_seq = 0
    for path in output_dir.glob("*.jpg"):
        stem = path.stem
        seq_text = stem.rsplit("_", 1)[-1]
        if seq_text.isdigit():
            max_seq = max(max_seq, int(seq_text))
    return max_seq + 1


def open_camera(index: int, size: tuple[int, int]):
    import cv2

    cap = cv2.VideoCapture(index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, size[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, size[1])
    if not cap.isOpened():
        raise RuntimeError(f"could not open camera index {index}")
    return cap


def save_frame(session: CaptureSession, frame, seq: int) -> Path:
    import cv2

    image_path = build_image_path(
        session.output_dir, session.class_name, session.session, seq
    )
    session.output_dir.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(image_path), frame)
    if not ok:
        raise RuntimeError(f"failed to write {image_path}")
    height, width = frame.shape[:2]
    write_manifest_row(
        session.output_dir / "manifest.csv",
        {
            "file_name": image_path.name,
            "class_name": session.class_name,
            "session": session.session,
            "camera": session.camera,
            "width": width,
            "height": height,
            "captured_at": datetime.now().isoformat(timespec="seconds"),
        },
    )
    return image_path


def run_capture(session: CaptureSession) -> None:
    if session.dry_run:
        print(f"output_dir: {session.output_dir}")
        print(f"class_name: {session.class_name}")
        print(f"session: {session.session}")
        return

    import cv2

    cap = open_camera(session.camera, session.size)
    seq = next_sequence(session.output_dir)
    saved = 0
    last_capture = 0.0
    interval = capture_interval(session)

    try:
        for _ in range(session.warmup):
            cap.read()

        print(f"Saving to: {session.output_dir}")
        if session.count:
            print(f"Auto mode: {session.count} images at {session.fps:g} fps")
        else:
            print("Manual mode: Space saves, q exits")

        while True:
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError("camera read failed")

            now = time.time()
            should_save = False
            key = -1
            if session.preview:
                cv2.imshow("usb_burst_capture", frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == ord(" "):
                    should_save = True

            if session.count and now - last_capture >= interval:
                should_save = True
                last_capture = now

            if should_save:
                image_path = save_frame(session, frame, seq)
                print(f"saved {image_path}")
                seq += 1
                saved += 1
                if session.count and saved >= session.count:
                    break

            if not session.preview and not session.count:
                raise RuntimeError("manual mode requires preview; use --count for headless capture")
    finally:
        cap.release()
        if session.preview:
            cv2.destroyAllWindows()


def main(argv=None) -> None:
    args = parse_args(argv)
    session = build_session(args)
    run_capture(session)


if __name__ == "__main__":
    main()
