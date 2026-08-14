#!/usr/bin/env python3
"""Generate CNN-sized images and OpenCV lane diagnostics without vehicle code."""

import argparse
import json
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from smartcar.whalesbot.tools.lane_collect import LaneAnalyzerConfig, OpenCVLaneAnalyzer

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--threshold", type=int, default=175)
    p.add_argument("--segmentation", choices=("dark", "orange_boundary"), default="dark")
    p.add_argument("--work-width", type=int, default=320)
    p.add_argument("--work-height", type=int, default=240)
    return p.parse_args()


def discover_images(path: Path):
    if path.is_file():
        return [path]
    return sorted(p for p in path.rglob("*")
                  if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)


def main() -> int:
    args = parse_args()
    images = discover_images(args.input)
    if args.limit is not None:
        images = images[: max(args.limit, 0)]
    if not images:
        print(f"No supported images found under {args.input}", file=sys.stderr)
        return 2

    cnn_dir, debug_dir, binary_dir = (args.output / name for name in
                                      ("cnn_images", "cv_debug", "cv_binary"))
    for directory in (cnn_dir, debug_dir, binary_dir):
        directory.mkdir(parents=True, exist_ok=True)
    analyzer = OpenCVLaneAnalyzer(LaneAnalyzerConfig(
        work_size=(args.work_width, args.work_height), threshold=args.threshold,
        segmentation=args.segmentation))
    record_path = args.output / "opencv_lane.jsonl"
    valid_count = 0
    with record_path.open("w", encoding="utf-8") as records:
        for index, image_path in enumerate(images):
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                records.write(json.dumps({"source_path": str(image_path),
                                          "valid": False,
                                          "reason": "decode failed"},
                                         ensure_ascii=False) + "\n")
                continue
            result = analyzer.process(image)
            name = f"{index:06d}_{image_path.stem}.jpg"
            cv2.imwrite(str(cnn_dir / name), analyzer.make_cnn_image(image))
            cv2.imwrite(str(debug_dir / name), analyzer.draw_debug(image, result))
            if result.binary_mask is not None:
                cv2.imwrite(str(binary_dir / name), result.binary_mask)
            records.write(json.dumps({
                "source_path": str(image_path),
                "cnn_img_path": str((cnn_dir / name).relative_to(args.output)),
                "debug_img_path": str((debug_dir / name).relative_to(args.output)),
                **result.to_record(),
            }, ensure_ascii=False) + "\n")
            valid_count += int(result.valid)
    print(f"Analyzed {len(images)} images: {valid_count} valid, "
          f"{len(images) - valid_count} invalid")
    print(f"Results: {record_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
