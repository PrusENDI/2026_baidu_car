#!/usr/bin/env python3
"""Render detailed review sheets for invalid OpenCV lane-analysis frames."""

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from smartcar.whalesbot.tools.lane_collect import LaneAnalyzerConfig, OpenCVLaneAnalyzer


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lap-dir", required=True, type=Path)
    parser.add_argument("--comparison-csv", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--threshold", type=int, default=175)
    parser.add_argument("--top-roi", type=float, default=0.30)
    parser.add_argument("--bottom-roi", type=float, default=0.20)
    return parser.parse_args()


def label(image, text, color=(255, 255, 255)):
    result = image.copy()
    cv2.rectangle(result, (0, 0), (result.shape[1] - 1, 24), (20, 20, 20), -1)
    cv2.putText(result, text, (6, 17), cv2.FONT_HERSHEY_SIMPLEX,
                0.48, color, 1, cv2.LINE_AA)
    return result


def mask_panel(mask, title):
    if mask is None:
        panel = np.zeros((240, 320, 3), dtype=np.uint8)
    else:
        panel = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    return label(panel, title)


def contiguous_ranges(frames):
    if not frames:
        return []
    groups = [[frames[0]]]
    for frame in frames[1:]:
        if frame == groups[-1][-1] + 1:
            groups[-1].append(frame)
        else:
            groups.append([frame])
    return [{"start": group[0], "end": group[-1], "count": len(group)}
            for group in groups]


def main():
    args = parse_args()
    with args.comparison_csv.open(newline="", encoding="utf-8-sig") as file:
        source_rows = list(csv.DictReader(file))
    invalid_rows = [row for row in source_rows if row["cv_valid"].lower() == "false"]
    analyzer = OpenCVLaneAnalyzer(LaneAnalyzerConfig(
        threshold=args.threshold, segmentation="dark", work_size=(320, 240),
        roi_top_ratio=args.top_roi, roi_bottom_ratio=args.bottom_roi))

    frames_dir = args.output / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    details = []
    for row in invalid_rows:
        frame = int(row["frame"])
        image_path = args.lap_dir / row["img_path"]
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            details.append({"frame": frame, "reason": "decode failed"})
            continue
        result = analyzer.process(image)
        center_rows = (np.flatnonzero(np.isfinite(result.center_line))
                       if result.center_line is not None else np.empty(0, dtype=int))
        partial_heading = None
        if center_rows.size >= 2:
            forward = (result.work_size[1] - 1 - center_rows).astype(np.float64)
            slope, _ = np.polyfit(
                forward, result.center_line[center_rows].astype(np.float64), 1)
            partial_heading = float(np.arctan(slope))
        work = analyzer.make_work_image(image)
        debug = analyzer.draw_debug(image, result)
        original = label(work, f"frame {frame}  manual_yaw={float(row['manual_yaw']):+.3f}")
        binary = mask_panel(result.binary_mask, "threshold mask")
        connected = mask_panel(result.lane_mask, "seed-connected track")
        debug = label(debug, "boundaries / center / corner")
        sheet = np.vstack((np.hstack((original, binary)),
                           np.hstack((connected, debug))))
        cv2.imencode(".jpg", sheet, [cv2.IMWRITE_JPEG_QUALITY, 95])[1].tofile(
            str(frames_dir / f"{frame}.jpg"))
        details.append({
            "frame": frame,
            "manual_yaw": float(row["manual_yaw"]),
            "valid": result.valid,
            "reason": result.reason,
            "center_first_y": int(center_rows[0]) if center_rows.size else None,
            "center_last_y": int(center_rows[-1]) if center_rows.size else None,
            "partial_heading": partial_heading,
            **result.metrics,
        })

    frames = sorted(item["frame"] for item in details)
    report = {
        "invalid_count": len(details),
        "contiguous_ranges": contiguous_ranges(frames),
        "frames": details,
        "legend": {
            "blue": "observed left boundary",
            "red": "observed right boundary",
            "yellow": "estimated center",
            "magenta": "detected near-horizontal corner segment",
        },
    }
    (args.output / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report["contiguous_ranges"], ensure_ascii=False, indent=2))
    print(f"Review frames: {frames_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
