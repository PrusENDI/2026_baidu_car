#!/usr/bin/env python3
"""Prepare threshold/ROI comparison artifacts for human CV-lane review."""

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

SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--thresholds", default="160,165,170,175,180")
    p.add_argument("--roi-top-ratios", default="0.00,0.08,0.15")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--sample-stride", type=int, default=500)
    return p.parse_args()


def parse_floats(value):
    return [float(x.strip()) for x in value.split(",") if x.strip()]


def images(path):
    if path.is_file():
        return [path]
    return sorted(x for x in path.rglob("*")
                  if x.is_file() and x.suffix.lower() in SUFFIXES)


def quantile(values, q):
    return float(np.quantile(np.asarray(values, dtype=np.float32), q)) if values else None


def config_name(threshold, roi):
    return f"t{threshold}_roi{roi:.2f}".replace(".", "p")


def main():
    a = args()
    files = images(a.input)
    if a.limit is not None:
        files = files[:max(a.limit, 0)]
    thresholds = [int(x) for x in parse_floats(a.thresholds)]
    rois = parse_floats(a.roi_top_ratios)
    if not files or not thresholds or not rois:
        print("Input images, thresholds, and ROI values are required", file=sys.stderr)
        return 2

    a.output.mkdir(parents=True, exist_ok=True)
    summary = []
    rows = []
    for threshold in thresholds:
        for roi in rois:
            name = config_name(threshold, roi)
            sample_dir = a.output / "samples" / name
            sample_dir.mkdir(parents=True, exist_ok=True)
            analyzer = OpenCVLaneAnalyzer(LaneAnalyzerConfig(
                threshold=threshold, segmentation="dark", roi_top_ratio=roi,
                work_size=(320, 240)))
            valid = 0
            lateral, heading, confidence, widths = [], [], [], []
            selected = []
            for index, path in enumerate(files):
                image = cv2.imread(str(path), cv2.IMREAD_COLOR)
                result = analyzer.process(image) if image is not None else None
                if result is None:
                    record = {"index": index, "path": str(path), "valid": False,
                              "reason": "decode failed"}
                else:
                    record = {"index": index, "path": str(path), **result.to_record()}
                    if result.valid:
                        valid += 1
                        lateral.append(result.raw_lateral)
                        heading.append(result.raw_heading)
                        confidence.append(result.confidence)
                        widths.append(result.metrics.get("median_lane_width"))
                    if index % max(a.sample_stride, 1) == 0:
                        selected.append((index, path, image, result))
                rows.append({"config": name, **record})

            for index, path, image, result in selected:
                debug = analyzer.draw_debug(image, result)
                cv2.imwrite(str(sample_dir / f"{index:06d}_{path.stem}.jpg"), debug)
            summary.append({
                "config": name,
                "threshold": threshold,
                "roi_top_ratio": roi,
                "image_count": len(files),
                "valid_count": valid,
                "valid_rate": valid / float(len(files)),
                "lateral_q05": quantile(lateral, .05),
                "lateral_median": quantile(lateral, .50),
                "lateral_q95": quantile(lateral, .95),
                "heading_q05": quantile(heading, .05),
                "heading_median": quantile(heading, .50),
                "heading_q95": quantile(heading, .95),
                "confidence_median": quantile(confidence, .50),
                "lane_width_median": quantile(widths, .50),
                "lane_width_q95": quantile(widths, .95),
                "sample_dir": str(sample_dir.relative_to(a.output)),
            })
            print(name, f"valid={valid}/{len(files)}",
                  f"width_med={quantile(widths, .5)}",
                  f"confidence_med={quantile(confidence, .5)}")

    (a.output / "review_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    with (a.output / "review_records.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=sorted({k for row in rows for k in row}))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Summary: {a.output / 'review_summary.json'}")
    print(f"Records: {a.output / 'review_records.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
