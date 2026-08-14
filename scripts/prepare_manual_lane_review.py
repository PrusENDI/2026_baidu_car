#!/usr/bin/env python3
"""Create side-by-side OpenCV lane overlays for quick human review."""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from smartcar.whalesbot.tools.lane_collect import (
    LaneAnalyzerConfig,
    OpenCVLaneAnalyzer,
    SharpTurnStateMachine,
)


SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--stride", type=int, default=200)
    parser.add_argument("--thresholds", default="165,175")
    parser.add_argument("--top-rois", default="0.00,0.15,0.30")
    parser.add_argument("--bottom-rois", default="0.00")
    parser.add_argument("--start-frame", type=int)
    parser.add_argument("--end-frame", type=int)
    parser.add_argument("--columns", type=int, default=2)
    parser.add_argument("--turn-state", action="store_true")
    return parser.parse_args()


def comma_values(value, converter):
    return [converter(item.strip()) for item in value.split(",") if item.strip()]


def image_files(path):
    if path.is_file():
        return [path]
    return sorted(
        item for item in path.rglob("*")
        if item.is_file() and item.suffix.lower() in SUFFIXES
    )


def add_panel_label(image, label, status=None):
    header = 46 if status else 26
    panel = np.zeros((image.shape[0] + header, image.shape[1], 3), dtype=np.uint8)
    panel[header:] = image
    cv2.putText(panel, label, (7, 18), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (255, 255, 255), 1, cv2.LINE_AA)
    if status:
        cv2.putText(panel, status, (7, 39), cv2.FONT_HERSHEY_SIMPLEX,
                    0.43, (0, 255, 255), 1, cv2.LINE_AA)
    return panel


def write_jpeg(path, image):
    """Write through NumPy so non-ASCII Windows paths work reliably."""
    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 92])
    if not ok:
        raise RuntimeError(f"Could not encode review sheet: {path}")
    encoded.tofile(str(path))


def main():
    args = parse_args()
    files = image_files(args.input)
    if args.start_frame is not None:
        files = [path for path in files if path.stem.isdigit()
                 and int(path.stem) >= args.start_frame]
    if args.end_frame is not None:
        files = [path for path in files if path.stem.isdigit()
                 and int(path.stem) <= args.end_frame]
    if not files:
        print("No images found", file=sys.stderr)
        return 2

    args.output.mkdir(parents=True, exist_ok=True)
    thresholds = comma_values(args.thresholds, int)
    top_rois = comma_values(args.top_rois, float)
    bottom_rois = comma_values(args.bottom_rois, float)
    configurations = [
        (threshold, top_roi, bottom_roi)
        for bottom_roi in bottom_rois
        for top_roi in top_rois
        for threshold in thresholds
    ]
    analyzers = {
        config: OpenCVLaneAnalyzer(LaneAnalyzerConfig(
            threshold=threshold,
            segmentation="dark",
            roi_top_ratio=top_roi,
            roi_bottom_ratio=bottom_roi,
            work_size=(320, 240),
        ))
        for config in configurations
        for threshold, top_roi, bottom_roi in [config]
    }
    turn_machines = ({config: SharpTurnStateMachine(
        error_mapping=analyzers[config].config.error_mapping)
                      for config in configurations}
                     if args.turn_state else {})

    stride = max(1, args.stride)
    selected = list(range(0, len(files), stride))
    if selected[-1] != len(files) - 1:
        selected.append(len(files) - 1)

    for review_number, index in enumerate(selected, start=1):
        path = files[index]
        source = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if source is None:
            continue
        panels = []
        for threshold, top_roi, bottom_roi in configurations:
            analyzer = analyzers[(threshold, top_roi, bottom_roi)]
            result = analyzer.process(source)
            debug = analyzer.draw_debug(source, result)
            status = None
            if args.turn_state:
                command = turn_machines[(threshold, top_roi, bottom_roi)].update(result)
                heading = ("none" if command.raw_heading is None
                           else f"{command.raw_heading:+.3f}")
                status = (f"state={command.state} cmd_head={heading} "
                          f"source={command.source}")
            panels.append(add_panel_label(
                debug, (f"T={threshold} top={int(top_roi * 100)}% "
                        f"bottom={int(bottom_roi * 100)}%"), status))
        columns = max(1, args.columns)
        blank = np.zeros_like(panels[0])
        while len(panels) % columns:
            panels.append(blank)
        rows = [cv2.hconcat(panels[offset:offset + columns])
                for offset in range(0, len(panels), columns)]
        sheet = cv2.vconcat(rows)
        write_jpeg(args.output / (
            f"review_{review_number:03d}_index_{index:05d}_"
            f"{path.parent.name}_{path.stem}.jpg"), sheet)

    print(f"Created {len(selected)} review sheets in {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
