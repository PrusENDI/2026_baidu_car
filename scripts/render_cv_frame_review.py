#!/usr/bin/env python3
"""Render every requested lap frame with CV geometry and control diagnostics."""

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from smartcar.whalesbot.tools.lane_collect import (
    CvLanePidController,
    ErrorMapping,
    LaneAnalyzerConfig,
    OpenCVLaneAnalyzer,
    PreviewTimingFilter,
    StandardLaneReference,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lap-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--start-frame", required=True, type=int)
    parser.add_argument("--end-frame", required=True, type=int)
    parser.add_argument("--standard-root", type=Path, default=ROOT / "standard")
    parser.add_argument("--temporal-filter", action="store_true")
    return parser.parse_args()


def fmt(value, digits=3):
    if value is None:
        return "none"
    try:
        if not np.isfinite(float(value)):
            return "none"
    except (TypeError, ValueError):
        return str(value)
    return f"{float(value):+.{digits}f}"


def add_header(image, lines, warning=False):
    height = 76
    panel = np.zeros((image.shape[0] + height, image.shape[1], 3), dtype=np.uint8)
    panel[height:] = image
    colors = [(255, 255, 255), (0, 255, 255),
              ((60, 60, 255) if warning else (80, 230, 80))]
    for index, line in enumerate(lines[:3]):
        cv2.putText(panel, line, (8, 20 + index * 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, colors[index], 1,
                    cv2.LINE_AA)
    return panel


def main():
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    frame_dir = args.output / "frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    reference = StandardLaneReference.from_files(
        args.standard_root / "standard_lane.json",
        args.standard_root / "perspective.json")
    mapping = ErrorMapping(lateral_scale=-0.10, heading_scale=-0.40)
    analyzer = OpenCVLaneAnalyzer(
        LaneAnalyzerConfig(error_mapping=mapping), reference)
    timing = PreviewTimingFilter(error_mapping=mapping)
    controller = CvLanePidController()
    records = json.loads((args.lap_dir / "data.json").read_text(encoding="utf-8"))
    by_frame = {
        int(Path(item["img_path"]).stem): item
        for item in records if Path(item["img_path"]).stem.isdigit()
    }
    output_records = []
    last_command = None
    invalid_frames = 0
    warmup_frames = 0

    # Temporal diagnostics must have the same history as a complete replay.
    # Process earlier frames without writing them so a requested late window
    # does not start with freshly reset analyzer/filter/controller state.
    for frame in range(1, args.end_frame + 1):
        item = by_frame.get(frame, {})
        image_path = args.lap_dir / f"{frame:04d}.jpg"
        if item.get("img_path"):
            candidate = Path(item["img_path"])
            image_path = candidate if candidate.is_absolute() else args.lap_dir / candidate
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            continue
        base = analyzer.process(image)
        filtered = timing.update(base) if args.temporal_filter else base
        command = controller.compute(filtered)
        if command.valid:
            invalid_frames = 0
            last_command = command
        else:
            invalid_frames += 1
            if last_command is not None and invalid_frames <= 5:
                command = replace(last_command, reason="short_invalid_hold")

        if frame < args.start_frame:
            warmup_frames += 1
            continue

        metrics = filtered.metrics
        state = item.get("state", [None, None, None])
        manual_wz = state[2] if len(state) > 2 else None
        cv_wz = command.angular_speed if command.valid else None
        mismatch = (manual_wz is not None and cv_wz is not None and
                    abs(float(manual_wz)) >= 0.03 and abs(float(cv_wz)) >= 0.03 and
                    np.sign(float(manual_wz)) != np.sign(float(cv_wz)))
        debug = analyzer.draw_debug(image, filtered)
        combined = cv2.hconcat([image, debug])
        lines = [
            (f"frame={frame} mode={metrics.get('reference_tracking_mode', 'none')} "
             f"base_head={fmt(base.raw_heading)} filtered_head={fmt(filtered.raw_heading)}"),
            (f"far_offset={fmt(metrics.get('preview_offset_normalized'))} "
             f"preview_dir={metrics.get('preview_direction', 0)} "
             f"arrival={fmt(metrics.get('arrival_weight'))} "
             f"held={bool(metrics.get('direction_held', False))}"),
            (f"CV wz={fmt(cv_wz)} manual wz={fmt(manual_wz)} "
             f"corner={bool(metrics.get('corner_detected', False))} "
             f"score={fmt(metrics.get('corner_score'))} "
             f"SIGN_MISMATCH={mismatch}"),
        ]
        review = add_header(combined, lines, warning=mismatch)
        output_path = frame_dir / f"{frame:04d}.jpg"
        cv2.imwrite(str(output_path), review,
                    [cv2.IMWRITE_JPEG_QUALITY, 94])
        output_records.append({
            "frame": frame,
            "source": str(image_path),
            "review": str(output_path),
            "manual_wz": manual_wz,
            "cv_wz": cv_wz,
            "sign_mismatch": bool(mismatch),
            "base": base.to_record(),
            "filtered": filtered.to_record(),
        })

    (args.output / "review_records.json").write_text(
        json.dumps(output_records, ensure_ascii=False, indent=2),
        encoding="utf-8")
    mismatch_count = sum(item["sign_mismatch"] for item in output_records)
    summary = {
        "start_frame": args.start_frame,
        "end_frame": args.end_frame,
        "rendered_frames": len(output_records),
        "warmup_frames": warmup_frames,
        "temporal_filter": bool(args.temporal_filter),
        "sign_mismatch_frames": mismatch_count,
        "frame_directory": str(frame_dir),
    }
    (args.output / "review_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
