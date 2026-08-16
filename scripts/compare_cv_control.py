#!/usr/bin/env python3
"""Run the complete fixed-track CV controller and compare steering trends."""

import argparse
import csv
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
    parser.add_argument("--standard-root", type=Path, default=ROOT / "standard")
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--start-frame", type=int, default=1)
    parser.add_argument("--end-frame", type=int, default=3432)
    parser.add_argument(
        "--temporal-filter", action="store_true",
        help="apply far-direction/near-arrival temporal filtering")
    return parser.parse_args()


def correlation(first, second):
    mask = np.isfinite(first) & np.isfinite(second)
    if np.count_nonzero(mask) < 3:
        return None
    first, second = first[mask], second[mask]
    if np.std(first) < 1e-9 or np.std(second) < 1e-9:
        return None
    return float(np.corrcoef(first, second)[0, 1])


def best_lag_metrics(values, manual, max_lag=50):
    """Return best correlation and the shift needed to align manual data.

    A positive lag means the manual curve is shifted later, i.e. CV is early.
    """
    candidates = []
    for lag in range(-max_lag, max_lag + 1):
        if lag > 0:
            score = correlation(values[:-lag], manual[lag:])
        elif lag < 0:
            score = correlation(values[-lag:], manual[:lag])
        else:
            score = correlation(values, manual)
        if score is not None:
            candidates.append((score, lag))
    if not candidates:
        return None, None
    score, lag = max(candidates, key=lambda item: item[0])
    return score, lag


def steering_metrics(values, manual, limit=None):
    values = np.asarray(values, dtype=np.float64)
    manual = np.asarray(manual, dtype=np.float64)
    valid = np.isfinite(values) & np.isfinite(manual)
    turning = valid & (np.abs(manual) >= 0.03)
    directional = turning & (np.abs(values) >= 0.03)
    false_active = valid & (np.abs(manual) < 0.03) & (np.abs(values) >= 0.03)
    finite_values = values[np.isfinite(values)]
    differences = np.abs(np.diff(finite_values))
    active_signs = np.sign(finite_values[np.abs(finite_values) >= 0.03])
    best_correlation, best_lag = best_lag_metrics(values, manual)
    return {
        "correlation_all": correlation(values[valid], manual[valid]),
        "correlation_manual_turning": correlation(values[turning], manual[turning]),
        "best_lag_correlation": best_correlation,
        "best_lag_frames": best_lag,
        "same_direction_rate": (float(np.mean(
            np.sign(values[directional]) == np.sign(manual[directional])))
            if np.count_nonzero(directional) else None),
        "active_frame_count": int(np.count_nonzero(np.abs(finite_values) >= 0.03)),
        "manual_zero_cv_active_count": int(np.count_nonzero(false_active)),
        "minimum": float(np.min(finite_values)) if finite_values.size else None,
        "maximum": float(np.max(finite_values)) if finite_values.size else None,
        "mean_abs_step": float(np.mean(differences)) if differences.size else None,
        "total_variation": float(np.sum(differences)) if differences.size else None,
        "large_step_count_0p05": int(np.count_nonzero(differences >= 0.05)),
        "sign_reversal_count": int(np.count_nonzero(
            active_signs[1:] != active_signs[:-1])) if active_signs.size > 1 else 0,
        "limit_frame_count": (int(np.count_nonzero(
            np.abs(finite_values) >= float(limit) - 1e-6))
            if limit is not None else None),
    }


def window_metrics(rows, start, end):
    selected = [row for row in rows if start <= row["index"] <= end]
    values = np.asarray([row["angular_speed"] for row in selected], dtype=np.float64)
    manual = np.asarray([row["hand_angular"] for row in selected], dtype=np.float64)
    def active_runs(field):
        runs = []
        for row in selected:
            if abs(float(row[field])) < 0.03:
                continue
            frame = int(row["index"])
            if not runs or frame > runs[-1][1] + 1:
                runs.append([frame, frame])
            else:
                runs[-1][1] = frame
        return runs

    cv_runs = active_runs("angular_speed")
    manual_runs = active_runs("hand_angular")
    cv_dominant = max(cv_runs, key=lambda run: run[1] - run[0], default=None)
    manual_dominant = max(
        manual_runs, key=lambda run: run[1] - run[0], default=None)
    cv_onset = cv_dominant[0] if cv_dominant else None
    manual_onset = manual_dominant[0] if manual_dominant else None
    return {
        "start": start,
        "end": end,
        "count": len(selected),
        "cv_first_active": cv_runs[0][0] if cv_runs else None,
        "manual_first_active": manual_runs[0][0] if manual_runs else None,
        "cv_onset": cv_onset,
        "manual_onset": manual_onset,
        "onset_error_frames": (cv_onset - manual_onset
                                if cv_onset is not None and
                                manual_onset is not None else None),
        "cv_mean": float(np.mean(values)) if values.size else None,
        "cv_min": float(np.min(values)) if values.size else None,
        "cv_max": float(np.max(values)) if values.size else None,
        "manual_mean": float(np.mean(manual)) if manual.size else None,
        "same_direction_rate": steering_metrics(values, manual)["same_direction_rate"],
    }


def write_chart(path, frames, manual, baseline, filtered):
    width, height = 1600, 600
    x0, x1, y0, y1 = 75, width - 25, 55, height - 50
    canvas = np.full((height, width, 3), 24, dtype=np.uint8)
    bound = max(1.0, *(float(np.nanmax(np.abs(series)))
                       for series in (manual, baseline, filtered)
                       if np.any(np.isfinite(series))))

    def point(index, value):
        x = int(round(x0 + index * (x1 - x0) / max(len(frames) - 1, 1)))
        y = int(round((y0 + y1) / 2 - value / bound * (y1 - y0) * 0.46))
        return x, y

    cv2.line(canvas, (x0, (y0 + y1) // 2), (x1, (y0 + y1) // 2),
             (100, 100, 100), 1)
    for series, color in ((manual, (80, 230, 80)),
                          (baseline, (80, 120, 255)),
                          (filtered, (0, 220, 255))):
        previous = None
        for index, value in enumerate(series):
            if not np.isfinite(value):
                previous = None
                continue
            current = point(index, float(value))
            if previous is not None:
                cv2.line(canvas, previous, current, color, 1, cv2.LINE_AA)
            previous = current
    labels = (("manual", (80, 230, 80), 75),
              ("baseline CV", (80, 120, 255), 210),
              ("filtered CV", (0, 220, 255), 410))
    for label, color, x in labels:
        cv2.putText(canvas, label, (x, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, color, 2, cv2.LINE_AA)
    cv2.putText(canvas, str(frames[0]), (x0, height - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)
    cv2.putText(canvas, str(frames[-1]), (x1 - 45, height - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)
    cv2.imwrite(str(path), canvas)


def main():
    args = parse_args()
    records = json.loads((args.lap_dir / "data.json").read_text(encoding="utf-8"))
    records = [item for item in records
               if Path(item["img_path"]).stem.isdigit()
               and args.start_frame <= int(Path(item["img_path"]).stem) <= args.end_frame]
    records.sort(key=lambda item: int(Path(item["img_path"]).stem))
    reference = StandardLaneReference.from_files(
        args.standard_root / "standard_lane.json",
        args.standard_root / "perspective.json")
    analyzer = OpenCVLaneAnalyzer(LaneAnalyzerConfig(
        error_mapping=ErrorMapping(
            lateral_scale=-0.10,
            heading_scale=-0.40,
        )), reference)
    temporal_filter = (PreviewTimingFilter(
        error_mapping=analyzer.config.error_mapping)
                       if args.temporal_filter else None)
    controller = CvLanePidController()
    last_valid_command = None
    invalid_hold_frames = 0
    rows = []

    for item in records:
        image_path = Path(item["img_path"])
        if not image_path.is_absolute():
            image_path = args.lap_dir / image_path
        frame = int(image_path.stem)
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        base_analysis = analyzer.process(image) if image is not None else None
        if base_analysis is None:
            raise RuntimeError(f"could not decode {image_path}")
        base_valid = base_analysis.valid
        controlled = (temporal_filter.update(base_analysis)
                      if temporal_filter is not None else base_analysis)
        command = controller.compute(controlled)
        held_invalid = False
        if not command.valid:
            invalid_hold_frames += 1
            if last_valid_command is not None and invalid_hold_frames <= 5:
                command = replace(last_valid_command, reason="short_invalid_hold")
                held_invalid = True
        else:
            invalid_hold_frames = 0
            last_valid_command = command
        state = item.get("state", [None, None, None])
        metrics = controlled.metrics
        rows.append({
            "index": frame,
            "base_valid": base_valid,
            "controlled_valid": command.valid,
            "held_invalid": held_invalid,
            "raw_lateral": controlled.raw_lateral,
            "raw_heading": controlled.raw_heading,
            "error_y": command.error_y,
            "error_angle": command.error_angle,
            "angular_speed": command.angular_speed,
            "forward_speed": command.forward_speed,
            "control_state": ("PREVIEW_TIMING_FILTER"
                               if temporal_filter is not None else "DISABLED"),
            "control_source": "ipm_lane_pid",
            "reference_mode": metrics.get("reference_tracking_mode"),
            "false_double_rejected": metrics.get("false_double_rejected", False),
            "false_double_rejected_side": metrics.get("false_double_rejected_side", "none"),
            "preview_direction": metrics.get("preview_direction"),
            "arrival_weight": metrics.get("arrival_weight"),
            "direction_held": metrics.get("direction_held", False),
            "hand_speed": state[0] if len(state) > 0 else None,
            "hand_lateral": state[1] if len(state) > 1 else None,
            "hand_angular": state[2] if len(state) > 2 else None,
            "reason": controlled.reason or command.reason,
        })

    baseline_by_frame = {}
    if args.baseline and args.baseline.exists():
        baseline_by_frame = {
            int(row["index"]): float(row["angular_speed"])
            for row in json.loads(args.baseline.read_text(encoding="utf-8"))}
    frames = [row["index"] for row in rows]
    filtered = np.asarray([row["angular_speed"] for row in rows], dtype=np.float64)
    manual = np.asarray([row["hand_angular"] for row in rows], dtype=np.float64)
    baseline = np.asarray([baseline_by_frame.get(frame, np.nan) for frame in frames],
                          dtype=np.float64)
    summary = {
        "frame_start": frames[0],
        "frame_end": frames[-1],
        "frame_count": len(rows),
        "base_invalid_count": int(sum(not row["base_valid"] for row in rows)),
        "controlled_invalid_count": int(sum(not row["controlled_valid"] for row in rows)),
        "held_invalid_count": int(sum(row["held_invalid"] for row in rows)),
        "false_double_rejected_count": int(sum(
            row["false_double_rejected"] for row in rows)),
        "filtered": steering_metrics(filtered, manual, controller.config.heading_limit),
        "baseline": steering_metrics(baseline, manual, 0.35),
        "filtered_vs_baseline_correlation": correlation(filtered, baseline),
        "windows": {
            "first_turn": window_metrics(rows, 150, 278),
            "first_cross": window_metrics(rows, 217, 278),
            "second_cross": window_metrics(rows, 1161, 1181),
            "right_turn_2700": window_metrics(rows, 2700, 2854),
            "left_turn_2855": window_metrics(rows, 2855, 2920),
            "known_invalid": window_metrics(rows, 2777, 2780),
            "late_section": window_metrics(rows, 3180, 3397),
        },
    }
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "control_summary.csv").open(
            "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.output / "control_summary.json").write_text(
        json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    (args.output / "comparison_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_chart(args.output / "trend_angular_output.png",
                frames, manual, baseline, filtered)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
