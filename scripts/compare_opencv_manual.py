#!/usr/bin/env python3
"""Compare stateless OpenCV lane heading with recorded manual yaw commands."""

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
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--end-frame", type=int)
    parser.add_argument("--threshold", type=int, default=175)
    parser.add_argument("--top-roi", type=float, default=0.30)
    parser.add_argument("--bottom-roi", type=float, default=0.20)
    parser.add_argument("--max-lag", type=int, default=100)
    return parser.parse_args()


def correlation(x, y):
    if len(x) < 3 or np.std(x) < 1e-9 or np.std(y) < 1e-9:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def finite(value):
    return value is not None and np.isfinite(value)


def write_chart(path, frames, manual, fitted_cv, valid):
    width, height = 1500, 560
    margin_left, margin_right, margin_top, margin_bottom = 75, 25, 55, 55
    canvas = np.full((height, width, 3), 24, dtype=np.uint8)
    values = np.concatenate((manual[np.isfinite(manual)], fitted_cv[np.isfinite(fitted_cv)]))
    bound = max(float(np.max(np.abs(values))) if values.size else 1.0, 0.1)
    x0, x1 = margin_left, width - margin_right
    y0, y1 = margin_top, height - margin_bottom

    def point(index, value):
        x = int(round(x0 + index * (x1 - x0) / max(len(frames) - 1, 1)))
        y = int(round((y0 + y1) / 2 - value / bound * (y1 - y0) * 0.46))
        return x, y

    cv2.line(canvas, (x0, (y0 + y1) // 2), (x1, (y0 + y1) // 2),
             (100, 100, 100), 1)
    for series, color in ((manual, (80, 230, 80)), (fitted_cv, (0, 220, 255))):
        previous = None
        for index, value in enumerate(series):
            if not np.isfinite(value):
                previous = None
                continue
            current = point(index, float(value))
            if previous is not None:
                cv2.line(canvas, previous, current, color, 2, cv2.LINE_AA)
            previous = current
    for index, is_valid in enumerate(valid):
        if not is_valid:
            x, _ = point(index, 0.0)
            cv2.line(canvas, (x, y1 - 8), (x, y1), (0, 0, 255), 1)

    cv2.putText(canvas, "manual yaw command", (80, 27), cv2.FONT_HERSHEY_SIMPLEX,
                0.65, (80, 230, 80), 2, cv2.LINE_AA)
    cv2.putText(canvas, "fitted stateless OpenCV heading", (330, 27),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 220, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas, "red ticks = invalid OpenCV", (720, 27),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas, str(frames[0]), (x0, height - 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)
    end_label = str(frames[-1])
    size = cv2.getTextSize(end_label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)[0]
    cv2.putText(canvas, end_label, (x1 - size[0], height - 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)
    cv2.putText(canvas, f"+{bound:.3f}", (8, y0 + 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 200, 200), 1)
    cv2.putText(canvas, f"-{bound:.3f}", (8, y1),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 200, 200), 1)
    ok, encoded = cv2.imencode(".jpg", canvas, [cv2.IMWRITE_JPEG_QUALITY, 94])
    if not ok:
        raise RuntimeError("Could not encode comparison chart")
    encoded.tofile(str(path))


def main():
    args = parse_args()
    records = json.loads((args.lap_dir / "data.json").read_text(encoding="utf-8"))
    end_frame = args.end_frame if args.end_frame is not None else 10**12
    records = [item for item in records
               if Path(item["img_path"]).stem.isdigit()
               and args.start_frame <= int(Path(item["img_path"]).stem) <= end_frame]
    if not records:
        print("No matching records", file=sys.stderr)
        return 2

    analyzer = OpenCVLaneAnalyzer(LaneAnalyzerConfig(
        threshold=args.threshold, segmentation="dark", work_size=(320, 240),
        roi_top_ratio=args.top_roi, roi_bottom_ratio=args.bottom_roi))
    rows = []
    for item in records:
        path = args.lap_dir / item["img_path"]
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        result = analyzer.process(image) if image is not None else None
        state = item.get("state", [None, None, None])
        manual_yaw = float(state[2]) if len(state) > 2 else None
        rows.append({
            "frame": int(path.stem),
            "img_path": item["img_path"],
            "manual_forward": state[0] if len(state) > 0 else None,
            "manual_lateral": state[1] if len(state) > 1 else None,
            "manual_yaw": manual_yaw,
            "cv_valid": bool(result and result.valid),
            "cv_raw_lateral": result.raw_lateral if result else None,
            "cv_raw_heading": result.raw_heading if result else None,
            "cv_tracking_mode": (result.metrics.get("tracking_mode")
                                 if result else "decode_failed"),
            "cv_reason": result.reason if result else "decode failed",
        })

    manual = np.asarray([row["manual_yaw"] for row in rows], dtype=np.float64)
    cv_heading = np.asarray([
        row["cv_raw_heading"] if finite(row["cv_raw_heading"]) else np.nan
        for row in rows], dtype=np.float64)
    valid = np.asarray([row["cv_valid"] for row in rows], dtype=bool)
    paired = valid & np.isfinite(manual) & np.isfinite(cv_heading)
    turning = paired & (np.abs(manual) >= 0.03)
    directional = turning & (np.abs(cv_heading) >= 0.05)

    fit_mask = turning if np.count_nonzero(turning) >= 10 else paired
    if np.count_nonzero(fit_mask) >= 2 and np.std(cv_heading[fit_mask]) > 1e-9:
        scale, bias = np.polyfit(cv_heading[fit_mask], manual[fit_mask], 1)
    else:
        scale, bias = 1.0, 0.0
    fitted = cv_heading * scale + bias
    mapped_directional = turning & (np.abs(fitted) >= 0.03)
    rmse = (float(np.sqrt(np.mean((fitted[fit_mask] - manual[fit_mask]) ** 2)))
            if np.count_nonzero(fit_mask) else None)

    lag_results = []
    for lag in range(-max(0, args.max_lag), max(0, args.max_lag) + 1):
        if lag >= 0:
            cv_part, manual_part = cv_heading[:len(rows) - lag or None], manual[lag:]
            valid_part = valid[:len(rows) - lag or None]
        else:
            cv_part, manual_part = cv_heading[-lag:], manual[:len(rows) + lag]
            valid_part = valid[-lag:]
        mask = valid_part & np.isfinite(cv_part) & np.isfinite(manual_part)
        corr = correlation(cv_part[mask], manual_part[mask])
        if corr is not None:
            lag_results.append((lag, corr, int(np.count_nonzero(mask))))
    best_lag = max(lag_results, key=lambda item: abs(item[1])) if lag_results else None

    aligned_summary = None
    aligned_frames = None
    if best_lag is not None:
        lag = best_lag[0]
        if lag >= 0:
            aligned_cv = cv_heading[:len(rows) - lag or None]
            aligned_manual = manual[lag:]
            aligned_valid = valid[:len(rows) - lag or None]
            aligned_frames = [row["frame"] for row in rows[:len(rows) - lag or None]]
        else:
            aligned_cv = cv_heading[-lag:]
            aligned_manual = manual[:len(rows) + lag]
            aligned_valid = valid[-lag:]
            aligned_frames = [row["frame"] for row in rows[-lag:]]
        aligned_paired = (aligned_valid & np.isfinite(aligned_cv) &
                          np.isfinite(aligned_manual))
        aligned_turning = aligned_paired & (np.abs(aligned_manual) >= 0.03)
        aligned_fit_mask = (aligned_turning if np.count_nonzero(aligned_turning) >= 10
                            else aligned_paired)
        if (np.count_nonzero(aligned_fit_mask) >= 2 and
                np.std(aligned_cv[aligned_fit_mask]) > 1e-9):
            aligned_scale, aligned_bias = np.polyfit(
                aligned_cv[aligned_fit_mask], aligned_manual[aligned_fit_mask], 1)
        else:
            aligned_scale, aligned_bias = 1.0, 0.0
        aligned_fitted = aligned_cv * aligned_scale + aligned_bias
        aligned_directional = (aligned_turning &
                               (np.abs(aligned_fitted) >= 0.03))
        aligned_summary = {
            "lag_frames": int(lag),
            "pair_count": int(np.count_nonzero(aligned_paired)),
            "turning_pair_count": int(np.count_nonzero(aligned_turning)),
            "correlation_all": correlation(
                aligned_cv[aligned_paired], aligned_manual[aligned_paired]),
            "correlation_turning": correlation(
                aligned_cv[aligned_turning], aligned_manual[aligned_turning]),
            "manual_yaw_fit_from_cv": {
                "scale": float(aligned_scale), "bias": float(aligned_bias)},
            "mapped_same_direction_rate": (
                float(np.mean(np.sign(aligned_fitted[aligned_directional]) ==
                              np.sign(aligned_manual[aligned_directional])))
                if np.count_nonzero(aligned_directional) else None),
            "fit_rmse_on_turning": (
                float(np.sqrt(np.mean(
                    (aligned_fitted[aligned_fit_mask] -
                     aligned_manual[aligned_fit_mask]) ** 2)))
                if np.count_nonzero(aligned_fit_mask) else None),
        }

    summary = {
        "frame_start": rows[0]["frame"],
        "frame_end": rows[-1]["frame"],
        "frame_count": len(rows),
        "opencv_valid_count": int(np.count_nonzero(valid)),
        "opencv_invalid_count": int(len(rows) - np.count_nonzero(valid)),
        "manual_turning_frame_count": int(np.count_nonzero(np.abs(manual) >= 0.03)),
        "paired_turning_frame_count": int(np.count_nonzero(turning)),
        "same_direction_rate": (float(np.mean(np.sign(cv_heading[directional]) ==
                                                 np.sign(manual[directional])))
                                if np.count_nonzero(directional) else None),
        "mapped_same_direction_rate": (
            float(np.mean(np.sign(fitted[mapped_directional]) ==
                          np.sign(manual[mapped_directional])))
            if np.count_nonzero(mapped_directional) else None),
        "zero_lag_correlation_all": correlation(cv_heading[paired], manual[paired]),
        "zero_lag_correlation_turning": correlation(cv_heading[turning], manual[turning]),
        "manual_yaw_fit_from_cv": {"scale": float(scale), "bias": float(bias)},
        "fit_rmse_on_turning": rmse,
        "manual_yaw_min": float(np.min(manual)),
        "manual_yaw_max": float(np.max(manual)),
        "cv_heading_min_valid": float(np.min(cv_heading[paired])),
        "cv_heading_max_valid": float(np.max(cv_heading[paired])),
        "best_lag_frames": best_lag[0] if best_lag else None,
        "best_lag_correlation": best_lag[1] if best_lag else None,
        "best_lag_pair_count": best_lag[2] if best_lag else 0,
        "best_lag_aligned": aligned_summary,
        "lag_definition": "positive means manual command occurs later than CV image",
        "notes": "No turn state machine; OpenCV raw heading is an angle, manual yaw is an angular velocity command.",
    }

    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "opencv_vs_manual.csv").open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_chart(args.output / "angle_comparison.jpg",
                [row["frame"] for row in rows], manual, fitted, valid)
    if aligned_summary is not None:
        write_chart(args.output / "angle_comparison_best_lag_aligned.jpg",
                    aligned_frames, aligned_manual, aligned_fitted, aligned_valid)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
