#!/usr/bin/env python3
"""Measure how heading slew changes a recorded CV teacher session."""

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def slew(requested, previous, entry, release, reverse):
    same_direction = (requested == 0.0 or previous == 0.0 or
                      np.sign(requested) == np.sign(previous))
    if not same_direction:
        step = reverse
    elif abs(requested) > abs(previous):
        step = entry
    else:
        step = release
    return float(np.clip(requested, previous - step, previous + step))


def metrics(values, reference):
    values = np.asarray(values, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    steps = np.abs(np.diff(values))
    active = (np.abs(values) >= 0.03) & (np.abs(reference) >= 0.03)
    return {
        "correlation_with_recorded": float(np.corrcoef(values, reference)[0, 1]),
        "mae_from_recorded": float(np.mean(np.abs(values - reference))),
        "max_error_from_recorded": float(np.max(np.abs(values - reference))),
        "same_direction_rate": (float(np.mean(
            np.sign(values[active]) == np.sign(reference[active])))
            if np.any(active) else None),
        "mean_abs_step": float(np.mean(steps)),
        "p95_abs_step": float(np.percentile(steps, 95)),
        "max_abs_step": float(np.max(steps)),
        "large_step_count_0p10": int(np.count_nonzero(steps >= 0.10)),
        "large_step_count_0p20": int(np.count_nonzero(steps >= 0.20)),
        "total_variation": float(np.sum(steps)),
    }


def main():
    args = parse_args()
    records = json.loads((args.session / "data.json").read_text(encoding="utf-8"))
    config = json.loads((args.session / "session.json").read_text(
        encoding="utf-8"))["controller"]
    period = max(float(config.get("control_period_s", 0.05)), 1e-6)
    entry_rate = float(config.get("pure_pursuit_entry_step", 0.10)) / period
    release_rate = float(config.get("max_heading_release_step", 0.04)) / period
    reverse_rate = float(config.get("max_heading_reverse_step", 0.10)) / period
    heading_limit = float(config.get("heading_limit", 1.5))

    recorded = []
    no_slew = []
    frame_slew = []
    real_dt_slew = []
    previous_frame = 0.0
    previous_dt = 0.0
    previous_time = None
    previous_valid_frame = 0.0
    previous_valid_dt = 0.0
    rows = []

    for index, item in enumerate(records):
        reference_wz = float(item["control"][2])
        timestamp = float(item.get("timestamp", index * period))
        dt = period if previous_time is None else max(timestamp - previous_time, 1e-4)
        previous_time = timestamp
        held = bool(item.get("held", False))
        launch = item.get("command_source") == "initial_straight_guard"
        override = bool(item.get("cv", {}).get("metrics", {}).get(
            "route_right_turn_override", False))
        curvature = item.get("cv", {}).get("metrics", {}).get(
            "pure_pursuit_curvature_m_inv")
        valid_curvature = curvature is not None and np.isfinite(float(curvature))

        if held or not valid_curvature:
            raw = previous_valid_frame
            frame_value = previous_valid_frame
            dt_value = previous_valid_dt
        else:
            raw = float(np.clip(
                float(item["control"][0]) * float(curvature),
                -heading_limit, heading_limit))
            if override:
                frame_value = raw
                dt_value = raw
            else:
                frame_value = slew(
                    raw, previous_frame,
                    float(config.get("pure_pursuit_entry_step", 0.10)),
                    float(config.get("max_heading_release_step", 0.04)),
                    float(config.get("max_heading_reverse_step", 0.10)))
                dt_value = slew(
                    raw, previous_dt,
                    entry_rate * dt, release_rate * dt, reverse_rate * dt)
            previous_frame = frame_value
            previous_dt = dt_value
            previous_valid_frame = frame_value
            previous_valid_dt = dt_value

        output_raw = 0.0 if launch else raw
        output_frame = 0.0 if launch else frame_value
        output_dt = 0.0 if launch else dt_value
        recorded.append(reference_wz)
        no_slew.append(output_raw)
        frame_slew.append(output_frame)
        real_dt_slew.append(output_dt)
        rows.append({
            "index": index,
            "img_path": item.get("img_path"),
            "dt_s": dt,
            "recorded_wz": reference_wz,
            "raw_wz": output_raw,
            "frame_slew_wz": output_frame,
            "real_dt_slew_wz": output_dt,
            "held": held,
            "launch_guard": launch,
            "right_turn_override": override,
        })

    difference = np.abs(np.asarray(no_slew) - np.asarray(frame_slew))
    largest = np.argsort(difference)[-20:][::-1]
    summary = {
        "session": str(args.session),
        "frame_count": len(records),
        "duration_s": float(records[-1]["timestamp"] - records[0]["timestamp"]),
        "median_dt_s": float(np.median([row["dt_s"] for row in rows[1:]])),
        "recorded": metrics(recorded, recorded),
        "replayed_original_frame_slew": metrics(frame_slew, recorded),
        "real_dt_slew": metrics(real_dt_slew, recorded),
        "no_slew": metrics(no_slew, recorded),
        "no_slew_vs_frame_slew": {
            "mean_abs_difference": float(np.mean(difference)),
            "p95_abs_difference": float(np.percentile(difference, 95)),
            "frames_over_0p05": int(np.count_nonzero(difference >= 0.05)),
            "frames_over_0p10": int(np.count_nonzero(difference >= 0.10)),
            "largest_difference_frames": [rows[int(i)] for i in largest],
        },
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.output / "rows.json").write_text(
        json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
