from __future__ import annotations

import math
from pathlib import Path
from typing import Sequence

import numpy as np

from smartcar.whalesbot.tools.curvature_control import CurvatureSpeedController


def derive_sequence_keys(rows: Sequence[dict]) -> list[str]:
    return [
        str(row.get("session_id") or Path(row["image_path"]).parent)
        for row in rows
    ]


def _runs(active: np.ndarray, keys: Sequence[str]) -> list[tuple[int, int]]:
    runs = []
    start = None
    for index, value in enumerate(active):
        boundary = index > 0 and keys[index] != keys[index - 1]
        if start is not None and (not value or boundary):
            runs.append((start, index - 1))
            start = None
        if value and start is None:
            start = index
    if start is not None:
        runs.append((start, len(active) - 1))
    return runs


def _optional_float(value: float) -> float | None:
    return float(value) if math.isfinite(float(value)) else None


def summarize_action_sequence(
    target,
    prediction,
    *,
    active_threshold: float = 0.05,
    sequence_keys: Sequence[str] | None = None,
) -> dict:
    target = np.asarray(target, dtype=np.float64).reshape(-1)
    prediction = np.asarray(prediction, dtype=np.float64).reshape(-1)
    if target.shape != prediction.shape or not np.all(np.isfinite(target)) or not np.all(
        np.isfinite(prediction)
    ):
        raise ValueError("target and prediction must be equal finite sequences")
    if active_threshold <= 0:
        raise ValueError("active_threshold must be positive")
    keys = list(sequence_keys) if sequence_keys is not None else ["sequence"] * len(target)
    if len(keys) != len(target):
        raise ValueError("sequence_keys must match the sequence length")
    active = np.abs(target) >= active_threshold
    predicted_active = np.abs(prediction) >= active_threshold
    context = ~active
    active_count = int(active.sum())
    context_count = int(context.sum())
    target_runs = _runs(active, keys)
    predicted_runs = _runs(predicted_active, keys)
    onset_offsets = []
    exit_offsets = []
    for start, end in target_runs:
        candidates = [
            (max(0, min(end, other_end) - max(start, other_start) + 1), other_start, other_end)
            for other_start, other_end in predicted_runs
            if keys[other_start] == keys[start]
        ]
        if candidates:
            overlap, other_start, other_end = max(candidates)
            if overlap > 0:
                onset_offsets.append(other_start - start)
                exit_offsets.append(other_end - end)

    adjacent_deltas = []
    for index in range(1, len(prediction)):
        if keys[index] == keys[index - 1]:
            adjacent_deltas.append(abs(prediction[index] - prediction[index - 1]))
    correlation = None
    if len(target) > 1 and np.std(target) > 0 and np.std(prediction) > 0:
        correlation = _optional_float(np.corrcoef(target, prediction)[0, 1])
    target_impulse = float(np.abs(target[active]).sum())
    return {
        "row_count": len(target),
        "active_count": active_count,
        "context_count": context_count,
        "kappa_mae": float(np.mean(np.abs(prediction - target))) if len(target) else None,
        "direction_accuracy": (
            float(np.mean(np.sign(prediction[active]) == np.sign(target[active])))
            if active_count else None
        ),
        "active_recall": float(np.mean(predicted_active[active])) if active_count else None,
        "false_steer_rate": float(np.mean(predicted_active[context])) if context_count else None,
        "sequence_correlation": correlation,
        "target_mean_abs_kappa": float(np.mean(np.abs(target))) if len(target) else None,
        "prediction_mean_abs_kappa": (
            float(np.mean(np.abs(prediction))) if len(prediction) else None
        ),
        "impulse_ratio": (
            float(np.abs(prediction[active]).sum() / target_impulse)
            if target_impulse > 0 else None
        ),
        "turn_count": len(target_runs),
        "median_onset_offset_frames": (
            float(np.median(onset_offsets)) if onset_offsets else None
        ),
        "median_exit_offset_frames": (
            float(np.median(exit_offsets)) if exit_offsets else None
        ),
        "peak_abs_kappa": float(np.max(np.abs(prediction))) if len(prediction) else None,
        "p95_abs_adjacent_delta": (
            float(np.percentile(adjacent_deltas, 95)) if adjacent_deltas else None
        ),
    }


def replay_vehicle_commands(
    rows: Sequence[dict], predictions, *, sequence_keys: Sequence[str] | None = None,
) -> dict:
    predictions = np.asarray(predictions, dtype=np.float64)
    if predictions.shape != (len(rows), 2) or not np.all(np.isfinite(predictions)):
        raise ValueError("predictions must have finite shape [N, 2]")
    keys = list(sequence_keys) if sequence_keys is not None else derive_sequence_keys(rows)
    if len(keys) != len(rows):
        raise ValueError("sequence_keys must match rows")
    controller = CurvatureSpeedController()
    actual_vx = []
    wz_values = []
    target_vx = []
    clipped = 0
    fallback_dt_count = 0
    for index, (row, prediction) in enumerate(zip(rows, predictions)):
        if index == 0 or keys[index] != keys[index - 1]:
            controller.reset()
        raw_dt = row.get("effective_dt_s")
        try:
            dt = float(raw_dt)
            valid_dt = math.isfinite(dt) and dt > 0
        except (TypeError, ValueError):
            valid_dt = False
            dt = controller.fallback_dt
        if not valid_dt:
            dt = controller.fallback_dt
            fallback_dt_count += 1
        speed_demand, curvature = map(float, prediction)
        vx, _, wz, target = controller.command(
            curvature, dt_s=dt, speed_demand=speed_demand,
        )
        unclipped = vx * curvature
        clipped += int(abs(unclipped) > controller.max_angular_speed)
        actual_vx.append(float(vx))
        wz_values.append(float(wz))
        target_vx.append(float(target))
    abs_wz = np.abs(wz_values)
    return {
        "actual_vx": actual_vx,
        "wz": wz_values,
        "target_vx": target_vx,
        "min_vx": min(actual_vx) if actual_vx else None,
        "mean_vx": float(np.mean(actual_vx)) if actual_vx else None,
        "max_vx": max(actual_vx) if actual_vx else None,
        "min_abs_wz": float(np.min(abs_wz)) if len(abs_wz) else None,
        "mean_abs_wz": float(np.mean(abs_wz)) if len(abs_wz) else None,
        "max_abs_wz": float(np.max(abs_wz)) if len(abs_wz) else None,
        "angular_clip_rate": clipped / len(rows) if rows else None,
        "p95_speed_delta": (
            float(np.percentile(np.abs(np.diff(actual_vx)), 95))
            if len(actual_vx) > 1 else None
        ),
        "p95_wz_delta": (
            float(np.percentile(np.abs(np.diff(wz_values)), 95))
            if len(wz_values) > 1 else None
        ),
        "fallback_dt_count": fallback_dt_count,
    }
