"""Ordered yaw-curve discovery and losses for the frozen CNN experiment."""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import paddle
from paddle.nn import functional as F


YAW_SCALE = 0.94
CURVE_LOSS_WEIGHTS = {
    "point": 1.0,
    "slope": 0.5,
    "curvature": 0.1,
    "integral": 0.5,
    "phase": 0.25,
    "saturation": 0.5,
}


def discover_turns(
    yaw: Sequence[float], *, threshold: float = 0.05,
    max_gap: int = 3, min_active: int = 5,
) -> list[tuple[int, int, int]]:
    """Return same-sign active runs as ``(start, end, sign)`` tuples.

    ``max_gap`` counts inactive frames between active observations. A sign
    change always terminates a run, even when it occurs inside the gap.
    ``min_active`` counts active observations rather than span length.
    """
    values = np.asarray(yaw, dtype=np.float64).reshape(-1)
    active = np.flatnonzero(np.abs(values) >= threshold)
    if active.size == 0:
        return []

    turns: list[tuple[int, int, int]] = []
    start = previous = int(active[0])
    sign = int(np.sign(values[start]))
    active_count = 1
    for index_value in active[1:]:
        index = int(index_value)
        index_sign = int(np.sign(values[index]))
        joins = index - previous - 1 <= max_gap and index_sign == sign
        if joins:
            previous = index
            active_count += 1
            continue
        if active_count >= min_active:
            turns.append((start, previous, sign))
        start = previous = index
        sign = index_sign
        active_count = 1
    if active_count >= min_active:
        turns.append((start, previous, sign))
    return turns


def build_curve_windows(
    rows: Sequence[dict], *, context: int = 10, max_length: int = 64,
) -> list[list[dict]]:
    """Build ordered, contextual row windows for every discovered turn."""
    if context < 0 or max_length < 1:
        raise ValueError("context must be nonnegative and max_length must be positive")
    yaw = np.asarray([float(row["yaw"]) for row in rows], dtype=np.float64)
    windows = []
    for start, end, _ in discover_turns(yaw):
        left = max(0, start - context)
        right = min(len(rows), end + context + 1)
        if right - left > max_length:
            center = (start + end) // 2
            left = max(0, min(center - max_length // 2, len(rows) - max_length))
            right = left + max_length
        windows.append([dict(row) for row in rows[left:right]])
    return windows


def selective_yaw_target(
    teacher_yaw: paddle.Tensor,
    label_yaw: paddle.Tensor,
    *,
    active_threshold: float = 0.05,
    minimum_deficit: float = 0.03,
    correction_fraction: float = 0.5,
    maximum_delta: float = 0.08,
) -> tuple[paddle.Tensor, paddle.Tensor]:
    """Build bounded targets only for active yaw that the teacher understeers."""
    if teacher_yaw.shape != label_yaw.shape:
        raise ValueError("teacher_yaw and label_yaw must have identical shapes")
    active = paddle.abs(label_yaw) >= active_threshold
    teacher_quiet = paddle.abs(teacher_yaw) < active_threshold
    opposite = teacher_yaw * label_yaw < 0.0
    same_direction_deficit = paddle.logical_and(
        teacher_yaw * label_yaw >= 0.0,
        paddle.abs(label_yaw) - paddle.abs(teacher_yaw) >= minimum_deficit,
    )
    correction_mask = paddle.logical_and(
        active,
        paddle.logical_or(
            opposite,
            paddle.logical_or(teacher_quiet, same_direction_deficit),
        ),
    )
    requested_delta = paddle.clip(
        correction_fraction * (label_yaw - teacher_yaw),
        min=-maximum_delta,
        max=maximum_delta,
    )
    target = paddle.where(
        correction_mask, teacher_yaw + requested_delta, teacher_yaw
    )
    return target, correction_mask


def _zero(value: paddle.Tensor) -> paddle.Tensor:
    return paddle.zeros([], dtype=value.dtype)


def _masked_huber(prediction, target, mask, *, scale=1.0):
    selected_prediction = paddle.masked_select(prediction, mask)
    selected_target = paddle.masked_select(target, mask)
    if selected_prediction.shape[0] == 0:
        return _zero(prediction)
    return F.smooth_l1_loss(
        selected_prediction / scale, selected_target / scale, reduction="mean"
    )


def _same_sequence_mask(sequence_ids: Sequence[int], offset: int = 1):
    if len(sequence_ids) <= offset:
        return None
    return paddle.to_tensor(
        [sequence_ids[index] == sequence_ids[index - offset]
         for index in range(offset, len(sequence_ids))],
        dtype="bool",
    )


def yaw_curve_loss(
    prediction: paddle.Tensor,
    target: paddle.Tensor,
    sequence_ids: Sequence[int],
    *,
    weights: dict[str, float] | None = None,
    active_mask: paddle.Tensor | None = None,
) -> tuple[paddle.Tensor, dict[str, paddle.Tensor]]:
    """Compute point and ordered-curve yaw losses without touching ``vy``."""
    if prediction.ndim != 2 or target.shape != prediction.shape or prediction.shape[1] != 2:
        raise ValueError("prediction and target must have shape [N, 2]")
    if len(sequence_ids) != prediction.shape[0]:
        raise ValueError("sequence_ids must match the batch row count")
    if active_mask is None:
        active_mask = paddle.ones([prediction.shape[0]], dtype="bool")
    elif active_mask.ndim != 1 or active_mask.shape[0] != prediction.shape[0]:
        raise ValueError("active_mask must have shape [N]")
    elif active_mask.dtype != paddle.bool:
        raise ValueError("active_mask must be boolean")
    active_weights = dict(CURVE_LOSS_WEIGHTS)
    if weights is not None:
        active_weights.update(weights)
    if any(not np.isfinite(value) or value < 0 for value in active_weights.values()):
        raise ValueError("curve loss weights must be finite and nonnegative")

    zero = _zero(prediction)
    if prediction.shape[0] == 0:
        parts = {name: zero for name in active_weights}
        return zero, parts

    pred_yaw = prediction[:, 1]
    target_yaw = target[:, 1]
    point = _masked_huber(
        pred_yaw, target_yaw, active_mask, scale=YAW_SCALE
    )

    first_mask = _same_sequence_mask(sequence_ids, 1)
    if first_mask is None:
        slope = zero
    else:
        first_mask = paddle.logical_and(
            first_mask,
            paddle.logical_and(active_mask[1:], active_mask[:-1]),
        )
        slope = _masked_huber(
            pred_yaw[1:] - pred_yaw[:-1],
            target_yaw[1:] - target_yaw[:-1],
            first_mask,
            scale=YAW_SCALE,
        )

    second_mask = _same_sequence_mask(sequence_ids, 2)
    if second_mask is None:
        curvature = zero
    else:
        second_mask = paddle.logical_and(
            second_mask,
            paddle.logical_and(
                paddle.logical_and(active_mask[2:], active_mask[1:-1]),
                active_mask[:-2],
            ),
        )
        pred_second = pred_yaw[2:] - 2 * pred_yaw[1:-1] + pred_yaw[:-2]
        target_second = target_yaw[2:] - 2 * target_yaw[1:-1] + target_yaw[:-2]
        curvature = _masked_huber(pred_second, target_second, second_mask, scale=YAW_SCALE)

    integral_terms = []
    phase_terms = []
    target_numpy = target_yaw.numpy()
    active_numpy = active_mask.numpy()
    for sequence_id in dict.fromkeys(sequence_ids):
        indices = np.asarray([index for index, value in enumerate(sequence_ids) if value == sequence_id])
        indices = indices[active_numpy[indices]]
        if indices.size == 0:
            continue
        target_slice = target_numpy[indices]
        target_impulse = float(np.sum(np.abs(target_slice)))
        if target_impulse < 1e-6:
            continue
        direction = 1.0 if float(np.sum(target_slice)) >= 0.0 else -1.0
        selected = paddle.to_tensor(indices, dtype="int64")
        pred_slice = paddle.index_select(pred_yaw, selected, axis=0)
        target_mean = paddle.to_tensor(
            float(np.mean(np.abs(target_slice))) / YAW_SCALE,
            dtype=prediction.dtype,
        )
        predicted_mean = paddle.mean(pred_slice * direction) / YAW_SCALE
        integral_terms.append(F.smooth_l1_loss(predicted_mean, target_mean))
        sequence_indices = np.asarray(
            [index for index, value in enumerate(sequence_ids) if value == sequence_id]
        )
        for phase_indices in np.array_split(sequence_indices, 3):
            phase_indices = phase_indices[active_numpy[phase_indices]]
            if phase_indices.size == 0:
                continue
            phase_target = float(np.sum(np.abs(target_numpy[phase_indices])))
            if phase_target < 1e-6:
                continue
            phase_selected = paddle.to_tensor(phase_indices, dtype="int64")
            phase_prediction = paddle.index_select(pred_yaw, phase_selected, axis=0)
            phase_target_mean = paddle.to_tensor(
                float(np.mean(np.abs(target_numpy[phase_indices]))) / YAW_SCALE,
                dtype=prediction.dtype,
            )
            phase_prediction_mean = paddle.mean(phase_prediction * direction) / YAW_SCALE
            phase_terms.append(
                F.smooth_l1_loss(phase_prediction_mean, phase_target_mean)
            )
    integral = paddle.mean(paddle.stack(integral_terms)) if integral_terms else zero
    phase = paddle.mean(paddle.stack(phase_terms)) if phase_terms else zero

    saturation_mask = paddle.logical_and(
        active_mask,
        paddle.logical_and(paddle.abs(target_yaw) < 0.5, paddle.abs(pred_yaw) > 0.5),
    )
    saturation_excess = paddle.masked_select(
        paddle.square(F.relu(paddle.abs(pred_yaw) - 0.5)), saturation_mask
    )
    saturation = paddle.mean(saturation_excess) if saturation_excess.shape[0] else zero
    parts = {
        "point": point, "slope": slope, "curvature": curvature,
        "integral": integral, "phase": phase, "saturation": saturation,
    }
    total = sum((active_weights[name] * value for name, value in parts.items()), zero)
    return total, parts
