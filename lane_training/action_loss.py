from __future__ import annotations
import paddle
from paddle.nn import functional as F


def masked_smooth_l1_loss(
        prediction, target, target_mask, *, kappa_weight=1.0, kappa_scale=5.0):
    """Mask-normalized Smooth L1 for [speed_demand, kappa_action]."""
    if prediction.ndim != 2 or prediction.shape[-1] != 2:
        raise ValueError("prediction must have shape [N, 2]")
    if target.shape != prediction.shape or target_mask.shape != prediction.shape:
        raise ValueError("target and target_mask must match prediction shape")
    if kappa_scale <= 0:
        raise ValueError("kappa_scale must be positive")
    mask = paddle.cast(target_mask, prediction.dtype)
    scale = paddle.to_tensor(
        [1.0, float(kappa_scale)], dtype=prediction.dtype)
    per_value = F.smooth_l1_loss(
        prediction / scale, target / scale, reduction="none")
    weights = paddle.to_tensor([1.0, float(kappa_weight)], dtype=prediction.dtype)
    weighted = per_value * mask * weights
    denom = paddle.sum(mask * weights)
    eps = paddle.to_tensor(1e-6, dtype=prediction.dtype)
    total = paddle.sum(weighted) / paddle.maximum(denom, eps)
    parts = {
        "speed": paddle.sum(per_value[:, 0] * mask[:, 0]) / paddle.maximum(paddle.sum(mask[:, 0]), eps),
        "kappa": paddle.sum(per_value[:, 1] * mask[:, 1]) / paddle.maximum(paddle.sum(mask[:, 1]), eps),
        "valid_speed": paddle.sum(mask[:, 0]), "valid_kappa": paddle.sum(mask[:, 1]), "total": total,
    }
    return total, parts
