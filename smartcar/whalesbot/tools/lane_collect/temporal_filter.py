"""Route-independent preview-direction and near-field timing filter."""

from dataclasses import dataclass, replace
from typing import Optional

import numpy as np

from .calibration import ErrorMapping
from .opencv_lane import LaneAnalysisResult


@dataclass(frozen=True)
class PreviewTimingConfig:
    """Continuous geometry gates; no route positions or event counters."""

    corner_gate_low: float = 0.60
    corner_gate_high: float = 0.85
    lateral_motion_low: float = 0.003
    lateral_motion_high: float = 0.015
    coverage_loss_low: float = 0.15
    coverage_loss_high: float = 0.40
    arrival_decay: float = 0.97
    direction_confirm_frames: int = 3
    protected_heading_min: float = 0.40
    protected_corner_score: float = 0.55
    preview_direction_threshold: float = 0.02
    preview_ema_alpha: float = 0.20


class PreviewTimingFilter:
    """Use far-field direction and near-field arrival evidence continuously.

    The filter deliberately has no ``NORMAL/TURNING/CROSS`` states.  Its
    memory consists only of exponentially decaying timing confidence and the
    last reliable direction needed to bridge a single-boundary observation.
    """

    def __init__(self, config: Optional[PreviewTimingConfig] = None,
                 error_mapping: Optional[ErrorMapping] = None) -> None:
        self.config = config or PreviewTimingConfig()
        self.error_mapping = error_mapping or ErrorMapping()
        self.reset()

    def reset(self) -> None:
        self._previous_lateral: Optional[float] = None
        self._lateral_motion_ema = 0.0
        self._arrival_weight = 0.0
        self._protected_mode: Optional[str] = None
        self._direction_streak = 0
        self._preview_direction = 0
        self._preview_signal_ema = 0.0
        self._preview_peak = 0.0

    def update(self, result: LaneAnalysisResult) -> LaneAnalysisResult:
        metrics = dict(result.metrics or {})
        mode = str(metrics.get("reference_tracking_mode", "none"))
        raw_heading = (float(result.raw_heading)
                       if result.raw_heading is not None else None)
        raw_lateral = (float(result.raw_lateral)
                       if result.raw_lateral is not None else None)
        if raw_lateral is not None and self._previous_lateral is not None:
            delta = abs(raw_lateral - self._previous_lateral)
            self._lateral_motion_ema = (
                0.80 * self._lateral_motion_ema + 0.20 * delta)
        if raw_lateral is not None:
            self._previous_lateral = raw_lateral

        corner_score = float(metrics.get("corner_score", 0.0))
        if not metrics.get("corner_detected", False):
            corner_score = 0.0
        rows = max(float(metrics.get("reference_rows", 0.0)), 0.0)
        roi_height = max(
            float(metrics.get("roi_bottom_y", 192.0)) -
            float(metrics.get("roi_top_y", 72.0)), 1.0)
        coverage_loss = float(np.clip(1.0 - rows / roi_height, 0.0, 1.0))
        preview_signal = metrics.get("preview_offset_normalized")
        if preview_signal is not None and np.isfinite(preview_signal):
            alpha = float(np.clip(self.config.preview_ema_alpha, 0.0, 1.0))
            self._preview_signal_ema = (
                (1.0 - alpha) * self._preview_signal_ema +
                alpha * float(preview_signal))
        far_direction_reliable = (
            abs(self._preview_signal_ema) >=
            self.config.preview_direction_threshold)
        if far_direction_reliable:
            self._preview_direction = (
                1 if self._preview_signal_ema > 0 else -1)
        corner_signal = self._smoothstep(
            corner_score, self.config.corner_gate_low,
            self.config.corner_gate_high)
        lateral_motion_signal = self._smoothstep(
            self._lateral_motion_ema, self.config.lateral_motion_low,
            self.config.lateral_motion_high)
        coverage_signal = self._smoothstep(
            coverage_loss, self.config.coverage_loss_low,
            self.config.coverage_loss_high)
        geometry_signal = max(corner_signal, coverage_signal)
        supported_motion_signal = lateral_motion_signal
        signal = max(
            corner_signal, coverage_signal, supported_motion_signal)
        decay = float(np.clip(self.config.arrival_decay, 0.0, 1.0))
        self._arrival_weight = max(signal, self._arrival_weight * decay)

        expected = self._expected_direction(mode)
        direction_held = False
        if mode in ("both", "mixed"):
            self._clear_protection()
            if raw_heading is not None and abs(raw_heading) >= 1e-3:
                self._preview_peak = max(self._preview_peak,
                                         abs(raw_heading))
        elif expected:
            if self._protected_mode is not None and \
                    self._protected_mode != mode:
                self._clear_protection()
            agrees = (raw_heading is not None and
                      raw_heading * expected >= 0.02)
            if agrees:
                self._direction_streak = (
                    self._direction_streak + 1
                    if self._protected_mode == mode or
                    self._preview_direction == expected else 1)
                if not far_direction_reliable:
                    self._preview_direction = expected
                self._preview_peak = max(self._preview_peak,
                                         abs(raw_heading))
                if (self._protected_mode is None and
                        self._direction_streak >= max(
                            1, self.config.direction_confirm_frames)):
                    self._protected_mode = mode
            elif self._protected_mode is None:
                self._direction_streak = 0

            if (raw_heading is not None and self._protected_mode == mode and
                    raw_heading * expected < 0):
                if corner_score >= self.config.protected_corner_score:
                    raw_heading = expected * max(
                        self.config.protected_heading_min,
                        self._preview_peak)
                    self._arrival_weight = max(self._arrival_weight, 0.75)
                    direction_held = True
                else:
                    # A single boundary may not create an opposite turn while
                    # the evidence is still ambiguous.
                    raw_heading = 0.0
                    direction_held = True

        # Direction comes from far-field displacement.  Middle/near evidence
        # only controls arrival and magnitude, so a noisy near tangent cannot
        # reverse a stable preview direction.
        if (raw_heading is not None and mode in ("both", "mixed") and
                self._preview_direction and
                self._arrival_weight > 0.05 and
                raw_heading * self._preview_direction < 0):
            raw_heading = self._preview_direction * abs(raw_heading)
            direction_held = True

        if raw_heading is None or not result.valid:
            metrics.update({
                "preview_direction": int(self._preview_direction),
                "preview_signal_ema": float(self._preview_signal_ema),
                "arrival_weight": float(self._arrival_weight),
                "direction_held": bool(direction_held),
                "timing_source": "invalid_or_missing",
            })
            return replace(result, metrics=metrics)

        filtered_heading = float(raw_heading * self._arrival_weight)
        mapped = self.error_mapping.map(
            raw_lateral if raw_lateral is not None else 0.0,
            filtered_heading)
        metrics.update({
            "preview_direction": int(self._preview_direction),
            "preview_signal_ema": float(self._preview_signal_ema),
            "preview_heading": float(raw_heading),
            "arrival_weight": float(self._arrival_weight),
            "corner_arrival_signal": float(corner_signal),
            "coverage_arrival_signal": float(coverage_signal),
            "lateral_motion_signal": float(lateral_motion_signal),
            "supported_motion_signal": float(supported_motion_signal),
            "lateral_motion_ema": float(self._lateral_motion_ema),
            "coverage_loss": float(coverage_loss),
            "direction_held": bool(direction_held),
            "timing_source": "far_direction_near_arrival",
        })
        return replace(
            result,
            error_y=mapped["error_y"],
            error_angle=mapped["error_angle"],
            raw_heading=filtered_heading,
            metrics=metrics,
        )

    @staticmethod
    def _expected_direction(mode: str) -> int:
        if mode == "left_only":
            return 1
        if mode == "right_only":
            return -1
        return 0

    @staticmethod
    def _smoothstep(value: float, low: float, high: float) -> float:
        if high <= low:
            return 1.0 if value >= high else 0.0
        x = float(np.clip((value - low) / (high - low), 0.0, 1.0))
        return x * x * (3.0 - 2.0 * x)

    def _clear_protection(self) -> None:
        self._protected_mode = None
        self._direction_streak = 0
        self._preview_peak = 0.0
