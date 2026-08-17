"""OpenCV lane teacher used by data collection only."""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np

from .calibration import ErrorMapping
from .cross import detect_cross_candidate

Size = Tuple[int, int]


@dataclass(frozen=True)
class StandardLaneReference:
    """Fixed-track boundary and perspective reference for one camera setup."""

    image_size: Size
    roi_top: int
    roi_bottom: int
    left_boundary: np.ndarray
    right_boundary: np.ndarray
    perspective: np.ndarray
    lane_width: float

    @classmethod
    def from_files(cls, lane_path: Path, perspective_path: Path):
        lane_data = json.loads(Path(lane_path).read_text(encoding="utf-8"))
        perspective_data = json.loads(
            Path(perspective_path).read_text(encoding="utf-8"))
        image_size = tuple(int(v) for v in lane_data["image_size"])
        roi = lane_data["roi"]
        width, height = image_size
        left = np.asarray(lane_data["left_boundary"], dtype=np.float64)
        right = np.asarray(lane_data["right_boundary"], dtype=np.float64)
        perspective = np.asarray(perspective_data["arr"], dtype=np.float64)
        roi_top, roi_bottom = int(roi["top"]), int(roi["bottom"])
        if (width <= 0 or height <= 0 or roi_top < 0 or
                roi_bottom > height or roi_top >= roi_bottom):
            raise ValueError("invalid standard lane image/ROI")
        if left.size != height or right.size != height:
            raise ValueError("standard boundary length must equal image height")
        if perspective.size != roi_bottom - roi_top:
            raise ValueError("perspective length must equal ROI height")
        if (not np.all(np.isfinite(perspective)) or
                np.any(perspective <= 0)):
            raise ValueError("perspective values must be positive and finite")
        left[left < 0] = np.nan
        right[right < 0] = np.nan
        return cls(image_size, roi_top, roi_bottom, left, right,
                   perspective, float(perspective_data.get("wid", 1.0)))


@dataclass(frozen=True)
class LaneAnalyzerConfig:
    cnn_size: Size = (128, 128)
    work_size: Optional[Size] = (320, 240)
    # None enables per-frame global Otsu segmentation.  Clamp the result so
    # course geometry cannot drive the histogram threshold to an unsafe
    # extreme when an intersection or a sharp corner fills most of the ROI.
    threshold: Optional[int] = None
    adaptive_threshold_min: int = 150
    adaptive_threshold_max: int = 165
    segmentation: str = "dark"
    orange_hue_low: int = 5
    orange_hue_high: int = 35
    orange_saturation_low: int = 50
    orange_value_low: int = 50
    orange_dilate_kernel: int = 5
    # Provisional ROI selected from the supplied 2026 track photos.
    roi_top_ratio: float = 0.30
    roi_bottom_ratio: float = 0.20
    seed_band_ratio: float = 0.25
    min_seed_run_ratio: float = 0.05
    min_lane_width_ratio: float = 0.08
    min_valid_rows_ratio: float = 0.18
    edge_margin_ratio: float = 0.015
    expected_lane_width_far_ratio: float = 0.50
    expected_lane_width_near_ratio: float = 0.90
    max_center_step_ratio: float = 0.08
    min_center_reach_ratio: float = 0.60
    # A long far-field center is still useful when a wide corner fills the
    # near field, but must not bypass a confirmed sharp-corner fallback.
    relaxed_center_reach_ratio: float = 0.50
    min_partial_center_rows_ratio: float = 0.10
    min_partial_heading: float = 0.35
    corner_min_line_ratio: float = 0.12
    corner_min_horizontalness: float = 0.65
    corner_min_score: float = 0.55
    reference_y_ratio: float = 0.90
    # Fit the complete equivalent-IPM trace.  A near-field-only image-space
    # fit cannot distinguish lateral displacement from vehicle heading.
    fit_far_y_ratio: float = 0.0
    fit_near_y_ratio: float = 1.0
    # perspective.json stores centimetres (wid=track width, arr=forward
    # distance).  Pure pursuit deliberately samples one mid/near-field point
    # instead of steering from the far-field tangent of the complete fit.
    pure_pursuit_lookahead_m: float = 0.30
    ipm_units_per_meter: float = 100.0
    pure_pursuit_sharp_ratio_threshold: float = 0.80
    pure_pursuit_sharp_gain: float = 1.8
    # The final right acute turn is already tighter in raw pure-pursuit
    # geometry than the three left corners.  Its left-only negative-curvature
    # phase must not receive the generic sharp-corner boost.
    pure_pursuit_right_turn_sharp_gain: float = 1.20
    # A confirmed entering corner can expose a stable contour tangent before
    # the fixed lookahead point moves laterally.  Use that tangent only for
    # this narrow failure mode; ordinary bends remain pure-pursuit driven.
    corner_pursuit_min_score: float = 0.70
    corner_pursuit_max_lateral_ratio: float = 0.15
    corner_pursuit_max_near_progress: float = 0.50
    # A weaker early gate is allowed only when both reference types are
    # present and the fitted path still looks straight.  This targets the
    # wide acute-corner entrance where the contour tangent leads the IPM fit.
    corner_pursuit_early_min_score: float = 0.58
    corner_pursuit_early_max_abs_heading: float = 0.15
    corner_pursuit_early_min_near_progress: float = 0.15
    # The right acute-turn gate is stateless.  Its command comes from the
    # current contour heading and the calibrated forward distance of the
    # contour's near endpoint, rather than a fixed curvature or frame count.
    right_turn_candidate_min_score: float = 0.70
    right_turn_candidate_max_heading: float = -1.35
    right_turn_candidate_max_lateral_ratio: float = 0.20
    right_turn_candidate_min_near_progress: float = 0.15
    right_turn_candidate_max_near_progress: float = 0.45
    # The 0.15 near-progress gate observes the final right acute corner
    # farther away than the former 0.20 gate.  Compensate only that strict
    # route override so early entry does not also weaken its curvature.
    right_turn_curvature_gain: float = 0.78
    right_turn_min_distance_m: float = 0.20
    right_turn_max_curvature_m_inv: float = 10.0
    morphology_kernel: int = 3
    error_mapping: ErrorMapping = field(default_factory=ErrorMapping)
    boundary_max_step: float = 10.0
    boundary_min_length: int = 10
    boundary_max_gap: int = 3
    false_double_max_width_ratio: float = 0.22
    false_double_min_slope_ratio: float = 1.8
    false_double_min_horizontal_slope: float = 0.25


@dataclass
class LaneAnalysisResult:
    valid: bool
    error_y: Optional[float]
    error_angle: Optional[float]
    raw_lateral: Optional[float]
    raw_heading: Optional[float]
    confidence: float
    cross_status: str
    cross_score: float
    reason: Optional[str]
    work_size: Size
    seed_point: Optional[Tuple[int, int]] = None
    left_line: Optional[np.ndarray] = None
    right_line: Optional[np.ndarray] = None
    center_line: Optional[np.ndarray] = None
    lane_mask: Optional[np.ndarray] = None
    binary_mask: Optional[np.ndarray] = None
    metrics: Dict[str, Any] = field(default_factory=dict)

    def to_record(self) -> Dict[str, Any]:
        return {
            "valid": bool(self.valid), "error_y": self.error_y,
            "error_angle": self.error_angle, "raw_lateral": self.raw_lateral,
            "raw_heading": self.raw_heading, "confidence": float(self.confidence),
            "cross_status": self.cross_status, "cross_score": float(self.cross_score),
            "reason": self.reason, "work_size": list(self.work_size),
            "seed_point": list(self.seed_point) if self.seed_point else None,
            "metrics": self.metrics,
        }


class OpenCVLaneAnalyzer:
    """Extract a connected dark track and fit its center line."""

    def __init__(self, config: Optional[LaneAnalyzerConfig] = None,
                 reference: Optional[StandardLaneReference] = None) -> None:
        self.config = config or LaneAnalyzerConfig()
        if reference is None:
            raise ValueError("standard lane reference is required")
        expected = self.config.work_size
        if expected is None or tuple(reference.image_size) != tuple(expected):
            raise ValueError("standard reference image size must match work_size")
        self.reference = reference

    def make_cnn_image(self, image: np.ndarray) -> np.ndarray:
        self._validate_image(image)
        return cv2.resize(image, self.config.cnn_size, interpolation=cv2.INTER_LINEAR)

    def make_work_image(self, image: np.ndarray) -> np.ndarray:
        self._validate_image(image)
        if self.config.work_size is None:
            return image.copy()
        return cv2.resize(image, self.config.work_size, interpolation=cv2.INTER_AREA)

    def process(self, image: np.ndarray) -> LaneAnalysisResult:
        try:
            work = self.make_work_image(image)
        except (TypeError, ValueError) as exc:
            return self._invalid((0, 0), str(exc))
        height, width = work.shape[:2]
        binary, segmentation_metrics = self._segment_track(work)
        seed = self._find_seed(binary)
        if seed is None:
            return self._invalid(
                (width, height), "no dark seed connected to near field",
                binary, metrics=segmentation_metrics)
        lane_mask = self._connected_seed_component(binary, seed)
        if lane_mask is None:
            return self._invalid(
                (width, height), "seed component is empty", binary,
                metrics=segmentation_metrics)
        roi_top, roi_bottom = self._roi_bounds(height)
        roi_height = max(roi_bottom - roi_top, 1)
        left, right, center, widths, modes = self._extract_boundaries(lane_mask)
        left = self._clean_boundary_trace(left)
        right = self._clean_boundary_trace(right)
        left, right, boundary_filter = self._reject_false_double_boundary(
            left, right, width)
        modes = self._boundary_modes(left, right)
        center = self._center_from_boundaries(left, right)
        corner = self._detect_steep_corner(lane_mask)
        return self._standard_relative_result(
            width, height, left, right, center, widths, modes, lane_mask,
            binary, seed, corner, boundary_filter, segmentation_metrics)

    def _standard_relative_result(self, width, height, left, right, center,
                                  widths, modes, lane_mask, binary, seed,
                                  corner, boundary_filter,
                                  segmentation_metrics):
        ref = self.reference
        roi_top, roi_bottom = self._roi_bounds(height)
        rows = np.arange(roi_top, roi_bottom, dtype=np.int32)
        standard_left = ref.left_boundary[rows]
        standard_right = ref.right_boundary[rows]
        standard_width = standard_right - standard_left
        standard_center = (standard_left + standard_right) / 2.0
        current_left = left[rows]
        current_right = right[rows]
        standard_ok = (
            np.isfinite(standard_left) & np.isfinite(standard_right) &
            np.isfinite(standard_width) & (standard_width > 1.0))
        left_ok = np.isfinite(current_left) & standard_ok
        right_ok = np.isfinite(current_right) & standard_ok
        both_ok = left_ok & right_ok
        left_only = left_ok & ~right_ok
        right_only = right_ok & ~left_ok
        current_center = np.full(rows.shape, np.nan, dtype=np.float64)
        current_center[both_ok] = (
            current_left[both_ok] + current_right[both_ok]) / 2.0
        current_center[left_only] = (
            current_left[left_only] + standard_width[left_only] / 2.0)
        current_center[right_only] = (
            current_right[right_only] - standard_width[right_only] / 2.0)
        valid = np.isfinite(current_center) & standard_ok
        side_counts = {
            "left": int(np.count_nonzero(left_only)),
            "right": int(np.count_nonzero(right_only)),
            "both": int(np.count_nonzero(both_ok)),
        }
        reference_rows = int(np.count_nonzero(valid))
        if reference_rows < self.config.boundary_min_length:
            return self._invalid(
                (width, height),
                "standard reference has too few valid boundary rows", binary,
                lane_mask, seed, left, right, center,
                 {"reference_rows": reference_rows,
                  "reference_tracking_mode": self._reference_mode(side_counts),
                  **segmentation_metrics,
                  **boundary_filter,
                  **self._corner_metrics(corner)})
        # This is the same row-wise perspective normalization used by the
        # open-source implementation, expressed as an equivalent bird's-eye
        # center trace instead of independent left/right residual lines.
        ipm_x = (
            (current_center - standard_center) * float(ref.lane_width) /
            standard_width)
        ipm_s = ref.perspective[
            np.clip(rows - ref.roi_top, 0, ref.perspective.size - 1)]
        fit_far = float(np.clip(self.config.fit_far_y_ratio, 0.0, 1.0))
        fit_near = float(np.clip(self.config.fit_near_y_ratio, 0.0, 1.0))
        if fit_near < fit_far:
            fit_far, fit_near = fit_near, fit_far
        fit_top = roi_top + (roi_bottom - roi_top - 1) * fit_far
        fit_bottom = roi_top + (roi_bottom - roi_top - 1) * fit_near
        fit_mask = (
            valid & (rows >= fit_top) & (rows <= fit_bottom) &
            np.isfinite(ipm_x) & np.isfinite(ipm_s))
        if int(np.count_nonzero(fit_mask)) < self.config.boundary_min_length:
            fit_mask = valid & np.isfinite(ipm_x) & np.isfinite(ipm_s)
        fit_s = ipm_s[fit_mask]
        fit_x = ipm_x[fit_mask]
        design = np.column_stack([
            fit_s * fit_s, fit_s, np.ones_like(fit_s)])
        quadratic, linear, intercept = np.linalg.lstsq(
            design, fit_x, rcond=None)[0]
        reference_y = roi_top + int(
            round((roi_bottom - roi_top - 1) * self.config.reference_y_ratio))
        reference_index = int(np.clip(
            reference_y - ref.roi_top, 0, ref.perspective.size - 1))
        reference_s = float(ref.perspective[reference_index])
        reference_x = float(
            quadratic * reference_s * reference_s +
            linear * reference_s + intercept)
        far_s = float(np.max(fit_s))
        far_x = float(
            quadratic * far_s * far_s + linear * far_s + intercept)
        preview_offset = float(far_x - reference_x)
        preview_offset_normalized = float(
            preview_offset / max(float(ref.lane_width) / 2.0, 1e-9))
        local_slope = float(2.0 * quadratic * reference_s + linear)
        raw_lateral = float(
            reference_x / max(float(ref.lane_width) / 2.0, 1e-9))
        raw_heading = float(np.arctan(local_slope))
        curvature = float(
            (2.0 * quadratic) /
            max((1.0 + local_slope * local_slope) ** 1.5, 1e-9))
        lookahead_units = (
            max(float(self.config.pure_pursuit_lookahead_m), 1e-3) *
            max(float(self.config.ipm_units_per_meter), 1e-6))
        pursuit = self._pure_pursuit_target(
            fit_s, fit_x, lookahead_units,
            self.config.ipm_units_per_meter)
        sharp_pursuit_control = self._apply_sharp_curvature_gain(
            pursuit, self.config.pure_pursuit_sharp_ratio_threshold,
            self.config.pure_pursuit_sharp_gain,
            self._reference_mode(side_counts),
            self.config.pure_pursuit_right_turn_sharp_gain)
        pursuit_control = self._apply_corner_curvature_fallback(
            sharp_pursuit_control, corner,
            self._reference_mode(side_counts),
            raw_heading, self.config.pure_pursuit_lookahead_m,
            self.config.corner_pursuit_min_score,
            self.config.corner_pursuit_max_lateral_ratio,
            self.config.corner_pursuit_max_near_progress,
            self.config.corner_pursuit_early_min_score,
            self.config.corner_pursuit_early_max_abs_heading,
            self.config.corner_pursuit_early_min_near_progress)
        right_turn_reason = self._right_turn_reason(
            corner, self._reference_mode(side_counts),
            pursuit_control["target_lateral_ratio"],
            self.config.right_turn_candidate_min_score,
            self.config.right_turn_candidate_max_heading,
            self.config.right_turn_candidate_max_lateral_ratio,
            self.config.right_turn_candidate_min_near_progress,
            self.config.right_turn_candidate_max_near_progress)
        right_turn_candidate = right_turn_reason is not None
        right_turn_waiting = (
            right_turn_reason is None and
            self._right_turn_geometry(
                corner, self._reference_mode(side_counts),
                self.config.right_turn_candidate_max_heading))
        if right_turn_waiting:
            # A right-acute contour tangent has the opposite sign from the
            # generic corner fallback.  Before the near-field gate is ready,
            # retain ordinary pure pursuit instead of steering left.
            pursuit_control = dict(sharp_pursuit_control)
            pursuit_control.update({
                "corner_fallback": False,
                "corner_fallback_reason": "right_turn_wait",
                "corner_curvature_m_inv": None,
            })
        right_turn_geometry = self._right_turn_curvature(
            corner, ref.perspective, ref.roi_top,
            self.config.ipm_units_per_meter,
            self.config.right_turn_curvature_gain,
            self.config.right_turn_min_distance_m,
            self.config.right_turn_max_curvature_m_inv)
        pursuit_control = self._apply_right_turn_override(
            pursuit_control, right_turn_candidate, right_turn_geometry)
        mapped = self.config.error_mapping.map(raw_lateral, raw_heading)
        cross = detect_cross_candidate(widths[np.isfinite(widths)])
        center = np.asarray(center, dtype=np.float64).copy()
        center[rows[valid]] = current_center[valid]
        perspective_fit_sides = int(bool(np.any(left_ok))) + int(
            bool(np.any(right_ok)))
        metrics = {
            **segmentation_metrics,
            "valid_rows": int(np.count_nonzero(fit_mask)),
            "reference_rows": reference_rows,
            "reference_tracking_mode": self._reference_mode(side_counts),
            "reference_side_counts": side_counts,
            "fit_far_y": float(fit_top),
            "fit_near_y": float(fit_bottom),
            "fit_rows": int(np.count_nonzero(fit_mask)),
            "perspective_k": local_slope,
            "perspective_b": reference_x,
            "perspective_fit_sides": perspective_fit_sides,
            "reference_fit_slope": local_slope,
            "reference_fit_intercept": float(intercept),
            "ipm_quadratic": float(quadratic),
            "ipm_linear": float(linear),
            "ipm_intercept": float(intercept),
            "ipm_reference_s": reference_s,
            "ipm_reference_x": reference_x,
            "ipm_far_s": far_s,
            "ipm_far_x": far_x,
            "preview_offset": preview_offset,
            "preview_offset_normalized": preview_offset_normalized,
            "curvature": curvature,
            "pure_pursuit_lookahead_m": float(
                self.config.pure_pursuit_lookahead_m),
            "pure_pursuit_target_s_m": pursuit["target_s_m"],
            "pure_pursuit_target_x_m": pursuit["target_x_m"],
            "pure_pursuit_raw_curvature_m_inv": pursuit["curvature_m_inv"],
            "pure_pursuit_target_lateral_ratio": pursuit_control[
                "target_lateral_ratio"],
            "pure_pursuit_sharp_boosted": pursuit_control["sharp_boosted"],
            "pure_pursuit_sharp_gain_applied": pursuit_control[
                "sharp_gain_applied"],
            "pure_pursuit_corner_fallback": pursuit_control[
                "corner_fallback"],
            "pure_pursuit_corner_fallback_reason": pursuit_control[
                "corner_fallback_reason"],
            "pure_pursuit_corner_curvature_m_inv": pursuit_control[
                "corner_curvature_m_inv"],
            "route_right_turn_candidate": right_turn_candidate,
            "route_right_turn_reason": right_turn_reason,
            "route_right_turn_waiting": right_turn_waiting,
            "route_right_turn_override": pursuit_control[
                "right_turn_override"],
            "route_right_turn_distance_m": (
                right_turn_geometry["distance_m"]
                if right_turn_candidate else None),
            "route_right_turn_curvature_m_inv": (
                right_turn_geometry["curvature_m_inv"]
                if right_turn_candidate else None),
            "pure_pursuit_curvature_m_inv": pursuit_control[
                "curvature_m_inv"],
            "pure_pursuit_lookahead_clipped": pursuit["lookahead_clipped"],
            "perspective_median": float(np.median(fit_s)),
            "roi_top_y": roi_top,
            "roi_bottom_y": roi_bottom,
            "tracking_mode": self._dominant_tracking_mode(
                self._tracking_mode_counts(modes, np.arange(height))),
            **boundary_filter,
            **self._corner_metrics(corner),
        }
        return LaneAnalysisResult(
            True, mapped["error_y"], mapped["error_angle"],
            raw_lateral, raw_heading, float(np.clip(
                reference_rows / max(roi_bottom - roi_top, 1), 0.0, 1.0)),
            cross.status, cross.score, None, (width, height), seed, left,
            right, center, lane_mask, binary, metrics)

    @staticmethod
    def _pure_pursuit_target(fit_s, fit_x, lookahead_units,
                             units_per_meter):
        """Interpolate one path point and return geometric curvature.

        Selecting the point directly from the IPM trace prevents far rows
        from changing the steering command before the bend reaches the
        configured lookahead distance.  The quadratic fit remains available
        for heading/curvature diagnostics only.
        """
        fit_s = np.asarray(fit_s, dtype=np.float64)
        fit_x = np.asarray(fit_x, dtype=np.float64)
        valid = np.isfinite(fit_s) & np.isfinite(fit_x)
        if not np.any(valid):
            raise ValueError("pure pursuit requires a valid IPM trace")
        fit_s, fit_x = fit_s[valid], fit_x[valid]
        order = np.argsort(fit_s)
        fit_s, fit_x = fit_s[order], fit_x[order]
        unique_s, inverse = np.unique(fit_s, return_inverse=True)
        if unique_s.size != fit_s.size:
            sums = np.bincount(inverse, weights=fit_x)
            counts = np.bincount(inverse)
            fit_s, fit_x = unique_s, sums / np.maximum(counts, 1)
        requested_s = float(lookahead_units)
        target_s = float(np.clip(requested_s, fit_s[0], fit_s[-1]))
        target_x = float(np.interp(target_s, fit_s, fit_x))
        scale = max(float(units_per_meter), 1e-6)
        target_s_m = target_s / scale
        target_x_m = target_x / scale
        distance_sq = target_s_m * target_s_m + target_x_m * target_x_m
        # Image/IPM x grows to the right, while chassis-positive yaw turns
        # left.  Negate the geometric image curvature at this boundary.
        curvature = -2.0 * target_x_m / max(distance_sq, 1e-9)
        return {
            "target_s_m": float(target_s_m),
            "target_x_m": float(target_x_m),
            "curvature_m_inv": float(curvature),
            "lookahead_clipped": bool(abs(target_s - requested_s) > 1e-6),
        }

    @staticmethod
    def _apply_sharp_curvature_gain(pursuit, ratio_threshold, sharp_gain,
                                    reference_mode="none",
                                    right_turn_sharp_gain=1.0):
        target_s = abs(float(pursuit["target_s_m"]))
        target_x = abs(float(pursuit["target_x_m"]))
        ratio = target_x / max(target_s, 1e-6)
        threshold = max(float(ratio_threshold), 0.0)
        boosted = bool(ratio >= threshold)
        gain = 1.0
        if boosted:
            gain = max(float(sharp_gain), 0.0)
            if (reference_mode == "left_only" and
                    float(pursuit["curvature_m_inv"]) < 0.0):
                gain = max(float(right_turn_sharp_gain), 0.0)
        return {
            "curvature_m_inv": float(pursuit["curvature_m_inv"]) * gain,
            "target_lateral_ratio": float(ratio),
            "sharp_boosted": boosted,
            "sharp_gain_applied": float(gain),
        }

    @staticmethod
    def _apply_corner_curvature_fallback(pursuit_control, corner,
                                         reference_mode, raw_heading,
                                         lookahead_m,
                                         min_score, max_lateral_ratio,
                                         max_near_progress,
                                         early_min_score,
                                         early_max_abs_heading,
                                         early_min_near_progress):
        """Use a confirmed entering-corner tangent when lookahead stays flat."""
        result = dict(pursuit_control)
        result.update({
            "corner_fallback": False,
            "corner_fallback_reason": None,
            "corner_curvature_m_inv": None,
        })
        if corner is None or reference_mode == "none":
            return result
        if (not corner.get("direction_known", False) or
                float(result["target_lateral_ratio"]) >=
                float(max_lateral_ratio) or
                float(corner.get("near_progress", 1.0)) >
                float(max_near_progress)):
            return result
        score = float(corner.get("score", 0.0))
        near_progress = float(corner.get("near_progress", 1.0))
        strong_gate = score >= float(min_score)
        early_gate = (
            score >= float(early_min_score) and
            reference_mode == "mixed" and
            abs(float(raw_heading)) <= float(early_max_abs_heading) and
            near_progress >= float(early_min_near_progress))
        if not strong_gate and not early_gate:
            return result
        lookahead = max(float(lookahead_m), 1e-3)
        # Image-positive heading points right, while chassis-positive yaw is
        # left, matching the sign conversion in _pure_pursuit_target().
        curvature = float(
            -2.0 * np.sin(float(corner["heading"])) / lookahead)
        if abs(curvature) <= abs(float(result["curvature_m_inv"])):
            return result
        result["curvature_m_inv"] = curvature
        result["corner_fallback"] = True
        result["corner_fallback_reason"] = (
            "strong_score" if strong_gate else "early_mixed_flat")
        result["corner_curvature_m_inv"] = curvature
        return result

    @staticmethod
    def _right_turn_reason(corner, reference_mode, lateral_ratio,
                           min_score, max_heading,
                           max_lateral_ratio, min_near_progress,
                           max_near_progress):
        if corner is None:
            return None
        score = float(corner.get("score", 0.0))
        near_progress = float(corner.get("near_progress", 1.0))
        entry = (
            reference_mode == "mixed" and
            corner.get("direction_known", False) and
            score >= float(min_score) and
            float(corner.get("heading", 0.0)) <= float(max_heading) and
            float(lateral_ratio) < float(max_lateral_ratio) and
            near_progress >= float(min_near_progress) and
            near_progress <= float(max_near_progress))
        if entry:
            return "near_entry"
        return None

    @staticmethod
    def _right_turn_geometry(corner, reference_mode, max_heading):
        return bool(
            corner is not None and reference_mode == "mixed" and
            corner.get("direction_known", False) and
            float(corner.get("heading", 0.0)) <= float(max_heading))

    @staticmethod
    def _right_turn_curvature(corner, perspective, roi_top,
                              units_per_meter, curvature_gain, min_distance_m,
                              max_curvature_m_inv):
        """Convert the current corner tangent and endpoint to curvature."""
        if corner is None or not corner.get("direction_known", False):
            return None
        segment = corner.get("segment")
        if not segment or len(segment) < 2:
            return None
        perspective = np.asarray(perspective, dtype=np.float64)
        if perspective.size == 0:
            return None
        near_y = int(round(float(segment[1])))
        index = int(np.clip(near_y - int(roi_top), 0,
                            perspective.size - 1))
        scale = max(float(units_per_meter), 1e-6)
        measured_distance_m = float(perspective[index]) / scale
        distance_m = max(measured_distance_m, float(min_distance_m), 1e-3)
        heading = abs(float(corner.get("heading", 0.0)))
        gain = max(float(curvature_gain), 0.0)
        curvature = -2.0 * np.sin(heading) / distance_m * gain
        limit = max(float(max_curvature_m_inv), 0.0)
        curvature = float(np.clip(curvature, -limit, 0.0))
        return {
            "distance_m": measured_distance_m,
            "curvature_m_inv": curvature,
        }

    @staticmethod
    def _apply_right_turn_override(pursuit_control, candidate, geometry):
        result = dict(pursuit_control)
        enabled = bool(candidate and geometry is not None)
        result["right_turn_override"] = enabled
        if not enabled:
            return result
        curvature = float(geometry["curvature_m_inv"])
        result["curvature_m_inv"] = float(curvature)
        result["corner_fallback"] = True
        result["corner_fallback_reason"] = "right_turn_direct"
        result["corner_curvature_m_inv"] = float(curvature)
        return result

    @staticmethod
    def _reference_mode(counts):
        if counts["both"]:
            return "both" if not (counts["left"] or counts["right"]) else "mixed"
        if counts["left"]:
            return "left_only"
        if counts["right"]:
            return "right_only"
        return "none"

    def _clean_boundary_trace(self, line):
        line = np.asarray(line, dtype=np.float64)
        clean = np.full_like(line, np.nan)
        rows = np.flatnonzero(np.isfinite(line))
        if rows.size == 0:
            return clean
        max_step = float(self.config.boundary_max_step)
        max_gap = max(int(self.config.boundary_max_gap), 0)
        segments = []
        start = 0
        for index in range(1, rows.size):
            row_gap = int(rows[index] - rows[index - 1])
            value_jump = abs(float(line[rows[index]]) -
                             float(line[rows[index - 1]]))
            if row_gap > max_gap + 1 or value_jump > max_step:
                segments.append(rows[start:index])
                start = index
        segments.append(rows[start:])
        segment = max(segments, key=lambda r: r.size)
        if segment.size < self.config.boundary_min_length:
            return clean
        clean[segment] = line[segment]
        start, end = int(segment[0]), int(segment[-1])
        for row in range(start + 1, end):
            if np.isfinite(clean[row]):
                continue
            before = row - 1
            while before >= start and not np.isfinite(clean[before]):
                before -= 1
            after = row + 1
            while after <= end and not np.isfinite(line[after]):
                after += 1
            if (before >= start and after <= end and
                    after - before <= self.config.boundary_max_gap + 1):
                clean[row] = np.interp(row, [before, after],
                                       [clean[before], line[after]])
        return clean

    def _reject_false_double_boundary(self, left, right, image_width):
        """Drop a likely second trace cut from one sharply curved boundary.

        The open-source implementation used near-field width and line slope.
        This version additionally requires a strong slope contrast so a real,
        narrow two-sided lane is left untouched.
        """
        left = np.asarray(left, dtype=np.float64).copy()
        right = np.asarray(right, dtype=np.float64).copy()
        metrics = {
            "false_double_rejected": False,
            "false_double_rejected_side": "none",
        }
        roi_top, roi_bottom = self._roi_bounds(left.size)
        near_top = roi_top + (roi_bottom - roi_top) // 2
        rows = np.flatnonzero(
            np.isfinite(left) & np.isfinite(right) &
            (np.arange(left.size) >= near_top) &
            (np.arange(left.size) < roi_bottom))
        if rows.size < self.config.boundary_min_length:
            return left, right, metrics

        separation = right[rows] - left[rows]
        separation = separation[np.isfinite(separation) & (separation > 0)]
        if separation.size < self.config.boundary_min_length:
            return left, right, metrics
        median_width = float(np.median(separation))
        metrics["false_double_median_width"] = median_width
        if median_width >= image_width * self.config.false_double_max_width_ratio:
            return left, right, metrics

        left_slope = abs(float(np.polyfit(rows, left[rows], 1)[0]))
        right_slope = abs(float(np.polyfit(rows, right[rows], 1)[0]))
        metrics["false_double_left_slope"] = left_slope
        metrics["false_double_right_slope"] = right_slope
        smaller = min(left_slope, right_slope)
        larger = max(left_slope, right_slope)
        slope_ratio = larger / max(smaller, 1e-6)
        if (larger < self.config.false_double_min_horizontal_slope or
                slope_ratio < self.config.false_double_min_slope_ratio):
            return left, right, metrics

        rejected = "right" if left_slope < right_slope else "left"
        if rejected == "right":
            right[:] = np.nan
        else:
            left[:] = np.nan
        metrics["false_double_rejected"] = True
        metrics["false_double_rejected_side"] = rejected
        metrics["false_double_slope_ratio"] = slope_ratio
        return left, right, metrics

    @staticmethod
    def _boundary_modes(left, right):
        modes = np.zeros(len(left), dtype=np.uint8)
        left_ok = np.isfinite(left)
        right_ok = np.isfinite(right)
        modes[left_ok] = 1
        modes[right_ok] = 2
        modes[left_ok & right_ok] = 3
        return modes

    @staticmethod
    def _center_from_boundaries(left, right):
        center = np.full_like(left, np.nan)
        both = np.isfinite(left) & np.isfinite(right)
        center[both] = (left[both] + right[both]) / 2.0
        left_only = np.isfinite(left) & ~np.isfinite(right)
        right_only = np.isfinite(right) & ~np.isfinite(left)
        center[left_only] = left[left_only]
        center[right_only] = right[right_only]
        return center
        finite_widths = widths[np.isfinite(widths)]
        median_width = float(np.median(finite_widths)) if finite_widths.size else 0.0
        forward = (height - 1 - fit_rows).astype(np.float64)
        slope, intercept = np.polyfit(forward, center[fit_rows].astype(np.float64), 1)
        ref_y = roi_top + int(round((roi_height - 1) * self.config.reference_y_ratio))
        ref_x = float(slope * (height - 1 - ref_y) + intercept)
        raw_lateral = (ref_x - (width - 1) / 2.0) / max((width - 1) / 2.0, 1.0)
        raw_heading = float(np.arctan(slope))
        mapped = self.config.error_mapping.map(raw_lateral, raw_heading)
        cross = detect_cross_candidate(finite_widths)
        coverage = min(valid_rows.size / float(roi_height), 1.0)
        area = float(np.count_nonzero(lane_mask)) / float(width * roi_height)
        observed_sides = (np.isfinite(left[valid_rows]).astype(np.float32) +
                          np.isfinite(right[valid_rows]).astype(np.float32))
        boundary_quality = float(np.mean(observed_sides) / 2.0)
        confidence = float(np.clip(
            0.55 * coverage + 0.25 * boundary_quality +
            0.20 * min(area / 0.35, 1.0), 0.0, 1.0))
        return LaneAnalysisResult(
            True, mapped["error_y"], mapped["error_angle"], float(raw_lateral),
            raw_heading, confidence, cross.status, cross.score, None, (width, height),
            seed, left, right, center, lane_mask, binary,
            {"valid_rows": int(valid_rows.size), "median_lane_width": median_width,
             "fit_slope": float(slope), "fit_intercept": float(intercept),
             "reference_x": ref_x, "cross_width_ratio": cross.width_ratio,
             "cross_expanded_rows": cross.expanded_rows,
             "roi_top_y": roi_top, "roi_bottom_y": roi_bottom,
             "tracking_mode": self._dominant_tracking_mode(mode_counts),
             "tracking_mode_counts": mode_counts,
             "boundary_quality": boundary_quality,
             **self._corner_metrics(corner)})

    def draw_debug(self, image: np.ndarray, result: LaneAnalysisResult) -> np.ndarray:
        debug = self.make_work_image(image)
        roi_top, roi_bottom = self._roi_bounds(debug.shape[0])
        if roi_top > 0:
            debug[:roi_top] = (debug[:roi_top].astype(np.float32) * 0.35).astype(np.uint8)
        if roi_bottom < debug.shape[0]:
            debug[roi_bottom:] = (debug[roi_bottom:].astype(np.float32) * 0.35).astype(np.uint8)
        cv2.line(debug, (0, roi_top), (debug.shape[1] - 1, roi_top), (255, 255, 255), 1)
        cv2.line(debug, (0, max(roi_bottom - 1, 0)),
                 (debug.shape[1] - 1, max(roi_bottom - 1, 0)), (255, 255, 255), 1)
        if result.lane_mask is not None:
            overlay = np.zeros_like(debug)
            overlay[:, :, 1] = result.lane_mask
            debug = cv2.addWeighted(debug, 0.72, overlay, 0.28, 0.0)
        if result.seed_point is not None:
            cv2.circle(debug, result.seed_point, 4, (0, 255, 255), -1)
        segment = result.metrics.get("corner_segment")
        if segment is not None:
            cv2.line(debug, tuple(segment[:2]), tuple(segment[2:]), (255, 0, 255), 3)
        self._draw_line_points(debug, result.left_line, (255, 0, 0))
        self._draw_line_points(debug, result.right_line, (0, 0, 255))
        self._draw_line_points(debug, result.center_line, (0, 255, 255))
        if result.valid:
            mode = result.metrics.get("tracking_mode", "unknown")
            label = (f"lat={result.raw_lateral:+.4f} head={result.raw_heading:+.4f} "
                     f"mode={mode} cross={result.cross_status}:{result.cross_score:.2f}")
            color = (0, 255, 0)
        else:
            label, color = f"invalid: {result.reason}", (0, 0, 255)
        cv2.putText(debug, label, (6, 18), cv2.FONT_HERSHEY_SIMPLEX,
                    0.42, color, 1, cv2.LINE_AA)
        return debug

    def _segment_track(self, image: np.ndarray):
        if self.config.segmentation == "orange_boundary":
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            orange = cv2.inRange(
                hsv,
                np.array([self.config.orange_hue_low,
                          self.config.orange_saturation_low,
                          self.config.orange_value_low], dtype=np.uint8),
                np.array([self.config.orange_hue_high, 255, 255], dtype=np.uint8),
            )
            size = max(1, int(self.config.orange_dilate_kernel))
            if size > 1:
                kernel = np.ones((size, size), dtype=np.uint8)
                orange = cv2.morphologyEx(orange, cv2.MORPH_CLOSE, kernel)
                orange = cv2.dilate(orange, kernel, iterations=1)
            # White means traversable candidate. Orange boundary pixels are
            # barriers for the seed-connected-component search.
            return cv2.bitwise_not(orange), {
                "segmentation_threshold_mode": "orange_boundary",
                "segmentation_threshold_otsu": None,
                "segmentation_threshold_used": None,
            }

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if self.config.threshold is None:
            otsu_threshold, _ = cv2.threshold(
                gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
            lower = int(np.clip(self.config.adaptive_threshold_min, 0, 255))
            upper = int(np.clip(self.config.adaptive_threshold_max, 0, 255))
            if upper < lower:
                lower, upper = upper, lower
            threshold_used = float(np.clip(otsu_threshold, lower, upper))
            _, binary = cv2.threshold(
                gray, threshold_used, 255, cv2.THRESH_BINARY_INV)
            threshold_metrics = {
                "segmentation_threshold_mode": "otsu_clamped",
                "segmentation_threshold_otsu": float(otsu_threshold),
                "segmentation_threshold_used": threshold_used,
                "segmentation_threshold_min": lower,
                "segmentation_threshold_max": upper,
            }
        else:
            threshold_used = int(np.clip(self.config.threshold, 0, 255))
            _, binary = cv2.threshold(gray, threshold_used, 255,
                                      cv2.THRESH_BINARY_INV)
            threshold_metrics = {
                "segmentation_threshold_mode": "fixed",
                "segmentation_threshold_otsu": None,
                "segmentation_threshold_used": threshold_used,
            }
        roi_top, roi_bottom = self._roi_bounds(binary.shape[0])
        binary[:roi_top, :] = 0
        binary[roi_bottom:, :] = 0
        size = max(1, int(self.config.morphology_kernel))
        if size > 1:
            kernel = np.ones((size, size), dtype=np.uint8)
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        return binary, threshold_metrics

    def _find_seed(self, binary: np.ndarray) -> Optional[Tuple[int, int]]:
        height, width = binary.shape
        center_x = (width - 1) / 2.0
        min_run = max(3, int(width * self.config.min_seed_run_ratio))
        roi_top, roi_bottom = self._roi_bounds(height)
        roi_height = roi_bottom - roi_top
        if roi_height <= 0:
            return None
        band_top = max(roi_top, roi_bottom - max(1, int(roi_height * self.config.seed_band_ratio)))
        best = None
        for y in range(roi_bottom - 1, band_top - 1, -1):
            xs = np.flatnonzero(binary[y] > 0)
            if xs.size == 0:
                continue
            splits = np.where(np.diff(xs) > 1)[0] + 1
            for run in np.split(xs, splits):
                if run.size < min_run:
                    continue
                run_center = (float(run[0]) + float(run[-1])) / 2.0
                score = float(run.size) - 0.35 * abs(run_center - center_x)
                if best is None or score > best[0]:
                    best = (score, int(round(run_center)), y)
        return None if best is None else (best[1], best[2])

    def _roi_bounds(self, height: int) -> Tuple[int, int]:
        top_ratio = float(np.clip(self.config.roi_top_ratio, 0.0, 1.0))
        bottom_ratio = float(np.clip(self.config.roi_bottom_ratio, 0.0, 1.0))
        top = int(height * top_ratio)
        bottom = int(height * (1.0 - bottom_ratio))
        return min(max(top, 0), height), min(max(bottom, 0), height)

    @staticmethod
    def _connected_seed_component(binary: np.ndarray, seed: Tuple[int, int]) -> Optional[np.ndarray]:
        count, labels = cv2.connectedComponents((binary > 0).astype(np.uint8), connectivity=4)
        if count <= 1:
            return None
        label = int(labels[seed[1], seed[0]])
        if label == 0:
            return None
        return np.where(labels == label, 255, 0).astype(np.uint8)

    def _extract_boundaries(self, lane_mask: np.ndarray):
        height, width = lane_mask.shape
        left = np.full(height, np.nan, dtype=np.float32)
        right = np.full(height, np.nan, dtype=np.float32)
        center = np.full(height, np.nan, dtype=np.float32)
        widths = np.full(height, np.nan, dtype=np.float32)
        modes = np.zeros(height, dtype=np.uint8)
        min_width = max(3, int(width * self.config.min_lane_width_ratio))
        edge_margin = max(1, int(width * self.config.edge_margin_ratio))
        roi_top, roi_bottom = self._roi_bounds(height)
        roi_height = max(roi_bottom - roi_top, 1)
        for y in range(height):
            xs = np.flatnonzero(lane_mask[y] > 0)
            if xs.size < min_width:
                continue
            raw_left, raw_right = float(xs[0]), float(xs[-1])
            widths[y] = raw_right - raw_left + 1.0
            left_visible = raw_left > edge_margin
            right_visible = raw_right < (width - 1 - edge_margin)
            if left_visible:
                left[y] = raw_left
            if right_visible:
                right[y] = raw_right

            progress = np.clip((y - roi_top) / float(max(roi_height - 1, 1)), 0.0, 1.0)
            expected_width = width * (
                self.config.expected_lane_width_far_ratio +
                progress * (self.config.expected_lane_width_near_ratio -
                            self.config.expected_lane_width_far_ratio))
            if left_visible and right_visible:
                center[y] = (raw_left + raw_right) / 2.0
                modes[y] = 3
            elif left_visible:
                center[y] = raw_left + expected_width / 2.0
                modes[y] = 1
            elif right_visible:
                center[y] = raw_right - expected_width / 2.0
                modes[y] = 2
        return left, right, center, widths, modes

    def _keep_longest_center_segment(self, center: np.ndarray, width: int) -> np.ndarray:
        rows = np.flatnonzero(np.isfinite(center))
        filtered = np.full_like(center, np.nan)
        if rows.size == 0:
            return filtered
        max_step = max(2.0, float(width) * self.config.max_center_step_ratio)
        split_at = np.flatnonzero(
            (np.diff(rows) != 1) |
            (np.abs(np.diff(center[rows])) > max_step)) + 1
        segments = np.split(rows, split_at)
        segment = max(segments, key=lambda values: (values.size, values[-1]))
        filtered[segment] = center[segment]
        return filtered

    def _detect_steep_corner(self, lane_mask: np.ndarray) -> Optional[Dict[str, Any]]:
        """Find a near-horizontal boundary that row-wise x(y) cannot represent."""
        height, width = lane_mask.shape
        roi_top, roi_bottom = self._roi_bounds(height)
        roi_height = max(roi_bottom - roi_top, 1)
        edges = cv2.Canny(lane_mask, 40, 120)
        border = 3
        edges[:min(roi_top + border, height)] = 0
        edges[max(roi_bottom - border, 0):] = 0
        edges[:, :border] = 0
        edges[:, max(width - border, 0):] = 0
        min_length = max(12, int(width * self.config.corner_min_line_ratio))
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180.0, threshold=18,
                                minLineLength=min_length, maxLineGap=14)
        if lines is None:
            return None

        best = None
        for raw in np.asarray(lines).reshape(-1, 4):
            x1, y1, x2, y2 = (int(value) for value in raw)
            dx, dy = x2 - x1, y2 - y1
            length = float(np.hypot(dx, dy))
            horizontalness = abs(dx) / max(abs(dx) + abs(dy), 1.0)
            if length < min_length or horizontalness < self.config.corner_min_horizontalness:
                continue
            if y1 >= y2:
                near_x, near_y, far_x, far_y = x1, y1, x2, y2
            else:
                near_x, near_y, far_x, far_y = x2, y2, x1, y1
            forward_y = near_y - far_y
            # A nearly horizontal segment is a strong corner observation but
            # its left/right sign is dominated by one-pixel Hough jitter.
            min_direction_dy = max(6, int(abs(far_x - near_x) * 0.12))
            direction_known = forward_y >= min_direction_dy
            heading = (float(np.arctan2(far_x - near_x, forward_y))
                       if direction_known else 0.0)
            near_progress = float(np.clip(
                (near_y - roi_top) / float(roi_height), 0.0, 1.0))
            length_score = min(length / max(width * 0.50, 1.0), 1.0)
            score = float(0.55 * horizontalness + 0.25 * length_score +
                          0.20 * near_progress)
            candidate = {
                "score": score,
                "heading": heading,
                "direction_known": bool(direction_known),
                "near_progress": near_progress,
                "segment": [near_x, near_y, far_x, far_y],
            }
            if best is None or candidate["score"] > best["score"]:
                best = candidate
        return best

    def _direct_corner_geometry(self, center: np.ndarray, rows: np.ndarray,
                                corner: Optional[Dict[str, Any]],
                                height: int, width: int) -> Optional[Dict[str, Any]]:
        """Produce a stateless corner heading only when this frame has direction evidence."""
        heading = None
        source = None
        corner_usable = (corner is not None and
                         corner["score"] >= self.config.corner_min_score)
        roi_top, roi_bottom = self._roi_bounds(height)
        roi_height = max(roi_bottom - roi_top, 1)
        min_partial_rows = (8 if corner_usable else max(
            8, int(roi_height * self.config.min_partial_center_rows_ratio)))
        if (rows.size >= min_partial_rows and
                int(rows[-1] - rows[0]) >= min_partial_rows - 1):
            forward = (height - 1 - rows).astype(np.float64)
            slope, _ = np.polyfit(forward, center[rows].astype(np.float64), 1)
            partial_heading = float(np.arctan(slope))
            heading_threshold = (0.18 if corner_usable
                                 else self.config.min_partial_heading)
            if abs(partial_heading) >= heading_threshold:
                heading = partial_heading
                source = "partial_center"

        if corner_usable and corner["direction_known"]:
            contour_heading = float(corner["heading"])
            if heading is None:
                heading = contour_heading
                source = "contour_tangent"
            elif heading * contour_heading > 0 and abs(contour_heading) > abs(heading):
                heading = contour_heading
                source = "partial_center_and_contour"

        if heading is None:
            return None
        heading = float(np.clip(heading, -1.45, 1.45))
        anchor_x = (float(center[rows[-1]]) if rows.size
                    else (width - 1) / 2.0)
        lateral = ((anchor_x - (width - 1) / 2.0) /
                   max((width - 1) / 2.0, 1.0))
        corner_confirmed = bool(corner_usable)
        confidence = (float(np.clip(0.35 + 0.25 * corner["score"], 0.0, 0.60))
                      if corner_confirmed else 0.30)
        return {"raw_lateral": float(lateral), "raw_heading": heading,
                "direction_source": source,
                "tracking_mode": ("corner_direct" if corner_confirmed
                                  else "partial_center_direct"),
                "corner_confirmed": corner_confirmed,
                "confidence": confidence}

    @staticmethod
    def _corner_metrics(corner: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        if corner is None:
            return {"corner_detected": False}
        return {
            "corner_detected": True,
            "corner_score": corner["score"],
            "corner_heading": corner["heading"],
            "corner_direction_known": corner["direction_known"],
            "corner_segment": corner["segment"],
            "corner_near_progress": corner["near_progress"],
        }

    @staticmethod
    def _tracking_mode_counts(modes: np.ndarray, rows: np.ndarray) -> Dict[str, int]:
        selected = modes[rows] if rows.size else np.empty(0, dtype=np.uint8)
        return {
            "left_only": int(np.count_nonzero(selected == 1)),
            "right_only": int(np.count_nonzero(selected == 2)),
            "both": int(np.count_nonzero(selected == 3)),
        }

    @staticmethod
    def _dominant_tracking_mode(counts: Dict[str, int]) -> str:
        if not counts or max(counts.values(), default=0) == 0:
            return "none"
        nonzero = [name for name, count in counts.items() if count > 0]
        if len(nonzero) > 1:
            return "mixed"
        return nonzero[0]

    @staticmethod
    def _validate_image(image: np.ndarray) -> None:
        if not isinstance(image, np.ndarray):
            raise TypeError("image must be a numpy array")
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("image must have shape HxWx3")
        if image.size == 0:
            raise ValueError("image must not be empty")

    @staticmethod
    def _draw_line_points(image: np.ndarray, line: Optional[np.ndarray], color) -> None:
        if line is None:
            return
        for y, x in enumerate(line):
            if np.isfinite(x):
                cv2.circle(image, (int(round(float(x))), y), 1, color, -1)

    @staticmethod
    def _invalid(work_size: Size, reason: str, binary_mask=None,
                 lane_mask=None, seed_point=None, left_line=None,
                 right_line=None, center_line=None, metrics=None) -> LaneAnalysisResult:
        return LaneAnalysisResult(
            False, None, None, None, None, 0.0, "unknown", 0.0, reason,
            work_size, seed_point, left_line, right_line, center_line,
            lane_mask, binary_mask, metrics or {})
