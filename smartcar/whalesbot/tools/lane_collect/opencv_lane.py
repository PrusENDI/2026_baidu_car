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
    # The old implementation used 190, but the new camera/lighting makes
    # that too permissive on the supplied photos.  175 is provisional and
    # must be rechecked against the complete course before closed-loop use.
    threshold: Optional[int] = 175
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
        binary = self._segment_track(work)
        seed = self._find_seed(binary)
        if seed is None:
            return self._invalid((width, height), "no dark seed connected to near field", binary)
        lane_mask = self._connected_seed_component(binary, seed)
        if lane_mask is None:
            return self._invalid((width, height), "seed component is empty", binary)
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
            binary, seed, corner, boundary_filter)

    def _standard_relative_result(self, width, height, left, right, center,
                                  widths, modes, lane_mask, binary, seed,
                                  corner, boundary_filter):
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
        mapped = self.config.error_mapping.map(raw_lateral, raw_heading)
        cross = detect_cross_candidate(widths[np.isfinite(widths)])
        center = np.asarray(center, dtype=np.float64).copy()
        center[rows[valid]] = current_center[valid]
        perspective_fit_sides = int(bool(np.any(left_ok))) + int(
            bool(np.any(right_ok)))
        metrics = {
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

    def _segment_track(self, image: np.ndarray) -> np.ndarray:
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
            return cv2.bitwise_not(orange)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if self.config.threshold is None:
            _, binary = cv2.threshold(gray, 0, 255,
                                      cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        else:
            _, binary = cv2.threshold(gray, self.config.threshold, 255,
                                      cv2.THRESH_BINARY_INV)
        roi_top, roi_bottom = self._roi_bounds(binary.shape[0])
        binary[:roi_top, :] = 0
        binary[roi_bottom:, :] = 0
        size = max(1, int(self.config.morphology_kernel))
        if size > 1:
            kernel = np.ones((size, size), dtype=np.uint8)
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        return binary

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
