import unittest

import cv2
import numpy as np

from smartcar.whalesbot.tools.lane_collect import (
    LaneAnalyzerConfig,
    OpenCVLaneAnalyzer,
    StandardLaneReference,
)


def make_track(offset=0, cross=False):
    image = np.full((240, 320, 3), 210, dtype=np.uint8)
    polygon = np.array([[118 + offset, 239], [145 + offset, 35],
                        [175 + offset, 35], [202 + offset, 239]], dtype=np.int32)
    cv2.fillPoly(image, [polygon], (35, 35, 35))
    if cross:
        cv2.rectangle(image, (20, 95), (300, 135), (35, 35, 35), -1)
    return image


def make_right_boundary_only_track():
    image = np.full((240, 320, 3), 210, dtype=np.uint8)
    polygon = np.array([[0, 239], [0, 35], [80, 35], [180, 239]], dtype=np.int32)
    cv2.fillPoly(image, [polygon], (35, 35, 35))
    return image


def make_near_horizontal_corner():
    image = np.full((240, 320, 3), 210, dtype=np.uint8)
    polygon = np.array([[0, 110], [319, 80], [319, 239], [0, 239]], dtype=np.int32)
    cv2.fillPoly(image, [polygon], (35, 35, 35))
    return image


def make_reference(roi_top=35, roi_bottom=192):
    rows = np.arange(240, dtype=np.float64)
    scale = (rows - 35.0) / (239.0 - 35.0)
    left = 145.0 + (118.0 - 145.0) * scale
    right = 175.0 + (202.0 - 175.0) * scale
    return StandardLaneReference(
        image_size=(320, 240),
        roi_top=roi_top,
        roi_bottom=roi_bottom,
        left_boundary=left,
        right_boundary=right,
        perspective=np.linspace(80.0, 10.0, roi_bottom - roi_top),
        lane_width=40.0,
    )


class OpenCVLaneAnalyzerTests(unittest.TestCase):
    def setUp(self):
        self.analyzer = OpenCVLaneAnalyzer(
            LaneAnalyzerConfig(work_size=(320, 240),
                               roi_top_ratio=35.0 / 240.0,
                               segmentation="dark"),
            reference=make_reference(),
        )

    def test_cnn_image_matches_runtime_geometry(self):
        image = self.analyzer.make_cnn_image(make_track())
        self.assertEqual(image.shape, (128, 128, 3))
        self.assertEqual(image.dtype, np.uint8)

    def test_straight_centered_track_is_valid(self):
        result = self.analyzer.process(make_track())
        self.assertTrue(result.valid, result.reason)
        self.assertAlmostEqual(result.raw_lateral, 0.0, delta=0.08)
        self.assertAlmostEqual(result.raw_heading, 0.0, delta=0.08)
        self.assertAlmostEqual(result.metrics["perspective_k"], 0.0, delta=0.02)
        self.assertAlmostEqual(result.metrics["perspective_b"], 0.0, delta=0.08)
        self.assertEqual(result.metrics["perspective_fit_sides"], 2)

    def test_right_shift_has_positive_lateral_error(self):
        centered = self.analyzer.process(make_track())
        shifted = self.analyzer.process(make_track(offset=35))
        self.assertTrue(shifted.valid, shifted.reason)
        self.assertGreater(shifted.raw_lateral, centered.raw_lateral + 0.10)
        self.assertAlmostEqual(shifted.raw_heading, 0.0, delta=0.13)
        self.assertIn("curvature", shifted.metrics)
        self.assertLess(
            shifted.metrics["pure_pursuit_curvature_m_inv"], 0.0)

    def test_pure_pursuit_uses_only_the_configured_target_point(self):
        result = self.analyzer._pure_pursuit_target(
            fit_s=np.array([10.0, 20.0, 40.0, 80.0]),
            fit_x=np.array([0.0, 2.0, 30.0, 60.0]),
            lookahead_units=20.0,
            units_per_meter=100.0,
        )
        self.assertAlmostEqual(result["target_s_m"], 0.20)
        self.assertAlmostEqual(result["target_x_m"], 0.02)
        self.assertAlmostEqual(
            result["curvature_m_inv"], -2.0 * 0.02 / (0.20 ** 2 + 0.02 ** 2))
        self.assertFalse(result["lookahead_clipped"])

    def test_sharp_gain_only_applies_to_large_lateral_ratio(self):
        ordinary = self.analyzer._apply_sharp_curvature_gain({
            "target_s_m": 0.30,
            "target_x_m": 0.12,
            "curvature_m_inv": -2.3,
        }, ratio_threshold=0.80, sharp_gain=1.8)
        sharp = self.analyzer._apply_sharp_curvature_gain({
            "target_s_m": 0.30,
            "target_x_m": 0.27,
            "curvature_m_inv": -3.3,
        }, ratio_threshold=0.80, sharp_gain=1.8)
        self.assertFalse(ordinary["sharp_boosted"])
        self.assertAlmostEqual(ordinary["curvature_m_inv"], -2.3)
        self.assertTrue(sharp["sharp_boosted"])
        self.assertAlmostEqual(sharp["curvature_m_inv"], -5.94)
        right_sharp = self.analyzer._apply_sharp_curvature_gain({
            "target_s_m": 0.30,
            "target_x_m": 0.27,
            "curvature_m_inv": -3.3,
        }, ratio_threshold=0.80, sharp_gain=1.8,
            reference_mode="left_only", right_turn_sharp_gain=1.20)
        self.assertTrue(right_sharp["sharp_boosted"])
        self.assertAlmostEqual(right_sharp["sharp_gain_applied"], 1.20)
        self.assertAlmostEqual(right_sharp["curvature_m_inv"], -3.96)
        left_sharp = self.analyzer._apply_sharp_curvature_gain({
            "target_s_m": 0.30,
            "target_x_m": -0.27,
            "curvature_m_inv": 3.3,
        }, ratio_threshold=0.80, sharp_gain=1.8,
            reference_mode="right_only", right_turn_sharp_gain=1.0)
        self.assertAlmostEqual(left_sharp["sharp_gain_applied"], 1.8)
        self.assertAlmostEqual(left_sharp["curvature_m_inv"], 5.94)

    def test_global_otsu_threshold_is_clamped_to_configured_bounds(self):
        analyzer = OpenCVLaneAnalyzer(
            LaneAnalyzerConfig(
                threshold=None,
                adaptive_threshold_min=150,
                adaptive_threshold_max=165,
                work_size=(320, 240),
                roi_top_ratio=35.0 / 240.0,
                segmentation="dark"),
            reference=make_reference(),
        )
        low_histogram = np.full((240, 320, 3), 100, dtype=np.uint8)
        low_histogram[:, :160] = 20
        _, low_metrics = analyzer._segment_track(low_histogram)
        self.assertLess(low_metrics["segmentation_threshold_otsu"], 150)
        self.assertEqual(low_metrics["segmentation_threshold_used"], 150)

        high_histogram = np.full((240, 320, 3), 240, dtype=np.uint8)
        high_histogram[:, :160] = 180
        _, high_metrics = analyzer._segment_track(high_histogram)
        self.assertGreater(high_metrics["segmentation_threshold_otsu"], 165)
        self.assertEqual(high_metrics["segmentation_threshold_used"], 165)

    def test_corner_tangent_fallback_only_replaces_flat_entering_target(self):
        pursuit = {
            "curvature_m_inv": 0.1,
            "target_lateral_ratio": 0.02,
            "sharp_boosted": False,
        }
        corner = {
            "score": 0.78,
            "heading": -1.40,
            "direction_known": True,
            "near_progress": 0.35,
        }
        result = self.analyzer._apply_corner_curvature_fallback(
            pursuit, corner, "mixed", 0.40, 0.30, 0.70, 0.15, 0.50,
            0.58, 0.15, 0.15)
        self.assertTrue(result["corner_fallback"])
        self.assertEqual(result["corner_fallback_reason"], "strong_score")
        self.assertGreater(result["curvature_m_inv"], 6.0)
        self.assertAlmostEqual(
            result["curvature_m_inv"],
            result["corner_curvature_m_inv"])

        exiting = dict(corner, near_progress=0.75)
        result = self.analyzer._apply_corner_curvature_fallback(
            pursuit, exiting, "mixed", 0.40, 0.30, 0.70, 0.15, 0.50,
            0.58, 0.15, 0.15)
        self.assertFalse(result["corner_fallback"])
        self.assertAlmostEqual(result["curvature_m_inv"], 0.1)

    def test_weak_corner_fallback_requires_mixed_near_straight_entry(self):
        pursuit = {
            "curvature_m_inv": 0.0,
            "target_lateral_ratio": 0.01,
            "sharp_boosted": False,
        }
        corner = {
            "score": 0.60,
            "heading": -1.43,
            "direction_known": True,
            "near_progress": 0.20,
        }
        args = (pursuit, corner, "mixed", 0.02, 0.30, 0.70, 0.15,
                0.50, 0.58, 0.15, 0.15)
        result = self.analyzer._apply_corner_curvature_fallback(*args)
        self.assertTrue(result["corner_fallback"])
        self.assertEqual(
            result["corner_fallback_reason"], "early_mixed_flat")

        result = self.analyzer._apply_corner_curvature_fallback(
            pursuit, corner, "left_only", 0.02, 0.30, 0.70, 0.15,
            0.50, 0.58, 0.15, 0.15)
        self.assertFalse(result["corner_fallback"])
        result = self.analyzer._apply_corner_curvature_fallback(
            pursuit, corner, "mixed", 0.30, 0.30, 0.70, 0.15,
            0.50, 0.58, 0.15, 0.15)
        self.assertFalse(result["corner_fallback"])

    def test_right_turn_trigger_is_stateless_near_entry_only(self):
        corner = {
            "score": 0.75,
            "heading": -1.40,
            "direction_known": True,
            "near_progress": 0.30,
            "segment": [260, 100, 205, 90],
        }
        args = (corner, "mixed", 0.10, 0.70, -1.35, 0.20,
                0.20, 0.45)
        self.assertEqual(
            self.analyzer._right_turn_reason(*args), "near_entry")
        continuation = dict(corner, score=0.82, direction_known=False)
        result = self.analyzer._right_turn_reason(
            continuation, "left_only", 0.25, 0.70, -1.35,
            0.20, 0.20, 0.45)
        self.assertIsNone(result)
        result = self.analyzer._right_turn_reason(
            dict(corner, near_progress=0.48), "mixed", 0.10,
            0.70, -1.35, 0.20, 0.20, 0.45)
        self.assertIsNone(result)
        self.assertTrue(self.analyzer._right_turn_geometry(
            corner, "mixed", -1.35))
        self.assertFalse(self.analyzer._right_turn_geometry(
            corner, "left_only", -1.35))
        pursuit = {
            "curvature_m_inv": 0.1,
            "target_lateral_ratio": 0.10,
            "sharp_boosted": False,
            "corner_fallback": False,
            "corner_fallback_reason": None,
            "corner_curvature_m_inv": None,
        }
        geometry = self.analyzer._right_turn_curvature(
            corner, self.analyzer.reference.perspective,
            self.analyzer.reference.roi_top, 100.0, 0.65, 0.20, 10.0)
        overridden = self.analyzer._apply_right_turn_override(
            pursuit, True, geometry)
        self.assertTrue(overridden["right_turn_override"])
        expected_index = 100 - self.analyzer.reference.roi_top
        expected_distance = (
            self.analyzer.reference.perspective[expected_index] / 100.0)
        expected_curvature = (
            -2.0 * np.sin(1.40) / expected_distance * 0.65)
        self.assertAlmostEqual(geometry["distance_m"], expected_distance)
        self.assertAlmostEqual(
            overridden["curvature_m_inv"], expected_curvature)
        self.assertEqual(
            overridden["corner_fallback_reason"], "right_turn_direct")

    def test_cross_expansion_is_reported(self):
        result = self.analyzer.process(make_track(cross=True))
        self.assertTrue(result.valid, result.reason)
        self.assertGreater(result.cross_score, 0.0)

    def test_top_and_bottom_roi_keep_seed_inside_active_region(self):
        analyzer = OpenCVLaneAnalyzer(LaneAnalyzerConfig(
            work_size=(320, 240), segmentation="dark",
            roi_top_ratio=0.30, roi_bottom_ratio=0.20),
            reference=make_reference(72, 192))
        result = analyzer.process(make_track())
        self.assertTrue(result.valid, result.reason)
        self.assertEqual(np.count_nonzero(result.binary_mask[:72]), 0)
        self.assertEqual(np.count_nonzero(result.binary_mask[192:]), 0)
        self.assertGreaterEqual(result.seed_point[1], 72)
        self.assertLess(result.seed_point[1], 192)

    def test_frame_edge_is_missing_boundary_and_uses_single_side(self):
        result = self.analyzer.process(make_right_boundary_only_track())
        self.assertTrue(result.valid, result.reason)
        self.assertEqual(result.metrics["tracking_mode"], "right_only")
        self.assertEqual(np.count_nonzero(np.isfinite(result.left_line)), 0)
        self.assertGreater(np.count_nonzero(np.isfinite(result.right_line)), 20)
        self.assertEqual(result.metrics["perspective_fit_sides"], 1)

    def test_both_missing_boundaries_are_not_reported_as_straight(self):
        image = np.full((240, 320, 3), 35, dtype=np.uint8)
        result = self.analyzer.process(image)
        self.assertFalse(result.valid)
        self.assertIsNone(result.raw_lateral)
        self.assertIsNone(result.raw_heading)
        self.assertEqual(result.metrics.get("tracking_mode", "none"), "none")

    def test_near_horizontal_corner_without_reference_rows_is_invalid(self):
        result = self.analyzer.process(make_near_horizontal_corner())
        self.assertFalse(result.valid)
        self.assertEqual(result.reason, "standard reference has too few valid boundary rows")

    def test_exact_horizontal_corner_without_direction_stays_invalid(self):
        image = np.full((240, 320, 3), 210, dtype=np.uint8)
        image[100:] = 35
        result = self.analyzer.process(image)
        self.assertFalse(result.valid)

    def test_strong_partial_center_does_not_require_corner_contour(self):
        center = np.full(240, np.nan, dtype=np.float32)
        rows = np.arange(72, 92)
        center[rows] = 250 - 2.0 * (rows - rows[0])
        result = self.analyzer._direct_corner_geometry(
            center, rows, None, height=240, width=320)
        self.assertIsNotNone(result)
        self.assertEqual(result["tracking_mode"], "partial_center_direct")
        self.assertGreater(abs(result["raw_heading"]), 0.35)

    def test_invalid_input_does_not_raise(self):
        result = self.analyzer.process(None)
        self.assertFalse(result.valid)
        self.assertIsNone(result.error_y)
        self.assertIsNotNone(result.reason)

    def test_record_omits_image_arrays(self):
        record = self.analyzer.process(make_track()).to_record()
        self.assertNotIn("lane_mask", record)
        self.assertNotIn("left_line", record)
        self.assertEqual(record["work_size"], [320, 240])


if __name__ == "__main__":
    unittest.main()
