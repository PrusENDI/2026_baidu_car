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


def make_reference(roi_top=0, roi_bottom=192):
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
        perspective=np.ones(roi_bottom - roi_top, dtype=np.float64),
        lane_width=float(np.median(right - left)),
    )


class OpenCVLaneAnalyzerTests(unittest.TestCase):
    def setUp(self):
        self.analyzer = OpenCVLaneAnalyzer(
            LaneAnalyzerConfig(work_size=(320, 240), roi_top_ratio=0.0,
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

    def test_right_shift_has_positive_lateral_error(self):
        centered = self.analyzer.process(make_track())
        shifted = self.analyzer.process(make_track(offset=35))
        self.assertTrue(shifted.valid, shifted.reason)
        self.assertGreater(shifted.raw_lateral, centered.raw_lateral + 0.10)

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
