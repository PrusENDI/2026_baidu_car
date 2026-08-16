import unittest

from smartcar.whalesbot.tools.lane_collect import (
    LaneAnalysisResult,
    PreviewTimingFilter,
)


def observation(heading=0.1, lateral=0.0, mode="both", rows=120,
                corner=False, corner_score=0.0):
    return LaneAnalysisResult(
        valid=True,
        error_y=lateral,
        error_angle=heading,
        raw_lateral=lateral,
        raw_heading=heading,
        confidence=1.0,
        cross_status="normal",
        cross_score=0.0,
        reason=None,
        work_size=(320, 240),
        metrics={
            "reference_tracking_mode": mode,
            "reference_rows": rows,
            "roi_top_y": 72,
            "roi_bottom_y": 192,
            "corner_detected": corner,
            "corner_score": corner_score,
        },
    )


class PreviewTimingFilterTests(unittest.TestCase):
    def test_far_direction_is_preserved_when_single_boundary_reverses(self):
        timing = PreviewTimingFilter()
        for _ in range(3):
            timing.update(observation(
                heading=0.20, mode="left_only", rows=120))

        result = timing.update(observation(
            heading=-0.20, mode="left_only", rows=70,
            corner=True, corner_score=0.80))

        self.assertGreater(result.raw_heading, 0.0)
        self.assertTrue(result.metrics["direction_held"])
        self.assertEqual(result.metrics["preview_direction"], 1)

    def test_middle_near_arrival_controls_when_preview_is_executed(self):
        timing = PreviewTimingFilter()
        preview_only = timing.update(observation(
            heading=0.20, lateral=0.0, mode="both", rows=120))
        arrived = timing.update(observation(
            heading=0.20, lateral=0.05, mode="both", rows=70,
            corner=True, corner_score=0.85))

        self.assertAlmostEqual(preview_only.raw_heading, 0.0, places=6)
        self.assertGreater(arrived.metrics["arrival_weight"], 0.95)
        self.assertGreater(arrived.raw_heading, 0.15)


if __name__ == "__main__":
    unittest.main()
