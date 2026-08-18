import numpy as np
import pytest

from tools.lane_training.evaluate_new_track import summarize_predictions


def test_summary_captures_curve_shape_false_turn_and_vy_drift():
    labels = np.array([
        [0.0, 0.0], [0.0, 0.0],
        [0.0, 0.1], [0.0, 0.2], [0.0, 0.1], [0.0, 0.0],
    ])
    predictions = np.array([
        [0.01, 0.06], [0.02, 0.0],
        [0.01, 0.1], [0.01, 0.1], [0.01, 0.1], [0.01, 0.0],
    ])

    report = summarize_predictions(labels, predictions, min_active=2)

    assert report["straight_false_turn_rate"] == pytest.approx(1 / 3)
    assert report["direction_consistency"] == pytest.approx(1.0)
    assert report["normalized_curve_error"] == pytest.approx(0.25)
    assert report["median_impulse_ratio"] == pytest.approx(0.75)
    assert report["mean_abs_vy"] == pytest.approx(0.0116666667)
