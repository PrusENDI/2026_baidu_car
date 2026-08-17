import numpy as np
import pytest

from lane_training.action_sequence_metrics import (
    replay_vehicle_commands,
    summarize_action_sequence,
)


def test_sequence_summary_measures_turn_behavior():
    target = np.array([0.0, 0.0, 1.0, 2.0, 1.0, 0.0])
    prediction = np.array([0.06, 0.0, 1.0, 1.0, 1.0, 0.0])
    report = summarize_action_sequence(target, prediction, active_threshold=0.05)
    assert report["active_count"] == 3
    assert report["direction_accuracy"] == pytest.approx(1.0)
    assert report["active_recall"] == pytest.approx(1.0)
    assert report["false_steer_rate"] == pytest.approx(1 / 3)
    assert report["impulse_ratio"] == pytest.approx(0.75)
    assert report["turn_count"] == 1
    assert report["median_onset_offset_frames"] == pytest.approx(0.0)


def test_vehicle_replay_uses_speed_demand_real_dt_and_wz_clip():
    rows = [
        {"image_path": "session/a.jpg", "effective_dt_s": 0.05},
        {"image_path": "session/b.jpg", "effective_dt_s": 0.05},
    ]
    predictions = np.array([[0.0, 0.0], [1.0, 10.0]], dtype=np.float32)
    report = replay_vehicle_commands(rows, predictions)
    assert report["actual_vx"][0] == pytest.approx(0.30)
    assert report["actual_vx"][1] == pytest.approx(0.30 - 0.194 * 0.05)
    assert report["wz"][1] == pytest.approx(1.50)
    assert report["angular_clip_rate"] == pytest.approx(0.5)
