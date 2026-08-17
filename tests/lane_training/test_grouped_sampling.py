import pytest

from lane_training.grouped_sampling import (
    build_epoch_batches,
    partition_action_rows,
)


def test_partition_action_rows_uses_source_mask_and_active_threshold():
    rows = [
        {"source": "cv", "target_mask": [1.0, 1.0], "kappa_action": 0.0},
        {"source": "manual", "target_mask": [0.0, 1.0], "kappa_action": 0.05},
        {"source": "manual", "target_mask": [0.0, 1.0], "kappa_action": -0.049},
    ]
    assert partition_action_rows(rows, active_threshold=0.05) == {
        "cv": [0], "manual_active": [1], "manual_context": [2],
    }

    rows[1]["target_mask"] = [1.0, 1.0]
    with pytest.raises(ValueError, match="manual target_mask"):
        partition_action_rows(rows)


def test_epoch_batches_cover_cv_and_are_reproducible():
    groups = {
        "cv": list(range(2822)),
        "manual_active": list(range(2822, 2896)),
        "manual_context": list(range(2896, 3468)),
    }
    batches, report = build_epoch_batches(
        groups, batch_size=64, seed=20260817, epoch=1,
    )
    assert len(batches) == 63
    assert all(len(batch) == 64 for batch in batches)
    assert report["draw_counts"] == {
        "cv": 2835, "manual_active": 599, "manual_context": 598,
    }
    assert set(groups["cv"]).issubset({item for batch in batches for item in batch})
    assert batches == build_epoch_batches(
        groups, batch_size=64, seed=20260817, epoch=1,
    )[0]
    assert batches != build_epoch_batches(
        groups, batch_size=64, seed=20260817, epoch=2,
    )[0]
