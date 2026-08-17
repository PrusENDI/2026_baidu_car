import json
from pathlib import Path

import pytest
from PIL import Image

from lane_training.manifest import (
    Sample,
    load_cv_action_session,
    load_labeled_session,
    load_manual_action_session,
    validate_samples,
)


def test_load_cv_action_session_uses_two_output_contract(tmp_path: Path):
    image = tmp_path / "000000.jpg"
    Image.new("RGB", (320, 240), "white").save(image)
    data_json = tmp_path / "data.json"
    data_json.write_text(json.dumps([{
        "img_path": image.name,
        "state": [0.2, 0.0, -0.24],
        "model_target": {"speed_demand": 0.75, "kappa_action": -1.2},
        "target_mask": [1.0, 1.0],
        "label_semantics": "raw_control_with_model_target",
    }]), encoding="utf-8")

    rows = load_cv_action_session(data_json)

    assert rows[0]["speed_demand"] == 0.75
    assert rows[0]["kappa_action"] == -1.2
    assert rows[0]["target_mask"] == [1.0, 1.0]


def test_load_manual_action_session_masks_speed_target(tmp_path: Path):
    image = tmp_path / "manual.jpg"
    Image.new("RGB", (320, 240), "white").save(image)
    data_json = tmp_path / "data.json"
    data_json.write_text(json.dumps([{
        "img_path": image.name,
        "state": [0.20, 0.0, -0.30],
    }]), encoding="utf-8")

    rows = load_manual_action_session(data_json)

    assert rows[0]["target_mask"] == [0.0, 1.0]
    assert rows[0]["kappa_action"] == pytest.approx(-1.5)


def test_rejects_same_image_in_conflicting_splits(tmp_path: Path):
    image = tmp_path / "same.jpg"
    rows = [
        Sample(image, 0.0, 0.0, "collected", "lap_001", "full_lap", "mixed", "train"),
        Sample(image, 0.0, 0.0, "collected", "lap_001", "full_lap", "mixed", "validation"),
    ]

    with pytest.raises(ValueError, match="conflicting splits"):
        validate_samples(rows)


def test_rejects_session_split_across_train_and_validation(tmp_path: Path):
    rows = [
        Sample(tmp_path / "a.jpg", 0.0, 0.0, "collected", "lap_001", "full_lap", "mixed", "train"),
        Sample(tmp_path / "b.jpg", 0.0, 0.0, "collected", "lap_001", "full_lap", "mixed", "validation"),
    ]

    with pytest.raises(ValueError, match="session lap_001 has conflicting splits"):
        validate_samples(rows)


def test_rejects_non_finite_command(tmp_path: Path):
    row = Sample(
        tmp_path / "a.jpg",
        float("nan"),
        0.0,
        "official",
        "official",
        "official",
        "official",
        "train",
    )

    with pytest.raises(ValueError, match="non-finite command"):
        validate_samples([row])


def test_loads_vy_and_yaw_but_not_forward_velocity(tmp_path: Path):
    image = tmp_path / "0000.jpg"
    Image.new("RGB", (16, 12), "white").save(image)
    data_json = tmp_path / "data.json"
    data_json.write_text(
        json.dumps([{"img_path": "0000.jpg", "state": [0.28, 0.05, -0.4]}]),
        encoding="utf-8",
    )

    samples = load_labeled_session(
        data_json,
        source="collected",
        session="lap_009",
        session_type="hard_segment",
        scene_type="intersection",
        split="train",
    )

    assert len(samples) == 1
    assert samples[0].vy == pytest.approx(0.05)
    assert samples[0].yaw == pytest.approx(-0.4)


def test_rejects_image_path_outside_session(tmp_path: Path):
    data_json = tmp_path / "data.json"
    data_json.write_text(
        json.dumps([{"img_path": "../outside.jpg", "state": [0.28, 0.0, 0.0]}]),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unsafe image path"):
        load_labeled_session(
            data_json,
            source="collected",
            session="lap_009",
            session_type="hard_segment",
            scene_type="intersection",
            split="train",
        )
