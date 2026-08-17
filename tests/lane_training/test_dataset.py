import numpy as np
import pytest
from PIL import Image

from lane_training.dataset import (
    CLEAN_ACTION_AUGMENTATION,
    MILD_FIXED_TRACK_ACTION_AUGMENTATION,
    ActionAugmentationConfig,
    LaneDataset,
    augment_rgb,
    horizontal_flip,
    preprocess_rgb,
)


def test_fixed_track_augmentation_profiles_match_approved_ranges():
    assert CLEAN_ACTION_AUGMENTATION.application_probability == 0.0
    mild = MILD_FIXED_TRACK_ACTION_AUGMENTATION
    assert mild.application_probability == pytest.approx(0.30)
    assert mild.brightness_range == pytest.approx((0.95, 1.05))
    assert mild.contrast_range == pytest.approx((0.95, 1.05))
    assert mild.saturation_range == pytest.approx((0.97, 1.03))
    assert mild.hue_delta == 2
    assert mild.blur_probability == 0.0
    assert mild.noise_probability == 0.0
    assert mild.horizontal_flip_probability == 0.0


def test_action_augmentation_config_rejects_invalid_probability():
    with pytest.raises(ValueError, match="application_probability"):
        ActionAugmentationConfig(application_probability=1.1)


def test_preprocess_contract_and_normalization():
    image = np.full((240, 320, 3), 255, dtype=np.uint8)

    tensor = preprocess_rgb(image)

    assert tensor.shape == (3, 128, 128)
    assert tensor.dtype == np.float32
    assert np.allclose(tensor, 1.0)


def test_horizontal_flip_negates_both_commands():
    image = np.arange(18, dtype=np.uint8).reshape(2, 3, 3)

    flipped, label = horizontal_flip(
        image, np.array([0.05, -0.4], dtype=np.float32)
    )

    assert np.array_equal(flipped, image[:, ::-1])
    assert np.allclose(label, [-0.05, 0.4])


def test_horizontal_flip_keeps_speed_demand_and_reverses_curvature():
    image = np.arange(18, dtype=np.uint8).reshape(2, 3, 3)

    flipped, label = horizontal_flip(
        image,
        np.array([0.75, -1.2], dtype=np.float32),
        label_semantics="speed_demand_action_curvature",
    )

    assert np.array_equal(flipped, image[:, ::-1])
    assert np.allclose(label, [0.75, 1.2])


def test_action_augmentation_can_disable_horizontal_flip():
    image = np.full((16, 16, 3), 128, dtype=np.uint8)
    label = np.array([0.75, -1.2], dtype=np.float32)

    for seed in range(20):
        _, augmented_label = augment_rgb(
            image,
            label,
            np.random.default_rng(seed),
            label_semantics="speed_demand_action_curvature",
            horizontal_flip_probability=0.0,
        )
        assert np.allclose(augmented_label, label)


def test_clean_action_augmentation_is_identical_and_not_applied():
    image = np.arange(16 * 16 * 3, dtype=np.uint8).reshape(16, 16, 3)
    label = np.array([0.75, -1.2], dtype=np.float32)

    for seed in range(5):
        augmented, target, applied = augment_rgb(
            image,
            label,
            np.random.default_rng(seed),
            label_semantics="speed_demand_action_curvature",
            config=CLEAN_ACTION_AUGMENTATION,
            return_applied=True,
        )
        assert np.array_equal(augmented, image)
        assert np.array_equal(target, label)
        assert applied is False


def test_forced_mild_augmentation_changes_pixels_without_disabled_operations(
    monkeypatch,
):
    image = np.full((16, 16, 3), [90, 130, 180], dtype=np.uint8)
    label = np.array([0.75, -1.2], dtype=np.float32)
    forced = ActionAugmentationConfig(
        application_probability=1.0,
        brightness_range=(1.05, 1.05),
        contrast_range=(1.0, 1.0),
        saturation_range=(1.0, 1.0),
        hue_delta=0,
        blur_probability=0.0,
        noise_probability=0.0,
        horizontal_flip_probability=0.0,
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("disabled spatial operation was called")

    monkeypatch.setattr("lane_training.dataset.cv2.GaussianBlur", forbidden)
    monkeypatch.setattr("lane_training.dataset.horizontal_flip", forbidden)
    augmented, target, applied = augment_rgb(
        image,
        label,
        np.random.default_rng(7),
        label_semantics="speed_demand_action_curvature",
        config=forced,
        return_applied=True,
    )

    assert not np.array_equal(augmented, image)
    assert np.array_equal(target, label)
    assert applied is True


def test_dataset_counts_applied_action_augmentations(tmp_path):
    image = tmp_path / "action.jpg"
    Image.new("RGB", (320, 240), (90, 130, 180)).save(image)
    rows = [{
        "image_path": str(image),
        "speed_demand": 0.75,
        "kappa_action": -1.2,
        "label_semantics": "speed_demand_action_curvature",
        "target_mask": [1.0, 1.0],
    }]
    dataset = LaneDataset(
        rows,
        training=True,
        return_target_mask=True,
        action_augmentation_config=CLEAN_ACTION_AUGMENTATION,
    )
    dataset.set_epoch(1)
    dataset[0]
    assert dataset.augmentation_applied_count == 0

    forced = ActionAugmentationConfig(
        application_probability=1.0,
        blur_probability=0.0,
        noise_probability=0.0,
        horizontal_flip_probability=0.0,
    )
    dataset.set_action_augmentation_config(forced)
    dataset.set_epoch(2)
    dataset[0]
    assert dataset.augmentation_applied_count == 1
    dataset.set_epoch(3)
    assert dataset.augmentation_applied_count == 0


def test_dataset_preload_caches_resized_rgb(tmp_path):
    image = tmp_path / "frame.jpg"
    Image.new("RGB", (320, 240), "white").save(image)
    dataset = LaneDataset(
        [
            {
                "image_path": str(image),
                "vy": 0.0,
                "yaw": 0.0,
            }
        ],
        training=False,
    )

    dataset.preload()

    assert dataset.cached_image_count == 1
    features, label = dataset[0]
    assert features.shape == (3, 128, 128)
    assert np.allclose(label, [0.0, 0.0])


def test_dataset_returns_manual_target_mask_when_requested(tmp_path):
    image = tmp_path / "manual.jpg"
    Image.new("RGB", (320, 240), "white").save(image)
    dataset = LaneDataset([{
        "image_path": str(image),
        "speed_demand": 0.0,
        "kappa_action": 1.25,
        "label_semantics": "speed_demand_action_curvature",
        "target_mask": [0.0, 1.0],
    }], training=False, return_target_mask=True)

    _, label, mask = dataset[0]

    assert np.allclose(label, [0.0, 1.25])
    assert np.allclose(mask, [0.0, 1.0])
