import numpy as np
from PIL import Image

from lane_training.dataset import (
    LaneDataset,
    augment_rgb,
    horizontal_flip,
    preprocess_rgb,
)


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
