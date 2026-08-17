from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
from PIL import Image
from paddle.io import Dataset
from .corruptions import CorruptionConfig, generate_light_corruption


@dataclass(frozen=True)
class ActionAugmentationConfig:
    application_probability: float = 1.0
    brightness_range: tuple[float, float] = (0.8, 1.2)
    contrast_range: tuple[float, float] = (0.8, 1.2)
    saturation_range: tuple[float, float] = (0.85, 1.15)
    hue_delta: int = 8
    blur_probability: float = 0.15
    noise_probability: float = 0.15
    horizontal_flip_probability: float = 0.5

    def __post_init__(self) -> None:
        for name in (
            "application_probability",
            "blur_probability",
            "noise_probability",
            "horizontal_flip_probability",
        ):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        for name in (
            "brightness_range",
            "contrast_range",
            "saturation_range",
        ):
            value = getattr(self, name)
            if len(value) != 2 or value[0] <= 0.0 or value[0] > value[1]:
                raise ValueError(f"{name} must be an ordered positive range")
        if self.hue_delta < 0:
            raise ValueError("hue_delta must be nonnegative")


LEGACY_ACTION_AUGMENTATION = ActionAugmentationConfig()
CLEAN_ACTION_AUGMENTATION = ActionAugmentationConfig(
    application_probability=0.0,
    brightness_range=(1.0, 1.0),
    contrast_range=(1.0, 1.0),
    saturation_range=(1.0, 1.0),
    hue_delta=0,
    blur_probability=0.0,
    noise_probability=0.0,
    horizontal_flip_probability=0.0,
)
MILD_FIXED_TRACK_ACTION_AUGMENTATION = ActionAugmentationConfig(
    application_probability=0.30,
    brightness_range=(0.95, 1.05),
    contrast_range=(0.95, 1.05),
    saturation_range=(0.97, 1.03),
    hue_delta=2,
    blur_probability=0.0,
    noise_probability=0.0,
    horizontal_flip_probability=0.0,
)


def preprocess_rgb(image: np.ndarray) -> np.ndarray:
    resized = cv2.resize(image.astype(np.uint8), (128, 128), interpolation=cv2.INTER_LINEAR)
    normalized = resized.astype(np.float32) / 127.5 - 1.0
    return np.transpose(normalized, (2, 0, 1)).astype(np.float32, copy=False)


def horizontal_flip(
    image: np.ndarray, label: np.ndarray, *,
    label_semantics: str = "legacy_vy_yaw",
) -> tuple[np.ndarray, np.ndarray]:
    flipped_image = image[:, ::-1].copy()
    label = np.asarray(label, dtype=np.float32)
    if label_semantics == "speed_demand_action_curvature":
        # Mirroring does not change bend strength, only turn direction.
        flipped_label = np.asarray([label[0], -label[1]], dtype=np.float32)
    else:
        flipped_label = -label
    return flipped_image, flipped_label


def row_label(row: dict) -> tuple[np.ndarray, str]:
    """Return the two-output target while preserving legacy datasets."""
    semantics = str(row.get("label_semantics", "legacy_vy_yaw"))
    if semantics == "speed_demand_action_curvature" or (
            "speed_demand" in row and "kappa_action" in row):
        return np.asarray(
            [row["speed_demand"], row["kappa_action"]], dtype=np.float32
        ), "speed_demand_action_curvature"
    return np.asarray([row["vy"], row["yaw"]], dtype=np.float32), semantics


def row_target(row: dict) -> tuple[np.ndarray, np.ndarray, str]:
    label, semantics = row_label(row)
    mask = np.asarray(row.get("target_mask", [1.0, 1.0]), dtype=np.float32)
    if mask.shape != (2,) or np.any(mask < 0.0) or np.any(mask > 1.0):
        raise ValueError("target_mask must contain two values in [0, 1]")
    return label, mask, semantics


def augment_rgb(
    image: np.ndarray,
    label: np.ndarray,
    rng: np.random.Generator,
    label_semantics: str = "legacy_vy_yaw",
    horizontal_flip_probability: float | None = None,
    config: ActionAugmentationConfig = LEGACY_ACTION_AUGMENTATION,
    return_applied: bool = False,
):
    if horizontal_flip_probability is not None:
        config = replace(
            config,
            horizontal_flip_probability=float(horizontal_flip_probability),
        )
    if rng.random() >= config.application_probability:
        result = (image, np.asarray(label, dtype=np.float32))
        return result + (False,) if return_applied else result

    brightness = rng.uniform(*config.brightness_range)
    contrast = rng.uniform(*config.contrast_range)
    saturation = rng.uniform(*config.saturation_range)
    work = image.astype(np.float32) * brightness
    mean = work.mean(axis=(0, 1), keepdims=True)
    work = (work - mean) * contrast + mean
    augmented = np.clip(work, 0, 255).astype(np.uint8)
    hsv = cv2.cvtColor(augmented, cv2.COLOR_RGB2HSV)
    hue_shift = (
        int(rng.integers(-config.hue_delta, config.hue_delta + 1))
        if config.hue_delta
        else 0
    )
    hsv[..., 0] = (hsv[..., 0].astype(np.int16) + hue_shift) % 180
    hsv[..., 1] = np.clip(hsv[..., 1].astype(np.float32) * saturation, 0, 255).astype(
        np.uint8
    )
    augmented = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    if config.blur_probability and rng.random() < config.blur_probability:
        augmented = cv2.GaussianBlur(augmented, (3, 3), rng.uniform(0.3, 0.8))
    if config.noise_probability and rng.random() < config.noise_probability:
        noise = rng.normal(0.0, rng.uniform(2.0, 6.0), augmented.shape)
        augmented = np.clip(augmented.astype(np.float32) + noise, 0, 255).astype(
            np.uint8
        )
    if (
        config.horizontal_flip_probability
        and rng.random() < config.horizontal_flip_probability
    ):
        augmented, label = horizontal_flip(
            augmented, label, label_semantics=label_semantics)
    result = (augmented, np.asarray(label, dtype=np.float32))
    return result + (True,) if return_applied else result


class LaneDataset(Dataset):
    def __init__(
        self,
        rows: Sequence[dict],
        *,
        training: bool,
        seed: int = 20260807,
        illumination_probability: float = 0.4,
        illumination_severity: int = 3,
        corruption_config: CorruptionConfig | None = None,
        return_target_mask: bool = False,
        horizontal_flip_probability: float = 0.5,
        action_augmentation_config: ActionAugmentationConfig | None = None,
    ) -> None:
        super().__init__()
        self.rows = list(rows)
        self.training = training
        self.seed = seed
        self.epoch = 0
        if not 0 <= illumination_probability <= 1 or illumination_severity not in range(1, 6):
            raise ValueError("invalid illumination augmentation configuration")
        self.illumination_probability = illumination_probability
        self.illumination_severity = illumination_severity
        self.corruption_config = corruption_config or CorruptionConfig()
        self.return_target_mask = bool(return_target_mask)
        self.horizontal_flip_probability = float(horizontal_flip_probability)
        self.action_augmentation_config = (
            replace(
                LEGACY_ACTION_AUGMENTATION,
                horizontal_flip_probability=self.horizontal_flip_probability,
            )
            if action_augmentation_config is None
            else action_augmentation_config
        )
        self.augmentation_applied_count = 0
        self._cached_images: list[np.ndarray | None] = [None] * len(self.rows)

    def __len__(self) -> int:
        return len(self.rows)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch
        self.augmentation_applied_count = 0

    def set_action_augmentation_config(
        self, config: ActionAugmentationConfig,
    ) -> None:
        if not isinstance(config, ActionAugmentationConfig):
            raise TypeError("config must be an ActionAugmentationConfig")
        self.action_augmentation_config = config

    @property
    def cached_image_count(self) -> int:
        return sum(image is not None for image in self._cached_images)

    def _read_resized(self, index: int) -> np.ndarray:
        image_path = Path(self.rows[index]["image_path"])
        bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError(f"cannot decode image: {image_path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return cv2.resize(rgb, (128, 128), interpolation=cv2.INTER_LINEAR)

    def preload(self) -> None:
        for index in range(len(self.rows)):
            if self._cached_images[index] is None:
                self._cached_images[index] = self._read_resized(index)

    def __getitem__(self, index):
        if isinstance(index, tuple):
            _, index = index
        row = self.rows[index]
        if self._cached_images[index] is None:
            self._cached_images[index] = self._read_resized(index)
        rgb = self._cached_images[index].copy()
        label, target_mask, label_semantics = row_target(row)
        if self.training:
            rng = np.random.default_rng(self.seed + self.epoch * 1_000_003 + int(index))
            rgb, label, applied = augment_rgb(
                rgb,
                label,
                rng,
                label_semantics=label_semantics,
                config=self.action_augmentation_config,
                return_applied=True,
            )
            self.augmentation_applied_count += int(applied)
        if self.return_target_mask:
            return preprocess_rgb(rgb), label, target_mask
        if not np.allclose(target_mask, 1.0):
            raise ValueError(
                "masked targets require LaneDataset(return_target_mask=True)")
        return preprocess_rgb(rgb), label

    def get_training_views(
        self,
        index: int,
        view_seed: int | None = None,
        *,
        severity: int | None = None,
        probability: float | None = None,
    ):
        """Return clean/augmented normalized views and the unchanged command label."""
        if self._cached_images[index] is None:
            self._cached_images[index] = self._read_resized(index)
        clean = self._cached_images[index].copy()
        seed = self.seed + int(index) if view_seed is None else int(view_seed)
        augmented = generate_light_corruption(
            clean,
            np.random.default_rng(seed),
            severity=self.illumination_severity if severity is None else severity,
            probability=self.illumination_probability if probability is None else probability,
            config=self.corruption_config,
        ).image
        label, target_mask, _ = row_target(self.rows[index])
        result = (preprocess_rgb(clean), preprocess_rgb(augmented), label)
        if self.return_target_mask:
            return result + (target_mask,)
        if not np.allclose(target_mask, 1.0):
            raise ValueError(
                "masked targets require LaneDataset(return_target_mask=True)")
        return result
