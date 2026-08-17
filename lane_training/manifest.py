from __future__ import annotations

import math
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from PIL import Image


@dataclass(frozen=True)
class Sample:
    image_path: Path
    vy: float
    yaw: float
    source: str
    session: str
    session_type: str
    scene_type: str
    split: str


def validate_samples(samples: Iterable[Sample]) -> None:
    image_splits: dict[Path, str] = {}
    session_splits: dict[str, str] = {}
    for sample in samples:
        if not math.isfinite(sample.vy) or not math.isfinite(sample.yaw):
            raise ValueError(f"non-finite command for {sample.image_path}")

        image = sample.image_path.resolve()
        prior_image_split = image_splits.setdefault(image, sample.split)
        if prior_image_split != sample.split:
            raise ValueError(f"image has conflicting splits: {sample.image_path}")

        if sample.source == "collected":
            prior_session_split = session_splits.setdefault(sample.session, sample.split)
            if prior_session_split != sample.split:
                raise ValueError(
                    f"session {sample.session} has conflicting splits: "
                    f"{prior_session_split} and {sample.split}"
                )


def load_labeled_session(
    data_json: Path,
    *,
    source: str,
    session: str,
    session_type: str,
    scene_type: str,
    split: str,
) -> list[Sample]:
    records = json.loads(data_json.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise ValueError(f"label file must contain a list: {data_json}")

    session_root = data_json.parent.resolve()
    samples: list[Sample] = []
    for index, record in enumerate(records):
        if not isinstance(record, dict) or "img_path" not in record or "state" not in record:
            raise ValueError(f"invalid label record {index} in {data_json}")
        image_path = (session_root / str(record["img_path"])).resolve()
        if image_path != session_root and session_root not in image_path.parents:
            raise ValueError(f"unsafe image path in {data_json}: {record['img_path']}")
        state = record["state"]
        if not isinstance(state, list) or len(state) < 3:
            raise ValueError(f"record {index} requires three state values")
        try:
            vy = float(state[1])
            yaw = float(state[2])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"record {index} has invalid state values") from error
        if not image_path.is_file():
            raise FileNotFoundError(f"missing labeled image: {image_path}")
        try:
            with Image.open(image_path) as image:
                image.verify()
        except OSError as error:
            raise ValueError(f"unreadable labeled image: {image_path}") from error
        samples.append(
            Sample(
                image_path=image_path,
                vy=vy,
                yaw=yaw,
                source=source,
                session=session,
                session_type=session_type,
                scene_type=scene_type,
                split=split,
            )
        )

    validate_samples(samples)
    return samples


def load_cv_action_session(data_json: Path) -> list[dict]:
    """Load the CV teacher's [speed_demand, kappa_action] records."""
    records = json.loads(data_json.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise ValueError(f"label file must contain a list: {data_json}")
    session_root = data_json.parent.resolve()
    rows = []
    for index, record in enumerate(records):
        if not isinstance(record, dict) or "img_path" not in record:
            raise ValueError(f"invalid label record {index} in {data_json}")
        if record.get("label_semantics") != "raw_control_with_model_target":
            raise ValueError(f"record {index} does not use CV action labels")
        target = record.get("model_target")
        mask = record.get("target_mask")
        if not isinstance(target, dict) or not isinstance(mask, list) or len(mask) != 2:
            raise ValueError(f"record {index} requires model_target and target_mask")
        try:
            speed_demand = float(target["speed_demand"])
            kappa_action = float(target["kappa_action"])
            target_mask = [float(mask[0]), float(mask[1])]
        except (TypeError, ValueError) as error:
            raise ValueError(f"record {index} has invalid model outputs") from error
        if (not math.isfinite(speed_demand) or
                not math.isfinite(kappa_action) or
                not 0.0 <= speed_demand <= 1.0):
            raise ValueError(f"record {index} has out-of-range model outputs")
        image_path = (session_root / str(record["img_path"])).resolve()
        if image_path != session_root and session_root not in image_path.parents:
            raise ValueError(f"unsafe image path in {data_json}: {record['img_path']}")
        if not image_path.is_file():
            raise FileNotFoundError(f"missing labeled image: {image_path}")
        try:
            with Image.open(image_path) as image:
                image.verify()
        except OSError as error:
            raise ValueError(f"unreadable labeled image: {image_path}") from error
        rows.append({
            "image_path": str(image_path),
            "speed_demand": speed_demand,
            "kappa_action": kappa_action,
            "label_semantics": "speed_demand_action_curvature",
            "target_mask": target_mask,
            "held": bool(record.get("held", False)),
            "command_source": str(record.get("command_source", "opencv")),
        })
    return rows


def load_manual_action_session(data_json: Path) -> list[dict]:
    """Convert joystick [vx, vy, wz] records to curvature-only targets."""
    records = json.loads(data_json.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise ValueError(f"label file must contain a list: {data_json}")
    session_root = data_json.parent.resolve()
    rows = []
    for index, record in enumerate(records):
        state = record.get("state") if isinstance(record, dict) else None
        if not isinstance(state, list) or len(state) < 3 or "img_path" not in record:
            raise ValueError(f"invalid joystick record {index} in {data_json}")
        try:
            vx, _, wz = map(float, state[:3])
        except (TypeError, ValueError) as error:
            raise ValueError(f"record {index} has invalid chassis command") from error
        image_path = (session_root / str(record["img_path"])).resolve()
        if image_path != session_root and session_root not in image_path.parents:
            raise ValueError(f"unsafe image path in {data_json}: {record['img_path']}")
        if not image_path.is_file():
            raise FileNotFoundError(f"missing labeled image: {image_path}")
        try:
            with Image.open(image_path) as image:
                image.verify()
        except OSError as error:
            raise ValueError(f"unreadable labeled image: {image_path}") from error
        kappa_action = wz / max(abs(vx), 0.12)
        rows.append({
            "image_path": str(image_path),
            "speed_demand": 0.0,
            "kappa_action": kappa_action,
            "label_semantics": "speed_demand_action_curvature",
            # Manual throttle is not assumed to follow the CV speed curve.
            "target_mask": [0.0, 1.0],
            "source": "manual",
        })
    return rows
