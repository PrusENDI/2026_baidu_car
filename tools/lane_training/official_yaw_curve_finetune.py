"""One-epoch official-initialized yaw-curve fine tuning."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import sys
from typing import Sequence

import numpy as np
import paddle
from paddle.nn import functional as F

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lane_training.dataset import LaneDataset
from lane_training.model import CnnModel, set_feature_trainable
from lane_training.yaw_curve import (
    build_curve_windows,
    selective_yaw_target,
    yaw_curve_loss,
)


OFFICIAL_CHECKPOINT_SHA256 = "ebad35ba8b9f6df706b93ebdbc254a28ec4086f668f02b73468c6e16ef6e7f4c"
COMMAND_SCALES = (0.15, 0.94)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_rows(data_json: Path, image_dir: Path, source: str) -> list[dict]:
    payload = json.loads(data_json.read_text(encoding="utf-8"))
    return [
        {
            "image_path": str(image_dir / item["img_path"]),
            "vy": float(item["state"][1]),
            "yaw": float(item["state"][2]),
            "source": source,
            "frame": f"{index:04d}",
        }
        for index, item in enumerate(payload)
    ]


def _dedupe_rows(rows: Sequence[dict]) -> list[dict]:
    seen = set()
    result = []
    for row in rows:
        key = str(row["image_path"])
        if key not in seen:
            seen.add(key)
            result.append(row)
    return result


def build_source_batches(rows: Sequence[dict], *, chunk_size: int = 64) -> list[tuple[list[dict], bool]]:
    """Return turn windows followed by non-turn chunks for one source."""
    windows = build_curve_windows(rows, context=10, max_length=64)
    turn_paths = {str(row["image_path"]) for window in windows for row in window}
    general = [row for row in rows if str(row["image_path"]) not in turn_paths]
    batches: list[tuple[list[dict], bool]] = [(window, True) for window in windows]
    batches.extend((general[start:start + chunk_size], False)
                   for start in range(0, len(general), chunk_size))
    return [batch for batch in batches if batch[0]]


def mixed_batches(official_batches, lane_batches):
    """Schedule three official batches for every supplemental lane batch."""
    if not official_batches or not lane_batches:
        raise ValueError("both official and lane batches are required")
    scheduled = []
    official_index = 0
    for lane_batch in lane_batches:
        for _ in range(3):
            scheduled.append((official_batches[official_index % len(official_batches)], "official"))
            official_index += 1
        scheduled.append((lane_batch, "lane_imageset"))
    return scheduled


def balanced_official_batches(batches, *, seed: int):
    """Alternate equally weighted turn windows and general replay batches."""
    turns = [batch for batch in batches if batch[1]]
    general = [batch for batch in batches if not batch[1]]
    if not turns or not general:
        raise ValueError("official-only training requires turn and general batches")
    rng = random.Random(seed)
    rng.shuffle(turns)
    rng.shuffle(general)
    size = max(len(turns), len(general))
    schedule = []
    for index in range(size):
        schedule.append((turns[index % len(turns)], "official"))
        schedule.append((general[index % len(general)], "official"))
    return schedule


def configure_student_for_curve(
    student: CnnModel, *, unfreeze_last_conv: bool = False,
) -> None:
    """Freeze convolutional features and disable their dropout noise."""
    set_feature_trainable(student, False)
    if unfreeze_last_conv:
        for parameter in student.features[11].parameters():
            parameter.stop_gradient = False
    student.train()
    frozen_prefix_end = 11 if unfreeze_last_conv else 14
    for layer in student.features[:frozen_prefix_end]:
        layer.eval()


def curve_optimizer_parameters(
    student: CnnModel, *, learning_rate: float, final_conv_learning_rate: float,
):
    final_conv = list(student.features[11].parameters())
    trainable_final = [parameter for parameter in final_conv if not parameter.stop_gradient]
    final_ids = {id(parameter) for parameter in final_conv}
    head = [
        parameter for parameter in student.parameters()
        if not parameter.stop_gradient and id(parameter) not in final_ids
    ]
    if not trainable_final:
        return [{"params": [parameter for parameter in student.parameters() if not parameter.stop_gradient]}]
    return [
        {"params": trainable_final, "learning_rate": final_conv_learning_rate / learning_rate},
        {"params": head, "learning_rate": 1.0},
    ]


def combine_curve_training_loss(
    curve_loss: paddle.Tensor, vy_loss: paddle.Tensor,
    teacher_vy_loss: paddle.Tensor, *, is_curve: bool,
    curve_weight: float = 4.0, teacher_weight: float = 0.1,
) -> paddle.Tensor:
    """Keep general replay conservative while emphasizing ordered turns."""
    if not np.isfinite(curve_weight) or curve_weight < 0:
        raise ValueError("curve_weight must be finite and nonnegative")
    curve_scale = curve_weight if is_curve else 1.0
    return curve_scale * curve_loss + vy_loss + teacher_weight * teacher_vy_loss


def combine_selective_training_loss(
    curve_loss: paddle.Tensor,
    preservation_yaw_loss: paddle.Tensor,
    teacher_vy_loss: paddle.Tensor,
    *,
    is_curve: bool,
    curve_weight: float = 2.0,
) -> paddle.Tensor:
    """Correct eligible yaw while preserving teacher yaw and all teacher vy."""
    if not np.isfinite(curve_weight) or curve_weight < 0:
        raise ValueError("curve_weight must be finite and nonnegative")
    curve_scale = curve_weight if is_curve else 0.0
    return curve_scale * curve_loss + preservation_yaw_loss + teacher_vy_loss


def selective_batch_training_loss(
    prediction: paddle.Tensor,
    labels: paddle.Tensor,
    teacher_prediction: paddle.Tensor,
    sequence_ids: Sequence[int],
    *,
    is_curve: bool,
    curve_weight: float = 2.0,
) -> tuple[paddle.Tensor, dict]:
    """Calculate one official-only batch without optimizing labelled ``vy``."""
    if is_curve:
        target_yaw, correction_mask = selective_yaw_target(
            teacher_prediction[:, 1], labels[:, 1]
        )
    else:
        target_yaw = teacher_prediction[:, 1]
        correction_mask = paddle.zeros([prediction.shape[0]], dtype="bool")
    curve_target = paddle.stack([teacher_prediction[:, 0], target_yaw], axis=1)
    curve_total, curve_parts = yaw_curve_loss(
        prediction,
        curve_target,
        sequence_ids,
        active_mask=correction_mask,
    )
    preserve_mask = (
        paddle.logical_not(correction_mask)
        if is_curve else paddle.ones([prediction.shape[0]], dtype="bool")
    )
    selected_yaw = paddle.masked_select(prediction[:, 1], preserve_mask)
    teacher_yaw = paddle.masked_select(teacher_prediction[:, 1], preserve_mask)
    preservation_yaw = (
        F.smooth_l1_loss(
            selected_yaw / COMMAND_SCALES[1],
            teacher_yaw / COMMAND_SCALES[1],
        )
        if selected_yaw.shape[0] else paddle.zeros([], dtype=prediction.dtype)
    )
    teacher_vy = F.smooth_l1_loss(
        prediction[:, 0] / COMMAND_SCALES[0],
        teacher_prediction[:, 0] / COMMAND_SCALES[0],
    )
    total = combine_selective_training_loss(
        curve_total,
        preservation_yaw,
        teacher_vy,
        is_curve=is_curve,
        curve_weight=curve_weight,
    )
    parts = dict(curve_parts)
    parts.update({
        "preservation_yaw": preservation_yaw,
        "teacher_vy": teacher_vy,
        "eligible_count": int(paddle.sum(correction_mask.astype("int64")).item()),
        "preservation_count": int(paddle.sum(preserve_mask.astype("int64")).item()),
    })
    return total, parts


def train(args) -> dict:
    random.seed(args.seed)
    np.random.seed(args.seed)
    paddle.seed(args.seed)
    paddle.set_device(args.device)
    checkpoint_sha = sha256_file(args.checkpoint)
    if checkpoint_sha != OFFICIAL_CHECKPOINT_SHA256:
        raise ValueError(f"official checkpoint SHA mismatch: {checkpoint_sha}")

    official_rows = load_rows(args.official_data_json, args.official_image_dir, "official")
    official_batches = build_source_batches(official_rows, chunk_size=args.batch_size)
    if args.selective_official_only:
        lane_rows = []
        lane_batches = []
        schedule = balanced_official_batches(official_batches, seed=args.seed)
    else:
        lane_rows = load_rows(args.lane_data_json, args.lane_image_dir, "lane_imageset")
        lane_batches = build_source_batches(lane_rows, chunk_size=args.batch_size)
        schedule = mixed_batches(official_batches, lane_batches)

    student = CnnModel()
    teacher = CnnModel()
    state = paddle.load(str(args.checkpoint))
    student.set_state_dict(state)
    teacher.set_state_dict(state)
    configure_student_for_curve(student, unfreeze_last_conv=args.unfreeze_last_conv)
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.stop_gradient = True
    optimizer_parameters = curve_optimizer_parameters(
        student,
        learning_rate=args.learning_rate,
        final_conv_learning_rate=args.final_conv_learning_rate,
    )
    optimizer = paddle.optimizer.Adam(learning_rate=args.learning_rate, parameters=optimizer_parameters)

    datasets = {
        "official": LaneDataset(official_rows, training=False, seed=args.seed),
    }
    if not args.selective_official_only:
        datasets["lane_imageset"] = LaneDataset(
            lane_rows, training=False, seed=args.seed
        )
    row_indices = {
        source: {str(row["image_path"]): index for index, row in enumerate(rows)}
        for source, rows in (("official", official_rows), ("lane_imageset", lane_rows))
        if rows
    }
    losses = {name: 0.0 for name in (
        "total", "vy", "teacher_vy", "preservation_yaw", "eligible_count",
        "preservation_count", "point", "slope", "curvature", "integral",
        "phase", "saturation",
    )}
    batch_count = 0
    for (rows, curve), source in schedule:
        if args.max_batches and batch_count >= args.max_batches:
            break
        dataset = datasets[source]
        images = np.stack([dataset[row_indices[source][str(row["image_path"])]][0] for row in rows])
        labels = np.asarray([[row["vy"], row["yaw"]] for row in rows], dtype=np.float32)
        x = paddle.to_tensor(images)
        y = paddle.to_tensor(labels)
        prediction = student(x)
        with paddle.no_grad():
            teacher_prediction = teacher(x)
        sequence_ids = [0] * len(rows) if curve else list(range(len(rows)))
        if args.selective_official_only:
            total, batch_parts = selective_batch_training_loss(
                prediction,
                y,
                teacher_prediction,
                sequence_ids,
                is_curve=curve,
                curve_weight=args.curve_weight,
            )
            curve_parts = {
                name: value for name, value in batch_parts.items()
                if isinstance(value, paddle.Tensor)
            }
            losses["eligible_count"] += batch_parts["eligible_count"]
            losses["preservation_count"] += batch_parts["preservation_count"]
        else:
            curve_total, curve_parts = yaw_curve_loss(prediction, y, sequence_ids)
            vy = F.smooth_l1_loss(
                prediction[:, 0] / COMMAND_SCALES[0], y[:, 0] / COMMAND_SCALES[0]
            )
            teacher_vy = F.smooth_l1_loss(
                prediction[:, 0] / COMMAND_SCALES[0],
                teacher_prediction[:, 0] / COMMAND_SCALES[0],
            )
            total = combine_curve_training_loss(
                curve_total, vy, teacher_vy, is_curve=curve,
                curve_weight=args.curve_weight,
            )
            losses["vy"] += float(vy)
            losses["teacher_vy"] += float(teacher_vy)
        total.backward()
        optimizer.step()
        optimizer.clear_grad()
        batch_count += 1
        for name, value in curve_parts.items():
            if name in losses:
                losses[name] += float(value)
        losses["total"] += float(total)

    args.output.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output / "best.pdparams"
    paddle.save(student.state_dict(), str(checkpoint))
    mean_losses = {name: value / max(batch_count, 1) for name, value in losses.items()}
    report = {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "initial_official_checkpoint_sha256": checkpoint_sha,
        "training_mode": (
            "selective_official_yaw"
            if args.selective_official_only else "official_yaw_curve"
        ),
        "epochs": 1,
        "batch_count": batch_count,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "curve_batch_weight": args.curve_weight,
        "unfreeze_last_conv": args.unfreeze_last_conv,
        "final_conv_learning_rate": args.final_conv_learning_rate,
        "official_row_count": len(official_rows),
        "lane_imageset_row_count": len(lane_rows),
        "official_batch_count": len(official_batches),
        "lane_imageset_batch_count": len(lane_batches),
        "scheduled_source_ratio": (
            "1 official turn batch : 1 official general batch"
            if args.selective_official_only
            else "3 official batches : 1 lane_imageset batch"
        ),
        "convolutional_features_frozen": not args.unfreeze_last_conv,
        "vy_optimized": False,
        "curve_loss_weights": {"point": 1.0, "slope": 0.5, "curvature": 0.1, "integral": 0.5, "phase": 0.25, "saturation": 0.5},
        "mean_losses": mean_losses,
        "eligible_frame_count": int(losses["eligible_count"]),
        "preservation_frame_count": int(losses["preservation_count"]),
        "locked_test_performed": False,
        "selected": False,
    }
    (args.output / "training_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--official-data-json", type=Path, required=True)
    parser.add_argument("--official-image-dir", type=Path, required=True)
    parser.add_argument("--lane-data-json", type=Path)
    parser.add_argument("--lane-image-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="gpu")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=2e-6)
    parser.add_argument("--curve-weight", type=float, default=4.0)
    parser.add_argument("--unfreeze-last-conv", action="store_true")
    parser.add_argument("--final-conv-learning-rate", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument("--max-batches", type=int, default=0)
    parser.add_argument("--selective-official-only", action="store_true")
    args = parser.parse_args(argv)
    if args.selective_official_only:
        if args.unfreeze_last_conv:
            parser.error("selective official-only mode freezes every convolution")
        args.learning_rate = 5e-6
        args.curve_weight = 2.0
    elif args.lane_data_json is None or args.lane_image_dir is None:
        parser.error("legacy mixed mode requires both lane_imageset paths")
    return args


if __name__ == "__main__":
    print(json.dumps(train(parse_args()), indent=2))
