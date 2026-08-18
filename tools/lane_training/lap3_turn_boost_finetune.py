"""Continue lap_003 training with extra active-turn curve supervision."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys

import numpy as np
import paddle
from paddle.nn import functional as F

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lane_training.dataset import LaneDataset
from lane_training.model import CnnModel
from lane_training.yaw_curve import yaw_curve_loss
from tools.lane_training.new_track_finetune import load_rows
from tools.lane_training.new_track_full_lap_finetune import (
    configure_full_lap_student,
    full_lap_optimizer_parameters,
)
from tools.lane_training.official_lap3_multiepoch_finetune import build_epoch_schedule
from tools.lane_training.official_yaw_curve_finetune import (
    OFFICIAL_CHECKPOINT_SHA256,
    sha256_file,
)


EPOCH5_CHECKPOINT_SHA256 = "4622109ab2d59e9ed8f0179190b89256b9a59d899e76f657ded67a81e85643fd"
VY_SCALE = 0.15
ACTIVE_CURVE_WEIGHTS = {
    "point": 1.0,
    "slope": 0.5,
    "curvature": 0.1,
    "integral": 1.0,
    "phase": 0.5,
    "saturation": 0.0,
}


def active_turn_mask(labels: paddle.Tensor, *, threshold: float = 0.05) -> paddle.Tensor:
    if labels.ndim != 2 or labels.shape[1] != 2:
        raise ValueError("labels must have shape [N, 2]")
    return paddle.abs(labels[:, 1]) >= threshold


def combine_turn_boost_loss(
    base_curve_loss: paddle.Tensor, active_curve_loss: paddle.Tensor,
    teacher_vy_loss: paddle.Tensor, *, active_weight: float = 2.0,
) -> paddle.Tensor:
    if not np.isfinite(active_weight) or active_weight < 0:
        raise ValueError("active_weight must be finite and nonnegative")
    return base_curve_loss + active_weight * active_curve_loss + teacher_vy_loss


def _chunk(rows, size):
    return [rows[start:start + size] for start in range(0, len(rows), size)]


def add_advanced_yaw_targets(rows, frames: int):
    if frames < 0:
        raise ValueError("yaw advance frames must be nonnegative")
    if not rows:
        return []
    advanced = []
    last = len(rows) - 1
    for index, row in enumerate(rows):
        copied = dict(row)
        copied["advanced_yaw"] = float(rows[min(index + frames, last)]["yaw"])
        advanced.append(copied)
    return advanced


def train(args) -> dict:
    random.seed(args.seed)
    np.random.seed(args.seed)
    paddle.seed(args.seed)
    paddle.set_device(args.device)
    initial_sha = sha256_file(args.checkpoint)
    if initial_sha != args.expected_checkpoint_sha256.lower():
        raise ValueError(f"initial checkpoint SHA mismatch: {initial_sha}")
    teacher_sha = sha256_file(args.teacher_checkpoint)
    if teacher_sha != OFFICIAL_CHECKPOINT_SHA256:
        raise ValueError(f"official teacher checkpoint SHA mismatch: {teacher_sha}")
    rows = add_advanced_yaw_targets([
        row for row in load_rows(args.lap3_data_json, args.lap3_image_dir, "lap_003_full")
        if row["image_exists"]
    ], args.yaw_advance_frames)
    batches = _chunk(rows, args.batch_size)
    dataset = LaneDataset(rows, training=False, seed=args.seed)
    row_indices = {str(row["image_path"]): index for index, row in enumerate(rows)}

    student = CnnModel()
    student.set_state_dict(paddle.load(str(args.checkpoint)))
    configure_full_lap_student(student)
    teacher = CnnModel()
    teacher.set_state_dict(paddle.load(str(args.teacher_checkpoint)))
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.stop_gradient = True
    optimizer = paddle.optimizer.Adam(
        learning_rate=args.learning_rate,
        parameters=full_lap_optimizer_parameters(
            student,
            head_learning_rate=args.learning_rate,
            convolution_learning_rates=args.convolution_learning_rates,
        ),
    )

    args.output.mkdir(parents=True, exist_ok=True)
    epoch_reports = []
    total_batches = 0
    stop = False
    for epoch in range(args.epochs):
        totals = {"total": 0.0, "base_yaw": 0.0, "active_yaw": 0.0, "teacher_vy": 0.0}
        epoch_batches = 0
        active_frames = 0
        for batch_rows in build_epoch_schedule(batches, seed=args.seed + epoch):
            if args.max_batches and total_batches >= args.max_batches:
                stop = True
                break
            images = np.stack([
                dataset[row_indices[str(row["image_path"])]][0] for row in batch_rows
            ])
            labels = np.asarray(
                [[row["vy"], row["yaw"]] for row in batch_rows], dtype=np.float32
            )
            advanced_yaw = np.asarray(
                [row["advanced_yaw"] for row in batch_rows], dtype=np.float32
            )
            x = paddle.to_tensor(images)
            y = paddle.to_tensor(labels)
            prediction = student(x)
            with paddle.no_grad():
                teacher_prediction = teacher(x)
            target = paddle.stack([teacher_prediction[:, 0], y[:, 1]], axis=1)
            advanced_target = paddle.stack([
                teacher_prediction[:, 0], paddle.to_tensor(advanced_yaw)
            ], axis=1)
            sequence_ids = [0] * len(batch_rows)
            base_yaw, _ = yaw_curve_loss(prediction, target, sequence_ids)
            mask = active_turn_mask(advanced_target)
            active_yaw, _ = yaw_curve_loss(
                prediction, advanced_target, sequence_ids,
                active_mask=mask, weights=ACTIVE_CURVE_WEIGHTS,
            )
            teacher_vy = F.smooth_l1_loss(
                prediction[:, 0] / VY_SCALE,
                teacher_prediction[:, 0] / VY_SCALE,
            )
            total = combine_turn_boost_loss(
                base_yaw, active_yaw, teacher_vy, active_weight=args.active_weight
            )
            total.backward()
            optimizer.step()
            optimizer.clear_grad()
            total_batches += 1
            epoch_batches += 1
            active_frames += int(paddle.sum(mask.astype("int64")).item())
            totals["total"] += float(total)
            totals["base_yaw"] += float(base_yaw)
            totals["active_yaw"] += float(active_yaw)
            totals["teacher_vy"] += float(teacher_vy)
        checkpoint = args.output / f"epoch_{epoch + 1}.pdparams"
        paddle.save(student.state_dict(), str(checkpoint))
        epoch_reports.append({
            "epoch": epoch + 1,
            "batch_count": epoch_batches,
            "all_lap_003_rows_used": epoch_batches == len(batches),
            "active_frame_count": active_frames,
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": sha256_file(checkpoint),
            "mean_losses": {
                name: value / max(epoch_batches, 1) for name, value in totals.items()
            },
        })
        if stop:
            break

    final_checkpoint = args.output / "best.pdparams"
    paddle.save(student.state_dict(), str(final_checkpoint))
    report = {
        "checkpoint": str(final_checkpoint),
        "checkpoint_sha256": sha256_file(final_checkpoint),
        "initial_epoch5_checkpoint_sha256": initial_sha,
        "teacher_official_checkpoint_sha256": teacher_sha,
        "epochs": args.epochs,
        "completed_epochs": len(epoch_reports),
        "lap_003_row_count": len(rows),
        "lap_003_batch_count_per_epoch": len(batches),
        "active_weight": args.active_weight,
        "yaw_advance_frames": args.yaw_advance_frames,
        "active_curve_weights": ACTIVE_CURVE_WEIGHTS,
        "epoch_reports": epoch_reports,
        "learning_rate": args.learning_rate,
        "convolution_learning_rates": list(args.convolution_learning_rates),
        "unfrozen": "last four convolutions and regression head",
        "vy_optimized": False,
        "locked_test_performed": False,
        "selected": False,
    }
    (args.output / "training_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expected-checkpoint-sha256", default=EPOCH5_CHECKPOINT_SHA256)
    parser.add_argument("--teacher-checkpoint", type=Path, required=True)
    parser.add_argument("--lap3-data-json", type=Path, required=True)
    parser.add_argument("--lap3-image-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="gpu")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument(
        "--convolution-learning-rates", type=float, nargs=4,
        default=(1e-7, 3e-7, 7e-7, 1.5e-6),
    )
    parser.add_argument("--active-weight", type=float, default=2.0)
    parser.add_argument("--yaw-advance-frames", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260811)
    parser.add_argument("--max-batches", type=int, default=0)
    return parser.parse_args(argv)


if __name__ == "__main__":
    print(json.dumps(train(parse_args()), indent=2))
