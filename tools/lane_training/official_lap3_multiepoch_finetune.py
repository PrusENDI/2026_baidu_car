"""Low-learning-rate multi-epoch lap_003 adaptation from the official model."""
from __future__ import annotations

import argparse
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
from lane_training.model import CnnModel
from lane_training.yaw_curve import yaw_curve_loss
from tools.lane_training.new_track_finetune import load_rows
from tools.lane_training.new_track_full_lap_finetune import (
    configure_full_lap_student,
    full_lap_optimizer_parameters,
)
from tools.lane_training.official_yaw_curve_finetune import (
    OFFICIAL_CHECKPOINT_SHA256,
    sha256_file,
)


VY_SCALE = 0.15


def _chunk(rows: Sequence[dict], size: int) -> list[list[dict]]:
    return [list(rows[start:start + size]) for start in range(0, len(rows), size)]


def build_epoch_schedule(batches: Sequence[list[dict]], *, seed: int) -> list[list[dict]]:
    if not batches:
        raise ValueError("lap_003 batches are required")
    schedule = list(batches)
    random.Random(seed).shuffle(schedule)
    return schedule


def train(args) -> dict:
    random.seed(args.seed)
    np.random.seed(args.seed)
    paddle.seed(args.seed)
    paddle.set_device(args.device)
    initial_sha = sha256_file(args.checkpoint)
    if initial_sha != OFFICIAL_CHECKPOINT_SHA256:
        raise ValueError(f"official checkpoint SHA mismatch: {initial_sha}")
    rows = [
        row for row in load_rows(args.lap3_data_json, args.lap3_image_dir, "lap_003_full")
        if row["image_exists"]
    ]
    batches = _chunk(rows, args.batch_size)
    dataset = LaneDataset(rows, training=False, seed=args.seed)
    row_indices = {str(row["image_path"]): index for index, row in enumerate(rows)}

    student = CnnModel()
    teacher = CnnModel()
    state = paddle.load(str(args.checkpoint))
    student.set_state_dict(state)
    teacher.set_state_dict(state)
    configure_full_lap_student(student)
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
        totals = {"total": 0.0, "yaw": 0.0, "teacher_vy": 0.0}
        epoch_batches = 0
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
            x = paddle.to_tensor(images)
            y = paddle.to_tensor(labels)
            prediction = student(x)
            with paddle.no_grad():
                teacher_prediction = teacher(x)
            target = paddle.stack([teacher_prediction[:, 0], y[:, 1]], axis=1)
            yaw_loss, _ = yaw_curve_loss(prediction, target, [0] * len(batch_rows))
            teacher_vy = F.smooth_l1_loss(
                prediction[:, 0] / VY_SCALE,
                teacher_prediction[:, 0] / VY_SCALE,
            )
            total = yaw_loss + teacher_vy
            total.backward()
            optimizer.step()
            optimizer.clear_grad()
            total_batches += 1
            epoch_batches += 1
            totals["total"] += float(total)
            totals["yaw"] += float(yaw_loss)
            totals["teacher_vy"] += float(teacher_vy)
        checkpoint = args.output / f"epoch_{epoch + 1}.pdparams"
        paddle.save(student.state_dict(), str(checkpoint))
        epoch_reports.append({
            "epoch": epoch + 1,
            "batch_count": epoch_batches,
            "all_lap_003_rows_used": epoch_batches == len(batches),
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
        "initial_official_checkpoint_sha256": initial_sha,
        "epochs": args.epochs,
        "completed_epochs": len(epoch_reports),
        "lap_003_row_count": len(rows),
        "lap_003_batch_count_per_epoch": len(batches),
        "epoch_reports": epoch_reports,
        "learning_rate": args.learning_rate,
        "convolution_learning_rates": list(args.convolution_learning_rates),
        "unfrozen": "last four convolutions and regression head",
        "training_sources": ["lap_003_full"],
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
    parser.add_argument("--lap3-data-json", type=Path, required=True)
    parser.add_argument("--lap3-image-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="gpu")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument(
        "--convolution-learning-rates", type=float, nargs=4,
        default=(3e-7, 6e-7, 1.5e-6, 3e-6),
    )
    parser.add_argument("--seed", type=int, default=20260811)
    parser.add_argument("--max-batches", type=int, default=0)
    return parser.parse_args(argv)


if __name__ == "__main__":
    print(json.dumps(train(parse_args()), indent=2))
