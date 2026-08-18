"""Aggressive one-pass adaptation over every labelled lap_003 frame."""
from __future__ import annotations

import argparse
import json
import math
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
from lane_training.yaw_curve import yaw_curve_loss
from tools.lane_training.new_track_finetune import load_rows, select_straight_rows
from tools.lane_training.official_yaw_curve_finetune import (
    OFFICIAL_CHECKPOINT_SHA256,
    sha256_file,
)


R2_CHECKPOINT_SHA256 = "7285cffe27a8b95129d623123fb783f52dda99e50417543ef3f9cd2ba880128c"
VY_SCALE = 0.15
CONVOLUTION_INDICES = (4, 6, 8, 11)


def _chunk(rows: Sequence[dict], size: int) -> list[list[dict]]:
    return [list(rows[start:start + size]) for start in range(0, len(rows), size)]


def build_full_lap_schedule(
    lap3_batches: Sequence[list[dict]], lap1_batches: Sequence[list[dict]],
    official_batches: Sequence[list[dict]], *, seed: int,
) -> list[tuple[list[dict], str, bool]]:
    """Use every lap_003 batch once plus approximately 10% replay per source."""
    if not lap3_batches or not lap1_batches or not official_batches:
        raise ValueError("all three training sources are required")
    total = int(math.ceil(len(lap3_batches) / 0.80))
    lap1_count = int(round(total * 0.10))
    official_count = total - len(lap3_batches) - lap1_count
    rng = random.Random(seed)

    def sample(pool, count):
        values = list(pool)
        rng.shuffle(values)
        return [values[index % len(values)] for index in range(count)]

    schedule = [
        *((batch, "lap_003_full", True) for batch in lap3_batches),
        *((batch, "lap_001_straight", False) for batch in sample(lap1_batches, lap1_count)),
        *((batch, "official_preserve", False) for batch in sample(official_batches, official_count)),
    ]
    rng.shuffle(schedule)
    return schedule


def configure_full_lap_student(student: CnnModel) -> None:
    set_feature_trainable(student, False)
    for index in CONVOLUTION_INDICES:
        for parameter in student.features[index].parameters():
            parameter.stop_gradient = False
    student.train()
    for layer in student.features[:4]:
        layer.eval()


def full_lap_optimizer_parameters(
    student: CnnModel, *, head_learning_rate: float,
    convolution_learning_rates: Sequence[float],
):
    if len(convolution_learning_rates) != len(CONVOLUTION_INDICES):
        raise ValueError("four convolution learning rates are required")
    convolution_parameters = [
        list(student.features[index].parameters()) for index in CONVOLUTION_INDICES
    ]
    convolution_ids = {
        id(parameter) for group in convolution_parameters for parameter in group
    }
    head = [
        parameter for parameter in student.parameters()
        if not parameter.stop_gradient and id(parameter) not in convolution_ids
    ]
    return [
        *(
            {"params": parameters, "learning_rate": rate / head_learning_rate}
            for parameters, rate in zip(convolution_parameters, convolution_learning_rates)
        ),
        {"params": head, "learning_rate": 1.0},
    ]


def _batch_arrays(dataset, row_indices, rows):
    images = np.stack([
        dataset[row_indices[str(row["image_path"])]][0] for row in rows
    ])
    labels = np.asarray([[row["vy"], row["yaw"]] for row in rows], dtype=np.float32)
    return paddle.to_tensor(images), paddle.to_tensor(labels)


def train(args) -> dict:
    random.seed(args.seed)
    np.random.seed(args.seed)
    paddle.seed(args.seed)
    paddle.set_device(args.device)
    student_sha = sha256_file(args.checkpoint)
    if student_sha != args.expected_checkpoint_sha256.lower():
        raise ValueError(f"student checkpoint SHA mismatch: {student_sha}")
    teacher_sha = sha256_file(args.teacher_checkpoint)
    if teacher_sha != OFFICIAL_CHECKPOINT_SHA256:
        raise ValueError(f"official teacher checkpoint SHA mismatch: {teacher_sha}")

    lap3 = [
        row for row in load_rows(args.lap3_data_json, args.lap3_image_dir, "lap_003_full")
        if row["image_exists"]
    ]
    lap1 = select_straight_rows(
        load_rows(args.lap1_data_json, args.lap1_image_dir, "lap_001_straight"),
        context=10,
    )
    official = select_straight_rows(
        load_rows(args.official_data_json, args.official_image_dir, "official_preserve"),
        context=10,
    )
    lap3_batches = _chunk(lap3, args.batch_size)
    lap1_batches = _chunk(lap1, args.batch_size)
    official_batches = _chunk(official, args.batch_size)
    schedule = build_full_lap_schedule(
        lap3_batches, lap1_batches, official_batches, seed=args.seed
    )

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

    source_rows = {
        "lap_003_full": lap3,
        "lap_001_straight": lap1,
        "official_preserve": official,
    }
    datasets = {
        source: LaneDataset(rows, training=False, seed=args.seed)
        for source, rows in source_rows.items()
    }
    row_indices = {
        source: {str(row["image_path"]): index for index, row in enumerate(rows)}
        for source, rows in source_rows.items()
    }
    totals = {"total": 0.0, "yaw": 0.0, "teacher_vy": 0.0}
    source_batch_counts = {source: 0 for source in source_rows}
    batch_count = 0
    for rows, source, ordered in schedule:
        if args.max_batches and batch_count >= args.max_batches:
            break
        x, labels = _batch_arrays(datasets[source], row_indices[source], rows)
        prediction = student(x)
        with paddle.no_grad():
            teacher_prediction = teacher(x)
        target_yaw = (
            teacher_prediction[:, 1] if source == "official_preserve" else labels[:, 1]
        )
        target = paddle.stack([teacher_prediction[:, 0], target_yaw], axis=1)
        sequence_ids = [0] * len(rows) if ordered else list(range(len(rows)))
        yaw_loss, _ = yaw_curve_loss(prediction, target, sequence_ids)
        teacher_vy = F.smooth_l1_loss(
            prediction[:, 0] / VY_SCALE,
            teacher_prediction[:, 0] / VY_SCALE,
        )
        total = yaw_loss + teacher_vy
        total.backward()
        optimizer.step()
        optimizer.clear_grad()
        batch_count += 1
        source_batch_counts[source] += 1
        totals["total"] += float(total)
        totals["yaw"] += float(yaw_loss)
        totals["teacher_vy"] += float(teacher_vy)

    args.output.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output / "best.pdparams"
    paddle.save(student.state_dict(), str(checkpoint))
    report = {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "initial_r2_checkpoint_sha256": student_sha,
        "teacher_official_checkpoint_sha256": teacher_sha,
        "epochs": args.epochs,
        "batch_count": batch_count,
        "source_batch_counts": source_batch_counts,
        "source_rows": {source: len(rows) for source, rows in source_rows.items()},
        "lap_003_all_rows_used": source_batch_counts["lap_003_full"] == len(lap3_batches),
        "learning_rate": args.learning_rate,
        "convolution_learning_rates": list(args.convolution_learning_rates),
        "unfrozen": "last four convolutions and regression head",
        "vy_optimized": False,
        "mean_losses": {
            name: value / max(batch_count, 1) for name, value in totals.items()
        },
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
    parser.add_argument("--expected-checkpoint-sha256", default=R2_CHECKPOINT_SHA256)
    parser.add_argument("--teacher-checkpoint", type=Path, required=True)
    parser.add_argument("--lap1-data-json", type=Path, required=True)
    parser.add_argument("--lap1-image-dir", type=Path, required=True)
    parser.add_argument("--lap3-data-json", type=Path, required=True)
    parser.add_argument("--lap3-image-dir", type=Path, required=True)
    parser.add_argument("--official-data-json", type=Path, required=True)
    parser.add_argument("--official-image-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="gpu")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument(
        "--convolution-learning-rates", type=float, nargs=4,
        default=(1e-6, 2e-6, 5e-6, 1e-5),
    )
    parser.add_argument("--seed", type=int, default=20260811)
    parser.add_argument("--max-batches", type=int, default=0)
    return parser.parse_args(argv)


if __name__ == "__main__":
    print(json.dumps(train(parse_args()), indent=2))
