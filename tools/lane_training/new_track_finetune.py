"""Fast one-epoch adaptation to the newly recorded track illumination and turns."""
from __future__ import annotations

import argparse
import hashlib
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
from lane_training.yaw_curve import build_curve_windows, discover_turns, yaw_curve_loss
from tools.lane_training.official_yaw_curve_finetune import (
    OFFICIAL_CHECKPOINT_SHA256,
    configure_student_for_curve,
    curve_optimizer_parameters,
    sha256_file,
)


YAW_SCALE = 0.94
VY_SCALE = 0.15


def load_rows(data_json: Path, image_dir: Path, source: str) -> list[dict]:
    payload = json.loads(data_json.read_text(encoding="utf-8"))
    rows = []
    for index, item in enumerate(payload):
        image_path = image_dir / item["img_path"]
        rows.append({
            "image_path": str(image_path),
            "image_exists": image_path.is_file(),
            "vy": float(item["state"][1]),
            "yaw": float(item["state"][2]),
            "source": source,
            "frame": f"{index:04d}",
        })
    return rows


def select_straight_rows(
    rows: Sequence[dict], *, threshold: float = 0.05, context: int = 10,
) -> list[dict]:
    """Select labelled straight frames outside every turn's context band."""
    excluded: set[int] = set()
    yaw = [float(row["yaw"]) for row in rows]
    for start, end, _ in discover_turns(yaw, threshold=threshold):
        excluded.update(range(max(0, start - context), min(len(rows), end + context + 1)))
    return [
        dict(row) for index, row in enumerate(rows)
        if index not in excluded
        and abs(float(row["yaw"])) < threshold
        and row.get("image_exists", True)
    ]


def _chunk(rows: Sequence[dict], size: int) -> list[list[dict]]:
    return [list(rows[start:start + size]) for start in range(0, len(rows), size)]


def build_mixed_schedule(
    lap1_batches: Sequence[list[dict]],
    curve_batches: Sequence[list[dict]],
    official_batches: Sequence[list[dict]],
    *, seed: int, straight_fraction: float = 0.30,
    curve_fraction: float = 0.40,
) -> list[tuple[list[dict], str, bool]]:
    """Schedule every curve exactly once with the requested source mix."""
    if not lap1_batches or not curve_batches or not official_batches:
        raise ValueError("all three training sources are required")
    if not 0 < straight_fraction < 1 or not 0 < curve_fraction < 1:
        raise ValueError("source fractions must be between zero and one")
    if straight_fraction + curve_fraction >= 1:
        raise ValueError("source fractions must leave official replay capacity")
    total = int(math.ceil(len(curve_batches) / curve_fraction))
    straight_count = int(round(total * straight_fraction))
    official_count = total - len(curve_batches) - straight_count
    rng = random.Random(seed)

    def sample(pool, count):
        order = list(pool)
        rng.shuffle(order)
        return [order[index % len(order)] for index in range(count)]

    schedule = [
        *((batch, "lap_001_straight", False) for batch in sample(lap1_batches, straight_count)),
        *((batch, "lap_003_curve", True) for batch in sample(curve_batches, len(curve_batches))),
        *((batch, "official_preserve", False) for batch in sample(official_batches, official_count)),
    ]
    rng.shuffle(schedule)
    return schedule


def configure_aggressive_student(student: CnnModel) -> None:
    """Train the final two convolutions and regression head only."""
    set_feature_trainable(student, False)
    for layer_index in (8, 11):
        for parameter in student.features[layer_index].parameters():
            parameter.stop_gradient = False
    student.train()
    for layer in student.features[:8]:
        layer.eval()


def aggressive_optimizer_parameters(
    student: CnnModel, *, head_learning_rate: float,
    last_conv_learning_rate: float, penultimate_conv_learning_rate: float,
):
    penultimate = list(student.features[8].parameters())
    final = list(student.features[11].parameters())
    convolution_ids = {id(parameter) for parameter in [*penultimate, *final]}
    head = [
        parameter for parameter in student.parameters()
        if not parameter.stop_gradient and id(parameter) not in convolution_ids
    ]
    return [
        {"params": penultimate, "learning_rate": penultimate_conv_learning_rate / head_learning_rate},
        {"params": final, "learning_rate": last_conv_learning_rate / head_learning_rate},
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
    initial_sha = sha256_file(args.checkpoint)
    if initial_sha != OFFICIAL_CHECKPOINT_SHA256:
        raise ValueError(f"official checkpoint SHA mismatch: {initial_sha}")

    lap1_all = load_rows(args.lap1_data_json, args.lap1_image_dir, "lap_001_straight")
    lap1 = select_straight_rows(lap1_all, context=10)
    lap3_all = [
        row for row in load_rows(args.lap3_data_json, args.lap3_image_dir, "lap_003_curve")
        if row["image_exists"]
    ]
    curves = build_curve_windows(lap3_all, context=10, max_length=args.batch_size)
    official_all = load_rows(
        args.official_data_json, args.official_image_dir, "official_preserve"
    )
    official = select_straight_rows(official_all, context=10)
    lap1_batches = _chunk(lap1, args.batch_size)
    official_batches = _chunk(official, args.batch_size)
    student = CnnModel()
    teacher = CnnModel()
    state = paddle.load(str(args.checkpoint))
    student.set_state_dict(state)
    teacher.set_state_dict(state)
    configure_aggressive_student(student)
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.stop_gradient = True
    optimizer = paddle.optimizer.Adam(
        learning_rate=args.learning_rate,
        parameters=aggressive_optimizer_parameters(
            student,
            head_learning_rate=args.learning_rate,
            last_conv_learning_rate=args.final_conv_learning_rate,
            penultimate_conv_learning_rate=args.penultimate_conv_learning_rate,
        ),
    )

    source_rows = {
        "lap_001_straight": lap1,
        "lap_003_curve": lap3_all,
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
    args.output.mkdir(parents=True, exist_ok=True)
    epoch_reports = []
    stop = False
    for epoch in range(args.epochs):
        schedule = build_mixed_schedule(
            lap1_batches, curves, official_batches, seed=args.seed + epoch,
            straight_fraction=0.20, curve_fraction=0.65,
        )
        epoch_start = batch_count
        for rows, source, is_curve in schedule:
            if args.max_batches and batch_count >= args.max_batches:
                stop = True
                break
            x, labels = _batch_arrays(datasets[source], row_indices[source], rows)
            prediction = student(x)
            with paddle.no_grad():
                teacher_prediction = teacher(x)
            if source == "official_preserve":
                target = teacher_prediction
            else:
                target = paddle.stack([teacher_prediction[:, 0], labels[:, 1]], axis=1)
            sequence_ids = [0] * len(rows) if is_curve else list(range(len(rows)))
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
        epoch_checkpoint = args.output / f"epoch_{epoch + 1}.pdparams"
        paddle.save(student.state_dict(), str(epoch_checkpoint))
        epoch_reports.append({
            "epoch": epoch + 1,
            "batch_count": batch_count - epoch_start,
            "checkpoint": str(epoch_checkpoint),
            "checkpoint_sha256": sha256_file(epoch_checkpoint),
        })
        if stop:
            break

    checkpoint = args.output / "best.pdparams"
    paddle.save(student.state_dict(), str(checkpoint))
    report = {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "initial_official_checkpoint_sha256": initial_sha,
        "epochs": args.epochs,
        "completed_epochs": len(epoch_reports),
        "epoch_checkpoints": epoch_reports,
        "batch_count": batch_count,
        "source_batch_counts": source_batch_counts,
        "source_rows": {source: len(rows) for source, rows in source_rows.items()},
        "lap_003_curve_count": len(curves),
        "learning_rate": args.learning_rate,
        "final_conv_learning_rate": args.final_conv_learning_rate,
        "penultimate_conv_learning_rate": args.penultimate_conv_learning_rate,
        "source_mix": {"lap_001_straight": 0.20, "lap_003_curve": 0.65, "official_preserve": 0.15},
        "unfrozen": "last two convolutions and regression head",
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
    parser.add_argument("--lap1-data-json", type=Path, required=True)
    parser.add_argument("--lap1-image-dir", type=Path, required=True)
    parser.add_argument("--lap3-data-json", type=Path, required=True)
    parser.add_argument("--lap3-image-dir", type=Path, required=True)
    parser.add_argument("--official-data-json", type=Path, required=True)
    parser.add_argument("--official-image-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="gpu")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--final-conv-learning-rate", type=float, default=5e-6)
    parser.add_argument("--penultimate-conv-learning-rate", type=float, default=2e-6)
    parser.add_argument("--seed", type=int, default=20260811)
    parser.add_argument("--max-batches", type=int, default=0)
    return parser.parse_args(argv)


if __name__ == "__main__":
    print(json.dumps(train(parse_args()), indent=2))
