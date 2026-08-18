"""Long D4 adaptation on manually confirmed physical track segments."""
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
from tools.lane_training.official_yaw_curve_finetune import sha256_file


D4_SHA256 = "3398fc6b2d2be73d1ec7eed6f54f51f0be6e2b6c5814f255485d23442167cb49"
VY_SCALE = 0.15
CURVE_WEIGHTS = {
    "point": 1.0, "slope": 0.5, "curvature": 0.2,
    "integral": 1.0, "phase": 0.5, "saturation": 0.0,
}
PHYSICAL_CURVES = {
    "lap_001": ((1666, 1842), (2164, 2307)),
    "lap_003": (
        (197, 277), (487, 1148), (1260, 1390), (1559, 1630),
        (2567, 2734), (3997, 4344),
    ),
}
STRAIGHT_CROSSES = {"lap_003": ((266, 472), (1148, 1219))}


def stage(epoch: int) -> tuple[float, float]:
    if epoch < 20:
        return 0.60, 1.0
    if epoch < 60:
        return 0.75, 0.3
    return 0.85, 0.1


def mark_rows(rows, source):
    marked = []
    curves = PHYSICAL_CURVES.get(source, ())
    crosses = STRAIGHT_CROSSES.get(source, ())
    for index, row in enumerate(rows):
        copied = dict(row)
        copied["frame_index"] = index
        copied["role"] = "preserve"
        copied["segment_id"] = None
        for segment_index, (start, end) in enumerate(curves):
            if start <= index <= end:
                copied["role"] = "curve"
                copied["segment_id"] = f"{source}_curve_{segment_index}"
                break
        for segment_index, (start, end) in enumerate(crosses):
            if start <= index <= end:
                copied["role"] = "cross"
                copied["segment_id"] = f"{source}_cross_{segment_index}"
                break
        marked.append(copied)
    return marked


def contiguous_batches(rows, role, batch_size):
    groups = {}
    for row in rows:
        if row["role"] == role:
            groups.setdefault(row["segment_id"], []).append(row)
    batches = []
    for group in groups.values():
        for start in range(0, len(group), batch_size):
            batches.append(group[start:start + batch_size])
    return batches


def sample_batches(pool, count, rng):
    values = list(pool)
    rng.shuffle(values)
    return [values[index % len(values)] for index in range(count)]


def epoch_schedule(curve_batches, cross_batches, preserve_rows, batch_size, seed):
    rng = random.Random(seed)
    total = max(20, len(curve_batches) * 2)
    curve_count = total // 2
    cross_count = total // 5
    preserve_count = total - curve_count - cross_count
    preserve = list(preserve_rows)
    rng.shuffle(preserve)
    preserve_batches = [
        preserve[start:start + batch_size]
        for start in range(0, min(len(preserve), preserve_count * batch_size), batch_size)
    ]
    schedule = [
        *((batch, "curve") for batch in sample_batches(curve_batches, curve_count, rng)),
        *((batch, "cross") for batch in sample_batches(cross_batches, cross_count, rng)),
        *((batch, "preserve") for batch in sample_batches(preserve_batches, preserve_count, rng)),
    ]
    rng.shuffle(schedule)
    return schedule


def direction_loss(predicted_yaw, target_yaw):
    active = paddle.abs(target_yaw) >= 0.05
    selected = paddle.masked_select(predicted_yaw * paddle.sign(target_yaw), active)
    if selected.shape[0] == 0:
        return paddle.zeros([], dtype=predicted_yaw.dtype)
    return paddle.mean(F.relu(0.02 - selected))


def make_optimizer(student, head_lr, conv_lrs):
    return paddle.optimizer.Adam(
        learning_rate=head_lr,
        parameters=full_lap_optimizer_parameters(
            student, head_learning_rate=head_lr,
            convolution_learning_rates=conv_lrs,
        ),
    )


def train(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    paddle.seed(args.seed)
    paddle.set_device(args.device)
    if sha256_file(args.checkpoint) != D4_SHA256:
        raise ValueError("D4 checkpoint SHA mismatch")

    sources = {}
    for name, data_json, image_dir in (
        ("lap_001", args.lap1_data_json, args.lap1_image_dir),
        ("lap_003", args.lap3_data_json, args.lap3_image_dir),
    ):
        rows = [row for row in load_rows(data_json, image_dir, name) if row["image_exists"]]
        sources[name] = mark_rows(rows, name)
    all_rows = sources["lap_001"] + sources["lap_003"]
    curve_batches = sum((contiguous_batches(rows, "curve", args.batch_size) for rows in sources.values()), [])
    cross_batches = sum((contiguous_batches(rows, "cross", args.batch_size) for rows in sources.values()), [])
    preserve_rows = [row for row in all_rows if row["role"] == "preserve"]
    datasets = {name: LaneDataset(rows, training=False, seed=args.seed) for name, rows in sources.items()}
    indices = {
        name: {str(row["image_path"]): index for index, row in enumerate(rows)}
        for name, rows in sources.items()
    }

    student = CnnModel()
    state = paddle.load(str(args.checkpoint))
    student.set_state_dict(state)
    configure_full_lap_student(student)
    teacher = CnnModel()
    teacher.set_state_dict(state)
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.stop_gradient = True

    args.output.mkdir(parents=True, exist_ok=True)
    reports = []
    optimizer = None
    current_scale = None
    for epoch in range(args.epochs):
        label_fraction, lr_scale = stage(epoch)
        if lr_scale != current_scale:
            optimizer = make_optimizer(
                student, args.learning_rate * lr_scale,
                [value * lr_scale for value in args.convolution_learning_rates],
            )
            current_scale = lr_scale
        schedule = epoch_schedule(
            curve_batches, cross_batches, preserve_rows, args.batch_size,
            args.seed + epoch,
        )
        totals = {"total": 0.0, "yaw": 0.0, "direction": 0.0, "consistency": 0.0, "vy": 0.0}
        roles = {"curve": 0, "cross": 0, "preserve": 0}
        for batch_rows, role in schedule:
            images, augmented = [], []
            labels = []
            for row in batch_rows:
                name = row["source"]
                dataset = datasets[name]
                index = indices[name][str(row["image_path"])]
                clean, light, label = dataset.get_training_views(
                    index, args.seed + epoch * 1_000_003 + row["frame_index"],
                    severity=args.light_severity, probability=1.0,
                )
                images.append(clean)
                augmented.append(light)
                labels.append(label)
            x = paddle.to_tensor(np.stack(images))
            x_light = paddle.to_tensor(np.stack(augmented))
            y = paddle.to_tensor(np.stack(labels))
            prediction = student(x)
            prediction_light = student(x_light)
            with paddle.no_grad():
                teacher_prediction = teacher(x)
            if role == "curve":
                target_yaw = label_fraction * y[:, 1] + (1.0 - label_fraction) * teacher_prediction[:, 1]
            elif role == "cross":
                target_yaw = paddle.zeros_like(y[:, 1])
            else:
                target_yaw = teacher_prediction[:, 1]
            target = paddle.stack([teacher_prediction[:, 0], target_yaw], axis=1)
            ordered = role != "preserve"
            sequence_ids = [0] * len(batch_rows) if ordered else list(range(len(batch_rows)))
            yaw, _ = yaw_curve_loss(
                prediction, target, sequence_ids, weights=CURVE_WEIGHTS,
            )
            direction = direction_loss(prediction[:, 1], target_yaw) if role == "curve" else paddle.zeros([])
            consistency = F.smooth_l1_loss(prediction_light[:, 1], prediction[:, 1].detach())
            vy = F.smooth_l1_loss(
                prediction[:, 0] / VY_SCALE, teacher_prediction[:, 0] / VY_SCALE,
            ) + F.smooth_l1_loss(
                prediction_light[:, 0] / VY_SCALE, teacher_prediction[:, 0] / VY_SCALE,
            )
            role_weight = 2.0 if role == "curve" else (3.0 if role == "cross" else 2.0)
            total = role_weight * yaw + 4.0 * direction + 3.0 * consistency + vy
            total.backward()
            optimizer.step()
            optimizer.clear_grad()
            roles[role] += 1
            for name, value in (("total", total), ("yaw", yaw), ("direction", direction), ("consistency", consistency), ("vy", vy)):
                totals[name] += float(value)
        checkpoint = args.output / f"epoch_{epoch + 1}.pdparams"
        paddle.save(student.state_dict(), str(checkpoint))
        reports.append({
            "epoch": epoch + 1, "label_fraction": label_fraction,
            "learning_rate_scale": lr_scale, "role_batches": roles,
            "checkpoint": str(checkpoint), "checkpoint_sha256": sha256_file(checkpoint),
            "mean_losses": {name: value / len(schedule) for name, value in totals.items()},
        })
        (args.output / "training_report.json").write_text(json.dumps({
            "initial_d4_sha256": D4_SHA256, "epochs": args.epochs,
            "completed_epochs": epoch + 1, "physical_curves": PHYSICAL_CURVES,
            "straight_crosses": STRAIGHT_CROSSES, "epoch_reports": reports,
            "unfrozen": "last four convolutions and regression head",
            "vy_optimized": False, "saturation_loss_enabled": False,
            "locked_test_performed": False, "selected": False,
        }, indent=2) + "\n", encoding="utf-8")
    paddle.save(student.state_dict(), str(args.output / "best.pdparams"))
    return reports[-1]


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--lap1-data-json", type=Path, required=True)
    parser.add_argument("--lap1-image-dir", type=Path, required=True)
    parser.add_argument("--lap3-data-json", type=Path, required=True)
    parser.add_argument("--lap3-image-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="gpu")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--convolution-learning-rates", nargs=4, type=float, default=(3e-7, 6e-7, 1.5e-6, 3e-6))
    parser.add_argument("--light-severity", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260812)
    return parser.parse_args(argv)


if __name__ == "__main__":
    print(json.dumps(train(parse_args()), indent=2))
