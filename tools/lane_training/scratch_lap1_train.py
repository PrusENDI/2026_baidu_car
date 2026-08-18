"""Experimental scratch training on lap_001 only.

This is intentionally separate from the D4 fine-tuning paths: it keeps the
deployed CnnModel architecture but initializes all weights randomly and uses
only lap_001 labels. lap_002 must remain evaluation-only.
"""
from __future__ import annotations

import argparse
import hashlib
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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


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


def _batches(rows, batch_size, seed):
    order = list(range(0, len(rows), batch_size))
    random.Random(seed).shuffle(order)
    return [rows[start:start + batch_size] for start in order]


def train(args) -> dict:
    random.seed(args.seed)
    np.random.seed(args.seed)
    paddle.seed(args.seed)
    paddle.set_device(args.device)

    rows = [
        row for row in load_rows(args.lap1_data_json, args.lap1_image_dir, "lap_001")
        if row["image_exists"]
    ]
    if not rows:
        raise ValueError("lap_001 contains no readable images")
    dataset = LaneDataset(rows, training=False, seed=args.seed)
    row_indices = {str(row["image_path"]): index for index, row in enumerate(rows)}

    model = CnnModel()
    initialization = "random CnnModel initialization"
    initial_checkpoint_sha256 = None
    if args.checkpoint is not None:
        model.set_state_dict(paddle.load(str(args.checkpoint)))
        initialization = "checkpoint initialization"
        initial_checkpoint_sha256 = sha256_file(args.checkpoint)
    model.train()
    optimizer = paddle.optimizer.Adam(
        learning_rate=args.learning_rate, parameters=model.parameters()
    )
    args.output.mkdir(parents=True, exist_ok=True)
    reports = []
    for epoch in range(args.epochs):
        totals = {"total": 0.0, "curve": 0.0, "supervised": 0.0, "light": 0.0}
        batches = _batches(rows, args.batch_size, args.seed + epoch)
        for batch_rows in batches:
            images, light_images, labels = [], [], []
            for row in batch_rows:
                index = row_indices[str(row["image_path"])]
                clean, light, label = dataset.get_training_views(
                    index,
                    args.seed + epoch * 1_000_003 + index,
                    severity=args.light_severity,
                    probability=1.0,
                )
                images.append(clean)
                light_images.append(light)
                labels.append(label)
            x = paddle.to_tensor(np.stack(images))
            x_light = paddle.to_tensor(np.stack(light_images))
            target = paddle.to_tensor(np.asarray(labels, dtype=np.float32))
            prediction = model(x)
            prediction_light = model(x_light)
            # Keep this scratch run independent of newer helper modules on the
            # server. The deployed output is still the same [vy, yaw] model.
            curve = F.smooth_l1_loss(prediction[:, 1], target[:, 1])
            supervised = F.smooth_l1_loss(prediction, target)
            light_loss = F.smooth_l1_loss(
                prediction_light[:, 1], prediction[:, 1].detach()
            )
            total = args.curve_weight * curve + supervised + args.light_weight * light_loss
            total.backward()
            optimizer.step()
            optimizer.clear_grad()
            for name, value in (("total", total), ("curve", curve),
                                ("supervised", supervised), ("light", light_loss)):
                totals[name] += float(value)
        checkpoint = args.output / f"epoch_{epoch + 1}.pdparams"
        paddle.save(model.state_dict(), str(checkpoint))
        reports.append({
            "epoch": epoch + 1,
            "batch_count": len(batches),
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": sha256_file(checkpoint),
            "mean_losses": {
                key: value / len(batches) for key, value in totals.items()
            },
        })
    final = args.output / "best.pdparams"
    paddle.save(model.state_dict(), str(final))
    report = {
        "checkpoint": str(final),
        "checkpoint_sha256": sha256_file(final),
        "initialization": initialization,
        "initial_checkpoint_sha256": initial_checkpoint_sha256,
        "training_sources": ["lap_001"],
        "lap_001_row_count": len(rows),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "curve_weight": args.curve_weight,
        "light_weight": args.light_weight,
        "light_severity": args.light_severity,
        "vy_trained_from_lap_001_labels": True,
        "lap_002_used_for_training": False,
        "locked_test_performed": False,
        "selected": False,
        "epoch_reports": reports,
    }
    (args.output / "training_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--lap1-data-json", type=Path, required=True)
    parser.add_argument("--lap1-image-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="gpu")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--curve-weight", type=float, default=2.0)
    parser.add_argument("--light-weight", type=float, default=1.0)
    parser.add_argument("--light-severity", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260816)
    return parser.parse_args(argv)


if __name__ == "__main__":
    print(json.dumps(train(parse_args()), indent=2))
