"""Fast unlocked comparison on lap_002 plus official straight preservation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import paddle

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lane_training.dataset import LaneDataset
from lane_training.model import CnnModel
from lane_training.yaw_curve import discover_turns
from tools.lane_training.new_track_finetune import load_rows, select_straight_rows


def _longest_run(mask) -> int:
    best = current = 0
    for value in mask:
        current = current + 1 if value else 0
        best = max(best, current)
    return best


def summarize_predictions(
    labels: np.ndarray, predictions: np.ndarray, *, min_active: int = 5,
) -> dict:
    labels = np.asarray(labels, dtype=np.float64)
    predictions = np.asarray(predictions, dtype=np.float64)
    target_yaw = labels[:, 1]
    predicted_yaw = predictions[:, 1]
    active = np.abs(target_yaw) >= 0.05
    straight = np.abs(target_yaw) < 0.05
    turns = discover_turns(target_yaw, min_active=min_active)
    curve_indices = np.concatenate([
        np.arange(start, end + 1) for start, end, _ in turns
    ]) if turns else np.asarray([], dtype=int)
    denominator = float(np.sum(np.abs(target_yaw[curve_indices])))
    normalized_error = (
        float(np.sum(np.abs(predicted_yaw[curve_indices] - target_yaw[curve_indices])))
        / denominator if denominator else float("nan")
    )
    impulse_ratios = []
    correlations = []
    onset_offsets = []
    exit_offsets = []
    for start, end, _ in turns:
        target = target_yaw[start:end + 1]
        predicted = predicted_yaw[start:end + 1]
        impulse_ratios.append(float(np.sum(np.abs(predicted))) / max(float(np.sum(np.abs(target))), 1e-9))
        if len(target) > 1 and np.std(target) > 1e-9 and np.std(predicted) > 1e-9:
            correlations.append(float(np.corrcoef(target, predicted)[0, 1]))
        predicted_active = np.flatnonzero(np.abs(predicted) >= 0.05)
        if predicted_active.size:
            onset_offsets.append(int(predicted_active[0]))
            exit_offsets.append(int(predicted_active[-1] - (end - start)))
    direction = np.sign(target_yaw[active]) == np.sign(predicted_yaw[active])
    return {
        "row_count": int(len(labels)),
        "turn_count": int(len(turns)),
        "yaw_mae": float(np.mean(np.abs(predicted_yaw - target_yaw))),
        "normalized_curve_error": normalized_error,
        "median_impulse_ratio": float(np.median(impulse_ratios)) if impulse_ratios else float("nan"),
        "median_curve_correlation": float(np.median(correlations)) if correlations else float("nan"),
        "median_onset_offset_frames": float(np.median(onset_offsets)) if onset_offsets else float("nan"),
        "median_exit_offset_frames": float(np.median(exit_offsets)) if exit_offsets else float("nan"),
        "direction_consistency": float(np.mean(direction)) if direction.size else float("nan"),
        "straight_false_turn_rate": float(np.mean(np.abs(predicted_yaw[straight]) >= 0.05)),
        "straight_mean_abs_yaw": float(np.mean(np.abs(predicted_yaw[straight]))),
        "pid_saturation_rate": float(np.mean(np.abs(predicted_yaw) >= 0.5)),
        "longest_pid_saturation_frames": _longest_run(np.abs(predicted_yaw) >= 0.5),
        "peak_abs_yaw": float(np.max(np.abs(predicted_yaw))),
        "mean_abs_vy": float(np.mean(np.abs(predictions[:, 0]))),
        "max_abs_vy": float(np.max(np.abs(predictions[:, 0]))),
    }


def infer(checkpoint: Path, rows: list[dict], batch_size: int) -> tuple[np.ndarray, np.ndarray]:
    model = CnnModel()
    model.set_state_dict(paddle.load(str(checkpoint)))
    model.eval()
    dataset = LaneDataset(rows, training=False)
    outputs = []
    labels = []
    with paddle.no_grad():
        for start in range(0, len(rows), batch_size):
            batch = [dataset[index] for index in range(start, min(len(rows), start + batch_size))]
            x = paddle.to_tensor(np.stack([item[0] for item in batch]))
            outputs.append(model(x).numpy())
            labels.append(np.stack([item[1] for item in batch]))
    return np.concatenate(labels), np.concatenate(outputs)


def parse_models(values):
    models = {}
    for value in values:
        name, separator, path = value.partition("=")
        if not separator or not name or name in models:
            raise ValueError("models must use unique name=path entries")
        models[name] = Path(path)
    return models


def evaluate(args) -> dict:
    paddle.set_device(args.device)
    models = parse_models(args.model)
    lap2 = [row for row in load_rows(args.lap2_data_json, args.lap2_image_dir, "lap_002") if row["image_exists"]]
    official_all = load_rows(args.official_data_json, args.official_image_dir, "official")
    official_straight = select_straight_rows(official_all, context=10)
    lap2_reports = {}
    official_predictions = {}
    for name, checkpoint in models.items():
        labels, prediction = infer(checkpoint, lap2, args.batch_size)
        lap2_reports[name] = summarize_predictions(labels, prediction)
        _, official_predictions[name] = infer(checkpoint, official_straight, args.batch_size)
    baseline = official_predictions["official"]
    preservation = {}
    for name, prediction in official_predictions.items():
        delta = np.abs(prediction - baseline)
        preservation[name] = {
            "mean_abs_yaw_delta": float(np.mean(delta[:, 1])),
            "p95_abs_yaw_delta": float(np.quantile(delta[:, 1], 0.95)),
            "mean_abs_vy_delta": float(np.mean(delta[:, 0])),
            "max_abs_vy_delta": float(np.max(delta[:, 0])),
        }
    candidate = lap2_reports["candidate"]
    base = lap2_reports["official"]
    preserve = preservation["candidate"]
    behavior_selected = bool(
        candidate["normalized_curve_error"] < base["normalized_curve_error"]
        and candidate["straight_false_turn_rate"] <= base["straight_false_turn_rate"] + 0.01
        and candidate["direction_consistency"] >= base["direction_consistency"] - 0.005
        and preserve["mean_abs_yaw_delta"] <= 0.03
        and preserve["mean_abs_vy_delta"] <= 0.01
    )
    report = {
        "lap_002": lap2_reports,
        "official_nonturn_preservation": preservation,
        "behavior_selected": behavior_selected,
        "selected": False,
        "locked_test_performed": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", action="append", required=True)
    parser.add_argument("--lap2-data-json", type=Path, required=True)
    parser.add_argument("--lap2-image-dir", type=Path, required=True)
    parser.add_argument("--official-data-json", type=Path, required=True)
    parser.add_argument("--official-image-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="gpu")
    parser.add_argument("--batch-size", type=int, default=128)
    return parser.parse_args(argv)


if __name__ == "__main__":
    print(json.dumps(evaluate(parse_args()), indent=2))
