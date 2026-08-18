"""Run the established unlocked A/B and global D4 preservation gates."""
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

from lane_training.model import CnnModel
from tools.lane_training.new_track_finetune import load_rows
from tools.lane_training.quick_hard_turn_finetune import (
    extended_turn_report,
    global_drift_report,
    predict_clean_rows,
    specialized_behavior_report,
)


def _model(path):
    model = CnnModel()
    model.set_state_dict(paddle.load(str(path)))
    return model


def run(args):
    paddle.set_device(args.device)
    rows = load_rows(args.data_json, args.image_dir, "image_set_lane")
    for index, row in enumerate(rows):
        row.update({"archive": "image_set_lane", "frame": index})
    labels = np.asarray([[row["vy"], row["yaw"]] for row in rows])
    baseline = predict_clean_rows(_model(args.baseline), rows, batch_size=args.batch_size)
    candidate = predict_clean_rows(_model(args.candidate), rows, batch_size=args.batch_size)

    def core(name, start, end):
        return specialized_behavior_report(
            name, list(range(start, end + 1)), labels[start:end + 1],
            baseline[start:end + 1], candidate[start:end + 1],
        )

    def extended(name, start, end):
        return extended_turn_report(
            name, list(range(start, end + 1)), labels[start:end + 1],
            baseline[start:end + 1], candidate[start:end + 1],
        )

    turn_a = core("turn_a", 152, 188)
    turn_b = core("turn_b", 3160, 3254)
    extended_a = extended("turn_a", 100, 220)
    extended_b = extended("turn_b", 3100, 3300)
    global_report = global_drift_report(rows, baseline, candidate)
    for report in (turn_a, turn_b, extended_a, extended_b, global_report):
        report.pop("rows")
    report = {
        "core": {"turn_a": turn_a, "turn_b": turn_b},
        "extended": {"turn_a": extended_a, "turn_b": extended_b},
        "global": global_report,
        "behavior_selected": bool(turn_a["passed"] and turn_b["passed"] and global_report["passed"]),
        "selected": False,
        "locked_test_performed": False,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--data-json", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="gpu")
    parser.add_argument("--batch-size", type=int, default=128)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2))
