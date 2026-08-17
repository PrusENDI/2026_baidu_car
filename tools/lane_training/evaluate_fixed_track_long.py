from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import paddle

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lane_training.action_sequence_metrics import (
    derive_sequence_keys,
    replay_vehicle_commands,
    summarize_action_sequence,
)
from lane_training.action_training import load_manifest, predict_action
from lane_training.model import CnnModel


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_set_spec(spec: str) -> tuple[str, Path, str]:
    if "=" not in spec or ":" not in spec:
        raise ValueError(f"invalid evaluation set: {spec}")
    name, location = spec.split("=", 1)
    manifest, selector = location.rsplit(":", 1)
    if not name or not manifest or not selector:
        raise ValueError(f"invalid evaluation set: {spec}")
    return name, Path(manifest), selector


def select_rows(rows: list[dict], selector: str) -> list[dict]:
    if selector == "train_manual":
        selected = [
            row for row in rows
            if row.get("source") == "manual" and row.get("split") == "train"
        ]
    elif selector == "validation_cv":
        selected = [
            row for row in rows
            if row.get("source") == "cv"
            and row.get("split") in {"val", "validation"}
        ]
    else:
        selected = [row for row in rows if row.get("split") == selector]
    if not selected:
        raise ValueError(f"selector {selector!r} produced no rows")
    return selected


def _checkpoint_details(checkpoint: Path) -> tuple[Path, int | None]:
    if checkpoint.is_dir():
        model_path = checkpoint / "model.pdparams"
        state_path = checkpoint / "state.json"
        epoch = None
        if state_path.is_file():
            epoch = int(json.loads(state_path.read_text(encoding="utf-8"))[
                "last_completed_epoch"
            ])
    else:
        model_path = checkpoint
        epoch = None
    if not model_path.is_file():
        raise FileNotFoundError(f"missing checkpoint model: {model_path}")
    return model_path, epoch


def evaluate_checkpoint(
    checkpoint: Path,
    evaluation_sets: dict[str, tuple[Path, str]],
    *,
    device: str = "gpu",
    batch_size: int = 64,
) -> dict:
    if not evaluation_sets:
        raise ValueError("at least one evaluation set is required")
    paddle.set_device("gpu:0" if device == "gpu" else "cpu")
    model_path, epoch = _checkpoint_details(Path(checkpoint))
    with paddle.utils.unique_name.guard():
        model = CnnModel()
        state = paddle.load(str(model_path))
        if set(state) != set(model.state_dict()):
            raise ValueError("checkpoint keys do not match CnnModel")
        model.set_state_dict(state)
        set_reports = {}
        for name, (manifest, selector) in evaluation_sets.items():
            rows = select_rows(load_manifest(manifest), selector)
            prediction, target, mask = predict_action(
                model, rows, batch_size=batch_size,
            )
            if not np.all(np.isfinite(prediction)):
                raise FloatingPointError(f"non-finite prediction in {name}")
            valid = mask.sum(axis=0)
            absolute_error = np.abs(prediction - target) * mask
            mae = np.divide(
                absolute_error.sum(axis=0), np.maximum(valid, 1e-6),
            )
            basic = {
                "count": len(rows),
                "valid_speed": int(valid[0]),
                "valid_kappa": int(valid[1]),
                "speed_mae": None if valid[0] == 0 else float(mae[0]),
                "kappa_mae": None if valid[1] == 0 else float(mae[1]),
                "masked_mae": float(
                    absolute_error.sum() / max(float(mask.sum()), 1e-6)
                ),
            }
            keys = derive_sequence_keys(rows)
            basic["sequence"] = summarize_action_sequence(
                target[:, 1], prediction[:, 1], sequence_keys=keys,
            )
            basic["per_sequence"] = {}
            for key in dict.fromkeys(keys):
                indices = [index for index, value in enumerate(keys) if value == key]
                basic["per_sequence"][key] = summarize_action_sequence(
                    target[indices, 1],
                    prediction[indices, 1],
                    sequence_keys=[key] * len(indices),
                )
            if float(mask[:, 0].sum()) > 0:
                basic["vehicle_replay"] = replay_vehicle_commands(
                    rows, prediction, sequence_keys=keys,
                )
            set_reports[name] = basic
    return {
        "schema_version": 1,
        "epoch": epoch,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256(model_path),
        "sets": set_reports,
    }


def write_evaluation_report(report: dict, output: Path) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    epoch = report.get("epoch")
    filename = f"epoch_{epoch:04d}.json" if epoch is not None else "checkpoint.json"
    path = output / filename
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
    return path


def append_evaluation_index(report: dict, report_path: Path, output: Path) -> None:
    index_path = output / "evaluation_index.json"
    if index_path.is_file():
        index = json.loads(index_path.read_text(encoding="utf-8"))
    else:
        index = {"reports": []}
    if any(item.get("epoch") == report.get("epoch") for item in index["reports"]):
        raise ValueError(f"evaluation already exists for epoch {report.get('epoch')}")
    indexed = dict(report)
    indexed["report_path"] = str(report_path)
    index["reports"].append(indexed)
    temporary = index_path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(index, indent=2, allow_nan=False) + "\n", encoding="utf-8",
    )
    temporary.replace(index_path)


def evaluate_checkpoints(
    checkpoints: list[Path], evaluation_sets: dict[str, tuple[Path, str]],
    *, output: Path, device: str, batch_size: int,
) -> list[dict]:
    reports = []
    for checkpoint in checkpoints:
        report = evaluate_checkpoint(
            checkpoint, evaluation_sets, device=device, batch_size=batch_size,
        )
        report_path = write_evaluation_report(report, output)
        report["report_path"] = str(report_path)
        reports.append(report)
    index = output / "evaluation_index.json"
    temporary = index.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps({"reports": reports}, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(index)
    return reports


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint", type=Path)
    source.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--set", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("gpu", "cpu"), default="gpu")
    parser.add_argument("--batch-size", type=int, default=64)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    parsed = [parse_set_spec(value) for value in args.set]
    evaluation_sets = {name: (manifest, selector) for name, manifest, selector in parsed}
    if len(evaluation_sets) != len(parsed):
        raise ValueError("duplicate evaluation set name")
    checkpoints = (
        [args.checkpoint]
        if args.checkpoint
        else sorted(
            path for path in args.checkpoint_dir.glob("epoch_[0-9][0-9][0-9][0-9]")
            if path.is_dir()
        )
    )
    if not checkpoints:
        raise ValueError("no checkpoints found")
    evaluate_checkpoints(
        checkpoints,
        evaluation_sets,
        output=args.output,
        device=args.device,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
