from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import paddle

from lane_training.action_loss import masked_smooth_l1_loss
from lane_training.action_training import load_manifest
from lane_training.dataset import (
    CLEAN_ACTION_AUGMENTATION,
    MILD_FIXED_TRACK_ACTION_AUGMENTATION,
    LaneDataset,
)
from lane_training.grouped_sampling import build_epoch_batches, partition_action_rows
from lane_training.model import CnnModel
from lane_training.staged_action import (
    build_optimizer,
    load_training_checkpoint,
    save_training_checkpoint,
    stage_for_epoch,
)


DEFAULT_EXPECTED_COUNTS = {
    "cv": 2822,
    "manual_active": 74,
    "manual_context": 572,
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--initial-checkpoint", type=Path)
    source.add_argument("--resume", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("gpu", "cpu"), default="gpu")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--max-epochs", type=int)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--kappa-weight", type=float, default=2.0)
    parser.add_argument("--checkpoint-interval", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260817)
    parser.add_argument("--expected-cv", type=int, default=2822)
    parser.add_argument("--expected-manual-active", type=int, default=74)
    parser.add_argument("--expected-manual-context", type=int, default=572)
    parser.add_argument("--eval-set", action="append", default=[])
    parser.add_argument("--evaluation-output", type=Path)
    return parser.parse_args(argv)


def audit_training_rows(groups: dict[str, list[int]], *, expected: dict) -> None:
    actual = {name: len(indices) for name, indices in groups.items()}
    if actual != expected:
        raise ValueError(
            f"expected training group counts {expected}, received {actual}"
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json_atomic(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _batch(dataset: LaneDataset, indices: list[int]):
    images, targets, masks = zip(*(dataset[index] for index in indices))
    return (
        paddle.to_tensor(np.stack(images)),
        paddle.to_tensor(np.stack(targets)),
        paddle.to_tensor(np.stack(masks)),
    )


def _validate_rows(rows: list[dict]) -> None:
    for index, row in enumerate(rows):
        try:
            values = [float(row["speed_demand"]), float(row["kappa_action"])]
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"invalid action label at row {index}") from error
        if not all(math.isfinite(value) for value in values):
            raise ValueError(f"non-finite action label at row {index}")


def _train(
    *,
    manifest: Path,
    output: Path,
    initial_checkpoint: Path | None = None,
    resume: Path | None = None,
    device: str = "gpu",
    epochs: int = 200,
    max_epochs: int | None = None,
    batch_size: int = 64,
    kappa_weight: float = 2.0,
    checkpoint_interval: int = 5,
    seed: int = 20260817,
    expected_cv: int = 2822,
    expected_manual_active: int = 74,
    expected_manual_context: int = 572,
    eval_set: list[str] | None = None,
    evaluation_output: Path | None = None,
) -> Path:
    if (initial_checkpoint is None) == (resume is None):
        raise ValueError("provide exactly one of initial_checkpoint or resume")
    if not 1 <= epochs <= 200 or max_epochs is not None and max_epochs < 1:
        raise ValueError("invalid epoch limit")
    if checkpoint_interval < 1 or kappa_weight <= 0:
        raise ValueError("invalid training configuration")
    if bool(eval_set) != (evaluation_output is not None):
        raise ValueError("eval_set and evaluation_output must be provided together")
    if resume is None and output.exists():
        raise FileExistsError(f"output already exists: {output}")
    if resume is not None and not output.is_dir():
        raise FileNotFoundError(f"resume output does not exist: {output}")

    paddle.set_device("gpu:0" if device == "gpu" else "cpu")
    random.seed(seed)
    np.random.seed(seed)
    paddle.seed(seed)
    rows = load_manifest(manifest)
    train_rows = [row for row in rows if row.get("split") == "train"]
    if not train_rows:
        raise ValueError("manifest has no train rows")
    _validate_rows(train_rows)
    groups = partition_action_rows(train_rows)
    expected = {
        "cv": expected_cv,
        "manual_active": expected_manual_active,
        "manual_context": expected_manual_context,
    }
    audit_training_rows(groups, expected=expected)

    model = CnnModel()
    history: list[dict]
    if initial_checkpoint is not None:
        initial_state = paddle.load(str(initial_checkpoint))
        if set(initial_state) != set(model.state_dict()):
            raise ValueError("checkpoint keys do not match CnnModel")
        model.set_state_dict(initial_state)
        start_epoch = 1
        optimizer, current_stage = build_optimizer(model, start_epoch)
        initial_checkpoint_sha256 = _sha256(initial_checkpoint)
        history = []
    else:
        restored = load_training_checkpoint(resume, model)
        start_epoch = restored.next_epoch
        optimizer = restored.optimizer
        current_stage = restored.stage
        history = restored.history
        prior_report = json.loads(
            (output / "training_report.json").read_text(encoding="utf-8")
        )
        initial_checkpoint_sha256 = prior_report["initial_checkpoint_sha256"]
    stop_epoch = min(epochs, max_epochs if max_epochs is not None else epochs)
    if start_epoch > stop_epoch:
        raise ValueError("resume checkpoint is already at or beyond requested epoch")

    output.mkdir(parents=True, exist_ok=resume is not None)
    checkpoint_root = output / "checkpoints"
    checkpoint_root.mkdir(exist_ok=True)
    dataset = LaneDataset(
        train_rows,
        training=True,
        return_target_mask=True,
        horizontal_flip_probability=0.0,
        action_augmentation_config=CLEAN_ACTION_AUGMENTATION,
    )
    report = {
        "schema_version": 1,
        "output_semantics": ["speed_demand", "kappa_action"],
        "manifest": str(manifest),
        "manifest_sha256": _sha256(manifest),
        "initial_checkpoint_sha256": initial_checkpoint_sha256,
        "data_counts": expected,
        "settings": {
            "epochs": epochs,
            "batch_size": batch_size,
            "kappa_weight": kappa_weight,
            "checkpoint_interval": checkpoint_interval,
            "seed": seed,
        },
        "history": history,
    }

    for epoch in range(start_epoch, stop_epoch + 1):
        stage = stage_for_epoch(epoch)
        if stage.name != current_stage.name:
            optimizer, current_stage = build_optimizer(model, epoch)
        profile = (
            MILD_FIXED_TRACK_ACTION_AUGMENTATION
            if stage.augmentation_profile == "mild_photometric"
            else CLEAN_ACTION_AUGMENTATION
        )
        dataset.set_action_augmentation_config(profile)
        dataset.set_epoch(epoch)
        batches, sampler_report = build_epoch_batches(
            groups, batch_size=batch_size, seed=seed, epoch=epoch,
        )
        model.train()
        started = time.monotonic()
        totals = []
        speeds = []
        kappas = []
        valid_speed = 0
        valid_kappa = 0
        for indices in batches:
            images, target, mask = _batch(dataset, indices)
            prediction = model(images)
            if not bool(paddle.isfinite(prediction).all()):
                raise FloatingPointError("non-finite model prediction")
            loss, parts = masked_smooth_l1_loss(
                prediction, target, mask, kappa_weight=kappa_weight,
            )
            numeric = {
                name: float(parts[name]) for name in (
                    "total", "speed", "kappa", "valid_speed", "valid_kappa"
                )
            }
            if not all(math.isfinite(value) for value in numeric.values()):
                raise FloatingPointError("non-finite training loss")
            loss.backward()
            optimizer.step()
            optimizer.clear_grad()
            totals.append(numeric["total"])
            speeds.append(numeric["speed"])
            kappas.append(numeric["kappa"])
            valid_speed += int(numeric["valid_speed"])
            valid_kappa += int(numeric["valid_kappa"])

        save_now = epoch % checkpoint_interval == 0 or epoch == stop_epoch
        checkpoint_path = checkpoint_root / f"epoch_{epoch:04d}" if save_now else None
        entry = {
            "epoch": epoch,
            "stage": stage.name,
            "augmentation_profile": stage.augmentation_profile,
            "augmentation_applied_count": dataset.augmentation_applied_count,
            "head_learning_rate": stage.head_learning_rate,
            "backbone_learning_rate": stage.backbone_learning_rate,
            "sampler": sampler_report,
            "valid_speed": valid_speed,
            "valid_kappa": valid_kappa,
            "total_loss": float(np.mean(totals)),
            "speed_loss": float(np.mean(speeds)),
            "kappa_loss": float(np.mean(kappas)),
            "elapsed_seconds": time.monotonic() - started,
            "checkpoint": str(checkpoint_path) if checkpoint_path else None,
        }
        history.append(entry)
        if save_now:
            saved_checkpoint = save_training_checkpoint(
                checkpoint_root,
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                stage=stage,
                sampler_state={"base_seed": seed, "next_epoch": epoch + 1},
                history=history,
            )
            entry["checkpoint_sha256"] = _sha256(
                saved_checkpoint / "model.pdparams"
            )
            state_path = saved_checkpoint / "state.json"
            checkpoint_state = json.loads(
                state_path.read_text(encoding="utf-8")
            )
            checkpoint_state["history"] = history
            state_path.write_text(
                json.dumps(checkpoint_state, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            if eval_set and epoch % checkpoint_interval == 0:
                from tools.lane_training.evaluate_fixed_track_long import (
                    append_evaluation_index,
                    evaluate_checkpoint,
                    parse_set_spec,
                    write_evaluation_report,
                )

                parsed_sets = [parse_set_spec(value) for value in eval_set]
                evaluation_sets = {
                    name: (path, selector)
                    for name, path, selector in parsed_sets
                }
                if len(evaluation_sets) != len(parsed_sets):
                    raise ValueError("duplicate evaluation set name")
                evaluation = evaluate_checkpoint(
                    saved_checkpoint,
                    evaluation_sets,
                    device=device,
                    batch_size=batch_size,
                )
                evaluation_path = write_evaluation_report(
                    evaluation, evaluation_output,
                )
                append_evaluation_index(
                    evaluation, evaluation_path, evaluation_output,
                )
                entry["evaluation_report"] = str(evaluation_path)
                checkpoint_state = json.loads(
                    state_path.read_text(encoding="utf-8")
                )
                checkpoint_state["history"] = history
                state_path.write_text(
                    json.dumps(checkpoint_state, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
        _write_json_atomic(output / "training_report.json", report)
    return output


def train(**kwargs) -> Path:
    # Paddle optimizer checkpoints use internal parameter names. A fresh guard
    # keeps those names stable for both initial training and same-process resume.
    with paddle.utils.unique_name.guard():
        return _train(**kwargs)


def main() -> None:
    args = parse_args()
    print(train(**vars(args)))


if __name__ == "__main__":
    main()
