from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import paddle


@dataclass(frozen=True)
class TrainingStage:
    name: str
    start_epoch: int
    end_epoch: int
    head_learning_rate: float
    backbone_learning_rate: float | None
    trainable_convolution_indices: tuple[int, ...]
    augmentation_profile: str


STAGES = (
    TrainingStage("head", 1, 20, 1e-5, None, (), "clean"),
    TrainingStage("rear", 21, 160, 5e-6, 1e-6, (8, 11), "mild_photometric"),
    TrainingStage(
        "full", 161, 200, 2e-6, 2e-7, (0, 2, 4, 6, 8, 11), "clean",
    ),
)
HEAD_LAYER_INDICES = (15, 17, 20)


def stage_for_epoch(epoch: int) -> TrainingStage:
    for stage in STAGES:
        if stage.start_epoch <= epoch <= stage.end_epoch:
            return stage
    raise ValueError("epoch must be in 1..200")


def configure_stage(model, epoch: int) -> TrainingStage:
    stage = stage_for_epoch(epoch)
    for parameter in model.parameters():
        parameter.stop_gradient = True
    for index in stage.trainable_convolution_indices + HEAD_LAYER_INDICES:
        for parameter in model.features[index].parameters():
            parameter.stop_gradient = False
    return stage


def optimizer_parameter_groups(model, stage: TrainingStage) -> list[dict]:
    backbone = []
    for index in stage.trainable_convolution_indices:
        backbone.extend(
            parameter for parameter in model.features[index].parameters()
            if not parameter.stop_gradient
        )
    head = []
    for index in HEAD_LAYER_INDICES:
        head.extend(
            parameter for parameter in model.features[index].parameters()
            if not parameter.stop_gradient
        )
    if not head or (stage.backbone_learning_rate is not None and not backbone):
        raise ValueError("optimizer parameter group must not be empty")
    all_parameters = backbone + head
    if len({id(parameter) for parameter in all_parameters}) != len(all_parameters):
        raise ValueError("duplicate optimizer parameter")

    groups = []
    if backbone:
        groups.append({
            "params": backbone,
            "learning_rate": stage.backbone_learning_rate
            / stage.head_learning_rate,
        })
    groups.append({"params": head, "learning_rate": 1.0})
    return groups


def build_optimizer(model, epoch: int):
    stage = configure_stage(model, epoch)
    optimizer = paddle.optimizer.Adam(
        learning_rate=stage.head_learning_rate,
        parameters=optimizer_parameter_groups(model, stage),
    )
    return optimizer, stage


@dataclass
class LoadedTrainingCheckpoint:
    checkpoint: Path
    last_completed_epoch: int
    stage: TrainingStage
    optimizer: paddle.optimizer.Optimizer
    sampler_state: dict
    history: list[dict]

    @property
    def next_epoch(self) -> int:
        return self.last_completed_epoch + 1


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_training_checkpoint(
    checkpoint_root: Path | str,
    *,
    epoch: int,
    model,
    optimizer,
    stage: TrainingStage,
    sampler_state: dict,
    history: list[dict],
) -> Path:
    if stage != stage_for_epoch(epoch):
        raise ValueError("checkpoint stage does not match epoch")
    root = Path(checkpoint_root)
    root.mkdir(parents=True, exist_ok=True)
    final = root / f"epoch_{epoch:04d}"
    incomplete = root / f"epoch_{epoch:04d}.incomplete"
    if final.exists() or incomplete.exists():
        raise FileExistsError(f"checkpoint already exists: {final}")
    incomplete.mkdir()
    model_path = incomplete / "model.pdparams"
    optimizer_path = incomplete / "optimizer.pdopt"
    paddle.save(model.state_dict(), str(model_path))
    paddle.save(optimizer.state_dict(), str(optimizer_path))
    hashes = {
        "model.pdparams": _sha256(model_path),
        "optimizer.pdopt": _sha256(optimizer_path),
    }
    state = {
        "schema_version": 1,
        "last_completed_epoch": int(epoch),
        "stage": stage.name,
        "augmentation_profile": stage.augmentation_profile,
        "sampler_state": sampler_state,
        "history": history,
        "sha256": hashes,
    }
    (incomplete / "state.json").write_text(
        json.dumps(state, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if any(_sha256(incomplete / name) != value for name, value in hashes.items()):
        raise ValueError("checkpoint SHA256 mismatch after save")
    incomplete.rename(final)
    return final


def load_training_checkpoint(checkpoint: Path | str, model) -> LoadedTrainingCheckpoint:
    checkpoint = Path(checkpoint)
    if checkpoint.name.endswith(".incomplete"):
        raise ValueError("cannot load an incomplete checkpoint")
    state = json.loads((checkpoint / "state.json").read_text(encoding="utf-8"))
    if state.get("schema_version") != 1:
        raise ValueError("unsupported checkpoint schema")
    for filename, expected in state.get("sha256", {}).items():
        if _sha256(checkpoint / filename) != expected:
            raise ValueError("checkpoint SHA256 mismatch")
    epoch = int(state["last_completed_epoch"])
    stage = stage_for_epoch(epoch)
    if (
        state.get("stage") != stage.name
        or state.get("augmentation_profile") != stage.augmentation_profile
    ):
        raise ValueError("checkpoint stage metadata mismatch")
    model_state = paddle.load(str(checkpoint / "model.pdparams"))
    if set(model_state) != set(model.state_dict()):
        raise ValueError("checkpoint model keys do not match CnnModel")
    model.set_state_dict(model_state)
    optimizer, configured_stage = build_optimizer(model, epoch)
    optimizer.set_state_dict(paddle.load(str(checkpoint / "optimizer.pdopt")))
    return LoadedTrainingCheckpoint(
        checkpoint=checkpoint,
        last_completed_epoch=epoch,
        stage=configured_stage,
        optimizer=optimizer,
        sampler_state=dict(state.get("sampler_state", {})),
        history=list(state.get("history", [])),
    )
