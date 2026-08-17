from __future__ import annotations

from dataclasses import dataclass

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
