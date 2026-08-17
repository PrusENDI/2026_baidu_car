import pytest
import paddle

from lane_training.model import CnnModel
from lane_training.staged_action import (
    build_optimizer,
    configure_stage,
    load_training_checkpoint,
    optimizer_parameter_groups,
    save_training_checkpoint,
    stage_for_epoch,
)


@pytest.mark.parametrize(
    ("epoch", "name", "head_lr", "backbone_lr", "augmentation_profile"),
    [
        (1, "head", 1e-5, None, "clean"),
        (20, "head", 1e-5, None, "clean"),
        (21, "rear", 5e-6, 1e-6, "mild_photometric"),
        (160, "rear", 5e-6, 1e-6, "mild_photometric"),
        (161, "full", 2e-6, 2e-7, "clean"),
        (200, "full", 2e-6, 2e-7, "clean"),
    ],
)
def test_stage_boundaries(
    epoch, name, head_lr, backbone_lr, augmentation_profile,
):
    stage = stage_for_epoch(epoch)
    assert stage.name == name
    assert stage.head_learning_rate == pytest.approx(head_lr)
    if backbone_lr is None:
        assert stage.backbone_learning_rate is None
    else:
        assert stage.backbone_learning_rate == pytest.approx(backbone_lr)
    assert stage.augmentation_profile == augmentation_profile


def _trainable_layer_indices(model):
    return {
        index for index, layer in enumerate(model.features)
        if list(layer.parameters())
        and any(not parameter.stop_gradient for parameter in layer.parameters())
    }


def test_stage_configuration_uses_approved_layers_and_rate_multipliers():
    model = CnnModel()
    head = configure_stage(model, epoch=1)
    assert _trainable_layer_indices(model) == {15, 17, 20}
    assert [group["learning_rate"] for group in optimizer_parameter_groups(
        model, head,
    )] == [1.0]

    rear = configure_stage(model, epoch=21)
    assert _trainable_layer_indices(model) == {8, 11, 15, 17, 20}
    assert [group["learning_rate"] for group in optimizer_parameter_groups(
        model, rear,
    )] == pytest.approx([0.2, 1.0])

    full = configure_stage(model, epoch=161)
    assert _trainable_layer_indices(model) == {0, 2, 4, 6, 8, 11, 15, 17, 20}
    assert [group["learning_rate"] for group in optimizer_parameter_groups(
        model, full,
    )] == pytest.approx([0.1, 1.0])


def test_training_checkpoint_round_trip_restores_complete_state(tmp_path):
    model = CnnModel()
    optimizer, stage = build_optimizer(model, epoch=21)
    checkpoint = save_training_checkpoint(
        tmp_path,
        epoch=21,
        model=model,
        optimizer=optimizer,
        stage=stage,
        sampler_state={"base_seed": 20260817, "next_epoch": 22},
        history=[{"epoch": 21, "train_loss": 0.2}],
    )

    restored = CnnModel()
    result = load_training_checkpoint(checkpoint, restored)

    assert result.last_completed_epoch == 21
    assert result.next_epoch == 22
    assert result.stage.name == "rear"
    assert result.sampler_state == {"base_seed": 20260817, "next_epoch": 22}
    assert result.history == [{"epoch": 21, "train_loss": 0.2}]
    for name, value in model.state_dict().items():
        assert paddle.allclose(value, restored.state_dict()[name])


def test_training_checkpoint_rejects_corrupted_model(tmp_path):
    model = CnnModel()
    optimizer, stage = build_optimizer(model, epoch=20)
    checkpoint = save_training_checkpoint(
        tmp_path,
        epoch=20,
        model=model,
        optimizer=optimizer,
        stage=stage,
        sampler_state={"next_epoch": 21},
        history=[],
    )
    model_path = checkpoint / "model.pdparams"
    with model_path.open("ab") as handle:
        handle.write(b"corrupt")

    with pytest.raises(ValueError, match="checkpoint SHA256 mismatch"):
        load_training_checkpoint(checkpoint, CnnModel())
