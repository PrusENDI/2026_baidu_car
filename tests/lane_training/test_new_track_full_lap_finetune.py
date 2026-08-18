import pytest

from lane_training.model import CnnModel
from tools.lane_training.new_track_full_lap_finetune import (
    build_full_lap_schedule,
    configure_full_lap_student,
    full_lap_optimizer_parameters,
    parse_args,
)


def test_full_lap_schedule_uses_every_lap3_batch_exactly_once():
    lap3 = [[{"id": f"l3-{index}"}] for index in range(8)]
    lap1 = [[{"id": f"l1-{index}"}] for index in range(5)]
    official = [[{"id": f"o-{index}"}] for index in range(5)]

    schedule = build_full_lap_schedule(lap3, lap1, official, seed=7)

    lap3_ids = [batch[0]["id"] for batch, source, _ in schedule if source == "lap_003_full"]
    assert sorted(lap3_ids) == sorted(batch[0]["id"] for batch in lap3)
    assert len(lap3_ids) == len(set(lap3_ids)) == 8
    assert len(schedule) == 10
    assert sum(source == "lap_001_straight" for _, source, _ in schedule) == 1
    assert sum(source == "official_preserve" for _, source, _ in schedule) == 1


def test_full_lap_student_unfreezes_last_four_convolutions_and_head():
    student = CnnModel()

    configure_full_lap_student(student)
    groups = full_lap_optimizer_parameters(
        student,
        head_learning_rate=3e-5,
        convolution_learning_rates=(1e-6, 2e-6, 5e-6, 1e-5),
    )

    assert all(parameter.stop_gradient for layer in student.features[:4] for parameter in layer.parameters())
    for index in (4, 6, 8, 11):
        assert all(not parameter.stop_gradient for parameter in student.features[index].parameters())
    assert [group["learning_rate"] for group in groups] == pytest.approx([
        1 / 30, 2 / 30, 5 / 30, 10 / 30, 1.0,
    ])


def test_full_lap_cli_defaults_to_one_epoch_and_approved_learning_rates():
    args = parse_args([
        "--checkpoint", "r2.pdparams", "--teacher-checkpoint", "official.pdparams",
        "--lap1-data-json", "lap1.json", "--lap1-image-dir", "lap1",
        "--lap3-data-json", "lap3.json", "--lap3-image-dir", "lap3",
        "--official-data-json", "official.json", "--official-image-dir", "official",
        "--output", "output",
    ])

    assert args.epochs == 1
    assert args.learning_rate == pytest.approx(3e-5)
    assert args.convolution_learning_rates == pytest.approx((1e-6, 2e-6, 5e-6, 1e-5))
