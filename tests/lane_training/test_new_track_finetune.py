import pytest

from lane_training.model import CnnModel

from tools.lane_training.new_track_finetune import (
    aggressive_optimizer_parameters,
    build_mixed_schedule,
    configure_aggressive_student,
    parse_args,
    select_straight_rows,
)


def _rows(yaw):
    return [
        {"image_path": f"{index:04d}.jpg", "yaw": value, "vy": 0.0}
        for index, value in enumerate(yaw)
    ]


def test_select_straight_rows_excludes_turn_context_and_unlabelled_images():
    rows = _rows([0.0] * 12 + [0.1] * 5 + [0.0] * 12)
    rows[0]["image_exists"] = False

    selected = select_straight_rows(rows, threshold=0.05, context=3)

    assert [row["image_path"] for row in selected] == [
        *[f"{index:04d}.jpg" for index in range(1, 9)],
        *[f"{index:04d}.jpg" for index in range(20, 29)],
    ]


def test_mixed_schedule_keeps_every_curve_once_and_uses_30_40_30_mix():
    lap1 = [[{"id": f"s{index}"}] for index in range(20)]
    curves = [[{"id": f"c{index}"}] for index in range(8)]
    official = [[{"id": f"o{index}"}] for index in range(20)]

    schedule = build_mixed_schedule(lap1, curves, official, seed=7)

    sources = [source for _, source, _ in schedule]
    assert len(schedule) == 20
    assert sources.count("lap_001_straight") == 6
    assert sources.count("lap_003_curve") == 8
    assert sources.count("official_preserve") == 6
    assert {batch[0]["id"] for batch, source, _ in schedule if source == "lap_003_curve"} == {
        f"c{index}" for index in range(8)
    }


def test_mixed_schedule_requires_all_three_sources():
    with pytest.raises(ValueError, match="all three"):
        build_mixed_schedule([[{}]], [], [[{}]], seed=7)


def test_aggressive_schedule_uses_twenty_sixty_five_fifteen_mix():
    lap1 = [[{"id": f"s{index}"}] for index in range(20)]
    curves = [[{"id": f"c{index}"}] for index in range(13)]
    official = [[{"id": f"o{index}"}] for index in range(20)]

    schedule = build_mixed_schedule(
        lap1, curves, official, seed=7,
        straight_fraction=0.20, curve_fraction=0.65,
    )

    sources = [source for _, source, _ in schedule]
    assert len(schedule) == 20
    assert sources.count("lap_001_straight") == 4
    assert sources.count("lap_003_curve") == 13
    assert sources.count("official_preserve") == 3


def test_aggressive_student_unfreezes_last_two_convolutions_and_head():
    student = CnnModel()

    configure_aggressive_student(student)
    groups = aggressive_optimizer_parameters(
        student,
        head_learning_rate=2e-5,
        last_conv_learning_rate=5e-6,
        penultimate_conv_learning_rate=2e-6,
    )

    assert all(parameter.stop_gradient for layer in student.features[:8] for parameter in layer.parameters())
    assert all(not parameter.stop_gradient for parameter in student.features[8].parameters())
    assert all(not parameter.stop_gradient for parameter in student.features[11].parameters())
    assert [group["learning_rate"] for group in groups] == pytest.approx([0.1, 0.25, 1.0])


def test_aggressive_cli_defaults_to_two_epochs_and_approved_learning_rates():
    args = parse_args([
        "--checkpoint", "official.pdparams",
        "--lap1-data-json", "lap1.json", "--lap1-image-dir", "lap1",
        "--lap3-data-json", "lap3.json", "--lap3-image-dir", "lap3",
        "--official-data-json", "official.json", "--official-image-dir", "official",
        "--output", "output",
    ])

    assert args.epochs == 2
    assert args.learning_rate == pytest.approx(2e-5)
    assert args.final_conv_learning_rate == pytest.approx(5e-6)
    assert args.penultimate_conv_learning_rate == pytest.approx(2e-6)
