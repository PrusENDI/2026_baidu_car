import pytest

from tools.lane_training.official_lap3_multiepoch_finetune import (
    build_epoch_schedule,
    parse_args,
)


def test_epoch_schedule_uses_every_lap3_batch_once_without_replay():
    batches = [[{"id": index}] for index in range(7)]

    schedule = build_epoch_schedule(batches, seed=9)

    assert len(schedule) == 7
    assert sorted(batch[0]["id"] for batch in schedule) == list(range(7))


def test_multiepoch_cli_defaults_to_five_epochs_and_low_learning_rates():
    args = parse_args([
        "--checkpoint", "official.pdparams",
        "--lap3-data-json", "lap3.json",
        "--lap3-image-dir", "lap3",
        "--output", "output",
    ])

    assert args.epochs == 5
    assert args.learning_rate == pytest.approx(1e-5)
    assert args.convolution_learning_rates == pytest.approx((3e-7, 6e-7, 1.5e-6, 3e-6))
