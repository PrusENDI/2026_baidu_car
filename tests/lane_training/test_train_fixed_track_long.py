import json

import paddle
import pytest
from PIL import Image

from lane_training.model import CnnModel
from tools.lane_training.train_fixed_track_long import (
    audit_training_rows,
    parse_args,
    train,
)


def test_long_cli_defaults_and_count_audit():
    args = parse_args([
        "--manifest", "joint.jsonl",
        "--initial-checkpoint", "d4.pdparams",
        "--output", "long_v1",
    ])
    assert (args.epochs, args.batch_size, args.kappa_weight) == (200, 64, 2.0)
    assert (args.checkpoint_interval, args.seed) == (5, 20260817)
    with pytest.raises(ValueError, match="expected training group counts"):
        audit_training_rows(
            {"cv": [0, 1], "manual_active": [2], "manual_context": [3]},
            expected={"cv": 2822, "manual_active": 74, "manual_context": 572},
        )


def test_one_epoch_cpu_training_writes_complete_report_and_checkpoint(tmp_path):
    rows = []
    labels = [
        ("cv", [1.0, 1.0], 0.0),
        ("cv", [1.0, 1.0], 0.2),
        ("manual", [0.0, 1.0], 0.5),
        ("manual", [0.0, 1.0], 0.0),
    ]
    for index, (source, mask, kappa) in enumerate(labels):
        image = tmp_path / f"frame_{index}.jpg"
        Image.new("RGB", (320, 240), (80 + index, 120, 160)).save(image)
        rows.append({
            "image_path": str(image),
            "speed_demand": 0.5 if source == "cv" else 0.0,
            "kappa_action": kappa,
            "target_mask": mask,
            "label_semantics": "speed_demand_action_curvature",
            "source": source,
            "split": "train",
        })
    manifest = tmp_path / "joint.jsonl"
    manifest.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8",
    )
    initial = tmp_path / "d4.pdparams"
    paddle.save(CnnModel().state_dict(), str(initial))
    output = tmp_path / "output"

    train(
        manifest=manifest,
        initial_checkpoint=initial,
        output=output,
        device="cpu",
        max_epochs=1,
        batch_size=4,
        expected_cv=2,
        expected_manual_active=1,
        expected_manual_context=1,
    )

    assert (output / "checkpoints/epoch_0001/model.pdparams").is_file()
    assert (output / "checkpoints/epoch_0001/optimizer.pdopt").is_file()
    report = json.loads((output / "training_report.json").read_text())
    assert report["data_counts"] == {
        "cv": 2, "manual_active": 1, "manual_context": 1,
    }
    assert report["history"][0]["stage"] == "head"
    assert report["history"][0]["valid_speed"] > 0
    assert report["history"][0]["valid_kappa"] > 0

    train(
        manifest=manifest,
        resume=output / "checkpoints/epoch_0001",
        output=output,
        device="cpu",
        epochs=2,
        batch_size=4,
        expected_cv=2,
        expected_manual_active=1,
        expected_manual_context=1,
    )
    resumed = json.loads((output / "training_report.json").read_text())
    assert [item["epoch"] for item in resumed["history"]] == [1, 2]
