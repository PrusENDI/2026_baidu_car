import json

import paddle
from PIL import Image

from lane_training.model import CnnModel
from tools.lane_training.evaluate_fixed_track_long import evaluate_checkpoint


def test_evaluator_keeps_manual_speed_masked_and_replays_cv_speed(tmp_path):
    image = tmp_path / "frame.jpg"
    Image.new("RGB", (320, 240), (90, 130, 180)).save(image)

    def manifest(name, *, source, split, mask, kappa):
        path = tmp_path / f"{name}.jsonl"
        row = {
            "image_path": str(image),
            "speed_demand": 0.5 if mask[0] else 0.0,
            "kappa_action": kappa,
            "target_mask": mask,
            "label_semantics": "speed_demand_action_curvature",
            "source": source,
            "split": split,
            "session_id": name,
            "effective_dt_s": 0.05,
        }
        path.write_text(json.dumps(row) + "\n", encoding="utf-8")
        return path

    sets = {
        "cross_train": (manifest(
            "cross", source="manual", split="train", mask=[0.0, 1.0], kappa=1.0,
        ), "train_manual"),
        "cv_validation": (manifest(
            "cv", source="cv", split="val", mask=[1.0, 1.0], kappa=0.2,
        ), "validation_cv"),
        "lap002": (manifest(
            "lap002", source="manual", split="validation", mask=[0.0, 1.0], kappa=0.5,
        ), "validation"),
        "official_train_reference": (manifest(
            "official_train", source="manual", split="validation", mask=[0.0, 1.0], kappa=0.5,
        ), "validation"),
        "official_eval": (manifest(
            "official_eval", source="manual", split="validation", mask=[0.0, 1.0], kappa=0.5,
        ), "validation"),
    }
    checkpoint = tmp_path / "model.pdparams"
    paddle.save(CnnModel().state_dict(), str(checkpoint))

    report = evaluate_checkpoint(checkpoint, sets, device="cpu", batch_size=1)

    assert set(report["sets"]) == set(sets)
    assert report["sets"]["cross_train"]["sequence"]["active_count"] == 1
    assert report["sets"]["cross_train"]["speed_mae"] is None
    assert "vehicle_replay" in report["sets"]["cv_validation"]
    assert "vehicle_replay" not in report["sets"]["lap002"]
