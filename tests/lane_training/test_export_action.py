import json

import paddle
from PIL import Image

from lane_training.model import CnnModel
from tools.lane_training.export_action import (
    build_deployment_metadata,
    export_shortlist,
    sha256,
)


def test_export_metadata_includes_long_training_provenance():
    metadata = build_deployment_metadata(
        checkpoint_sha256="abc", difference=1e-7, tested_batch_size=8,
        model_filename="cnn_lane.json",
        training={"manifest_sha256": "def", "initial_checkpoint_sha256": "ghi"},
        selection={"epoch": 100, "roles": ["crossroad_best", "balanced"]},
    )
    assert metadata["output_semantics"] == ["speed_demand", "kappa_action"]
    assert metadata["training"]["manifest_sha256"] == "def"
    assert metadata["selection"]["epoch"] == 100
    assert metadata["selected"] is False
    assert metadata["vehicle_test_performed"] is False


def test_single_shortlist_candidate_exports_with_verified_provenance(tmp_path):
    image = tmp_path / "frame.jpg"
    Image.new("RGB", (320, 240), (90, 130, 180)).save(image)
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(json.dumps({
        "image_path": str(image), "speed_demand": 0.5, "kappa_action": 0.2,
        "target_mask": [1.0, 1.0], "split": "validation", "source": "cv",
    }) + "\n", encoding="utf-8")
    checkpoint = tmp_path / "checkpoints/epoch_0005"
    checkpoint.mkdir(parents=True)
    paddle.save(CnnModel().state_dict(), str(checkpoint / "model.pdparams"))
    checkpoint_hash = sha256(checkpoint / "model.pdparams")
    (checkpoint / "state.json").write_text(json.dumps({
        "last_completed_epoch": 5,
    }), encoding="utf-8")
    training_report = tmp_path / "training_report.json"
    training_report.write_text(json.dumps({
        "manifest_sha256": sha256(manifest),
        "initial_checkpoint_sha256": "d4",
        "history": [{"epoch": 5, "checkpoint_sha256": checkpoint_hash}],
    }), encoding="utf-8")
    selection_report = tmp_path / "selection.json"
    selection_report.write_text(json.dumps({
        "candidates": [{
            "epoch": 5, "checkpoint_sha256": checkpoint_hash,
            "roles": ["crossroad_best", "balanced"],
        }],
    }), encoding="utf-8")

    results = export_shortlist(
        selection_report=selection_report,
        checkpoint_dir=tmp_path / "checkpoints",
        output_root=tmp_path / "exports",
        manifest=manifest,
        training_report=training_report,
        device="cpu",
    )

    assert len(results) == 1
    deployment = json.loads(next((tmp_path / "exports").glob("*/deployment.json")).read_text())
    assert deployment["checkpoint_sha256"] == checkpoint_hash
    assert deployment["selection"]["roles"] == ["crossroad_best", "balanced"]
    assert len(list((tmp_path / "exports").glob("*.tgz"))) == 1
