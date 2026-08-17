from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
import tempfile
from pathlib import Path

import numpy as np
import paddle

from lane_training.action_training import load_manifest
from lane_training.dataset import LaneDataset
from lane_training.model import CnnModel


def sha256(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_deployment_metadata(
    *, checkpoint_sha256: str, difference: float, tested_batch_size: int,
    model_filename: str, training: dict | None = None,
    selection: dict | None = None,
) -> dict:
    return {
        "output_semantics": ["speed_demand", "kappa_action"],
        "control_profile": "kappa_action",
        "lane_control": {
            "profile": "kappa_action", "max_forward_speed": 0.30,
            "min_forward_speed": 0.12, "full_turn_curvature_m_inv": 5.0,
            "speed_curve_exponent": 1.5, "max_deceleration_mps2": 0.194,
            "max_acceleration_mps2": 0.129, "max_angular_speed": 1.50,
        },
        "input_size": [128, 128], "source_image_size": [320, 240],
        "model_filename": model_filename, "params_filename": "cnn_lane.pdiparams",
        "checkpoint_sha256": checkpoint_sha256,
        "dynamic_static_max_abs_difference": float(difference),
        "tested_batch_size": int(tested_batch_size),
        "training": training, "selection": selection,
        "selected": False, "vehicle_test_performed": False,
    }


def _model_path(checkpoint: Path) -> Path:
    return checkpoint / "model.pdparams" if checkpoint.is_dir() else checkpoint


def export_checkpoint(
    *, checkpoint: Path, manifest: Path, output: Path, device: str,
    training: dict | None = None, selection: dict | None = None,
) -> dict:
    archive = output.parent / f"{output.name}.tgz"
    if output.exists() or archive.exists():
        raise FileExistsError(f"export output already exists: {output}")
    paddle.set_device("gpu:0" if device == "gpu" else "cpu")
    all_rows = load_manifest(manifest)
    rows = [row for row in all_rows if row.get("split") != "train"][:32] or all_rows[:32]
    dataset = LaneDataset(rows, training=False, return_target_mask=True)
    inputs = paddle.to_tensor(np.stack([dataset[index][0] for index in range(len(dataset))]))
    model_path = _model_path(checkpoint)
    checkpoint_hash = sha256(model_path)
    with paddle.utils.unique_name.guard():
        model = CnnModel()
        state = paddle.load(str(model_path))
        if set(state) != set(model.state_dict()):
            raise ValueError("checkpoint keys do not match CnnModel")
        model.set_state_dict(state); model.eval()
        with paddle.no_grad(): dynamic = model(inputs).numpy()
        output.mkdir(parents=True)
        static = paddle.jit.to_static(
            model,
            input_spec=[paddle.static.InputSpec([None, 3, 128, 128], dtype="float32")],
            full_graph=True,
        )
        from paddle.jit.dy2static import utils as dy2static_utils

        original_get_temp_dir = dy2static_utils.get_temp_dir
        with tempfile.TemporaryDirectory(prefix="lane-paddle-static-") as temp_dir:
            dy2static_utils.get_temp_dir = lambda: temp_dir
            try:
                paddle.jit.save(static, str(output / "cnn_lane"))
            finally:
                dy2static_utils.get_temp_dir = original_get_temp_dir
        loaded = paddle.jit.load(str(output / "cnn_lane")); loaded.eval()
        with paddle.no_grad(): exported = loaded(inputs).numpy()
    difference = float(np.max(np.abs(dynamic - exported)))
    if not np.isfinite(difference) or difference > 1e-5:
        raise RuntimeError(f"dynamic/static mismatch: {difference}")
    model_filename = "cnn_lane.pdmodel" if (output / "cnn_lane.pdmodel").exists() else "cnn_lane.json"
    metadata = build_deployment_metadata(
        checkpoint_sha256=checkpoint_hash, difference=difference,
        tested_batch_size=len(rows), model_filename=model_filename,
        training=training, selection=selection,
    )
    (output / "deployment.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    with tarfile.open(archive, "w:gz") as stream:
        stream.add(output, arcname=output.name)
    metadata.update(archive=str(archive), archive_sha256=sha256(archive))
    return metadata


def export_shortlist(
    *, selection_report: Path, checkpoint_dir: Path, output_root: Path,
    manifest: Path, training_report: Path, device: str,
) -> list[dict]:
    if output_root.exists():
        raise FileExistsError(f"export root already exists: {output_root}")
    selection = json.loads(selection_report.read_text(encoding="utf-8"))
    training = json.loads(training_report.read_text(encoding="utf-8"))
    if sha256(manifest) != training["manifest_sha256"]:
        raise ValueError("manifest SHA256 does not match training report")
    output_root.mkdir(parents=True)
    history = {int(item["epoch"]): item for item in training.get("history", [])}
    results = []
    for candidate in selection["candidates"]:
        epoch = int(candidate["epoch"])
        checkpoint = checkpoint_dir / f"epoch_{epoch:04d}"
        state = json.loads((checkpoint / "state.json").read_text(encoding="utf-8"))
        if int(state["last_completed_epoch"]) != epoch:
            raise ValueError("checkpoint epoch does not match selection report")
        checkpoint_hash = sha256(checkpoint / "model.pdparams")
        if (candidate.get("checkpoint_sha256") != checkpoint_hash or
                history.get(epoch, {}).get("checkpoint_sha256") != checkpoint_hash):
            raise ValueError("checkpoint SHA256 does not match training/selection reports")
        roles = list(candidate["roles"])
        name = f"epoch_{epoch:04d}_{'_'.join(roles)}"
        results.append(export_checkpoint(
            checkpoint=checkpoint, manifest=manifest, output=output_root / name,
            device=device,
            training={
                "manifest_sha256": training["manifest_sha256"],
                "initial_checkpoint_sha256": training["initial_checkpoint_sha256"],
            },
            selection={"epoch": epoch, "roles": roles},
        ))
    return results


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint", type=Path)
    source.add_argument("--selection-report", type=Path)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--training-report", type=Path)
    parser.add_argument("--selection-role")
    parser.add_argument("--device", choices=("gpu", "cpu"), default="gpu")
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    if args.checkpoint:
        if args.output is None:
            raise ValueError("--output is required for a single checkpoint")
        result = export_checkpoint(
            checkpoint=args.checkpoint, manifest=args.manifest,
            output=args.output, device=args.device,
        )
    else:
        if not args.checkpoint_dir or not args.output_root or not args.training_report:
            raise ValueError("shortlist export requires checkpoint-dir, output-root, and training-report")
        result = export_shortlist(
            selection_report=args.selection_report, checkpoint_dir=args.checkpoint_dir,
            output_root=args.output_root, manifest=args.manifest,
            training_report=args.training_report, device=args.device,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
