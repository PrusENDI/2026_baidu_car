"""Export a risk-accepted lane checkpoint in the vehicle deployment format."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tarfile
import tempfile

import numpy as np
import paddle

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lane_training.dataset import LaneDataset
from lane_training.model import CnnModel
from tools.lane_training.new_track_finetune import load_rows


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def deployment_metadata(
    *, source_checkpoint: str, checkpoint_sha256: str, jit_differences: dict,
    inference_differences: dict, file_hashes: dict,
) -> dict:
    return {
        "source_checkpoint": source_checkpoint,
        "source_checkpoint_sha256": checkpoint_sha256,
        "vehicle_deployment_format": True,
        "input_shape": [None, 3, 128, 128],
        "output_shape": [None, 2],
        "tested_batch_sizes": [1, 32],
        "jit_max_abs_difference": jit_differences,
        "paddle_inference_max_abs_difference": inference_differences,
        "files_sha256": file_hashes,
        "user_risk_acceptance": True,
        "override_reason": "user explicitly requested export without D4 or absolute A/B gates",
        "behavior_gates_passed": False,
        "selected": False,
        "locked_test_performed": False,
    }


def _inference_output(model_file: Path, params_file: Path, inputs: np.ndarray, *, gpu: bool):
    config = paddle.inference.Config(str(model_file), str(params_file))
    if gpu:
        config.enable_use_gpu(256, 0)
    else:
        config.disable_gpu()
    config.switch_ir_optim(True)
    predictor = paddle.inference.create_predictor(config)
    input_handle = predictor.get_input_handle(predictor.get_input_names()[0])
    input_handle.reshape(inputs.shape)
    input_handle.copy_from_cpu(inputs)
    predictor.run()
    return predictor.get_output_handle(predictor.get_output_names()[0]).copy_to_cpu()


def export(args) -> dict:
    paddle.set_device(args.device)
    checkpoint_sha = sha256_file(args.checkpoint)
    if args.expected_sha256 and checkpoint_sha != args.expected_sha256.lower():
        raise ValueError("checkpoint SHA256 mismatch")
    rows = [
        row for row in load_rows(args.validation_data_json, args.validation_image_dir, "lap_002")
        if row["image_exists"]
    ][:32]
    if len(rows) != 32:
        raise ValueError("export validation requires 32 decodable frames")
    dataset = LaneDataset(rows, training=False)
    inputs32 = np.stack([dataset[index][0] for index in range(32)]).astype("float32")
    inputs1 = inputs32[:1]

    model = CnnModel()
    model.set_state_dict(paddle.load(str(args.checkpoint)))
    model.eval()
    with paddle.no_grad():
        dynamic1 = model(paddle.to_tensor(inputs1)).numpy()
        dynamic32 = model(paddle.to_tensor(inputs32)).numpy()

    if args.output.exists():
        raise FileExistsError(f"output already exists: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=str(args.output.parent)) as temporary:
        staging = Path(temporary) / "lane_model"
        staging.mkdir()
        static = paddle.jit.to_static(
            model,
            input_spec=[paddle.static.InputSpec([None, 3, 128, 128], dtype="float32")],
            full_graph=True,
        )
        paddle.jit.save(static, str(staging / "cnn_lane"))
        loaded = paddle.jit.load(str(staging / "cnn_lane"))
        loaded.eval()
        with paddle.no_grad():
            jit1 = loaded(paddle.to_tensor(inputs1)).numpy()
            jit32 = loaded(paddle.to_tensor(inputs32)).numpy()
        jit_differences = {
            "batch_1": float(np.max(np.abs(dynamic1 - jit1))),
            "batch_32": float(np.max(np.abs(dynamic32 - jit32))),
        }
        model_file = staging / "cnn_lane.pdmodel"
        params_file = staging / "cnn_lane.pdiparams"
        cpu1 = _inference_output(model_file, params_file, inputs1, gpu=False)
        gpu1 = _inference_output(model_file, params_file, inputs1, gpu=True)
        inference_differences = {
            "cpu_batch_1": float(np.max(np.abs(dynamic1 - cpu1))),
            "gpu_batch_1": float(np.max(np.abs(dynamic1 - gpu1))),
        }
        all_differences = [*jit_differences.values(), *inference_differences.values()]
        if not np.isfinite(all_differences).all() or max(all_differences) >= 1e-5:
            raise RuntimeError(f"static inference mismatch: {all_differences}")
        artifact_names = [
            "cnn_lane.pdmodel", "cnn_lane.pdiparams", "cnn_lane.pdiparams.info"
        ]
        file_hashes = {
            name: sha256_file(staging / name) for name in artifact_names
        }
        metadata = deployment_metadata(
            source_checkpoint=f"{args.checkpoint.parent.name}/{args.checkpoint.name}",
            checkpoint_sha256=checkpoint_sha,
            jit_differences=jit_differences,
            inference_differences=inference_differences,
            file_hashes=file_hashes,
        )
        (staging / "deployment.json").write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
        )
        shutil.copytree(staging, args.output)
    archive = args.output.parent / f"{args.output.name}_risk_accepted.tgz"
    if archive.exists():
        raise FileExistsError(f"archive already exists: {archive}")
    with tarfile.open(archive, "w:gz") as stream:
        stream.add(args.output, arcname=args.output.name)
    metadata["archive"] = str(archive)
    metadata["archive_sha256"] = sha256_file(archive)
    return metadata


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expected-sha256")
    parser.add_argument("--validation-data-json", type=Path, required=True)
    parser.add_argument("--validation-image-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="gpu")
    return parser.parse_args(argv)


if __name__ == "__main__":
    print(json.dumps(export(parse_args()), indent=2))
