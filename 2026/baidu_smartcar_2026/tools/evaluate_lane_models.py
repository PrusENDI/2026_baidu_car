"""Offline, side-by-side evaluation for the old and new lane-following models."""

import argparse
import csv
import glob
import json
import math
import time
from pathlib import Path


def load_dataset(dataset_dir):
    dataset_dir = Path(dataset_dir)
    with (dataset_dir / "data.json").open(encoding="utf-8-sig") as handle:
        records = json.load(handle)
    samples = []
    missing = []
    for record in records:
        image_path = dataset_dir / record["img_path"]
        if image_path.is_file():
            samples.append(record)
        else:
            missing.append(record["img_path"])
    return samples, missing


class LanePredictor:
    def __init__(self, model_dir):
        import paddle.inference as paddle_infer

        self.cv2 = __import__("cv2")
        self.np = __import__("numpy")
        model_dir = Path(model_dir)
        model_path = glob.glob(str(model_dir / "*.pdmodel"))[0]
        params_path = glob.glob(str(model_dir / "*.pdiparams"))[0]
        config = paddle_infer.Config(model_path, params_path)
        config.disable_gpu()
        config.switch_ir_optim()
        config.enable_memory_optim()
        self.predictor = paddle_infer.create_predictor(config)

    def predict(self, image):
        image = self.cv2.resize(image, (128, 128)).astype("float32")
        image = image / 127.5 - 1.0
        image = image[:, :, ::-1].transpose((2, 0, 1))[self.np.newaxis, :]
        input_name = self.predictor.get_input_names()[0]
        self.predictor.get_input_handle(input_name).copy_from_cpu(image)
        self.predictor.run()
        output_name = self.predictor.get_output_names()[0]
        return self.predictor.get_output_handle(output_name).copy_to_cpu()[0].tolist()


def pearson(values_x, values_y):
    if len(values_x) < 2:
        return None
    mean_x = sum(values_x) / len(values_x)
    mean_y = sum(values_y) / len(values_y)
    numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(values_x, values_y))
    denominator_x = math.sqrt(sum((x - mean_x) ** 2 for x in values_x))
    denominator_y = math.sqrt(sum((y - mean_y) ** 2 for y in values_y))
    return None if denominator_x == 0 or denominator_y == 0 else numerator / (denominator_x * denominator_y)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--old-model", type=Path, required=True)
    parser.add_argument("--new-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    samples, missing = load_dataset(args.dataset)
    if not samples:
        raise SystemExit("No usable image/label pairs in dataset")
    args.output.mkdir(parents=True, exist_ok=True)
    models = {"old": LanePredictor(args.old_model), "new": LanePredictor(args.new_model)}
    rows = []
    for index, sample in enumerate(samples, start=1):
        image = models["old"].cv2.imread(str(args.dataset / sample["img_path"]))
        row = {"img_path": sample["img_path"], "forward": sample["state"][0], "lateral": sample["state"][1], "angular": sample["state"][2]}
        for name, model in models.items():
            started = time.perf_counter()
            output = model.predict(image)
            row[f"{name}_error_y"] = output[0]
            row[f"{name}_error_angle"] = output[1]
            row[f"{name}_latency_ms"] = (time.perf_counter() - started) * 1000
        rows.append(row)
        if index % 200 == 0:
            print(f"processed {index}/{len(samples)}")
    fieldnames = list(rows[0])
    with (args.output / "per_frame.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    summary = {"samples": len(rows), "missing_image_labels": missing}
    for name in models:
        y_values = [row[f"{name}_error_y"] for row in rows]
        angle_values = [row[f"{name}_error_angle"] for row in rows]
        latency_values = [row[f"{name}_latency_ms"] for row in rows]
        summary[name] = {
            "latency_ms_mean": sum(latency_values) / len(latency_values),
            "error_y_range": [min(y_values), max(y_values)],
            "error_angle_range": [min(angle_values), max(angle_values)],
            "corr_error_y_to_lateral": pearson(y_values, [row["lateral"] for row in rows]),
            "corr_error_angle_to_angular": pearson(angle_values, [row["angular"] for row in rows]),
        }
    with (args.output / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
