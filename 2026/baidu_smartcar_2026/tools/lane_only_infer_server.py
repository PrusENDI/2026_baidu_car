"""Standalone 2026 lane-model inference service; deliberately avoids smartcar imports."""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def model_paths(root):
    model_dir = root / "smartcar" / "paddlebaidu" / "models" / "lane_model"
    return model_dir / "cnn_lane.pdmodel", model_dir / "cnn_lane.pdiparams"


class LaneOnlyInfer:
    def __init__(self, root=ROOT):
        model, params = model_paths(root)
        if not model.is_file() or not params.is_file():
            raise FileNotFoundError(f"lane model files not found: {model}, {params}")
        from paddle.inference import Config, create_predictor

        config = Config(str(model), str(params))
        config.enable_use_gpu(100, 0)
        config.switch_ir_optim()
        config.enable_memory_optim()
        self.predictor = create_predictor(config)

    @staticmethod
    def preprocess(image):
        import cv2
        import numpy as np

        image = cv2.resize(image, (128, 128)).astype(np.float32) / 127.5
        image -= np.array([1.0, 1.0, 1.0])
        image = image[:, :, ::-1].astype("float32").transpose((2, 0, 1))
        return image[np.newaxis, :]

    def __call__(self, image):
        data = self.preprocess(image)
        input_name = self.predictor.get_input_names()[0]
        self.predictor.get_input_handle(input_name).copy_from_cpu(data)
        self.predictor.run()
        output_name = self.predictor.get_output_names()[0]
        return self.predictor.get_output_handle(output_name).copy_to_cpu()[0].tolist()


def serve(port=5001):
    import cv2
    import numpy as np
    import zmq

    infer = LaneOnlyInfer()
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind(f"tcp://127.0.0.1:{port}")
    print(f"LANE_ONLY_READY port={port}", flush=True)
    try:
        while True:
            request = socket.recv()
            if request == b"ATATA":
                response = True
            elif request.startswith(b"image"):
                image = cv2.imdecode(np.frombuffer(request[5:], dtype=np.uint8), cv2.IMREAD_COLOR)
                response = infer(image) if image is not None else {"error": "invalid_image"}
            else:
                response = {"error": "unknown_request"}
            socket.send_string(json.dumps(response))
    finally:
        socket.close(0)
        context.term()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=5001)
    args = parser.parse_args()
    serve(args.port)


if __name__ == "__main__":
    main()
