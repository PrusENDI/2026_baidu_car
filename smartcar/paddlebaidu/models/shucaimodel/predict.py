"""
Inference entry script for competition submission.
Usage:
    python predict.py data.txt result.json
"""
import os
import sys
import json
import yaml
import math
import cv2
import numpy as np
import paddle

# Add PaddleDetection helpers to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'PaddleDetection', 'deploy', 'python'))


OFFICIAL_TYPE_BY_LABEL = {
    'animal': 1,
    'ball_blue': 2,
    'ball_yellow': 3,
    'cylinder_1': 4,
    'cylinder_2': 5,
    'cylinder_3': 6,
    'cylinder_set': 7,
    'h_dou_jiao': 8,
    'h_fan_qie': 9,
    'h_jin_zhen_gu': 10,
    'h_mo_gu': 11,
    'h_qin_cai': 12,
    'h_qing_jiao': 13,
    'h_tu_dou': 14,
    'h_xi_lan_hua': 15,
    'h_you_cai': 16,
    'water': 17,
    'water_l1': 18,
    'water_l2': 19,
    'water_l3': 20,
}

DEFAULT_SCORE_THRESH = 0.50
CLASS_SCORE_THRESH = {
    1: 0.50,  # animal
    2: 0.50,  # ball_blue
    3: 0.50,  # ball_yellow
    4: 0.50,  # cylinder_1
    5: 0.50,  # cylinder_2
    6: 0.50,  # cylinder_3
    7: 0.50,  # cylinder_set
    8: 0.50,  # h_dou_jiao
    9: 0.50,  # h_fan_qie
    10: 0.50,  # h_jin_zhen_gu
    11: 0.50,  # h_mo_gu
    12: 0.50,  # h_qin_cai
    13: 0.50,  # h_qing_jiao
    14: 0.50,  # h_tu_dou
    15: 0.50,  # h_xi_lan_hua
    16: 0.50,  # h_you_cai
    17: 0.50,  # water
    18: 0.50,  # water_l1
    19: 0.50,  # water_l2
    20: 0.50,  # water_l3
}

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
TARGET_H = 640
TARGET_W = 640


def official_type_from_model_class(cls_id, labels):
    cls_idx = int(cls_id)
    if cls_idx < 0 or cls_idx >= len(labels):
        return None
    label = labels[cls_idx]
    if label == 'yellow_tower':
        return None
    return OFFICIAL_TYPE_BY_LABEL.get(label)


def score_threshold_for_type(type_id, default_threshold):
    return CLASS_SCORE_THRESH.get(type_id, default_threshold)


def get_test_images(data_txt):
    data_txt = os.path.abspath(data_txt)
    infer_root = os.path.dirname(data_txt)
    images = []
    with open(data_txt, 'r', encoding='utf-8') as f:
        for line in f:
            image_path = line.strip().lstrip('\ufeff')
            if not image_path:
                continue
            image_path = image_path.replace('\\', '/')
            if not os.path.isabs(image_path):
                image_path = os.path.join(infer_root, image_path)
            images.append(image_path)
    assert len(images) > 0, f"no image found in {data_txt}"
    return images


def load_predictor(model_dir, device='GPU'):
    infer_model = os.path.join(model_dir, 'model.pdmodel')
    infer_params = os.path.join(model_dir, 'model.pdiparams')
    deploy_file = os.path.join(model_dir, 'infer_cfg.yml')

    with open(deploy_file) as f:
        yml_conf = yaml.safe_load(f)

    def make_config(use_trt=False, fp16=False):
        config = paddle.inference.Config(infer_model, infer_params)
        config.disable_glog_info()
        config.enable_memory_optim()
        config.switch_use_feed_fetch_ops(False)
        return config

    def enable_trt(config, fp16=False):
        precision = paddle.inference.PrecisionType.Half if fp16 else paddle.inference.PrecisionType.Float32
        config.enable_tensorrt_engine(
            workspace_size=1 << 30,
            max_batch_size=8,
            min_subgraph_size=3,
            precision_mode=precision,
            use_static=False,
            use_calib_mode=False,
        )
        try:
            config.set_tensorrt_optimization_level(3)
        except Exception:
            pass

    if device == 'GPU':
        trt_mode = os.environ.get('USE_TRT', '0')
        gpu_trials = []
        if trt_mode != '0':
            gpu_trials.extend([('TensorRT FP16', True, True), ('TensorRT FP32', True, False)])
        gpu_trials.append(('GPU', False, False))

        for name, use_trt, fp16 in gpu_trials:
            try:
                config = make_config()
                config.enable_use_gpu(2000, 0)
                config.switch_ir_optim(use_trt)
                if use_trt:
                    enable_trt(config, fp16=fp16)
                predictor = paddle.inference.create_predictor(config)
                return predictor, yml_conf
            except Exception as e:
                print(f"{name} init failed ({e})")

        print("All GPU init attempts failed, falling back to CPU")

    config = make_config()
    
    config.disable_gpu()
    config.set_cpu_math_library_num_threads(4)
    predictor = paddle.inference.create_predictor(config)
    print("Using CPU inference")
    return predictor, yml_conf


def read_image(im_path):
    im = cv2.imread(im_path, cv2.IMREAD_COLOR)
    if im is None:
        data = np.fromfile(im_path, dtype=np.uint8)
        im = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if im is None:
        raise ValueError(f"failed to read image: {im_path}")
    return im


def preprocess_images(image_list):
    batch_size = len(image_list)
    images = np.empty((batch_size, 3, TARGET_H, TARGET_W), dtype=np.float32)
    im_shape = np.empty((batch_size, 2), dtype=np.float32)
    scale_factor = np.empty((batch_size, 2), dtype=np.float32)
    for idx, im_path in enumerate(image_list):
        im = read_image(im_path)
        orig_h, orig_w = im.shape[:2]
        im = cv2.resize(im, (TARGET_W, TARGET_H), interpolation=cv2.INTER_LINEAR)
        im = im[:, :, ::-1].astype(np.float32, copy=False)
        im *= 1.0 / 255.0
        im -= MEAN
        im /= STD
        images[idx] = im.transpose((2, 0, 1))
        im_shape[idx] = (TARGET_H, TARGET_W)
        scale_factor[idx] = (TARGET_H / float(orig_h), TARGET_W / float(orig_w))
    return {'image': images, 'im_shape': im_shape, 'scale_factor': scale_factor}


def predict_batch(predictor, inputs):
    predictor.run()
    output_names = predictor.get_output_names()
    boxes = predictor.get_output_handle(output_names[0]).copy_to_cpu()
    if len(output_names) == 1:
        boxes_num = np.array([len(boxes)])
    else:
        boxes_num = predictor.get_output_handle(output_names[1]).copy_to_cpu()
    return {'boxes': boxes, 'boxes_num': boxes_num}


def main():
    import argparse as _argparse
    parser = _argparse.ArgumentParser()
    parser.add_argument("data_txt", type=str)
    parser.add_argument("result_json", type=str)
    parser.add_argument("--device", type=str, default="GPU")
    parser.add_argument("--batch_size", type=int, default=3)
    parser.add_argument("--threshold", type=float, default=DEFAULT_SCORE_THRESH)
    parser.add_argument("--model_dir", type=str, default=None)
    FLAGS = parser.parse_args()
    FLAGS.device = FLAGS.device.upper()

    model_dir = FLAGS.model_dir or os.path.join(os.path.dirname(__file__), 'model')
    predictor, yml_conf = load_predictor(model_dir, FLAGS.device)

    labels = yml_conf['label_list']

    img_list = get_test_images(FLAGS.data_txt)
    batch_size = max(1, FLAGS.batch_size)

    save_results = {"result": []}
    for i in range(math.ceil(len(img_list) / batch_size)):
        image_offset = i * batch_size
        batch = img_list[i * batch_size:(i + 1) * batch_size]
        inputs = preprocess_images(batch)
        input_names = predictor.get_input_names()
        for name in input_names:
            tensor = predictor.get_input_handle(name)
            tensor.copy_from_cpu(inputs[name])

        result = predict_batch(predictor, inputs)
        boxes = result['boxes']
        boxes_num = result['boxes_num']
        start_idx = 0
        for img_idx in range(len(boxes_num)):
            num = boxes_num[img_idx]
            global_img_idx = image_offset + img_idx
            if num == 0:
                start_idx += int(num)
                continue
            file_name = os.path.basename(img_list[global_img_idx])
            image_id = os.path.splitext(file_name)[0]
            for box in boxes[start_idx:start_idx + int(num)]:
                cls_id, score, x_min, y_min, x_max, y_max = box.tolist()
                official_type = official_type_from_model_class(cls_id, labels)
                if official_type is None:
                    continue
                if float(score) < score_threshold_for_type(official_type, FLAGS.threshold):
                    continue
                save_results["result"].append({
                    'image_id': image_id,
                    'type': official_type,
                    'x': float(x_min),
                    'y': float(y_min),
                    'width': float(x_max - x_min),
                    'height': float(y_max - y_min),
                    'segmentation': []
                })
            start_idx += int(num)

    result_dir = os.path.dirname(os.path.abspath(FLAGS.result_json))
    if result_dir:
        os.makedirs(result_dir, exist_ok=True)
    with open(FLAGS.result_json, 'w', encoding='utf-8') as f:
        json.dump(save_results, f, ensure_ascii=False, separators=(',', ':'))


if __name__ == '__main__':
    paddle.enable_static()
    try:
        main()
    except Exception as e:
        print(f"FATAL ERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)
