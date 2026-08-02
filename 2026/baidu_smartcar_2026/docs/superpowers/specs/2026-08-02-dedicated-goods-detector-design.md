# Dedicated Goods Detector Design

## Goal

Use the Paddle detection model in `8_2/submission-8-2/model` for vegetable
detection during `get_order()` without replacing the existing `task2026`
detector or disrupting order-label and other task detections.

## Architecture

The inference backend will expose a new `goods` instance on its own ZMQ port.
`Car.paddle_infer_init()` will create a matching `goods_det` client alongside
`task_det`. Detection and alignment methods will accept an optional detector;
their default remains `task_det`, preserving every existing caller.
`find_goods()` alone will request the `goods` detector.

The model artifacts will live under
`smartcar/paddlebaidu/models/goods_8_2`. Its deployment configuration will use
the label list shipped with the submission and the same RGB, 640-by-640,
ImageNet mean/std preprocessing implemented by the submission's `predict.py`.

## Data Flow

1. `get_order()` continues to locate order labels with `task_det` and parse the
   cropped order image with the multimodal service.
2. After the requested vegetable name is mapped to an `h_*` label,
   `find_goods()` calls `move_to_detection_target(..., detector=goods_det)`.
3. `move_to_detection_target()` repeatedly obtains normalized detections from
   the selected detector, filters by the requested label, and performs the
   existing vehicle/arm alignment loop.
4. The selected label is returned to the existing pickup safety checks. A
   timeout still returns `(None, None)`, so `get_order()` continues to prohibit
   blind pickup.

## Compatibility and Failure Handling

- The detector argument is optional and defaults to `task_det`; existing
  motion and detection behavior is unchanged.
- A new backend port must be unique. Port `5005` will be used because the
  current configuration uses `5001` through `5004`.
- The goods client is initialized with the other inference clients, so missing
  model files or an unavailable backend fail during startup instead of during
  pickup.
- The model's labels already include all nine `goods_dict` values. Non-goods
  labels in the model are harmless because `find_goods()` supplies an exact
  label filter.

## Testing

Unit tests will load the relevant source modules with hardware and Paddle
dependencies stubbed where necessary. They will prove that:

- `get_detection_results()` uses `task_det` by default and a supplied detector
  when requested.
- `move_to_detection_target()` passes its detector selection through to every
  detection attempt.
- `find_goods()` selects `my_car.goods_det` for all retries.
- configuration declares the `goods` model with port `5005` and input size
  `640x640`.
- the copied model deployment configuration matches the preprocessing in the
  submission script and retains the expected vegetable labels.

Full tests and Python syntax compilation will be run after implementation.
Hardware behavior and real GPU inference require final verification on the
Orin with representative side-camera images.
