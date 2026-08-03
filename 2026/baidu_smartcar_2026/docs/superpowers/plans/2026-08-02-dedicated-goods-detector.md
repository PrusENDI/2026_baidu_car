# Dedicated Goods Detector Implementation Plan

> **For agentic workers:** Execute inline in the current user-selected worktree. The user explicitly requested direct implementation without TDD red/green cycles.

**Goal:** Use the `submission-8-2` Paddle model only for vegetable detection in `get_order()` while retaining `task2026` for every other detection.

**Architecture:** Add an independent `goods` inference service and client. Thread an optional detector through the existing normalized detection and alignment APIs, defaulting to `task_det`, and select `goods_det` only inside `find_goods()`.

**Tech Stack:** Python, Paddle Inference, PaddleDetection deployment wrapper, OpenCV, ZMQ, YAML.

---

### Task 1: Install and configure the goods model

**Files:**
- Copy: `8_2/submission-8-2/model/*`
- Create: `smartcar/paddlebaidu/models/goods_8_2/*`
- Modify: `smartcar/paddlebaidu/models/goods_8_2/infer_cfg.yml`
- Modify: `config_car.yml`

- [ ] Copy the exported model without modifying the user's submission copy.
- [ ] Change the copied deployment preprocessing to RGB ImageNet mean/std normalization, matching `predict.py`.
- [ ] Add a `goods` `YoloeInfer` instance on port `5005`, with a `640x640` client image size.

### Task 2: Add detector selection to the car wrapper

**Files:**
- Modify: `car_wrap_2026.py`

- [ ] Initialize `self.goods_det = ClintInterface("goods")` next to `task_det`.
- [ ] Add an optional `detector` argument to `get_detection_results()` and default it to `task_det`.
- [ ] Add an optional `detector` argument to `move_to_detection_target()` and forward it on every detection iteration.
- [ ] Document why the default preserves existing callers.

### Task 3: Use the goods detector only for vegetable search

**Files:**
- Modify: `car_task_function.py`

- [ ] Pass `my_car.goods_det` on every `find_goods()` alignment attempt.
- [ ] Keep order-label detection and unrelated task detection on the default `task_det`.
- [ ] Comment the model boundary at the search function.

### Task 4: Verify integration

**Files:**
- Create: `tests/test_dedicated_goods_detector.py`

- [ ] Add static integration assertions for configuration, preprocessing, labels, initialization, detector forwarding, and goods-only selection.
- [ ] Run the focused test module.
- [ ] Compile the modified Python files.
- [ ] Run the existing non-hardware test suite and report any environment-only failures separately.
- [ ] Review the final diff without staging or modifying unrelated user changes.
