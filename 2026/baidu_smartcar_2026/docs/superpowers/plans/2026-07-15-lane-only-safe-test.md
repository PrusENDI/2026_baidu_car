# Lane-only Safe Test Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Provide a lane-only inference service and a bounded, preflight-gated lane-following test that bypasses MOT/sklearn.

**Architecture:** `lane_only_infer_server.py` owns only Paddle lane prediction and a local ZMQ protocol. `safe_lane_test.py` owns process lifecycle, bounded readiness and output validation before it ever enters the existing motion loop.

**Tech Stack:** Python 3, Paddle Inference, OpenCV, NumPy, pyzmq, unittest.

---

### Task 1: Validate lane response shapes

**Files:**
- Modify: `tests/test_safe_lane_test.py`
- Modify: `tools/safe_lane_test.py`

- [ ] **Step 1: Write the failing test**

```python
def test_rejects_wrong_length_lane_output(self):
    self.assertEqual("invalid_model_output", safe.validate_output([0.1], 1.0))
    self.assertEqual("invalid_model_output", safe.validate_output([0.1, 0.2, 0.3], 1.0))
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m unittest tests/test_safe_lane_test.py -v`

Expected: FAIL because `validate_output` still accepts two scalar parameters rather than a response list.

- [ ] **Step 3: Write the minimal implementation**

```python
def validate_output(output, limit):
    if not isinstance(output, (list, tuple)) or len(output) != 2:
        return "invalid_model_output"
    error_y, error_angle = output
    if not all(isinstance(value, (int, float)) and math.isfinite(value)
               for value in output):
        return "invalid_model_output"
    if max(abs(error_y), abs(error_angle)) > limit:
        return "model_output_limit"
    return None
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `python -m unittest tests/test_safe_lane_test.py -v`

Expected: PASS.

### Task 2: Add the standalone lane inference service

**Files:**
- Create: `tools/lane_only_infer_server.py`
- Test: `tests/test_lane_only_infer_server.py`

- [ ] **Step 1: Write the failing test**

```python
def test_model_paths_select_2026_lane_weights(self):
    paths = server.model_paths(PROJECT_ROOT)
    self.assertEqual(paths[0].name, "cnn_lane.pdmodel")
    self.assertIn("smartcar/paddlebaidu/models/lane_model", str(paths[0]).replace("\\\\", "/"))
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m unittest tests/test_lane_only_infer_server.py -v`

Expected: FAIL because the module and `model_paths` do not exist.

- [ ] **Step 3: Write the minimal implementation**

```python
def model_paths(root):
    model_dir = root / "smartcar" / "paddlebaidu" / "models" / "lane_model"
    return model_dir / "cnn_lane.pdmodel", model_dir / "cnn_lane.pdiparams"
```

Implement a `LaneOnlyInfer` class with the preprocessing formerly used by `LaneInfer`, and a REP loop that replies to `ATATA` and `image` messages.  Imports must be only standard library, `cv2`, `numpy`, `zmq`, and `paddle.inference`.

- [ ] **Step 4: Run the test to verify it passes**

Run: `python -m unittest tests/test_lane_only_infer_server.py -v`

Expected: PASS.

### Task 3: Gate the safe test on a bounded lane preflight

**Files:**
- Modify: `tests/test_safe_lane_test.py`
- Modify: `tools/safe_lane_test.py`

- [ ] **Step 1: Write the failing test**

```python
def test_wait_ready_times_out_without_server(self):
    client = FakeClient([False, False])
    self.assertFalse(safe.wait_ready(client, timeout=0.0, sleep=lambda _: None))
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m unittest tests/test_safe_lane_test.py -v`

Expected: FAIL because `wait_ready` does not exist.

- [ ] **Step 3: Write the minimal implementation**

```python
def wait_ready(client, timeout, sleep=time.sleep):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if client.ready(timeout_ms=250):
            return True
        sleep(0.1)
    return False
```

Use a ZMQ client whose socket has `RCVTIMEO` and `SNDTIMEO`; start the standalone server, require readiness and one validated camera inference before the first `car.set_velocity` call, and terminate the child process in `finally`.

- [ ] **Step 4: Run the test to verify it passes**

Run: `python -m unittest tests/test_safe_lane_test.py -v`

Expected: PASS.

### Task 4: Verify on Orin without movement

**Files:**
- Modify: `docs/debug-daily/2026-07-14.md`

- [ ] **Step 1: Copy only the two tool files and tests to the Orin running copy**

Run: `scp tools/lane_only_infer_server.py tools/safe_lane_test.py jetson@192.168.0.155:/home/jetson/workspaces/baidu_car_2026_official_run_copy/tools/`

- [ ] **Step 2: Run a bounded service and camera/inference preflight**

Run: `python3 tools/safe_lane_test.py --preflight-only --startup-timeout 30`

Expected: It reports a two-number lane result and `SAFE_STOP`; there is no `set_velocity` call.

- [ ] **Step 3: Record the actual output and confirmation that no motion command was issued**

Append the command, output, model path, and non-motion confirmation to the daily debug log.
