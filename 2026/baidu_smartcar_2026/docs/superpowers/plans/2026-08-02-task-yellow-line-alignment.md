# Task-Point Yellow-Line Alignment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an opt-in, HSV-based front-camera function that aligns a stopped vehicle to a configured physical distance and heading relative to the same-side yellow line, without changing existing lane-following behavior.

**Architecture:** Keep `LaneInfer` and every `lane_dis_offset()` call unchanged. Add a hardware-independent yellow-line pose estimator, a separately testable stationary alignment controller, offline camera/ground calibration utilities, and a thin `MyCar.align_to_yellow_edge()` adapter. The first task integration is disabled by default and runs only after the existing lane movement has stopped.

**Tech Stack:** Python 3, OpenCV, NumPy, PyYAML, the existing project `PID`, `unittest`, synthetic-image tests, existing mecanum `set_velocity(vx, vy, wz)`.

---

## File structure

- Create `smartcar/whalesbot/tools/yellow_line_pose.py`: immutable result/config types, HSV mask extraction, ground-plane mapping, robust line fit, and confidence calculation. It must not import vehicle or serial modules.
- Create `smartcar/whalesbot/tools/yellow_line_aligner.py`: stopped-vehicle acquisition/control/settling state machine with injected frame, velocity, clock, sleep, and stop dependencies.
- Create `tools/calibrate_yellow_line.py`: offline chessboard intrinsics and image-to-ground homography CLI; it never commands hardware or edits `config_car.yml` automatically.
- Create `tools/yellow_line_read_only.py`: camera-only measurement/overlay/CSV tool; it never imports or initializes the chassis.
- Modify `car_wrap_2026.py`: lazy construction and thin public alignment adapter only; do not modify `lane_base()`, `lane_dis()`, or `lane_dis_offset()`.
- Modify `config_car.yml`: disabled-by-default yellow-line calibration, HSV, confidence, PID, and task-offset configuration.
- Modify `car_task_function.py`: add one disabled-by-default `auto_seeding` alignment call after its existing entry lane movement has stopped; retain the existing 2 cm compensation until A/B evidence supports removing it.
- Create `tests/test_yellow_line_pose.py`: synthetic perception and validation tests.
- Create `tests/test_yellow_line_aligner.py`: deterministic state-machine and stop-safety tests.
- Create `tests/test_calibrate_yellow_line.py`: calibration math and CLI payload tests.
- Create `tests/test_yellow_line_car_integration.py`: AST/config regression tests proving current lane interfaces remain unchanged.
- Create `tests/test_yellow_line_read_only.py`: read-only CLI and CSV record tests without opening a real camera.

## Task 1: Define yellow-line data contracts and configuration validation

**Files:**
- Create: `smartcar/whalesbot/tools/yellow_line_pose.py`
- Create: `tests/test_yellow_line_pose.py`

- [ ] **Step 1: Write failing tests for valid, invalid, and disabled configuration**

```python
# tests/test_yellow_line_pose.py
import unittest
import numpy as np

from smartcar.whalesbot.tools.yellow_line_pose import (
    YellowLineConfig,
    YellowLinePose,
)


def valid_mapping():
    return {
        "enabled": True,
        "side": "right",
        "frame_size": [640, 480],
        "calibration": {
            "camera_matrix": [[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]],
            "distortion": [0.0, 0.0, 0.0, 0.0, 0.0],
            "image_to_ground_homography": [[0.001, 0.0, -0.32], [0.0, -0.001, 0.48], [0.0, 0.0, 1.0]],
        },
        "hsv": {
            "roi": [[0, 240], [640, 240], [640, 480], [0, 480]],
            "lower": [15, 80, 80],
            "upper": [40, 255, 255],
            "open_kernel": 3,
            "close_kernel": 5,
        },
        "confidence": {
            "valid_threshold": 0.65,
            "acquire_frames": 5,
            "lost_frames": 3,
        },
    }


class YellowLineConfigTest(unittest.TestCase):
    def test_enabled_config_requires_metric_calibration(self):
        mapping = valid_mapping()
        mapping["calibration"]["image_to_ground_homography"] = []
        with self.assertRaisesRegex(ValueError, "homography"):
            YellowLineConfig.from_mapping(mapping)

    def test_valid_config_has_expected_numpy_shapes(self):
        config = YellowLineConfig.from_mapping(valid_mapping())
        self.assertEqual((640, 480), config.frame_size)
        self.assertEqual((3, 3), config.camera_matrix.shape)
        self.assertEqual((3, 3), config.homography.shape)
        self.assertEqual(np.uint8, config.hsv_lower.dtype)

    def test_invalid_pose_never_exposes_numeric_measurement(self):
        pose = YellowLinePose.invalid("no_line", timestamp=12.5)
        self.assertFalse(pose.valid)
        self.assertIsNone(pose.distance_m)
        self.assertIsNone(pose.heading_rad)
        self.assertEqual("no_line", pose.reason)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the tests and confirm the module is missing**

Run:

```powershell
python -m unittest tests.test_yellow_line_pose -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'smartcar.whalesbot.tools.yellow_line_pose'`.

- [ ] **Step 3: Add complete immutable contracts and strict enabled-config parsing**

```python
# smartcar/whalesbot/tools/yellow_line_pose.py
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class YellowLinePose:
    valid: bool
    distance_m: Optional[float]
    heading_rad: Optional[float]
    confidence: float
    reason: str
    timestamp: float
    diagnostics: Mapping[str, float] = field(default_factory=dict)

    @classmethod
    def invalid(cls, reason: str, timestamp: float, diagnostics=None):
        return cls(False, None, None, 0.0, reason, float(timestamp), diagnostics or {})


def _matrix(value, shape, name):
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f"yellow_line {name} must have shape {shape} and finite values")
    return array


@dataclass(frozen=True)
class YellowLineConfig:
    enabled: bool
    side: str
    frame_size: Tuple[int, int]
    camera_matrix: np.ndarray
    distortion: np.ndarray
    homography: np.ndarray
    roi: np.ndarray
    hsv_lower: np.ndarray
    hsv_upper: np.ndarray
    open_kernel: int
    close_kernel: int
    valid_threshold: float
    acquire_frames: int
    lost_frames: int

    @classmethod
    def from_mapping(cls, source: Mapping[str, Any]):
        if not bool(source.get("enabled", False)):
            raise ValueError("yellow_line is disabled")
        side = str(source.get("side", ""))
        if side not in {"left", "right"}:
            raise ValueError("yellow_line side must be left or right")
        frame_size = tuple(int(v) for v in source.get("frame_size", ()))
        if len(frame_size) != 2 or min(frame_size) <= 0:
            raise ValueError("yellow_line frame_size must contain positive width and height")
        calibration = source.get("calibration", {})
        camera_matrix = _matrix(calibration.get("camera_matrix", []), (3, 3), "camera_matrix")
        distortion = np.asarray(calibration.get("distortion", []), dtype=np.float64).reshape(-1)
        if distortion.size not in {4, 5, 8, 12, 14} or not np.isfinite(distortion).all():
            raise ValueError("yellow_line distortion must contain valid OpenCV coefficients")
        homography = _matrix(
            calibration.get("image_to_ground_homography", []),
            (3, 3),
            "homography",
        )
        hsv = source.get("hsv", {})
        roi = np.asarray(hsv.get("roi", []), dtype=np.int32)
        if roi.ndim != 2 or roi.shape[0] < 3 or roi.shape[1] != 2:
            raise ValueError("yellow_line hsv roi must contain at least three image points")
        lower = np.asarray(hsv.get("lower", []), dtype=np.uint8)
        upper = np.asarray(hsv.get("upper", []), dtype=np.uint8)
        if lower.shape != (3,) or upper.shape != (3,) or np.any(lower > upper):
            raise ValueError("yellow_line hsv bounds must be ordered HSV triplets")
        confidence = source.get("confidence", {})
        threshold = float(confidence.get("valid_threshold", 0.65))
        acquire_frames = int(confidence.get("acquire_frames", 5))
        lost_frames = int(confidence.get("lost_frames", 3))
        if not 0.0 < threshold <= 1.0 or acquire_frames < 1 or lost_frames < 1:
            raise ValueError("yellow_line confidence settings are invalid")
        open_kernel = int(hsv.get("open_kernel", 3))
        close_kernel = int(hsv.get("close_kernel", 5))
        if min(open_kernel, close_kernel) < 1:
            raise ValueError("yellow_line morphology kernels must be positive")
        return cls(
            True, side, frame_size, camera_matrix, distortion, homography, roi,
            lower, upper, open_kernel, close_kernel, threshold,
            acquire_frames, lost_frames,
        )
```

- [ ] **Step 4: Run the focused tests**

Run:

```powershell
python -m unittest tests.test_yellow_line_pose -v
```

Expected: 3 tests pass.

- [ ] **Step 5: Commit the contracts**

```powershell
git add smartcar/whalesbot/tools/yellow_line_pose.py tests/test_yellow_line_pose.py
git commit -m "feat: define yellow-line pose contracts"
```

## Task 2: Implement HSV mask extraction and metric line estimation

**Files:**
- Modify: `smartcar/whalesbot/tools/yellow_line_pose.py`
- Modify: `tests/test_yellow_line_pose.py`

- [ ] **Step 1: Add failing synthetic-image tests**

Append tests that construct a 640×480 black frame, draw one BGR yellow stripe inside the configured ROI, and use a known homography. Verify that `estimate()` returns a finite pose on the configured side. Also test an empty frame, a wrong-size frame, and two comparable yellow stripes.

```python
import cv2

from smartcar.whalesbot.tools.yellow_line_pose import HsvYellowLinePoseEstimator


class HsvYellowLinePoseEstimatorTest(unittest.TestCase):
    def setUp(self):
        mapping = valid_mapping()
        mapping["calibration"]["image_to_ground_homography"] = [
            [0.001, 0.0, -0.32],
            [0.0, -0.001, 0.48],
            [0.0, 0.0, 1.0],
        ]
        self.estimator = HsvYellowLinePoseEstimator(YellowLineConfig.from_mapping(mapping))

    def test_single_right_yellow_line_returns_metric_pose(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.line(frame, (500, 470), (470, 260), (0, 255, 255), 14)
        pose = self.estimator.estimate(frame, timestamp=1.0)
        self.assertTrue(pose.valid, pose.reason)
        self.assertGreater(pose.distance_m, 0.0)
        self.assertTrue(np.isfinite(pose.heading_rad))
        self.assertGreaterEqual(pose.confidence, 0.65)

    def test_empty_frame_is_rejected(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        self.assertEqual("no_line", self.estimator.estimate(frame, 2.0).reason)

    def test_wrong_frame_size_is_rejected(self):
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        self.assertEqual("frame_invalid", self.estimator.estimate(frame, 3.0).reason)

    def test_two_comparable_candidates_are_rejected(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.line(frame, (470, 470), (450, 260), (0, 255, 255), 14)
        cv2.line(frame, (570, 470), (550, 260), (0, 255, 255), 14)
        self.assertEqual("ambiguous_candidates", self.estimator.estimate(frame, 4.0).reason)
```

- [ ] **Step 2: Run the new estimator tests and confirm the class is missing**

Run:

```powershell
python -m unittest tests.test_yellow_line_pose.HsvYellowLinePoseEstimatorTest -v
```

Expected: FAIL because `HsvYellowLinePoseEstimator` is not defined.

- [ ] **Step 3: Implement the estimator with explicit rejection reasons**

Add `HsvYellowLinePoseEstimator` to `yellow_line_pose.py`. Its implementation must:

1. Verify exact `(height, width, 3)` against `frame_size`.
2. Undistort with the configured OpenCV camera matrix and coefficients.
3. Apply the polygon ROI before HSV thresholding.
4. Run configured open/close morphology.
5. Find connected components and reject no candidate or two candidates whose areas differ by less than 25%.
6. Transform candidate pixels through the homography with `cv2.perspectiveTransform`.
7. Reject transformed points that are non-finite or lie behind the vehicle working region.
8. Fit a line with `cv2.fitLine`, compute normal-form signed distance and heading, and normalize heading to `[-pi/2, pi/2]`.
9. Reject a candidate on the wrong configured side.
10. Compute confidence from area, vertical coverage, and RMS fit residual; return `valid=True` only at or above `valid_threshold`.

Use these concrete constants in the first implementation so behavior is deterministic and testable: minimum component area 150 pixels, ambiguity ratio 0.75, minimum 80 transformed points, area full-score at 1200 pixels, vertical coverage full-score at 160 pixels, residual full-score at or below 0.01 m and zero-score at or above 0.05 m. Keep them module constants so later configuration is an isolated change.

The public method is exactly:

```python
def estimate(self, frame_bgr, timestamp):
    """Return YellowLinePose; never raise for ordinary image rejection."""
```

Unexpected OpenCV errors must return `YellowLinePose.invalid("fit_failed", timestamp)` with a numeric diagnostic code; configuration errors remain construction-time exceptions.

- [ ] **Step 4: Run perception tests and static checks**

Run:

```powershell
python -m unittest tests.test_yellow_line_pose -v
python -m py_compile smartcar/whalesbot/tools/yellow_line_pose.py
```

Expected: all yellow-line pose tests pass and compilation exits 0.

- [ ] **Step 5: Commit metric HSV estimation**

```powershell
git add smartcar/whalesbot/tools/yellow_line_pose.py tests/test_yellow_line_pose.py
git commit -m "feat: estimate metric yellow-line pose from HSV"
```

## Task 3: Add offline intrinsics and homography calibration tools

**Files:**
- Create: `tools/calibrate_yellow_line.py`
- Create: `tests/test_calibrate_yellow_line.py`

- [ ] **Step 1: Write failing calibration math tests**

```python
# tests/test_calibrate_yellow_line.py
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tools.calibrate_yellow_line import compute_homography, reprojection_errors


class YellowLineCalibrationTest(unittest.TestCase):
    def test_known_planar_mapping_has_submillimeter_reprojection_error(self):
        image = np.array([[100, 100], [500, 100], [500, 400], [100, 400], [300, 250]], dtype=np.float64)
        ground = np.array([[-0.2, 0.6], [0.2, 0.6], [0.2, 0.1], [-0.2, 0.1], [0.0, 0.35]], dtype=np.float64)
        matrix, inliers = compute_homography(image, ground)
        errors = reprojection_errors(matrix, image, ground)
        self.assertEqual((3, 3), matrix.shape)
        self.assertEqual(5, int(inliers.sum()))
        self.assertLess(float(errors.max()), 0.001)

    def test_fewer_than_six_correspondences_are_rejected(self):
        points = np.zeros((5, 2), dtype=np.float64)
        with self.assertRaisesRegex(ValueError, "at least six"):
            compute_homography(points, points)
```

- [ ] **Step 2: Run the tests and confirm the calibration module is missing**

Run:

```powershell
python -m unittest tests.test_calibrate_yellow_line -v
```

Expected: FAIL with `ModuleNotFoundError: tools.calibrate_yellow_line`.

- [ ] **Step 3: Implement a hardware-free two-subcommand calibration CLI**

Implement these functions in `tools/calibrate_yellow_line.py`:

```python
def calibrate_intrinsics(image_paths, pattern_cols, pattern_rows, square_size_m):
    """Use cv2.findChessboardCorners and cv2.calibrateCamera; reject fewer than 8 valid images."""

def compute_homography(image_points, ground_points):
    """Require at least 6 paired points and call cv2.findHomography(image_points, ground_points, cv2.RANSAC, 0.015)."""

def reprojection_errors(matrix, image_points, ground_points):
    """Return Euclidean ground-plane errors in meters."""

def main(argv=None):
    """Parse `intrinsics` or `homography`, validate inputs, and write JSON output."""
```

The CLI contracts are:

```powershell
python tools/calibrate_yellow_line.py intrinsics `
  --images "calibration/chessboard/*.jpg" `
  --cols 9 --rows 6 --square-size-m 0.025 `
  --output calibration/front_intrinsics.json

python tools/calibrate_yellow_line.py homography `
  --correspondences calibration/ground_points.json `
  --intrinsics calibration/front_intrinsics.json `
  --output calibration/yellow_line_calibration.json
```

The correspondence JSON schema is exact:

```json
{
  "frame_size": [640, 480],
  "image_points": [[120.0, 420.0], [520.0, 420.0], [160.0, 330.0], [480.0, 330.0], [220.0, 260.0], [420.0, 260.0]],
  "ground_points_m": [[-0.30, 0.10], [0.30, 0.10], [-0.30, 0.35], [0.30, 0.35], [-0.30, 0.65], [0.30, 0.65]]
}
```

The homography output must include `frame_size`, `camera_matrix`, `distortion`, `image_to_ground_homography`, per-point errors, RMS error, maximum error, and inlier mask. Exit nonzero if RMS exceeds 0.01 m or maximum error exceeds 0.02 m. Never edit vehicle YAML automatically.

- [ ] **Step 4: Run calibration tests and CLI help**

Run:

```powershell
python -m unittest tests.test_calibrate_yellow_line -v
python tools/calibrate_yellow_line.py --help
python -m py_compile tools/calibrate_yellow_line.py
```

Expected: tests pass; help lists `intrinsics` and `homography`; compilation exits 0.

- [ ] **Step 5: Commit calibration tooling**

```powershell
git add tools/calibrate_yellow_line.py tests/test_calibrate_yellow_line.py
git commit -m "feat: add yellow-line camera calibration tools"
```

## Task 4: Implement the stationary alignment controller

**Files:**
- Create: `smartcar/whalesbot/tools/yellow_line_aligner.py`
- Create: `tests/test_yellow_line_aligner.py`

- [ ] **Step 1: Write deterministic failing state-machine tests**

Use fake frame reader, estimator, velocity sink, monotonic clock, and sleep. Cover these exact cases:

- five valid acquisition frames produce no motion before the fifth frame;
- ten settled frames return `success=True` and finish with `(0, 0, 0)`;
- invalid frames before acquisition never move and return `line_not_found` or `low_confidence`;
- three consecutive lost frames after acquisition command an immediate stop and return `low_confidence`;
- `_stop_flag` returns `stop_requested` and stops;
- timeout returns `timeout` and stops;
- distance and heading command signs, output limits, and per-cycle slew limits match configuration;
- any exception in frame reading still produces a final stop through `finally`.

Define the expected public result in the test:

```python
from smartcar.whalesbot.tools.yellow_line_aligner import (
    EdgeAlignConfig,
    EdgeAlignResult,
    YellowLineAligner,
)

config = EdgeAlignConfig.from_mapping({
    "max_vy": 0.08,
    "max_wz": 0.30,
    "max_vy_step": 0.02,
    "max_wz_step": 0.05,
    "distance_tolerance_m": 0.015,
    "heading_tolerance_deg": 1.5,
    "stable_frames": 10,
    "timeout_s": 5.0,
    "loop_interval_s": 0.05,
    "vy_sign": 1.0,
    "wz_sign": 1.0,
    "distance_pid": {"Kp": 1.0, "Ki": 0.0, "Kd": 0.0},
    "heading_pid": {"Kp": 1.0, "Ki": 0.0, "Kd": 0.0},
})
```

- [ ] **Step 2: Run the focused tests and confirm the aligner is missing**

Run:

```powershell
python -m unittest tests.test_yellow_line_aligner -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'smartcar.whalesbot.tools.yellow_line_aligner'`.

- [ ] **Step 3: Implement the dependency-injected controller**

`yellow_line_aligner.py` must import `PID` directly from `.tools_class`, not through the top-level `smartcar` package. Implement:

Define `EdgeAlignConfig` as a frozen dataclass with the exact fields exercised by the test mapping: output limits, step limits, distance/heading tolerances, stable frames, timeout, loop interval, command signs, and two PID mappings. Its `from_mapping()` rejects non-positive limits/timing/counts, signs other than `-1.0` or `1.0`, and missing PID gains.

Define `EdgeAlignResult` as a frozen dataclass with these exact fields:

```python
@dataclass(frozen=True)
class EdgeAlignResult:
    success: bool
    reason: str
    final_distance_m: Optional[float]
    distance_error_m: Optional[float]
    heading_error_rad: Optional[float]
    confidence: float
    elapsed_s: float
    frames: int
```

Define `YellowLineAligner.__init__(self, estimator, frame_reader, velocity_writer, stop_requested, config, monotonic=time.monotonic, sleep=time.sleep)` and `YellowLineAligner.align(self, target_distance_m)`. Store all injected dependencies without importing camera or vehicle modules.

The implementation sequence is mandatory:

1. Call `velocity_writer(0, 0, 0)` before reading a frame.
2. Construct fresh distance and heading PID instances for every call.
3. Require `estimator.config.acquire_frames` consecutive valid, above-threshold poses before motion.
4. Set distance PID setpoint to `target_distance_m` and heading PID setpoint to zero.
5. Apply `vy_sign`/`wz_sign`, absolute limits, and step limits after PID calculation.
6. Send only `(0, vy, wz)`.
7. Reset stable count whenever either error or either command leaves its threshold.
8. Stop immediately on an invalid acquired measurement; return after configured consecutive loss.
9. Return a structured result for settled, timeout, stop request, camera failure, and low confidence.
10. In `finally`, reset both PID objects if created and call `velocity_writer(0, 0, 0)` exactly once more.

Do not catch `KeyboardInterrupt`; `finally` must stop before it propagates.

- [ ] **Step 4: Run controller tests and compile**

Run:

```powershell
python -m unittest tests.test_yellow_line_aligner -v
python -m py_compile smartcar/whalesbot/tools/yellow_line_aligner.py
```

Expected: all aligner tests pass and compilation exits 0.

- [ ] **Step 5: Commit the stopped-vehicle controller**

```powershell
git add smartcar/whalesbot/tools/yellow_line_aligner.py tests/test_yellow_line_aligner.py
git commit -m "feat: add stationary yellow-line alignment controller"
```

## Task 5: Integrate the aligner into `MyCar` without changing lane following

**Files:**
- Modify: `config_car.yml`
- Modify: `car_wrap_2026.py`
- Create: `tests/test_yellow_line_car_integration.py`

- [ ] **Step 1: Write AST and configuration regression tests first**

The test must parse source without importing vehicle/serial modules and assert:

```python
import ast
from pathlib import Path
import unittest
import yaml

ROOT = Path(__file__).resolve().parents[1]


class YellowLineCarIntegrationTest(unittest.TestCase):
    def test_lane_dis_offset_signature_is_unchanged(self):
        tree = ast.parse((ROOT / "car_wrap_2026.py").read_text(encoding="utf-8"))
        method = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "lane_dis_offset")
        self.assertEqual(["self", "speed", "dis_hold", "stop"], [a.arg for a in method.args.args])

    def test_public_alignment_method_exists(self):
        tree = ast.parse((ROOT / "car_wrap_2026.py").read_text(encoding="utf-8"))
        names = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
        self.assertIn("align_to_yellow_edge", names)

    def test_feature_is_disabled_by_default(self):
        cfg = yaml.safe_load((ROOT / "config_car.yml").read_text(encoding="utf-8"))
        self.assertFalse(cfg["yellow_line"]["enabled"])
        self.assertEqual("hsv", cfg["yellow_line"]["mask_backend"])
```

- [ ] **Step 2: Run the integration test and confirm the method/config are absent**

Run:

```powershell
python -m unittest tests.test_yellow_line_car_integration -v
```

Expected: the unchanged-signature test passes; method/config tests fail.

- [ ] **Step 3: Add disabled configuration and the thin adapter**

Add the full `yellow_line` block from the approved design to `config_car.yml`, with these additional alignment fields:

```yaml
yellow_line:
  enabled: false
  side: right
  frame_size: [640, 480]
  mask_backend: hsv
  calibration:
    camera_matrix: []
    distortion: []
    image_to_ground_homography: []
  hsv:
    roi: []
    lower: []
    upper: []
    open_kernel: 3
    close_kernel: 5
  confidence:
    valid_threshold: 0.65
    acquire_frames: 5
    lost_frames: 3
  align_control:
    max_vy: 0.08
    max_wz: 0.30
    max_vy_step: 0.02
    max_wz_step: 0.05
    distance_tolerance_m: 0.015
    heading_tolerance_deg: 1.5
    stable_frames: 10
    timeout_s: 5.0
    loop_interval_s: 0.05
    vy_sign: 1.0
    wz_sign: 1.0
    distance_pid: {Kp: 1.0, Ki: 0.0, Kd: 0.0}
    heading_pid: {Kp: 1.0, Ki: 0.0, Kd: 0.0}
  task_edge_offsets: {}
```

In `MyCar.__init__`, retain the mapping and set the lazy instance to `None`; do not validate empty calibration while disabled:

```python
self.yellow_line_cfg = cfg.get("yellow_line", {"enabled": False})
self._yellow_line_aligner = None
```

Add a lazy helper and public method near other alignment methods:

```python
def _get_yellow_line_aligner(self):
    if not self.yellow_line_cfg.get("enabled", False):
        return None
    if self.yellow_line_cfg.get("mask_backend") != "hsv":
        raise ValueError("only hsv yellow-line backend is implemented")
    if self._yellow_line_aligner is None:
        from smartcar.whalesbot.tools.yellow_line_pose import YellowLineConfig, HsvYellowLinePoseEstimator
        from smartcar.whalesbot.tools.yellow_line_aligner import EdgeAlignConfig, YellowLineAligner

        pose_config = YellowLineConfig.from_mapping(self.yellow_line_cfg)
        estimator = HsvYellowLinePoseEstimator(pose_config)
        align_config = EdgeAlignConfig.from_mapping(self.yellow_line_cfg["align_control"])
        self._yellow_line_aligner = YellowLineAligner(
            estimator=estimator,
            frame_reader=lambda: self.cap_front.read().copy(),
            velocity_writer=self.set_velocity,
            stop_requested=lambda: self._stop_flag,
            config=align_config,
        )
    return self._yellow_line_aligner

def align_to_yellow_edge(self, edge_offset_m):
    """Align a stopped vehicle; return disabled result without motion when disabled."""
    from smartcar.whalesbot.tools.yellow_line_aligner import EdgeAlignResult

    aligner = self._get_yellow_line_aligner()
    if aligner is None:
        return EdgeAlignResult.disabled()
    self.set_velocity(0, 0, 0)
    return aligner.align(float(edge_offset_m))
```

Define `EdgeAlignResult.disabled()` in Task 4 to return `success=False`, `reason="disabled"`, no measurements, zero confidence, elapsed time, and frames.

- [ ] **Step 4: Run integration and focused unit tests**

Run:

```powershell
python -m unittest tests.test_yellow_line_car_integration tests.test_yellow_line_pose tests.test_yellow_line_aligner -v
python -m py_compile car_wrap_2026.py
```

Expected: all focused tests pass; compilation exits 0; no camera or serial device is opened by tests.

- [ ] **Step 5: Commit the opt-in vehicle adapter**

```powershell
git add config_car.yml car_wrap_2026.py tests/test_yellow_line_car_integration.py
git commit -m "feat: expose opt-in task yellow-line alignment"
```

## Task 6: Add a safe read-only measurement tool

**Files:**
- Create: `tools/yellow_line_read_only.py`
- Create: `tests/test_yellow_line_read_only.py`

- [ ] **Step 1: Write failing parser and record-format tests**

Test that default CLI arguments are camera index 1, duration 15 seconds, output directory `logs/yellow-line-read-only`, and that no motion-related option exists. Test a pure `build_record()` function with a fake pose and assert fixed CSV fields.

```python
EXPECTED_FIELDS = [
    "elapsed_s", "frame_index", "valid", "reason", "confidence",
    "distance_m", "heading_rad", "fit_residual_m", "visible_length_m",
]
```

- [ ] **Step 2: Run the tests and confirm the tool is missing**

Run:

```powershell
python -m unittest tests.test_yellow_line_read_only -v
```

Expected: FAIL because `tools.yellow_line_read_only` is missing.

- [ ] **Step 3: Implement a camera-only bounded recorder**

The tool must import `Camera` lazily only inside `main()`, load and validate `yellow_line` config, instantiate `HsvYellowLinePoseEstimator`, and for a bounded duration:

- read and copy one 640×480 frame;
- estimate pose;
- draw ROI, status, distance, heading, confidence, and rejection reason;
- write `annotated.mp4`, `frames.csv`, and `metadata.json` in a unique timestamped directory;
- close camera/video/CSV in `finally`;
- never import `MecanumDriver`, `car_wrap_2026`, or call `set_velocity`.

CLI:

```powershell
python tools/yellow_line_read_only.py `
  --config config_car.yml `
  --camera 1 --duration 15 `
  --output-dir logs/yellow-line-read-only
```

Reject duration outside `(0, 300]`. Exit before opening the camera when the feature is disabled or calibration is empty.

- [ ] **Step 4: Run read-only tests and import isolation check**

Run:

```powershell
python -m unittest tests.test_yellow_line_read_only -v
python -c "import tools.yellow_line_read_only; print('READ_ONLY_IMPORT_OK')"
python -m py_compile tools/yellow_line_read_only.py
```

Expected: tests pass, output contains `READ_ONLY_IMPORT_OK`, and no serial/camera error appears during import.

- [ ] **Step 5: Commit the read-only tool**

```powershell
git add tools/yellow_line_read_only.py tests/test_yellow_line_read_only.py
git commit -m "feat: add read-only yellow-line measurement tool"
```

## Task 7: Add the first disabled-by-default task integration and complete verification

**Files:**
- Modify: `car_task_function.py`
- Modify: `tests/test_yellow_line_car_integration.py`
- Modify: `docs/superpowers/specs/2026-08-02-single-front-camera-yellow-line-calibration-design.md` only if implementation uncovers an approved-design contradiction

- [ ] **Step 1: Write a failing AST test for call placement and preservation of current compensation**

Parse `auto_seeding()` and assert:

- its first `align_to_yellow_edge` call appears after the entry `lane_dis_offset` call;
- the existing `RIGHT_SEEDING_PRE_DETECTION_OFFSET = [0.0, -0.02, 0.0]` assignment still exists;
- the function checks result success and raises on an enabled alignment failure;
- disabled or unconfigured alignment does not alter the old flow.

- [ ] **Step 2: Run the test and confirm the task call is absent**

Run:

```powershell
python -m unittest tests.test_yellow_line_car_integration -v
```

Expected: existing tests pass and the new call-placement test fails.

- [ ] **Step 3: Add the opt-in task call after the existing entry stop**

Add a `MyCar` helper that resolves a task target without moving:

```python
def align_to_yellow_edge_for_task(self, task_name):
    offsets = self.yellow_line_cfg.get("task_edge_offsets", {})
    if not self.yellow_line_cfg.get("enabled", False):
        return self.align_to_yellow_edge(0.0)
    if task_name not in offsets:
        from smartcar.whalesbot.tools.yellow_line_aligner import EdgeAlignResult
        return EdgeAlignResult.not_configured(task_name)
    return self.align_to_yellow_edge(float(offsets[task_name]))
```

Define `EdgeAlignResult.not_configured()` in Task 4 with reason `task_not_configured`.

Immediately after `auto_seeding()` completes the entry `lane_dis_offset()` and its existing settle delay, call:

```python
edge_alignment = my_car.align_to_yellow_edge_for_task("auto_seeding")
if edge_alignment.reason not in {"disabled", "task_not_configured"} and not edge_alignment.success:
    raise RuntimeError(f"播种任务黄线精确校准失败: {edge_alignment.reason}")
```

Do not remove or change `RIGHT_SEEDING_PRE_DETECTION_OFFSET` in this phase.

- [ ] **Step 4: Run all local non-hardware verification**

Run:

```powershell
python -m unittest discover -s tests -v
python -m py_compile `
  smartcar/whalesbot/tools/yellow_line_pose.py `
  smartcar/whalesbot/tools/yellow_line_aligner.py `
  tools/calibrate_yellow_line.py `
  tools/yellow_line_read_only.py `
  car_wrap_2026.py car_task_function.py
git diff --check
```

Expected: all tests pass, every compilation exits 0, and `git diff --check` is silent. If unrelated pre-existing tests fail, record their exact names and reproduce them against the pre-feature commit before attributing them to this work.

- [ ] **Step 5: Commit the task integration**

```powershell
git add car_task_function.py tests/test_yellow_line_car_integration.py
git commit -m "feat: opt auto seeding into task edge alignment"
```

- [ ] **Step 6: Perform the user-authorized staged physical verification**

Do not run these steps without explicit user authorization and a populated calibration. Execute in this order:

1. Copy generated intrinsics/homography values into `config_car.yml`, leave `enabled: false`, and validate the read-only CLI configuration offline.
2. Set `enabled: true` only for the read-only tool; record multiple known distances and verify static 95th-percentile distance error is at most 1.5 cm and heading error at most 1.5°.
3. With wheels suspended or motion physically constrained, invoke one alignment and verify `vy_sign` and `wz_sign`; stop immediately if either sign is wrong.
4. On an empty bounded floor, use `|Vy| <= 0.08 m/s`, `|Wz| <= 0.30 rad/s`, and 5-second timeout. Test positive/negative lateral and heading starts.
5. Cover the yellow line before acquisition and during control; verify no acquisition motion and immediate stop after loss.
6. Run at least 20 repeated alignments at the `auto_seeding` target. Measure physical error independently and require at least 95% within 3 cm before enabling the task call in competition flow.

Record each run's configuration commit, physical truth measurement, final reported pose, result reason, elapsed time, and whether any stop-safety violation occurred.

## Plan self-review

- Spec coverage: HSV perception, metric calibration, structured confidence/rejection, stopped-only PID, failure-stop behavior, disabled default, read-only validation, first-task opt-in, 3 cm acceptance, and future mask-backend replacement boundaries are each assigned to a task.
- Scope boundary: no task changes `lane_base()`, `lane_dis()`, or `lane_dis_offset()`; no driving-time yellow-line correction is introduced.
- Type consistency: `YellowLinePose`, `YellowLineConfig`, `EdgeAlignConfig`, `EdgeAlignResult`, `HsvYellowLinePoseEstimator`, and `YellowLineAligner` use the same names and signatures throughout.
- Placeholder scan: the plan contains no deferred implementation markers; empty calibration and empty task offsets are deliberate disabled defaults that fail before motion.
