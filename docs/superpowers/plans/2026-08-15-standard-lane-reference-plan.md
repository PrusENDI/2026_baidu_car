# Standard Lane Reference Error Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace frame-only center estimation with fixed-track standard-boundary error using the prepared `standard_lane.json` and ROI-aligned perspective weights.

**Architecture:** Load and validate the standard reference once when constructing `OpenCVLaneAnalyzer`. Each frame keeps the existing segmentation, then filters/fills left and right boundary traces, compares each available side with the corresponding standard boundary, applies the per-row perspective weight, and fits lateral/heading residuals. Missing or mismatched references are hard errors; no legacy center-line fallback is retained.

**Tech Stack:** Python, NumPy, OpenCV, unittest.

---

### Task 1: Define the reference data contract

**Files:**
- Modify: `smartcar/whalesbot/tools/lane_collect/opencv_lane.py`
- Modify: `smartcar/whalesbot/tools/lane_collect/ssh_test.py`
- Create: `tests/test_standard_lane_reference.py`

- [ ] Add a `StandardLaneReference` loader that reads `standard_lane.json` and `perspective.json`, validates image size `320x240`, ROI `[72,192)`, 120 boundary entries, and 120 perspective entries, and exposes NumPy arrays.
- [ ] Make `OpenCVLaneAnalyzer` require a reference path/object for the CV collection mode; reject missing or invalid references with a clear `ValueError`.
- [ ] Add failing tests for valid loading and invalid length/size rejection.

### Task 2: Port line segment filtering and gap filling

**Files:**
- Modify: `smartcar/whalesbot/tools/lane_collect/opencv_lane.py`
- Modify: `tests/test_standard_lane_reference.py`

- [ ] Add a boundary-trace helper that selects the longest contiguous valid segment, splits on jumps larger than the configured gap threshold, and linearly fills only short internal gaps inside the selected segment.
- [ ] Apply it independently to left and right traces before standard comparison.
- [ ] Add failing tests proving short gaps are filled and long discontinuities are rejected.

### Task 3: Compute standard-relative, perspective-weighted errors

**Files:**
- Modify: `smartcar/whalesbot/tools/lane_collect/opencv_lane.py`
- Modify: `tests/test_standard_lane_reference.py`

- [ ] For every valid row, compute `current_boundary - standard_boundary` for each available side.
- [ ] Use both sides when available, otherwise use only the available side; never synthesize the missing side from an assumed lane width.
- [ ] Multiply residuals by the ROI-aligned perspective array and fit residual versus forward row coordinate to produce `raw_lateral` and `raw_heading`.
- [ ] Preserve diagnostics identifying `both`, `left_only`, or `right_only` reference tracking.
- [ ] Add tests for centered, shifted, left-only, and right-only synthetic traces.

### Task 4: Wire the reference into collection mode

**Files:**
- Modify: `smartcar/whalesbot/tools/lane_collect/ssh_test.py`
- Modify: `tests/test_cv_lane_pid_control.py`

- [ ] Construct `OpenCVLaneAnalyzer` with `standard/standard_lane.json` relative to the collection working directory, and fail startup if the reference cannot be loaded.
- [ ] Keep the existing control and recording interfaces unchanged; do not add turn-state or cross-road behavior in this change.
- [ ] Add a startup-path test using a temporary reference directory.

### Task 5: Defer full-lap validation

**Files:**
- Future: `scripts/compare_cv_lap_to_manual.py`

- [ ] Do not run automated tests or full-lap playback in this implementation pass, per request.
- [ ] After cross-road handling is implemented, replay `lane_sessions2/lane_sessions2/lap_001` and compare CV error trends against the standard controller record.
