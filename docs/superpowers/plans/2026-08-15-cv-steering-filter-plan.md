# CV Steering Filter Implementation Plan

> **For agentic workers:** Execute inline in the current feature branch because the CV work is already present as uncommitted, interdependent changes. Do not create a clean worktree or commit unrelated changes.

**Goal:** Reduce CV steering jitter while retaining fixed-track turn behavior and produce an evidence-based full-lap comparison.

**Architecture:** Add spatial false-boundary rejection inside `OpenCVLaneAnalyzer` and temporal heading filtering inside `CvLanePidController`. Keep the cross state machine and invalid-frame hold unchanged.

**Tech Stack:** Python, NumPy, OpenCV, unittest/pytest, existing offline lap scripts.

---

### Task 1: Spatial boundary rejection

**Files:**
- Modify: `smartcar/whalesbot/tools/lane_collect/opencv_lane.py`
- Verify: `tests/test_opencv_lane_analyzer.py`

- [ ] Add configurable conservative false-double-boundary criteria.
- [ ] Apply rejection after trace cleanup and before reference-relative fitting.
- [ ] Record the decision in analysis metrics for offline review.

### Task 2: Temporal steering filter

**Files:**
- Modify: `smartcar/whalesbot/tools/lane_collect/pid_control.py`
- Verify: `tests/test_cv_lane_pid_control.py`

- [ ] Add `heading_limit=0.60`, `max_heading_step=0.04`, `heading_ema_alpha=0.35`, and `heading_deadband=0.03` configuration.
- [ ] Filter scaled heading error before PID and clear all state in `reset()`.
- [ ] Preserve invalid-result and lateral-control behavior.

### Task 3: Focused regression

**Files:**
- Verify: `tests/test_cv_lane_pid_control.py`
- Verify: `tests/test_opencv_lane_analyzer.py`
- Verify: `tests/test_turn_state.py`

- [ ] Run the focused tests once after implementation.
- [ ] Compile the changed Python modules.

### Task 4: Full-lap comparison

**Files:**
- Reuse: `scripts/analyze_opencv_lane.py`
- Create: `artifacts/cv_lap001_filtered_0001_3432/*`

- [ ] Run frames 1 through 3432 with the current standard reference, cross state machine, controller, and invalid-frame hold.
- [ ] Save per-frame controls, a trend chart, and aggregate comparison metrics.
- [ ] Compare against `artifacts/cv_lap001_0001_3432/control_summary.json` and manual `data.json`, including cross and invalid-frame windows.

