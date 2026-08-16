# Preview Direction and Near-Field Timing Implementation Plan

> **For agentic workers:** This plan is executed inline because the user requested direct implementation. The user explicitly requested no red/green test cycle; verification uses existing focused tests and a complete offline lap replay.

**Goal:** Add a route-independent temporal teacher that uses far-field geometry for turn direction and middle/near-field evidence for steering onset and magnitude.

**Architecture:** `PreviewTimingFilter` receives one `LaneAnalysisResult` per frame and returns a mapped result. It maintains only continuous EMA/peak values, never route event states or fixed frame/odometry windows. Far-field direction is retained through single-boundary degradation; arrival weight is computed from corner confidence, boundary coverage loss, and near lateral motion; the PID continues to consume the filtered PID-before errors.

**Tech Stack:** Python, NumPy, existing `LaneAnalysisResult`, `ErrorMapping`, `CvLanePidController`, OpenCV replay scripts.

---

### Task 1: Add the continuous preview/timing filter

**Files:**
- Create: `smartcar/whalesbot/tools/lane_collect/temporal_filter.py`
- Modify: `smartcar/whalesbot/tools/lane_collect/__init__.py`

- [x] Implement `PreviewTimingConfig` with only geometry/time constants: corner gate range, lateral-motion gate range, coverage-loss gate range, EMA decay, direction confirmation count, and protected heading minimum.
- [x] Implement `PreviewTimingFilter.reset()` with no route counters or one-shot crossing flags.
- [x] Implement `PreviewTimingFilter.update(result)`:
  - derive expected direction from `left_only/right_only` only for detecting an implausible single-boundary reversal;
  - maintain preview heading/lateral continuity and a continuous arrival weight in `[0, 1]`;
  - scale normal steering heading by arrival weight;
  - when the same single boundary reverses during a confirmed corner, preserve the far-field direction and use the larger of the configured minimum and recent preview peak;
  - clear protected direction on side switch or `both/mixed` recovery;
  - recalculate `error_y/error_angle` with the existing `ErrorMapping` and add diagnostic metrics (`preview_direction`, `arrival_weight`, `direction_held`, `timing_source`).
- [x] Export the filter classes from `lane_collect/__init__.py`.

### Task 2: Integrate the filter into real-car CV collection

**Files:**
- Modify: `smartcar/whalesbot/tools/lane_collect/ssh_test.py`

- [x] Construct one filter per `OpenCVLaneSshTest` session using the analyzer’s `ErrorMapping`.
- [x] Reset it on `start` and `stop`.
- [x] Apply it immediately after `analyzer.process()` and before `CvLanePidController.compute()` when the safety flag is enabled.
- [x] Keep the existing 0.10 m launch-only odometry guard after PID calculation; do not reuse it for other turns.
- [x] Save filtered diagnostics in `cv` while retaining PID-before labels in `state[1:3]` and actual commands in `control`.

### Task 3: Add replay comparison for direction and timing

**Files:**
- Modify: `scripts/compare_cv_control.py`

- [x] Add `--temporal-filter` to enable the new filter while preserving the current baseline path.
- [x] Report zero-lag correlation, best lag/correlation, direction accuracy, false active frames, and separate windows `217–278`, `1161–1181`, right turn `2700–2854`, left turn `2855–2920`, and `3180–3397`; exclude terminal frames from 3398 onward.
- [x] Write filter diagnostics to the output CSV/JSON and draw baseline versus filtered angular speed.

### Task 4: Verify and document

**Files:**
- Modify: `docs/cv-lane-development-status-2026-08-15.md`

- [x] Run `py_compile` for changed modules.
- [x] Run the existing focused CV tests; do not run a red/green cycle.
- [x] Replay `lap_001` frames `1–3432` with and without `--temporal-filter` and compare direction and timing metrics.
- [x] Record that the best lag remains +9 frames and keep real-car activation disabled.

### Task 5: Remove premature near-field activation

**Files:**
- Modify: `smartcar/whalesbot/tools/lane_collect/temporal_filter.py`
- Modify: `tests/test_preview_timing_filter.py`
- Modify: `scripts/render_cv_frame_review.py`

- [x] Evaluate geometry-supported lateral motion and removal of forced arrival. Rejected after correlation fell from `0.6805` to `0.6317` and late valid steering disappeared.
- [x] Evaluate single-boundary-only motion support. Rejected after correlation remained below baseline at `0.6433` and the first turn remained 17 frames early.
- [x] Evaluate faster decay plus a higher corner gate. Rejected after correlation fell to `0.5850` and the first dominant turn became 31 frames late.
- [x] Restore the best first-version control behavior and keep `PREVIEW_TIMING_ENABLED = False`.
- [x] Add per-cue arrival diagnostics without changing control output.
- [x] Warm the analyzer, timing filter, controller, and invalid-frame hold from frame 1 in `render_cv_frame_review.py`, while writing images and records only for the requested review range.
- [x] Replay `lap_001` frames `1–3397` and document why onset and hold must be separated before the next implementation attempt.
