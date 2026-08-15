# CV Adaptive Speed Control Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the CV collection teacher run at maximum speed on straights and smoothly slow down as steering demand increases, while preserving exact applied-command CNN labels.

**Architecture:** `CvLanePidController` derives speed from the pre-slew angular request, applies asymmetric speed slew limits, and returns diagnostics with each command. `CvTestSessionWriter` persists those diagnostics and controller configuration. The analyzer retains its configurable near-field fit as the source of steering demand, and the crossing state machine stays disabled.

**Tech Stack:** Python 3, NumPy, OpenCV, existing PID controller, unittest/pytest-compatible tests

---

### Task 1: Implement adaptive speed in the CV controller

**Files:**
- Modify: `smartcar/whalesbot/tools/lane_collect/pid_control.py`

- [x] Replace fixed `forward_speed` configuration with maximum/minimum speed, steering-demand reference, speed curve exponent, and separate acceleration/deceleration step limits.
- [x] Compute normalized demand from the absolute PID angular request before angular slew limiting.
- [x] Compute the bounded target speed and apply faster deceleration than acceleration to the applied speed.
- [x] Reset speed state with the other controller state and expose target speed and normalized demand in `CvLaneControlCommand`.

### Task 2: Persist adaptive-speed labels and diagnostics

**Files:**
- Modify: `smartcar/whalesbot/tools/lane_collect/ssh_test.py`

- [x] Keep `state` and `control` equal to the complete command sent to `set_velocity()`.
- [x] Store normalized steering demand and target forward speed per frame.
- [x] Store the adaptive-speed configuration in `session.json`.
- [x] Preserve complete prior commands, including speed diagnostics, during the 10-frame invalid hold.

### Task 3: Update focused tests and operational documentation

**Files:**
- Modify: `tests/test_cv_lane_pid_control.py`
- Modify: `docs/cv-lane-collection-operation-guide.md`

- [x] Update existing fixed-speed expectations to the adaptive-speed contract.
- [x] Add direct assertions for monotonic bounded speed, asymmetric response, saved applied labels, and held command speed.
- [x] Document straight/bend speed behavior, configuration parameters, and recorded diagnostics.

### Task 4: Verify the direct implementation

**Files:**
- Verify: all modified Python and documentation files

- [x] Run Python compilation for the modified modules and focused tests.
- [x] Run the existing focused CV controller/analyzer/state-machine tests.
- [x] Run `git diff --check` and inspect the final diff for unrelated changes.
