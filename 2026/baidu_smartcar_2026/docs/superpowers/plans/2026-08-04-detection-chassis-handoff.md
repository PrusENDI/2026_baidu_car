# Detection Chassis Handoff Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Redirect visual `dy` correction from the arm X axis to chassis lateral motion when the arm reaches its movement boundary.

**Architecture:** Add a pure static helper beside the existing alignment helpers to split arm and chassis outputs. Call it from `move_to_detection_target()` after computing `out_y`, using caller bounds when supplied and `(0.01, 0.24)` otherwise.

**Tech Stack:** Python 3, `ast`, `unittest`

---

### Task 1: Implement chassis handoff

**Files:**
- Modify: `car_wrap_2026.py`

- [x] Add `_detection_y_control_with_chassis_handoff(out_y, arm_x_position, arm_x_bounds=None)` next to the existing alignment helpers.
- [x] Initialize and reset `out_chassis_y`, replace the old boundary suppression block with the helper, and pass the result to `set_velocity(out_x, out_chassis_y, 0)`.
- [x] Ensure successful alignment and missing detections clear both arm and chassis correction outputs.

### Task 2: Scoped inspection

**Files:**
- Inspect: `car_wrap_2026.py`

- [x] Inspect the scoped diff for unintended changes without running tests.
