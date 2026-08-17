# CV/CNN Curvature Control Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace frame-count steering/speed post-processing with a shared curvature-based control path using real `dt`, while preserving raw joystick-compatible actions.

**Architecture:** The OpenCV teacher computes a final action-equivalent curvature from its final `wz` and pre-ramp target speed. The CNN adapter consumes the same curvature semantic and known speed, computes `wz = actual_vx * kappa`, and applies only safety limits. Raw `[vx, vy, wz]` remains the canonical recorded action. CV and CNN share a time-aware speed slew helper.

**Tech Stack:** Python dataclasses, NumPy, existing lane collection controller, pytest.

---

### Task 1: Add time-aware slew primitives

**Files:**
- Modify: `smartcar/whalesbot/tools/lane_collect/pid_control.py`
- Test: `tests/test_cv_lane_pid_control.py`

- [ ] Add `dt_s` to `CvLanePidController.compute`, defaulting to the configured nominal period for existing callers.
- [ ] Clamp `dt_s` to a configurable safe interval and convert acceleration/deceleration rates from per-frame steps to units-per-second fields, preserving equivalent behavior at 20 Hz.
- [ ] Change heading slew to accept `dt_s` only if retained for safety; do not apply steering behavior shaping to action-equivalent CNN curvature.
- [ ] Add tests proving a 0.10 s interval changes the speed by twice the 0.05 s interval and that a dropped-frame interval is clamped.

### Task 2: Make CV teacher emit action-equivalent curvature and timestamped records

**Files:**
- Modify: `smartcar/whalesbot/tools/lane_collect/pid_control.py`
- Modify: `smartcar/whalesbot/tools/lane_collect/ssh_test.py`
- Test: `tests/test_cv_lane_pid_control.py`

- [ ] Add `action_curvature` to `CvLaneControlCommand` and its serialized record.
- [ ] Compute it from final `angular_speed / max(abs(target_forward_speed), 0.12)` with finite limits; retain raw `control` fields unchanged.
- [ ] Pass measured monotonic frame delta from `OpenCVLaneSshTest` into the controller and persist timestamp plus effective `dt` in each record.
- [ ] Ensure invalid hold and startup guard update the final action curvature after they modify the command.
- [ ] Update session metadata to declare raw action fields, curvature label semantics, denominator, and time policy.

### Task 3: Add CNN curvature-to-chassis adapter

**Files:**
- Locate and modify the CNN vehicle inference/control entry identified in `lane-cnn-training-handoff.md` (expected: `smartcar/paddlebaidu/paddle_jetson/base/infer_wrap.py` and its car control caller).
- Test: add focused adapter tests under the existing CNN/lane test directory.

- [ ] Define one adapter function that accepts `kappa`, known `actual_vx`, `target_vx`, and `dt_s`, then returns `vx, 0, wz`.
- [ ] Compute `wz = actual_vx * kappa`, apply only finite curvature and chassis `wz` safety limits, and call the shared real-time speed slew helper.
- [ ] Do not call CV heading slew, right-turn override, or line-loss steering hold from the CNN path; those behaviors must come from the label.
- [ ] Keep an explicit compatibility profile for the legacy `[vy, yaw_error]` model until the new curvature model is selected, rather than silently changing existing deployed models.

### Task 4: Update data/replay tooling for offline CV relabeling

**Files:**
- Modify: the existing lane collection/replay writer and comparison script used by `scripts/compare_cv_control.py`.
- Test: add a replay test using ordered synthetic frames and a reset controller.

- [ ] Replay old images in sequence, reset controller state at each lap, and use original timestamps or an explicitly recorded nominal period.
- [ ] Generate a separate relabeled session containing raw control, `kappa_action`, `target_vx`, validity/held state, source reason, and CV config hash.
- [ ] Never overwrite the original joystick session.
- [ ] Filter invalid detection, curvature spikes, and incompatible rescue frames instead of silently training on them.

### Task 5: Verify offline behavior and documentation

**Files:**
- Test: focused controller, adapter, and replay tests; no red/green full training run unless requested.
- Modify: `docs/lane-cnn-training-handoff.md` in the training worktree only if implementation details differ from section 18.

- [ ] Run focused pytest tests for time scaling, curvature labels, startup zero steering, held frames, and right-sharp-turn labels.
- [ ] Run `git diff --check` and inspect the final control schema.
- [ ] Compare final `vx/wz`, turn onset/return positions, and special windows against the CV teacher replay.
- [ ] Record any remaining incompatibility explicitly before real-car testing.

## Scope guard

Do not change CNN network architecture, training loss, or remote training scripts in this pass unless the current vehicle inference entry cannot consume the curvature adapter. Do not clean, overwrite, commit, or push unrelated files in the CNN training worktree.
