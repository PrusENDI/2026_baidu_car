# Fixed Shooting Target Mapping Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Keep animal recognition indices bound to the same four physical shooting targets, verify knockdown, retry up to four shots, and report per-target outcomes.

**Architecture:** Add shooting-only fixed-order selection controls to the existing visual alignment loop, while keeping its default behavior unchanged for every other task. Build a task-local target session in `car_task_function.py` that calibrates four stable ordered targets, retains logical indices in the movement plan, verifies removal of the selected target, retries, and emits structured telemetry.

**Tech Stack:** Python 3, existing OpenCV/Paddle detection output, existing vehicle and arm APIs.

---

### Task 1: Fixed-order visual alignment

**Files:**
- Modify: `car_wrap_2026.py:1615-1844`

- [x] Extend `move_to_detection_target()` with optional `fixed_order_num`, `expected_detection_count`, `max_selected_dx_jump`, and `target_index` arguments whose defaults preserve existing callers.
- [x] In fixed-order mode, filter by label, require the expected candidate count, order candidates once by the configured vehicle direction, and select the requested stable rank without removing/rebasing earlier candidates.
- [x] Reject transient candidate-count mismatches and implausible frame-to-frame jumps by stopping and waiting within the existing timeout.
- [x] Include logical target index, selected rank, expected count, and rejection reason in throttled alignment telemetry.

### Task 2: Shooting session and fixed target plan

**Files:**
- Modify: `car_task_function.py:853-1002`

- [x] Add shooting configuration for three stable calibration frames, calibration/recovery timeouts, five-second settling, three-frame knockdown confirmation, and four maximum shots.
- [x] Replace `relative_loc` with a plan containing both `target_index` and `move_distance`.
- [x] At the shooting entrance, collect four-animal frames until three stable ordered frames lock `target[0..3]`; abort before firing only if initial identity cannot be established.
- [x] Maintain `standing_indices`, confirmed-shot indices, per-target shot counts, successes, failures, and skipped indices for the lifetime of one call.

### Task 3: Knockdown verification and retry

**Files:**
- Modify: `car_task_function.py:898-1002`

- [x] Align the exact logical target rank among currently standing targets and retry recoverable association/alignment failures without consuming a shot.
- [x] Before each shot capture an ordered standing-target snapshot; after settling, require three stable frames showing only the selected target removed while other standing targets retain their positions/order.
- [x] If the selected target remains, reacquire the same logical index and retry up to four total shots.
- [x] If verification stays ambiguous, or four shots do not knock the target down, record a categorized target failure and continue the remaining plan.

### Task 4: Structured telemetry and verification

**Files:**
- Modify: `car_task_function.py:898-1002`

- [x] Emit `[TARGET_SHOOTING]` events for calibration, association, alignment, firing, verification, target outcome, and final summary with target index, attempt, ordered dx values, mapping, elapsed time, and reason.
- [x] Preserve the configured final course-distance compensation and debug return behavior even when individual targets fail.
- [x] Run `python -m py_compile car_task_function.py car_wrap_2026.py` and inspect `git diff --check` plus the focused diff. Do not add test files per user request.
