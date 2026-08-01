# Vertical Motion State Reset Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reset all per-command vertical-motion state and pace the control loop at 50 ms so stale counters and CPU-speed-dependent stop detection do not prematurely terminate a new command.

**Architecture:** Keep the change local to `ArmController.move_y_position()`. Recreate both `CountRecord` instances, establish a fresh position-difference baseline before setting the target, and sleep only on loop iterations that have not reached either exit condition.

**Tech Stack:** Python 3.8-compatible source, `unittest`, Python `ast` module.

---

### Task 1: Strengthen the vertical-motion regression test

**Files:**
- Modify: `tests/test_vertical_motion_state.py`
- Test: `tests/test_vertical_motion_state.py`

- [ ] **Step 1: Replace broad attribute checks with exact AST assertions**

Add helpers that locate assignments in `move_y_position()`, then assert:

```python
self.y_stop_flag = CountRecord(10)
self.y_pid_flag = CountRecord(5)
self.y_pose_now = self.y_get_position()
self.y_pose_last = self.y_pose_now
self.y_distance_change = 0
```

Also assert that the method's `while` body contains `time.sleep(0.05)` after both exit checks.

- [ ] **Step 2: Run the test and verify RED**

Run:

```powershell
& 'C:\weizijian\documents\baidu\2026_baidu_car\2026\baidu_smartcar_2026\.venv\Scripts\python.exe' -m unittest tests.test_vertical_motion_state
```

Expected: `FAIL` because the five resets are still comments and the loop has no active `time.sleep(0.05)`.

### Task 2: Implement the scoped reset and loop pacing

**Files:**
- Modify: `smartcar/whalesbot/vehicle/arm/arm_base.py:191-224`
- Test: `tests/test_vertical_motion_state.py`

- [ ] **Step 1: Activate and document the five reset assignments**

At the start of `move_y_position()`, before `self.y_pid.setpoint = target`, add the exact five assignments from Task 1. Place a Chinese comment before each assignment that states its internal reset value and control purpose.

- [ ] **Step 2: Add a vertical-loop-only 50 ms delay**

After the two exit checks in the `while True` body, add:

```python
# 每个未完成循环额外等待 50 ms，让步进电机执行命令并更新步数位置。
time.sleep(0.05)
```

- [ ] **Step 3: Run the focused test and verify GREEN**

Run the Task 1 command.

Expected: `Ran 1 test` and `OK`.

- [ ] **Step 4: Run syntax and patch checks**

Run:

```powershell
& 'C:\weizijian\documents\baidu\2026_baidu_car\2026\baidu_smartcar_2026\.venv\Scripts\python.exe' -m py_compile smartcar/whalesbot/vehicle/arm/arm_base.py tests/test_vertical_motion_state.py
git diff --check
```

Expected: both commands exit 0. Existing CRLF conversion warnings from Git are informational.

### Task 3: Review the resulting control behavior

**Files:**
- Review: `smartcar/whalesbot/vehicle/arm/arm_base.py`
- Review: `tests/test_vertical_motion_state.py`

- [ ] **Step 1: Inspect the final diff**

Confirm the diff changes only the vertical-state regression test, the five resets, the per-loop delay, and the approved design/plan documentation. Do not modify global `CountRecord`, thresholds, PID parameters, or runtime logging.

- [ ] **Step 2: Report exact reset values and verification evidence**

Report all five assignments, the earliest approximately 0.5 second stop window plus loop overhead, focused test results, and the limitation that open-loop lost steps are not corrected by this change.
