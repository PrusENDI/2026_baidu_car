# X Limit Switch Homing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace X-axis stall-as-success homing with one startup-only AI1 micro-switch homing operation, then use an encoder-based `x=0.015 m` safe retract with switch-release confirmation throughout task execution.

**Architecture:** `ArmController` owns the AI1 sensor, validated homing configuration, fresh-sample reads, the mechanical homing state machine, and a separate safe-retract operation. `reset_position()` propagates X failures and chooses between startup homing and later safe retract, while `car_task_function.py` stops calling `reset_x()` during normal tasks.

**Tech Stack:** Python 3, PyYAML, existing MC602 `AnalogInput2`/`sensor_analog_a` protocol path, existing `MotorWrap`, `PID`, and `CountRecord` helpers.

---

### Task 1: Add and validate X homing configuration

**Files:**
- Modify: `smartcar/whalesbot/vehicle/arm/arm_cfg.yaml`
- Modify: `smartcar/whalesbot/vehicle/arm/arm_base.py`

- [ ] **Step 1: Add the confirmed AI1 homing configuration**

Add this mapping below `horiz_cfg` without changing existing motor, PID, threshold, or hand calibration values:

```yaml
  homing:
    limit_port: 1
    active: above
    threshold: 2000
    stable_samples: 5
    speed: 0.06
    timeout: 8.0
    safe_position: 0.015
```

- [ ] **Step 2: Initialize and validate the sensor configuration**

Import `AnalogInput2`, extend `x_params_init()` with `homing`, and reject invalid port, active direction, non-finite threshold, non-positive sample count/speed/timeout, speed outside the configured motor range, or a safe position outside the X travel range. Store the validated values, construct only `AnalogInput2(limit_port)`, and initialize `x_zero_valid = False` regardless of persisted pose configuration.

- [ ] **Step 3: Add fresh sample and classification helpers**

Add `_read_x_limit_fresh()` that clears `self.x_limit_sensor.sensor_2.last_data`, calls the high-level `AnalogInput2.read()`, rejects a missing fresh response and non-finite values, and returns the raw float. Add `_x_limit_is_triggered(raw)` using `>= threshold` for `above` and `<= threshold` for `below`.

- [ ] **Step 4: Commit only the configuration/foundation files**

```powershell
git add -- smartcar/whalesbot/vehicle/arm/arm_cfg.yaml smartcar/whalesbot/vehicle/arm/arm_base.py
git commit -m "feat: configure X limit switch homing"
```

### Task 2: Implement mechanical homing and safe retract

**Files:**
- Modify: `smartcar/whalesbot/vehicle/arm/arm_base.py`

- [ ] **Step 1: Replace the `reset_x()` stall-success state machine**

Change the signature to `reset_x(self, out_time=None, speed_limit=None) -> bool`, so omitted values come from YAML while existing explicit calls remain compatible. The method must:

1. Stop X immediately and invalidate `x_zero_valid`.
2. Validate runtime overrides before any motion.
3. Read exactly `stable_samples` fresh released samples without moving; any trigger, timeout, read error, or unstable sample fails.
4. Command constant negative speed and sample AI1 while updating encoder deltas.
5. Clear the trigger count on every released sample and succeed only after `stable_samples` consecutive triggered samples.
6. Treat encoder stall, timeout, serial/sensor error, and unexpected exceptions as failure without changing `x_pose_start`.
7. On stable trigger only, stop X, set `x_pose_start` from the motor encoder, clear X position/delta state, set `x_zero_valid = True`, and return `True`.
8. In `finally`, stop X and restore the ordinary PID output limits and PID state.

- [ ] **Step 2: Add `retract_x_safe()`**

Implement `retract_x_safe(self, out_time=6.0) -> bool`. It must stop and fail without motion when `x_zero_valid` is false, call `move_x_position(self.x_safe_position, out_time=out_time)`, then require `stable_samples` consecutive fresh released AI1 samples. It must not modify `x_pose_start`; movement, sensor, or serial failure returns `False`, and `finally` always calls `x_speed(0)`.

- [ ] **Step 3: Commit the homing state machines**

```powershell
git add -- smartcar/whalesbot/vehicle/arm/arm_base.py
git commit -m "feat: home X axis with AI1 limit switch"
```

### Task 3: Propagate reset failures and leave the switch released

**Files:**
- Modify: `smartcar/whalesbot/vehicle/arm/arm_base.py`

- [ ] **Step 1: Extend `reset_position()` without breaking no-argument callers**

Change the signature to `reset_position(self, rehome_x=True)`. Run the existing Y reset thread as before. In the X thread, call `reset_x()` followed by `retract_x_safe()` when `rehome_x=True`, or call only `retract_x_safe()` when false. Capture the Boolean result and any exception in thread-owned result state.

- [ ] **Step 2: Stop initialization on X failure**

After both threads join, if the X operation did not return true, call `x_speed(0)` and raise `RuntimeError` before saving positions or continuing. Remove `self.x = 0`, because its property setter sends a real move back to the switch; retain the existing Y behavior and save the encoder-derived safe X position only after success.

- [ ] **Step 3: Commit reset orchestration**

```powershell
git add -- smartcar/whalesbot/vehicle/arm/arm_base.py
git commit -m "fix: propagate X homing failures"
```

### Task 4: Use one-time homing in task orchestration

**Files:**
- Modify: `car_task_function.py`

- [ ] **Step 1: Preserve the single startup mechanical home**

Leave `init()` calling the no-argument `my_car.arm.reset_position()`. This is the sole normal task entry that performs mechanical X homing.

- [ ] **Step 2: Replace normal-task `reset_x()` calls**

Replace every direct `my_car.arm.reset_x()` in seeding, water tower, crop harvesting, and sort/store paths with `my_car.arm.retract_x_safe()`. Preserve each existing failure boundary and update messages/comments from “重新机械归零” to “返回 1.5 cm 安全位置并确认开关释放.”

In the water-tower rotation helper, remove the preliminary unchecked `move_x_position(0)` and use only the checked `retract_x_safe()` before rotation.

- [ ] **Step 3: Avoid the second mechanical home in `get_order()`**

Change its call to:

```python
my_car.arm.reset_position(rehome_x=False)
```

This preserves Y reset while requiring X to return to the existing 1.5 cm safe position.

- [ ] **Step 4: Update explicit retraction-only targets**

Change only calls whose comments and control flow explicitly mean “task end retract” or “retract before rotation” to `retract_x_safe()` with checked failure handling. Do not alter `x=0` targets used as crop/ball/goods placement coordinates, and preserve special collision-avoidance positions such as `x=0.20 m`.

- [ ] **Step 5: Commit only the task orchestration file**

```powershell
git add -- car_task_function.py
git commit -m "refactor: reuse X zero during task execution"
```

### Task 5: Perform non-hardware static verification

**Files:**
- Inspect: `smartcar/whalesbot/vehicle/arm/arm_cfg.yaml`
- Inspect: `smartcar/whalesbot/vehicle/arm/arm_base.py`
- Inspect: `car_task_function.py`

- [ ] **Step 1: Check exact diffs and whitespace**

```powershell
git diff --check HEAD~3..HEAD
git show --stat --oneline HEAD~3..HEAD
git status --short
```

Expected: only the intended files are committed by this implementation; all unrelated pre-existing working-tree changes remain present.

- [ ] **Step 2: Run static syntax compilation without importing hardware modules**

```powershell
python -m py_compile smartcar/whalesbot/vehicle/arm/arm_base.py car_task_function.py
```

Expected: exit code 0. This parses/compiles source only and sends no hardware command.

- [ ] **Step 3: Audit safety-sensitive call sites**

```powershell
rg -n "reset_x\(|retract_x_safe\(|reset_position\(" car_task_function.py smartcar/whalesbot/vehicle/arm/arm_base.py
rg -n "x_pose_start\s*=" smartcar/whalesbot/vehicle/arm/arm_base.py
```

Expected: task code has no direct `reset_x()` calls; startup keeps no-argument `reset_position()`; `get_order()` uses `rehome_x=False`; only initialization/pose restoration and the stable-switch success branch assign `x_pose_start`.

> Per the user's explicit instruction, this implementation plan adds no tests and performs no red/green TDD cycle. It does not run the diagnostic reader, real switch reads, motor homing, or any `move_x_position()` hardware action.

### Task 6: Stop on the first triggered sample before debounce

**Files:**
- Modify: `smartcar/whalesbot/vehicle/arm/arm_base.py:511-552`

- [ ] **Step 1: Separate moving detection from stopped confirmation**

In `reset_x()`, keep the initial negative speed command. During the movement loop, read AI1 before refreshing the negative speed. On the first triggered sample, immediately call `self.x_speed(0)`, count that sample as the first stable sample, and leave the movement loop. Only released samples may refresh the negative speed and participate in encoder-stall detection.

```python
self.x_speed(-homing_speed)
triggered_samples = 0
while True:
    if time.time() > end_time:
        logger.error(
            f"水平轴寻零超时 actual={self.x_get_position():.6f}"
        )
        return False

    raw = self._read_x_limit_fresh()
    if self._x_limit_is_triggered(raw):
        self.x_speed(0)
        triggered_samples = 1
        break

    self.x_speed(-homing_speed)
    self.x_pose_now = self.x_get_position()
    self.x_distance_change = self.x_pose_now - self.x_pose_last
    self.x_pose_last = self.x_pose_now
    if self.x_stop_check():
        logger.error(
            f"X 轴归零期间编码器停滞，未建立零点 "
            f"actual={self.x_get_position():.6f}, raw={raw:.1f}"
        )
        return False
    time.sleep(0.05)
```

- [ ] **Step 2: Confirm the remaining samples while stopped**

After leaving the movement loop, keep X stopped and collect the remaining samples at the existing 20 ms debounce interval. A timeout, read error, or released sample returns `False`; a released sample must not restart the motor.

```python
while triggered_samples < self.x_limit_stable_samples:
    if time.time() > end_time:
        logger.error("X 轴触发后稳定确认超时，未建立零点")
        return False
    time.sleep(0.02)
    raw = self._read_x_limit_fresh()
    if not self._x_limit_is_triggered(raw):
        logger.error(
            f"X 轴触发后信号回落，未建立零点 "
            f"port=AI{self.x_limit_port}, raw={raw:.1f}, "
            f"stable_samples={triggered_samples}/"
            f"{self.x_limit_stable_samples}"
        )
        return False
    triggered_samples += 1
```

Leave the existing success branch after this loop: read the stopped encoder position, update `x_pose_start`, clear X position state, set `x_zero_valid = True`, and return `True`. Retain the unconditional `x_speed(0)` in `finally`.

- [ ] **Step 3: Perform static verification only**

Run:

```powershell
git diff --check -- smartcar/whalesbot/vehicle/arm/arm_base.py
python -m py_compile smartcar/whalesbot/vehicle/arm/arm_base.py
```

Expected: both commands exit with code 0. Do not import the hardware stack, run the diagnostic reader, or execute any real motor or homing command.
