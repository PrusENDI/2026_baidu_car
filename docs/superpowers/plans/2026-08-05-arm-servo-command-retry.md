# Arm Servo Command Retry Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add lane-test-style arm-servo retries and fresh-response handling to the 7-17 worktree while preserving its angle calibration and X/Y reset code.

**Architecture:** `ArmController.set_arm_angle()` owns retry policy and state updates. `ServoBus_2.set_angle()` exposes the current MC602 response; no reset sequencing or configuration values change.

**Tech Stack:** Python 3, MC602 serial controller wrappers

---

### Task 1: Expose the current servo response

**Files:**
- Modify: `smartcar/whalesbot/vehicle/base/mc602_ctl2.py:545-547`

- [ ] **Step 1: Return only the current arm-servo command response**

Replace the 7-17 implementation with:

```python
def set_angle(self, angle, speed=100):
    # 只接受本次舵机命令对应的新响应，避免沿用上一条命令的缓存结果。
    self.last_data = None
    return self.act_mode(1, speed, angle, mode=2)
```

### Task 2: Add arm-servo retries without changing calibration

**Files:**
- Modify: `smartcar/whalesbot/vehicle/arm/arm_base.py:32-40`
- Modify: `smartcar/whalesbot/vehicle/arm/arm_base.py:622-637`

- [ ] **Step 1: Add retry constants independent of X/Y constants**

```python
ARM_SERVO_COMMAND_RETRIES = 3
ARM_SERVO_COMMAND_RETRY_DELAY = 0.5
```

- [ ] **Step 2: Replace only `set_arm_angle()`**

```python
def set_arm_angle(self, angle: Union[str, int] = "RIGHT", speed=60):
    requested_angle = angle
    target_side = angle if isinstance(angle, str) else None
    resolved_angle = angle
    if isinstance(resolved_angle, str):
        assert resolved_angle in ("LEFT", "MID", "RIGHT"), "Direction should be LEFT, MID, or RIGHT"
        resolved_angle = self.hand_angle_list[resolved_angle]

    response_received = False
    last_response = None
    for attempt in range(1, ARM_SERVO_COMMAND_RETRIES + 1):
        result = self.arm_servo.set_angle(resolved_angle, speed)
        if result is not None:
            response_received = True
            last_response = result
            logger.info(
                f"机械臂翻转指令已获新响应 requested={requested_angle}, "
                f"resolved={resolved_angle}, speed={speed}, "
                f"attempt={attempt}/{ARM_SERVO_COMMAND_RETRIES}, response={result!r}"
            )
        else:
            logger.warning(
                f"机械臂翻转指令无新响应 requested={requested_angle}, "
                f"resolved={resolved_angle}, speed={speed}, "
                f"attempt={attempt}/{ARM_SERVO_COMMAND_RETRIES}"
            )
        if attempt < ARM_SERVO_COMMAND_RETRIES:
            time.sleep(ARM_SERVO_COMMAND_RETRY_DELAY)

    if response_received:
        if target_side is not None:
            self.side = target_side
        self._arm_angle_last = resolved_angle
        logger.info(
            f"机械臂翻转重复发送完成 requested={requested_angle}, "
            f"resolved={resolved_angle}, speed={speed}, "
            f"attempts={ARM_SERVO_COMMAND_RETRIES}, last_response={last_response!r}"
        )
        return True

    logger.error(
        f"机械臂翻转指令连续无新响应 requested={requested_angle}, "
        f"resolved={resolved_angle}, speed={speed}, attempts={ARM_SERVO_COMMAND_RETRIES}"
    )
    return False
```

### Task 3: Verify scope without running tests

**Files:**
- Verify: `smartcar/whalesbot/vehicle/arm/arm_base.py`
- Verify: `smartcar/whalesbot/vehicle/base/mc602_ctl2.py`
- Verify unchanged: `smartcar/whalesbot/vehicle/arm/arm_cfg.yaml`

- [ ] **Step 1: Check whitespace errors and exact diff**

Run: `git diff --check` and `git diff -- smartcar/whalesbot/vehicle/arm/arm_base.py smartcar/whalesbot/vehicle/base/mc602_ctl2.py smartcar/whalesbot/vehicle/arm/arm_cfg.yaml`

Expected: only two retry constants, `set_arm_angle()`, and `ServoBus_2.set_angle()` change; `arm_cfg.yaml` has no diff.

- [ ] **Step 2: Confirm preserved calibration and reset implementation**

Run read-only searches for `LEFT`, `MID`, `RIGHT`, `reset_position`, `reset_x`, and Y reset methods.

Expected: `LEFT=92`, `MID=0`, `RIGHT=-97`; no X/Y reset code changes.

No automated tests are added or run, per the user's explicit instruction.
