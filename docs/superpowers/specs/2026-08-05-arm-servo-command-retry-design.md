# Arm Servo Command Retry Design

## Goal

Bring the lane-test arm-servo command reliability behavior into the 7-17 worktree without changing its calibrated angles or its X/Y reset implementation.

## Scope

`ArmController.set_arm_angle()` will resolve symbolic directions through the existing 7-17 `hand_angle_list`, send the resolved angle three times with a 0.5-second interval, and treat any fresh MC602 response as command acceptance. It will update `side` and `_arm_angle_last` only after at least one response, returning `True` on acceptance and `False` after three unanswered attempts.

`ServoBus_2.set_angle()` will clear its cached response before sending and return the result of the existing `act_mode()` call. This makes the high-level response check observe only the current command. No other MC602 behavior changes.

## Preserved behavior

- `arm_cfg.yaml` remains unchanged: `LEFT=92`, `MID=0`, and `RIGHT=-97`.
- The default arm-servo speed remains `60`.
- `reset_position()`, `reset_x()`, Y-axis reset, cooperative scheduling, and all X/Y constants remain unchanged.
- No 1.5-second reset settle delay is imported from lane-test.
- Existing callers that ignore the return value remain compatible.

## Verification

Focused tests will cover symbolic-angle resolution using `RIGHT=-97`, three identical sends, two 0.5-second retry delays, success after any fresh response, failure after three `None` responses, and state updates only on success. A diff-scope check will verify that no angle configuration or X/Y reset code changed.
