# MC602 Servo Frame Length Hardware Test Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the raw-serial `testangle.py` experiment with a guarded MC602 hardware test that homes X, moves to 0.25 m, then sends exactly one RIGHT=-92° servo command using a selected one- or two-byte angle field.

**Architecture:** Keep protocol-frame construction and CLI parsing free of hardware imports so they can be unit tested on Windows. Put the physical sequence in an injected `run_test` function, and lazy-import `ArmController`, `StructData`, and controller state only from `main()` after CLI validation.

**Tech Stack:** Python 3, `argparse`, `struct`, project `ArmController`/MC602 `StructData`, pytest.

---

### Task 1: Specify the frame and guarded motion behavior

**Files:**
- Create: `tests/test_servo_frame_length_hardware_test.py`
- Test: `testangle.py`

- [ ] **Step 1: Write failing frame and CLI tests**

Add tests that import `testangle` without importing any `smartcar.whalesbot.vehicle` module, assert the exact angle8 and angle16 complete frames, and assert that `parse_args([])` plus unsupported choices fail with argparse exit code 2.

```python
def test_angle8_expected_frame():
    assert testangle.build_expected_frame(1) == bytes.fromhex(
        "77 68 0a 06 02 02 01 3c a4 0a"
    )


def test_angle16_expected_frame():
    assert testangle.build_expected_frame(2) == bytes.fromhex(
        "77 68 0b 06 02 02 01 3c a4 ff 0a"
    )
```

- [ ] **Step 2: Write failing orchestration tests**

Use fake arm, servo, confirmation, and struct factory objects to cover:

- non-MC602 rejection;
- homing failure without servo transmission;
- X=0.25 movement failure without servo transmission;
- operator cancellation without servo transmission;
- strict success order `reset_x -> move_x_position -> confirm -> one set_angle`;
- `x_speed(0)` on every return and exception path;
- selection of `bbbbb` for angle8 and `bbbbh` for angle16.

The success assertion must require exactly one call:

```python
assert arm.arm_servo.calls == [(-92, 60)]
assert events.index("confirm") < events.index(("set_angle", -92, 60))
assert arm.stop_calls == [0]
```

- [ ] **Step 3: Run tests to verify RED**

Run: `python -m pytest tests/test_servo_frame_length_hardware_test.py -q`

Expected: FAIL because `testangle.py` still opens `/dev/ttyUSB0` during import or because the new helper API does not exist.

### Task 2: Implement the safe hardware test script

**Files:**
- Modify: `testangle.py`
- Test: `tests/test_servo_frame_length_hardware_test.py`

- [ ] **Step 1: Add hardware-free frame construction and CLI parsing**

Define fixed test constants, a format map `{1: ("angle8", "bbbbb"), 2: ("angle16", "bbbbh")}`, `build_expected_frame(angle_bytes)`, and a required `--angle-bytes {1,2}` argument. Keep all project hardware imports out of module scope.

```python
def build_expected_frame(angle_bytes):
    if angle_bytes not in FRAME_FORMATS:
        raise ValueError(f"unsupported angle byte count: {angle_bytes}")
    angle_format = "b" if angle_bytes == 1 else "h"
    payload = struct.pack("<BBBBB", 0x06, 0x02, 2, 0x01, 60)
    payload += struct.pack(f"<{angle_format}", -92)
    return b"\x77\x68" + bytes([len(payload) + 4]) + payload + b"\x0a"
```

- [ ] **Step 2: Implement the injected guarded sequence**

Implement `run_test(angle_bytes, arm, struct_factory, controller_name, confirm=input, output=print)` with a `try/finally`. Validate MC602, call `reset_x()`, call `move_x_position(0.25)`, print actual X and expected TX, require the exact confirmation token `YES`, replace only `arm.arm_servo.servo_bus_2.data_struct`, call `arm.arm_servo.set_angle(-92, 60)` once, and print `last_data` plus final X. Every return or exception must execute `arm.x_speed(0)`; do not automatically retract X.

- [ ] **Step 3: Add lazy hardware loading and script entry point**

Import project hardware only inside `load_hardware()` and construct the arm in `main()`:

```python
def load_hardware():
    from smartcar.whalesbot.vehicle.arm.arm_base import ArmController
    from smartcar.whalesbot.vehicle.base.controller_wrap import serial_wrap
    from smartcar.whalesbot.vehicle.base.mc602_ctl2 import StructData

    return ArmController, StructData, serial_wrap.dev.name
```

Return `run_test(...)` from `main()` and use `raise SystemExit(main())` under the standard `__main__` guard.

- [ ] **Step 4: Run focused tests to verify GREEN**

Run: `python -m pytest tests/test_servo_frame_length_hardware_test.py -q`

Expected: all tests PASS without serial initialization or hardware access.

- [ ] **Step 5: Run syntax and import-isolation verification**

Run: `python -m py_compile testangle.py tests/test_servo_frame_length_hardware_test.py`

Run: `python -c "import sys, testangle; assert not any(name.startswith('smartcar.whalesbot.vehicle') for name in sys.modules)"`

Expected: both commands exit 0 with no serial scan.

- [ ] **Step 6: Commit the focused implementation**

```bash
git add testangle.py tests/test_servo_frame_length_hardware_test.py docs/superpowers/plans/2026-07-27-mc602-servo-frame-length-hardware-test.md
git commit -m "test: add guarded MC602 servo frame comparison"
```

### Task 3: Verify against the approved design

**Files:**
- Verify: `testangle.py`
- Verify: `tests/test_servo_frame_length_hardware_test.py`

- [ ] **Step 1: Re-run all relevant no-hardware tests**

Run: `python -m pytest tests/test_servo_frame_length_hardware_test.py tests/test_arm_reset_telemetry.py tests/test_horizontal_motion_safety.py -q`

Expected: all tests PASS. Do not execute `testangle.py` locally because that would initialize physical hardware.

- [ ] **Step 2: Check the diff for whitespace and scope**

Run: `git diff --check`

Run: `git status --short`

Expected: no whitespace errors; unrelated pre-existing user changes remain untouched.

