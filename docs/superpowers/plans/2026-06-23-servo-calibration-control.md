# Servo Calibration Control Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an interactive servo calibration tool for PWM and RS485 bus servos.

**Architecture:** Put pure parsing and state logic in `smartcar/whalesbot/tools/servo_calibration_control.py`, and delay imports of `ServoPwm` and `ServoBus` until runtime. Use standard library `unittest` for local tests.

**Tech Stack:** Python stdlib, existing `smartcar.whalesbot.vehicle.ServoPwm`, existing `smartcar.whalesbot.vehicle.ServoBus`.

---

### Task 1: Local Command Tests

**Files:**
- Create: `tests/test_servo_calibration_control.py`

- [ ] Add tests for home-list parsing, servo selection, angle stepping, center command, and home-all dispatch.
- [ ] Run `.\.venv\Scripts\python.exe tests\test_servo_calibration_control.py` and confirm it fails because the module does not exist yet.

### Task 2: Calibration Tool

**Files:**
- Create: `smartcar/whalesbot/tools/servo_calibration_control.py`

- [ ] Add pure helpers for parsing `port:angle` lists.
- [ ] Add a `ServoCalibrationSession` that accepts injected PWM and bus factory callables.
- [ ] Add commands: `pwm`, `bus`, `angle`, `speed`, `step`, `center`, `home-all`, `sweep`, `help`, and `quit`.
- [ ] Add a CLI entrypoint that imports hardware classes only in `main()`.

### Task 3: Verification

**Files:**
- Verify: `tests/test_servo_calibration_control.py`
- Verify: `smartcar/whalesbot/tools/servo_calibration_control.py`

- [ ] Run `.\.venv\Scripts\python.exe tests\test_servo_calibration_control.py`.
- [ ] Run no-pyc syntax compilation for the new tool and tests.
- [ ] Run sync plan preview with explicit `-Files`.
