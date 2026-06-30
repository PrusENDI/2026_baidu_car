# Chassis Pad Control Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a standalone Bluetooth gamepad file for chassis-only testing.

**Architecture:** Keep hardware imports inside runtime code so pure mapping tests can run locally. Reuse the existing `BluetoothPad` and `MecanumDriver` classes from the project.

**Tech Stack:** Python stdlib unittest, existing `smartcar.whalesbot.vehicle` hardware APIs.

---

### Task 1: Mapping Tests

**Files:**
- Create: `tests/test_chassis_pad_control.py`

- [ ] Add tests for disconnected pad detection, L1+L2 exit detection, default pad-to-velocity mapping, and low-speed override mapping.
- [ ] Run `python tests/test_chassis_pad_control.py` and confirm it fails because `smartcar.whalesbot.tools.chassis_pad_control` does not exist yet.

### Task 2: Standalone Tool

**Files:**
- Create: `smartcar/whalesbot/tools/chassis_pad_control.py`

- [ ] Add pure mapping helpers.
- [ ] Add a `ChassisPadController` that initializes only `MecanumDriver` and `BluetoothPad`.
- [ ] Add a command-line entrypoint with speed scale parameters and a loop delay parameter.
- [ ] Ensure every exit path calls `set_velocity(0, 0, 0)`.

### Task 3: Verification

**Files:**
- Verify: `tests/test_chassis_pad_control.py`
- Verify: `smartcar/whalesbot/tools/chassis_pad_control.py`

- [ ] Run `python tests/test_chassis_pad_control.py`.
- [ ] Run `python -m py_compile smartcar/whalesbot/tools/chassis_pad_control.py tests/test_chassis_pad_control.py`.
- [ ] Run `powershell -NoProfile -ExecutionPolicy Bypass -File .\tools\sync_to_orin.ps1 -Files smartcar/whalesbot/tools/chassis_pad_control.py,tests/test_chassis_pad_control.py -PlanOnly`.
