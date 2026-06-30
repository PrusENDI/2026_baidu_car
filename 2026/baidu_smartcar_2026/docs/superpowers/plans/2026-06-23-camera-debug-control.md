# Camera Debug Control Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a standalone camera debug tool that streams cam1 and cam2 without starting chassis or collection code.

**Architecture:** Keep argument parsing and camera configuration pure and testable. Import `Camera` and `Streamer` only in runtime entrypoint.

**Tech Stack:** Python stdlib unittest, existing `smartcar.whalesbot.tools.Camera`, existing `smartcar.whalesbot.tools.Streamer`.

---

### Task 1: Parser Tests

**Files:**
- Create: `tests/test_camera_debug_control.py`

- [ ] Test parsing `320x240`.
- [ ] Test default camera specs.
- [ ] Test single camera selection.
- [ ] Confirm test fails because `camera_debug_control.py` does not exist.

### Task 2: Camera Debug Tool

**Files:**
- Create: `smartcar/whalesbot/tools/camera_debug_control.py`

- [ ] Add pure helpers for parsing sizes and selecting camera specs.
- [ ] Add `CameraDebugRunner` with injectable camera and streamer factories.
- [ ] Add CLI options for camera indices, sizes, single-camera mode, stream port, fps, and loop delay.
- [ ] Close cameras and streamer on exit.

### Task 3: Verification

**Files:**
- Verify: `tests/test_camera_debug_control.py`
- Verify: `smartcar/whalesbot/tools/camera_debug_control.py`

- [ ] Run local unittest.
- [ ] Run no-pyc syntax compile.
- [ ] Run explicit sync plan preview.
