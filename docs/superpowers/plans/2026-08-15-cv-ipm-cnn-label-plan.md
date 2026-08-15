# CV IPM CNN Label Implementation Plan

> **For agentic workers:** Implement inline in the current branch; the user explicitly requested no red/green cycle.

**Goal:** Replace perspective-image residual steering with equivalent-IPM lane fitting and split PID-before CNN labels from post-PID chassis commands.

**Architecture:** `opencv_lane.py` owns single-frame geometry and label mapping. `pid_control.py` consumes those labels with the production PID parameters. `ssh_test.py` records labels and applied controls under separate schemas.

**Tech Stack:** Python, NumPy, OpenCV, existing PID helper, unittest/pytest.

---

### Task 1: Equivalent-IPM lane fit

- [x] Recover a center from either or both standard-relative boundaries.
- [x] Fit a quadratic in perspective coordinates across the complete ROI.
- [x] Export normalized lateral displacement, tangent heading, curvature, and fit diagnostics.

### Task 2: Production-compatible CV PID

- [x] Use the exact production lateral and angle PID gains and limits.
- [x] Remove fixed steering-onset distance and pre-PID history filtering.
- [x] Keep adaptive speed and post-PID angular slew limiting.

### Task 3: Training record contract

- [x] Write PID-before errors to `state[1:3]`.
- [x] Write applied mecanum commands to `control[1:3]`.
- [x] Update metadata and the operation guide.

### Task 4: Verification

- [x] Run focused tests without a red/green cycle.
- [x] Replay `lap_001` and inspect correlation, false steering, invalid holds, and selected straight frames.
- [x] Review the final diff and preserve unrelated untracked data.
