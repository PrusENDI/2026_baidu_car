# Shooting Functions Port Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Port the current shooting-detection and shooting behavior into `car_task_function_all_test.py` while removing detailed telemetry output and retaining final result prints.

**Architecture:** Copy the two shooting configuration dictionaries and their corresponding functions as one self-contained unit. Preserve target ordering, alignment, retry, knockdown verification, safety returns, and debug behavior; remove the nested shooting event emitter and all intermediate event calls.

**Tech Stack:** Python 3, `ast`, `unittest`

---

### Task 1: Define the structural contract

**Files:**
- Create: `tests/test_all_test_shooting_port.py`
- Test: `tests/test_all_test_shooting_port.py`

- [x] Add one AST-based test that requires both shooting configuration dictionaries, the requested function signatures, current alignment/verification keywords, no `shooting_event` calls, and exactly one final `print` in each function.
- [x] Run `uv run --offline --no-project python -m unittest tests.test_all_test_shooting_port -v` and confirm it fails because the target file still contains the legacy implementation.

### Task 2: Port the current shooting implementation

**Files:**
- Modify: `car_task_function_all_test.py`

- [x] Replace the legacy detection function with `target_shooting_detection(debug=False)` and add `TARGET_SHOOTING_DETECTION_POSES` directly above it.
- [x] Replace the legacy shooting function with `target_shooting(animal_list=[1, 1, 1, 0], debug=False, shooting_delta_x=None)` and add `TARGET_SHOOTING_POSES` directly above it.
- [x] Remove the nested detailed event emitter and all intermediate telemetry calls, retain one final result print per function, and add detailed Chinese comments around each processing stage.

### Task 3: Focused verification

**Files:**
- Test: `tests/test_all_test_shooting_port.py`

- [x] Run the single unittest and confirm it passes.
- [x] Compile `car_task_function_all_test.py` without importing hardware dependencies.
- [x] Inspect the final diff to confirm unrelated task flows were not changed.
