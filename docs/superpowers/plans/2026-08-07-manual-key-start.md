# Manual Key Start Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task with review checkpoints.

**Goal:** Require physical controller button 1 before each default-task run and allow button 3 to stop the current run so a later button 1 can start it again.

**Architecture:** Keep one controller reader in `MyCar.key_thread_func`; that thread sets a `threading.Event` when it sees key 1 and sets `_stop_flag` for key 3. The startup entry loops around the existing default-task call, clearing the event and resetting `_stop_flag` before each wait. No timeout, recovery protocol, menu, or legacy driver is added.

**Tech Stack:** Python 3, `threading.Event`, existing `unittest` tests.

---

### Task 1: Add the failing start-gate tests

**Files:**
- Create: `tests/test_manual_key_gate.py`

- [ ] **Step 1: Write tests for key filtering and blocking**

  Add a fake event and test a helper named `wait_for_start_key(event)`: it must call `event.wait()` and then `event.clear()`. Also assert the startup entry calls this helper before its default task call and the key thread sets the event only for key 1.

- [ ] **Step 2: Run the focused test and verify RED**

  Run: `python -m unittest tests.test_manual_key_gate -v`

  Expected: FAIL because the helper is not yet present.

### Task 2: Implement the minimal event-based gate

**Files:**
- Modify: `car_wrap_2026.py:340-348,647-665`
- Modify: `car_start_2026.py:20-35`
- Create: `manual_start.py`

- [ ] **Step 1: Add the start event to `MyCar` initialization**

  Import/use `threading.Event`, create `self._start_key_event = threading.Event()` beside the existing key-thread flags, and leave the existing thread startup unchanged.

- [ ] **Step 2: Set the event when key 1 is read**

  In `key_thread_func`, after `key_val = self.key.get_key()`, add `if key_val == 1: self._start_key_event.set()`. Preserve the existing key-3 stop behavior and polling interval.

- [ ] **Step 3: Expose a small wait method**

  Add `manual_start.py` with a dependency-free `wait_for_start_key(event)` helper that calls `event.wait()` and then `event.clear()`, matching the old blocking behavior without a second controller read loop.

- [ ] **Step 4: Gate the existing default task**

  In `car_start_2026.py`, after `init()` and immediately before the current default-task call, call `wait_for_start_key(my_car._start_key_event)` (or the equivalent existing `MyCar` reference established by `init()`). Do not alter task order, parameters, or task implementations.

- [ ] **Step 5: Run the focused tests and verify GREEN**

  Run: `python -m unittest tests.test_manual_key_gate -v`

  Expected: PASS.

### Task 3: Regression verification

**Files:**
- Test: `tests/test_manual_key_gate.py`, `tests/test_manual_start_contract.py`

- [ ] **Step 1: Run all repository tests**

  Run: `python -m unittest discover -s tests -v`

  Expected: all tests pass.

- [ ] **Step 2: Run syntax compilation for changed Python files**

  Run: `python -m py_compile car_start_2026.py car_wrap_2026.py tests/test_manual_key_gate.py`

  Expected: exit code 0 and no syntax errors.

- [ ] **Step 3: Check the final diff**

  Run: `git diff --check`

  Expected: no whitespace errors.
