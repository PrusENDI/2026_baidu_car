# MC602 Disconnect Recovery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Stop the task safely when MC602 becomes unavailable, wait for the controller to recover, and require a new key press before restarting.

**Architecture:** Add a small testable failure tracker to the serial layer and raise a dedicated disconnect exception after consecutive failed replies. The task entry catches that exception and performs best-effort safe cleanup; the Bash supervisor uses bounded MC602 initialization checks and returns to the device/key-wait loop after a disconnect exit.

**Tech Stack:** Python 3, pyserial, Bash, systemd, unittest/pytest.

---

### Task 1: Add failing communication-health tests

**Files:**
- Create: `tests/test_mc602_disconnect.py`
- Create: `smartcar/whalesbot/vehicle/base/controller_health.py`

- [ ] **Step 1: Write tests for consecutive-failure behavior**

```python
from smartcar.whalesbot.vehicle.base.controller_health import CommunicationHealth

def test_three_failures_raise_disconnect_and_success_resets():
    health = CommunicationHealth(failure_limit=3)
    health.record_success()
    health.record_failure()
    health.record_failure()
    assert health.record_failure() is True
    health.record_success()
    assert health.record_failure() is False
```

- [ ] **Step 2: Run the focused test and verify it fails because the helper is absent**

Run: `python -m pytest tests/test_mc602_disconnect.py -q`

Expected: collection failure mentioning missing `controller_health` or `CommunicationHealth`.

- [ ] **Step 3: Implement only the minimal tracker**

Implement `CommunicationHealth.record_failure()` to increment a counter and return `True` exactly when the configured limit is reached; `record_success()` resets the counter.

- [ ] **Step 4: Run the focused test and verify it passes**

Run: `python -m pytest tests/test_mc602_disconnect.py -q`

Expected: `1 passed`.

### Task 2: Integrate disconnect detection into SerialWrap

**Files:**
- Modify: `smartcar/whalesbot/vehicle/base/serial_wrap.py`
- Test: `tests/test_mc602_disconnect.py`

- [ ] **Step 1: Add a failing test for `None` replies**

Exercise an object created without `SerialWrap.__init__`, stub `reset_buffer`, `dev.send_cmd`, and `dev.get_anwser`, then assert two failed replies return `None` and the threshold reply raises `ControllerDisconnected`; assert a later successful reply clears the failure state.

- [ ] **Step 2: Run the focused test and verify the expected failure**

Run: `python -m pytest tests/test_mc602_disconnect.py -q`

Expected: failure because `ControllerDisconnected` and health integration do not exist.

- [ ] **Step 3: Implement the minimal integration**

Define `ControllerDisconnected`, create `self.communication_health` after the serial lock, record failures for exceptions and `None` replies, record success for valid replies, and raise only at the configured limit. Use `try/finally` around lock release. Keep the existing one-transaction lock.

- [ ] **Step 4: Run focused and syntax checks**

Run: `python -m pytest tests/test_mc602_disconnect.py -q` and `python -m py_compile smartcar/whalesbot/vehicle/base/serial_wrap.py smartcar/whalesbot/vehicle/base/controller_health.py`.

Expected: tests pass and compilation exits 0.

### Task 3: Make task entry safely terminate on controller loss

**Files:**
- Modify: `car_start_2026.py`
- Test: `tests/test_mc602_disconnect.py`

- [ ] **Step 1: Add a failing source-level behavior test**

Assert the task entry imports `ControllerDisconnected`, catches it around `main()` task execution, calls the car stop/cleanup path, and exits with a dedicated `MC602_DISCONNECTED_EXIT` code.

- [ ] **Step 2: Run the test and verify it fails**

Run: `python -m pytest tests/test_mc602_disconnect.py -q`

Expected: assertion failure because the entry currently calls `main()` directly without a disconnect handler.

- [ ] **Step 3: Implement minimal safe cleanup**

Wrap the existing task sequence in `try/except ControllerDisconnected`; keep a `my_car` reference once initialized; on disconnect set stop state, call `stop()` and `close()`/resource cleanup best-effort while suppressing secondary communication errors; return the dedicated exit code. Preserve normal task behavior and existing dirty user edits.

- [ ] **Step 4: Run focused tests and compile**

Run: `python -m pytest tests/test_mc602_disconnect.py -q` and `python -m py_compile car_start_2026.py`.

Expected: all focused tests pass.

### Task 4: Bound MC602 startup and recover in supervisor

**Files:**
- Modify: `scripts/wait_for_start_key.py`
- Modify: `scripts/start_with_key.sh`
- Test: `tests/test_key_start_helpers.py`

- [ ] **Step 1: Add failing helper/supervisor assertions**

Assert the listener supports a `--check-only` mode and the supervisor invokes MC602 readiness with a bounded `timeout`, recognizes the dedicated disconnect exit code, and returns to the MC602 wait before key listening.

- [ ] **Step 2: Run tests and verify they fail**

Run: `python -m pytest tests/test_key_start_helpers.py -q`

Expected: failures for the missing check mode/timeout/disconnect handling.

- [ ] **Step 3: Implement bounded readiness and recovery**

Add `--check-only` to initialize `Key4Btn(5)` and exit immediately. In Bash, run it through GNU `timeout` with a short configurable limit; retry on timeout or nonzero status. After the main process exits with `MC602_DISCONNECTED_EXIT`, log the condition, wait for MC602 readiness, then return to the key-wait loop without auto-starting.

- [ ] **Step 4: Run focused tests and shell/Python syntax checks**

Run: `python -m pytest tests/test_key_start_helpers.py -q`, `python -m py_compile scripts/wait_for_start_key.py`, and `bash -n scripts/start_with_key.sh` (on Orin).

Expected: all tests pass and syntax checks exit 0.

### Task 5: Final verification and commit

**Files:**
- Modify only files listed above; preserve unrelated dirty files.

- [ ] **Step 1: Run the focused regression suite**

Run: `python -m pytest tests/test_mc602_disconnect.py tests/test_key_start_helpers.py -q`.

- [ ] **Step 2: Run `git diff --check` and compile changed Python files**

Expected: no whitespace errors and compilation succeeds.

- [ ] **Step 3: Review the diff for unrelated changes and commit**

```bash
git add smartcar/whalesbot/vehicle/base/controller_health.py smartcar/whalesbot/vehicle/base/serial_wrap.py car_start_2026.py scripts/wait_for_start_key.py scripts/start_with_key.sh tests/test_mc602_disconnect.py tests/test_key_start_helpers.py
git commit -m "fix: recover from MC602 disconnects"
```

