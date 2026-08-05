# Key Start Supervisor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an Orin boot service that prewarms and watches the inference backend, waits for MC602 port 5 key value `12`, runs the existing full task program, and returns to key waiting after every run.

**Architecture:** A systemd unit starts a Bash supervisor. The supervisor keeps `infer_back_end.py` alive, uses a small ZMQ health probe to wait for all configured services, runs a separate hardware-key listener, then launches the unchanged `car_start_2026.py`. The key listener exits before the main program starts, so port 5 and the `MyCar` port 1 key instance never coexist in one Python process.

**Tech Stack:** Bash, Python 3, pyzmq, pytest, MC602 `Key4Btn`, systemd

---

## File map

- Create `scripts/inference_backend_probe.py`: bounded ZMQ readiness checks for ports 5001, 5002, and 5005.
- Create `scripts/wait_for_start_key.py`: lightweight port 5 key listener and screen feedback; never imports or constructs `MyCar`.
- Create `scripts/start_with_key.sh`: backend watchdog and repeated key-to-main lifecycle.
- Create `systemd/baidu-smart-key-start.service`: Orin boot integration.
- Create `tests/test_inference_backend_probe.py`: probe success, failure, cleanup, and retry behavior.
- Create `tests/test_wait_for_start_key.py`: key filtering, one-shot start, and invalid-value behavior.
- Create `tests/test_key_start_assets.py`: Shell syntax and systemd unit contract checks.
- Do not modify `car_start_2026.py`, `car_task_function.py`, or the MC602 driver for this feature.

Existing uncommitted Orin-synchronized files must remain outside every feature commit. Use path-limited `git add` commands exactly as shown below.

### Task 1: Bounded inference backend health probe

**Files:**
- Create: `scripts/inference_backend_probe.py`
- Create: `tests/test_inference_backend_probe.py`

- [ ] **Step 1: Write failing probe tests**

Create `tests/test_inference_backend_probe.py`:

```python
import importlib.util
import json
from pathlib import Path


MODULE_PATH = Path(__file__).parents[1] / "scripts" / "inference_backend_probe.py"


def load_module():
    spec = importlib.util.spec_from_file_location("inference_backend_probe", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeSocket:
    def __init__(self, response=b"true", error=None):
        self.response = response
        self.error = error
        self.options = {}
        self.connected = None
        self.sent = None
        self.closed_with = None

    def setsockopt(self, option, value):
        self.options[option] = value

    def connect(self, endpoint):
        self.connected = endpoint

    def send(self, payload):
        self.sent = payload
        if self.error:
            raise self.error

    def recv(self):
        if self.error:
            raise self.error
        return self.response

    def close(self, linger):
        self.closed_with = linger


class FakeContext:
    def __init__(self, socket):
        self.fake_socket = socket
        self.terminated = False

    def socket(self, socket_type):
        return self.fake_socket

    def term(self):
        self.terminated = True


def test_probe_service_accepts_only_json_true(monkeypatch):
    module = load_module()
    socket = FakeSocket(response=json.dumps(True).encode())
    context = FakeContext(socket)
    monkeypatch.setattr(module.zmq, "Context", lambda: context)

    assert module.probe_service(5001, timeout_ms=250) is True
    assert socket.connected == "tcp://127.0.0.1:5001"
    assert socket.sent == b"ATATA"
    assert socket.closed_with == 0
    assert context.terminated is True


def test_probe_service_returns_false_and_closes_on_zmq_error(monkeypatch):
    module = load_module()
    socket = FakeSocket(error=module.zmq.Again())
    context = FakeContext(socket)
    monkeypatch.setattr(module.zmq, "Context", lambda: context)

    assert module.probe_service(5002, timeout_ms=250) is False
    assert socket.closed_with == 0
    assert context.terminated is True


def test_all_services_ready_checks_every_configured_port():
    module = load_module()
    calls = []

    def probe(port, timeout_ms):
        calls.append((port, timeout_ms))
        return port != 5002

    assert module.all_services_ready(probe=probe, timeout_ms=300) is False
    assert calls == [(5001, 300), (5002, 300), (5005, 300)]


def test_wait_until_ready_retries_until_all_services_answer():
    module = load_module()
    attempts = iter([False, False, True])
    sleeps = []

    assert module.wait_until_ready(
        ready=lambda: next(attempts),
        sleep=sleeps.append,
        interval=0.2,
    ) is True
    assert sleeps == [0.2, 0.2]
```

- [ ] **Step 2: Run the tests and verify the expected failure**

Run:

```bash
python3 -m pytest tests/test_inference_backend_probe.py -q
```

Expected: FAIL because `scripts/inference_backend_probe.py` does not exist.

- [ ] **Step 3: Implement the health probe**

Create `scripts/inference_backend_probe.py`:

```python
#!/usr/bin/env python3
import argparse
import json
import time

import zmq


SERVICE_PORTS = (5001, 5002, 5005)


def probe_service(port, timeout_ms=1000):
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.setsockopt(zmq.LINGER, 0)
    socket.setsockopt(zmq.SNDTIMEO, timeout_ms)
    socket.setsockopt(zmq.RCVTIMEO, timeout_ms)
    socket.connect(f"tcp://127.0.0.1:{port}")
    try:
        socket.send(b"ATATA")
        return json.loads(socket.recv().decode("utf-8")) is True
    except (zmq.ZMQError, UnicodeDecodeError, json.JSONDecodeError):
        return False
    finally:
        socket.close(0)
        context.term()


def all_services_ready(probe=probe_service, timeout_ms=1000):
    results = [probe(port, timeout_ms) for port in SERVICE_PORTS]
    return all(results)


def wait_until_ready(ready, sleep=time.sleep, interval=1.0):
    while not ready():
        sleep(interval)
    return True


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wait", action="store_true")
    parser.add_argument("--timeout-ms", type=int, default=1000)
    parser.add_argument("--interval", type=float, default=1.0)
    return parser.parse_args()


def main():
    args = parse_args()
    ready = lambda: all_services_ready(timeout_ms=args.timeout_ms)
    if args.wait:
        wait_until_ready(ready=ready, interval=args.interval)
        return 0
    return 0 if ready() else 1


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run the focused tests**

Run:

```bash
python3 -m pytest tests/test_inference_backend_probe.py -q
```

Expected: `4 passed`.

- [ ] **Step 5: Commit the probe only**

```bash
git add scripts/inference_backend_probe.py tests/test_inference_backend_probe.py
git commit -m "feat: add inference backend readiness probe"
```

### Task 2: One-shot MC602 start-key listener

**Files:**
- Create: `scripts/wait_for_start_key.py`
- Create: `tests/test_wait_for_start_key.py`

- [ ] **Step 1: Write failing key-listener tests**

Create `tests/test_wait_for_start_key.py`:

```python
import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).parents[1] / "scripts" / "wait_for_start_key.py"


def load_module():
    spec = importlib.util.spec_from_file_location("wait_for_start_key", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class SequenceKey:
    def __init__(self, values):
        self.values = iter(values)
        self.read_count = 0

    def get_key(self):
        self.read_count += 1
        return next(self.values)


class RecordingDisplay:
    def __init__(self):
        self.messages = []

    def show(self, message):
        self.messages.append(message)


def test_wait_for_start_ignores_other_keys_and_starts_once():
    module = load_module()
    key = SequenceKey([0, 4, 8, 12, 12])
    display = RecordingDisplay()
    sleeps = []

    module.wait_for_start(key, display, sleep=sleeps.append, poll_interval=0.1)

    assert key.read_count == 4
    assert display.messages == ["wait to start", "started!!!"]
    assert sleeps == [0.1, 0.1, 0.1, 0.3]


def test_wait_for_start_treats_invalid_values_as_no_key():
    module = load_module()
    key = SequenceKey([None, "bad", "12"])
    display = RecordingDisplay()

    module.wait_for_start(key, display, sleep=lambda _: None)

    assert key.read_count == 3
    assert display.messages[-1] == "started!!!"
```

- [ ] **Step 2: Run the tests and verify the expected failure**

Run:

```bash
python3 -m pytest tests/test_wait_for_start_key.py -q
```

Expected: FAIL because `scripts/wait_for_start_key.py` does not exist.

- [ ] **Step 3: Implement the listener with lazy hardware imports**

Create `scripts/wait_for_start_key.py`:

```python
#!/usr/bin/env python3
import time


START_KEY_VALUE = 12
START_KEY_PORT = 5


def wait_for_start(key, display, sleep=time.sleep, poll_interval=0.1):
    display.show("wait to start")
    while True:
        try:
            value = int(key.get_key())
        except (TypeError, ValueError):
            value = 0
        if value == START_KEY_VALUE:
            display.show("started!!!")
            sleep(0.3)
            return
        sleep(poll_interval)


def main():
    from smartcar.whalesbot.vehicle import Key4Btn, ScreenShow

    wait_for_start(Key4Btn(START_KEY_PORT), ScreenShow())
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
```

- [ ] **Step 4: Run the focused tests and compile check**

Run:

```bash
python3 -m pytest tests/test_wait_for_start_key.py -q
python3 -m py_compile scripts/wait_for_start_key.py
```

Expected: `2 passed`; compile command exits 0.

- [ ] **Step 5: Commit the listener only**

```bash
git add scripts/wait_for_start_key.py tests/test_wait_for_start_key.py
git commit -m "feat: add MC602 start-key listener"
```

### Task 3: Repeating Shell supervisor

**Files:**
- Create: `scripts/start_with_key.sh`
- Create: `tests/test_key_start_assets.py`

- [ ] **Step 1: Write failing asset-contract tests**

Create `tests/test_key_start_assets.py`:

```python
import shutil
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "scripts" / "start_with_key.sh"
UNIT = ROOT / "systemd" / "baidu-smart-key-start.service"


def test_shell_script_has_valid_bash_syntax():
    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("bash is not installed in this test environment")
    result = subprocess.run([bash, "-n", str(SCRIPT)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_shell_supervisor_contract():
    text = SCRIPT.read_text(encoding="utf-8")
    assert "flock -n" in text
    assert "inference_backend_probe.py" in text
    assert "wait_for_start_key.py" in text
    assert "car_start_2026.py" in text
    assert "backend_watchdog" in text
    assert "while true" in text


def test_systemd_unit_contract():
    text = UNIT.read_text(encoding="utf-8")
    assert "User=jetson" in text
    assert "WorkingDirectory=/home/jetson/workspaces/baidu_smart_2026_7_17" in text
    assert "Restart=always" in text
    assert "KillMode=control-group" in text
```

At this task only the first two tests are targeted; the unit contract is completed in Task 4.

- [ ] **Step 2: Run supervisor tests and verify the expected failure**

Run:

```bash
python3 -m pytest tests/test_key_start_assets.py::test_shell_script_has_valid_bash_syntax tests/test_key_start_assets.py::test_shell_supervisor_contract -q
```

Expected: FAIL because `scripts/start_with_key.sh` does not exist.

- [ ] **Step 3: Implement the supervisor**

Create `scripts/start_with_key.sh`:

```bash
#!/usr/bin/env bash
set -u

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
BACKEND_PATH="${PROJECT_ROOT}/smartcar/paddlebaidu/infer_cs/base/infer_back_end.py"
PROBE_PATH="${SCRIPT_DIR}/inference_backend_probe.py"
KEY_WAITER_PATH="${SCRIPT_DIR}/wait_for_start_key.py"
MAIN_PATH="${PROJECT_ROOT}/car_start_2026.py"
WATCHDOG_INTERVAL="${WATCHDOG_INTERVAL:-2}"
RESTART_DELAY="${RESTART_DELAY:-2}"
LOCK_PATH="${XDG_RUNTIME_DIR:-/tmp}/baidu-smart-key-start.lock"

WATCHDOG_PID=""
MAIN_PID=""

log() {
    printf '%s %s\n' "$(date -Is)" "$*"
}

backend_running() {
    pgrep -f -- "${BACKEND_PATH}" >/dev/null 2>&1
}

start_backend_if_needed() {
    if backend_running; then
        return 0
    fi
    log "inference backend is not running; starting it"
    "${PYTHON_BIN}" "${BACKEND_PATH}" &
}

backend_watchdog() {
    while true; do
        start_backend_if_needed
        sleep "${WATCHDOG_INTERVAL}"
    done
}

cleanup() {
    trap - INT TERM EXIT
    if [[ -n "${MAIN_PID}" ]]; then
        kill "${MAIN_PID}" 2>/dev/null || true
        wait "${MAIN_PID}" 2>/dev/null || true
    fi
    if [[ -n "${WATCHDOG_PID}" ]]; then
        kill "${WATCHDOG_PID}" 2>/dev/null || true
        wait "${WATCHDOG_PID}" 2>/dev/null || true
    fi
}

exec 9>"${LOCK_PATH}"
if ! flock -n 9; then
    log "another key-start supervisor is already running"
    exit 1
fi

trap cleanup INT TERM EXIT
cd "${PROJECT_ROOT}"

backend_watchdog &
WATCHDOG_PID=$!

while true; do
    log "waiting for inference backend readiness"
    "${PYTHON_BIN}" "${PROBE_PATH}" --wait

    log "waiting for MC602 port 5 key value 12"
    "${PYTHON_BIN}" "${KEY_WAITER_PATH}"
    key_status=$?
    if [[ "${key_status}" -ne 0 ]]; then
        log "key listener exited without a start event: ${key_status}"
        sleep "${RESTART_DELAY}"
        continue
    fi

    log "start event received; rechecking inference backend"
    "${PYTHON_BIN}" "${PROBE_PATH}" --wait

    log "starting full task program"
    "${PYTHON_BIN}" "${MAIN_PATH}" &
    MAIN_PID=$!
    wait "${MAIN_PID}"
    main_status=$?
    MAIN_PID=""
    log "full task program exited with status ${main_status}; returning to key wait"
    sleep "${RESTART_DELAY}"
done
```

The script intentionally does not use `set -e`: nonzero key-listener and main-program statuses are lifecycle events handled by the loop.

- [ ] **Step 4: Run Shell tests**

Run:

```bash
python3 -m pytest tests/test_key_start_assets.py::test_shell_script_has_valid_bash_syntax tests/test_key_start_assets.py::test_shell_supervisor_contract -q
bash -n scripts/start_with_key.sh
```

Expected: `2 passed` (or one pass plus one explicit Bash-unavailable skip on Windows); syntax command exits 0 on Orin/MSYS2.

- [ ] **Step 5: Mark scripts executable and commit**

```bash
chmod +x scripts/start_with_key.sh scripts/inference_backend_probe.py scripts/wait_for_start_key.py
git add scripts/start_with_key.sh tests/test_key_start_assets.py
git update-index --chmod=+x scripts/start_with_key.sh scripts/inference_backend_probe.py scripts/wait_for_start_key.py
git commit -m "feat: supervise key-triggered task runs"
```

### Task 4: systemd boot unit

**Files:**
- Create: `systemd/baidu-smart-key-start.service`
- Modify: `tests/test_key_start_assets.py`

- [ ] **Step 1: Run the existing unit-contract test and verify failure**

Run:

```bash
python3 -m pytest tests/test_key_start_assets.py::test_systemd_unit_contract -q
```

Expected: FAIL because the service unit does not exist.

- [ ] **Step 2: Create the service unit**

Create `systemd/baidu-smart-key-start.service`:

```ini
[Unit]
Description=Baidu Smart Car key-start supervisor
After=multi-user.target

[Service]
Type=simple
User=jetson
WorkingDirectory=/home/jetson/workspaces/baidu_smart_2026_7_17
ExecStart=/bin/bash /home/jetson/workspaces/baidu_smart_2026_7_17/scripts/start_with_key.sh
Restart=always
RestartSec=3
KillMode=control-group
TimeoutStopSec=15
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
```

- [ ] **Step 3: Run unit tests and systemd validation**

Run locally:

```bash
python3 -m pytest tests/test_key_start_assets.py -q
```

Run on Orin when `systemd-analyze` is available:

```bash
systemd-analyze verify systemd/baidu-smart-key-start.service
```

Expected: all asset tests pass; systemd verification exits 0 without unit errors.

- [ ] **Step 4: Commit the unit**

```bash
git add systemd/baidu-smart-key-start.service tests/test_key_start_assets.py
git commit -m "feat: add key-start systemd service"
```

### Task 5: Full verification, Orin deployment, and boot installation

**Files:**
- Verify only; no additional source file is required.

- [ ] **Step 1: Run the complete local automated suite**

```bash
python3 -m pytest tests/test_inference_backend_probe.py tests/test_wait_for_start_key.py tests/test_key_start_assets.py -q
python3 -m py_compile scripts/inference_backend_probe.py scripts/wait_for_start_key.py
bash -n scripts/start_with_key.sh
git diff --check
```

Expected: all tests pass, Python compilation exits 0, Bash syntax exits 0, and `git diff --check` reports no errors. Existing Orin-synchronized working-tree changes may remain present but must not be included in feature commits.

- [ ] **Step 2: Sync only the new runtime assets to Orin**

From Windows, use the existing approved SSH/rsync setup to copy these exact paths into `/home/jetson/workspaces/baidu_smart_2026_7_17/`:

```text
scripts/inference_backend_probe.py
scripts/wait_for_start_key.py
scripts/start_with_key.sh
systemd/baidu-smart-key-start.service
```

Verify on Orin:

```bash
cd /home/jetson/workspaces/baidu_smart_2026_7_17
chmod +x scripts/start_with_key.sh scripts/inference_backend_probe.py scripts/wait_for_start_key.py
bash -n scripts/start_with_key.sh
python3 -m py_compile scripts/inference_backend_probe.py scripts/wait_for_start_key.py
systemd-analyze verify systemd/baidu-smart-key-start.service
```

Expected: every command exits 0.

- [ ] **Step 3: Install and enable the systemd unit on Orin**

```bash
sudo install -m 0644 \
  /home/jetson/workspaces/baidu_smart_2026_7_17/systemd/baidu-smart-key-start.service \
  /etc/systemd/system/baidu-smart-key-start.service
sudo systemctl daemon-reload
sudo systemctl enable --now baidu-smart-key-start.service
sudo systemctl status --no-pager baidu-smart-key-start.service
```

Expected: the service is enabled and active. The journal shows backend startup/readiness followed by key waiting, while no `car_start_2026.py` process exists before key value `12`.

- [ ] **Step 4: Perform Orin hardware acceptance checks**

Run:

```bash
journalctl -u baidu-smart-key-start.service -f
```

Verify all of the following in order:

1. Cold service start launches one `infer_back_end.py` process.
2. The screen does not show `wait to start` until ports 5001, 5002, and 5005 return JSON `true` to `ATATA`.
3. Short presses and keys other than port 5 long key 4 do not launch `car_start_2026.py`.
4. Port 5 long key 4 launches exactly one `car_start_2026.py`; its normal `init()` then moves/resets hardware.
5. Stopping the main program returns the display to `wait to start` after the configured delay.
6. Killing `infer_back_end.py` causes the watchdog to relaunch it; a later key event waits for readiness before starting the main program.
7. `sudo systemctl restart baidu-smart-key-start.service` leaves only one supervisor and one backend process.

- [ ] **Step 5: Record final repository and service state**

```bash
git status --short
git log -5 --oneline
sudo systemctl is-enabled baidu-smart-key-start.service
sudo systemctl is-active baidu-smart-key-start.service
```

Expected: feature files are committed, pre-existing unrelated local changes remain identifiable, and both systemd checks print their successful states.
