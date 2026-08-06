# Remove Key Start Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove the key-start supervisor and its MC602 disconnect recovery additions so the project is started manually with `python car_start_2026.py`.

**Architecture:** Delete the dedicated supervisor assets and documentation, restore the affected runtime modules to their pre-protection behavior, and surgically preserve unrelated local task changes in mixed files. A small source-contract test prevents the removed supervisor assets and exception path from being reintroduced accidentally.

**Tech Stack:** Python 3, unittest/pytest, Bash and systemd asset removal, Git verification.

---

### Task 1: Establish the manual-start contract

**Files:**
- Create: `tests/test_manual_start_contract.py`

- [ ] **Step 1: Write the failing test**

```python
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class ManualStartContractTest(unittest.TestCase):
    def test_key_start_runtime_assets_are_absent(self):
        assets = (
            "scripts/start_with_key.sh",
            "scripts/wait_for_start_key.py",
            "scripts/inference_backend_probe.py",
            "systemd/baidu-smart-key-start.service",
        )
        self.assertEqual([], [path for path in assets if (ROOT / path).exists()])

    def test_main_entry_has_no_supervisor_disconnect_protocol(self):
        source = (ROOT / "car_start_2026.py").read_text(encoding="utf-8")
        self.assertNotIn("ControllerDisconnected", source)
        self.assertNotIn("MC602_DISCONNECTED_EXIT", source)
        self.assertIn('if __name__ == "__main__":', source)
        self.assertIn("main()", source)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the test and verify RED**

Run: `python -m pytest tests/test_manual_start_contract.py -q`

Expected: FAIL because the four supervisor assets exist and `car_start_2026.py` still contains the disconnect protocol.

- [ ] **Step 3: Commit the failing contract test**

```bash
git add tests/test_manual_start_contract.py
git commit -m "test: define manual start contract"
```

### Task 2: Remove dedicated key-start assets

**Files:**
- Delete: `scripts/start_with_key.sh`
- Delete: `scripts/wait_for_start_key.py`
- Delete: `scripts/inference_backend_probe.py`
- Delete: `systemd/baidu-smart-key-start.service`
- Delete: `tests/test_key_start_helpers.py`
- Delete: `smartcar/whalesbot/vehicle/base/controller_health.py`
- Delete: `tests/test_mc602_disconnect.py`
- Delete: `docs/superpowers/specs/2026-08-05-key-start-supervisor-design.md`
- Delete: `docs/superpowers/plans/2026-08-05-key-start-supervisor.md`
- Delete: `docs/superpowers/specs/2026-08-06-mc602-disconnect-recovery-design.md`
- Delete: `docs/superpowers/plans/2026-08-06-mc602-disconnect-recovery.md`
- Delete: `docs/superpowers/specs/2026-08-06-mc602-disconnect-timeout-design.md`
- Delete: `docs/superpowers/plans/2026-08-06-mc602-disconnect-timeout.md`

- [ ] **Step 1: Delete every dedicated runtime, test and historical planning asset**

Use one explicit `apply_patch` deletion entry per path listed above. Do not delete unrelated `.dbg` files or general debugging notes.

- [ ] **Step 2: Confirm the asset half of the contract is green**

Run: `python -m pytest tests/test_manual_start_contract.py::ManualStartContractTest::test_key_start_runtime_assets_are_absent -q`

Expected: PASS.

### Task 3: Restore direct manual execution and baseline MC602 behavior

**Files:**
- Modify: `car_start_2026.py`
- Modify: `car_wrap_2026.py`
- Modify: `smartcar/whalesbot/vehicle/base/serial_wrap.py`
- Modify: `smartcar/whalesbot/vehicle/driver/mecanum.py`

- [ ] **Step 1: Restore `car_start_2026.py` to direct task execution**

Remove the `time`, `task_functions` and `ControllerDisconnected` imports, `MC602_DISCONNECTED_EXIT`, both disconnect helpers, the exception wrapper and `SystemExit`. Keep the original ordered calls to `init`, seeding, detection, irrigation, shooting, harvesting, storage, order capture and delivery. End with:

```python
if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Restore baseline serial behavior**

In `serial_wrap.py`, remove `CommunicationHealth`, `ControllerDisconnected`, failure-window bookkeeping and controller-restart retry additions. Restore `get_anwser()` to one locked transaction that logs an exception, releases the lock and returns the response. Restore the original controller-not-found wait and MC602 download/Ping behavior from `HEAD`.

- [ ] **Step 3: Restore baseline odometry behavior**

In `mecanum.py`, remove the disconnect import and flags. Restore `update_odometry_thread()` to its original loop from `HEAD`, without a `ControllerDisconnected` handler.

- [ ] **Step 4: Surgically remove disconnect propagation from `car_wrap_2026.py`**

Remove only the `ControllerDisconnected` import, `_controller_disconnected` initialization/preservation and the exception handler around `self.key.get_key()`. Preserve all camera, crop harvesting and task-specific local edits.

- [ ] **Step 5: Run the complete manual-start contract and verify GREEN**

Run: `python -m pytest tests/test_manual_start_contract.py -q`

Expected: 2 passed.

- [ ] **Step 6: Commit the runtime rollback**

Stage only the deleted feature assets, the four restored runtime files and `tests/test_manual_start_contract.py`, then commit with `git commit -m "revert: remove key-start supervisor"`.

### Task 4: Verify isolation and repository health

**Files:**
- Verify: `car_start_2026.py`
- Verify: `car_wrap_2026.py`
- Verify: `smartcar/whalesbot/vehicle/base/serial_wrap.py`
- Verify: `smartcar/whalesbot/vehicle/driver/mecanum.py`

- [ ] **Step 1: Compile affected Python entry points**

Run: `python -m py_compile car_start_2026.py car_wrap_2026.py smartcar/whalesbot/vehicle/base/serial_wrap.py smartcar/whalesbot/vehicle/driver/mecanum.py`

Expected: exit code 0.

- [ ] **Step 2: Run focused tests**

Run: `python -m pytest tests/test_manual_start_contract.py -q`

Expected: 2 passed.

- [ ] **Step 3: Scan runtime code for removed protocol references**

Run: `git grep -n -I -E "ControllerDisconnected|CommunicationHealth|MC602_DISCONNECTED_EXIT|start_with_key|wait_for_start_key|baidu-smart-key-start" -- '*.py' '*.sh' '*.service' '*.yml' '*.yaml' ':!docs/**'`

Expected: no matches.

- [ ] **Step 4: Verify the patch and preserved unrelated work**

Run: `git diff --check` and inspect `git status --short` plus focused diffs. Confirm task tuning in `car_task_function.py`, camera/crop logic in `car_wrap_2026.py`, `config_car.yml` and `arm_cfg.yaml` remain present.

Expected: no whitespace errors; unrelated changes remain unstaged and intact.
