# Baidu Smart 2026-7-17 Worktree Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Track the existing 2026-7-17 project snapshot on its own subtree worktree branch and retarget the prior worktree's Orin tooling to that local and remote root.

**Architecture:** The project-only subtree branch keeps project files at the worktree root. The prior worktree remains the home of the PowerShell utilities, but their defaults point at the new worktree and matching Orin directory; existing fake-rsync/SSH tests validate the defaults without writing remotely.

**Tech Stack:** Git worktrees and subtree branches, PowerShell, rsync/SSH test doubles

---

### Task 1: Commit the imported project snapshot

**Files:**
- Modify: all tracked and non-ignored files below `C:\weizijian\documents\baidu\baidu_smart_2026_7_17`

- [ ] **Step 1: Inspect the snapshot status**

Run: `git status --short --branch`

Expected: branch `baidu-smart-2026-7-17` with the existing snapshot represented as modifications, deletions, and additions relative to the source subtree.

- [ ] **Step 2: Stage the snapshot without force-adding ignored runtime/model artifacts**

Run: `git add -A`

Expected: all non-ignored snapshot changes staged; `.dbg`, `__pycache__`, and excluded model binaries remain ignored.

- [ ] **Step 3: Commit the snapshot**

Run: `git commit -m "chore: import baidu smart 2026-7-17 snapshot"`

Expected: a new commit on `baidu-smart-2026-7-17` and a clean status.

### Task 2: Test-drive the new synchronization defaults

**Files:**
- Modify: `C:\weizijian\documents\baidu\2026_baidu_car\.worktrees\lane-test-telemetry\2026\baidu_smartcar_2026\tools\test_sync_to_orin.ps1`
- Modify: `C:\weizijian\documents\baidu\2026_baidu_car\.worktrees\lane-test-telemetry\2026\baidu_smartcar_2026\tools\test_orin_status.ps1`
- Modify: `C:\weizijian\documents\baidu\2026_baidu_car\.worktrees\lane-test-telemetry\2026\baidu_smartcar_2026\tools\sync_to_orin.ps1`
- Modify: `C:\weizijian\documents\baidu\2026_baidu_car\.worktrees\lane-test-telemetry\2026\baidu_smartcar_2026\tools\sync_debug_tools_to_orin.ps1`
- Modify: `C:\weizijian\documents\baidu\2026_baidu_car\.worktrees\lane-test-telemetry\2026\baidu_smartcar_2026\tools\orin_status.ps1`

- [ ] **Step 1: Change test expectations first**

Replace expected default remote roots with `/home/jetson/workspaces/baidu_smart_2026_7_17` and add an assertion that `sync_to_orin.ps1` contains this default local root:

```powershell
C:\weizijian\documents\baidu\baidu_smart_2026_7_17
```

- [ ] **Step 2: Run tests and verify RED**

Run: `powershell -NoProfile -ExecutionPolicy Bypass -File .\tools\test_sync_to_orin.ps1`

Run: `powershell -NoProfile -ExecutionPolicy Bypass -File .\tools\test_orin_status.ps1`

Expected: failures reporting the old local/remote defaults.

- [ ] **Step 3: Apply the minimal production changes**

Set `sync_to_orin.ps1` and `sync_debug_tools_to_orin.ps1` local defaults to `C:\weizijian\documents\baidu\baidu_smart_2026_7_17`. Set all three production scripts' remote default to `/home/jetson/workspaces/baidu_smart_2026_7_17`.

- [ ] **Step 4: Run tests and verify GREEN**

Run the same two PowerShell test commands.

Expected: `All sync script tests passed.` and `All Orin status tests passed.`

- [ ] **Step 5: Commit the script and test changes on the prior worktree branch**

Run: `git add 2026/baidu_smartcar_2026/tools && git commit -m "chore: retarget Orin sync to 2026-7-17 worktree"`

Expected: one focused commit on `lane-test-telemetry-20260803`.

### Task 3: Verify worktree registration and safe sync planning

**Files:**
- Verify only: Git worktree metadata and sync command output

- [ ] **Step 1: Verify worktree registration**

Run: `git worktree list --porcelain`

Expected: `C:/weizijian/documents/baidu/baidu_smart_2026_7_17` associated with `refs/heads/baidu-smart-2026-7-17`.

- [ ] **Step 2: Verify both branches are clean**

Run `git status --short --branch` in each worktree.

Expected: no uncommitted changes created by this task.

- [ ] **Step 3: Verify plan-only synchronization**

Run: `powershell -NoProfile -ExecutionPolicy Bypass -File .\tools\sync_to_orin.ps1 -Files car_wrap_2026.py -PlanOnly`

Expected: local source under `baidu_smart_2026_7_17` and destination `jetson@192.168.0.155:/home/jetson/workspaces/baidu_smart_2026_7_17/`; no rsync or remote write occurs.
