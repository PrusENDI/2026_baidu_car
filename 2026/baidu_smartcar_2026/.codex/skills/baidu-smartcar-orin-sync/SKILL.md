---
name: baidu-smartcar-orin-sync
description: Use when working on this Baidu smartcar project with Jetson Orin, especially before editing, syncing, testing, running hardware commands, or interpreting Orin logs. Enforces local-first source changes, rsync-based synchronization to the confirmed Orin run copy, and Orin-as-execution-only workflow.
---

# Baidu Smartcar Orin Sync

Use this skill for the Baidu smartcar 2026 project:

```text
C:\weizijian\documents\baidu\2026_baidu_car\2026\baidu_smartcar_2026
```

## Core Rule

Treat Windows as the only source-editing side. Treat Orin as the run, test, and
hardware-debug side. Do not make long-lived source edits on Orin.

## Confirmed Orin Target

SSH:

```text
jetson@192.168.0.155
```

Confirmed run copy:

```text
/home/jetson/workspaces/baidu_car_2026_official_run_copy
```

This path was confirmed by inspecting `/home/jetson/workspaces` on Orin. The
`baidu_car_2026_official_code` directory is not the run copy.

## Local Workflow

Before changing code:

```powershell
git -c safe.directory=C:/weizijian/documents/baidu/2026_baidu_car `
  -C C:\weizijian\documents\baidu\2026_baidu_car\2026\baidu_smartcar_2026 `
  status --short --branch
```

Modify files only in the local project tree. Work with any user changes already
present; do not revert them unless explicitly asked.

## Sync Workflow

Use the project script:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\tools\sync_to_orin.ps1
```

Preview selected files without connecting to Orin:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\tools\sync_to_orin.ps1 `
  -Files car_wrap_2026.py,config_car.yml `
  -PlanOnly
```

Preview git-changed files:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\tools\sync_to_orin.ps1 `
  -Changed `
  -PlanOnly
```

Dry-run with Orin:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\tools\sync_to_orin.ps1 `
  -Changed `
  -DryRun
```

Sync and verify:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\tools\sync_to_orin.ps1 `
  -Changed `
  -Verify
```

Prefer explicit `-Files` for narrow edits. Use `-Changed` when all changed files
are intended for Orin.

## Orin Workflow

On Orin, run commands only for inspection, tests, startup scripts, hardware
debugging, and log collection. Example read-only path checks are allowed:

```bash
find /home/jetson/workspaces -maxdepth 2 -type d -print
```

Before diagnosing a runtime or hardware issue, capture live Orin status and
recent logs from the local project:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\tools\orin_status.ps1
```

This helper is read-only on Orin. It reports system resources, Jetson telemetry
when available, project processes, ports `5000` through `5006`, recent project
logs, matched error lines, and recent journal warnings. It saves a local snapshot
under `logs\orin-status\`.

After an Orin error, map the traceback or log line back to local source, edit
locally, sync again, then rerun on Orin.

## Reporting

Every project update should say:

- What changed locally.
- Which files were synced to Orin, if any.
- What ran on Orin, if anything.
- What the result was.
