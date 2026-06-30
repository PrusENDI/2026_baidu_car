# Sync To Orin

This project uses the Windows repository as the only source-editing side.
The Jetson Orin is the run, test, and hardware-debug side.

## Confirmed Paths

Local source root:

```text
C:\weizijian\documents\baidu\2026_baidu_car\2026\baidu_smartcar_2026
```

Confirmed Orin run copy:

```text
jetson@192.168.0.155:/home/jetson/workspaces/baidu_car_2026_official_run_copy
```

The Orin directory was confirmed by SSH inspection of `/home/jetson/workspaces`.
`baidu_car_2026_official_code` is not the run copy; it contains `data`, `logs`,
and `models`.

## Tooling

Windows uses MSYS2 rsync and ssh:

```text
C:\msys64\usr\bin\rsync.exe
C:\msys64\usr\bin\ssh.exe
```

## Common Commands

From this project root, preview the selected files without connecting to Orin:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\tools\sync_to_orin.ps1 `
  -Files car_wrap_2026.py,config_car.yml `
  -PlanOnly
```

Preview git-changed files without connecting to Orin:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\tools\sync_to_orin.ps1 `
  -Changed `
  -PlanOnly
```

Run rsync dry-run against Orin:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\tools\sync_to_orin.ps1 `
  -Changed `
  -DryRun
```

Sync git-changed files and verify remote file metadata:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\tools\sync_to_orin.ps1 `
  -Changed `
  -Verify
```

Sync explicit files and verify:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\tools\sync_to_orin.ps1 `
  -Files car_task_function.py,car_wrap_2026.py,config_car.yml `
  -Verify
```

## Orin Status And Logs

Use the read-only status helper before and after running hardware tests:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\tools\orin_status.ps1
```

The helper SSHes into Orin and prints:

- system time, host, uptime, memory, and disk usage
- Jetson telemetry when `tegrastats` is available
- Python, car startup, inference, ZMQ, Paddle, and collect-control processes
- listening ports for `5000` through `5006`
- tail output from project log-like files under `logs`, `log`, `debug`, and the
  project root
- recent matching error lines and recent system journal warnings

It also saves a local snapshot under:

```text
logs\orin-status\
```

Use a shorter or longer log tail with:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\tools\orin_status.ps1 `
  -LogLines 300
```

## Safety Rules

- Edit source locally first.
- Use `git status` before editing or syncing.
- Prefer explicit `-Files` for narrow changes.
- Use `-PlanOnly` before the first real sync in a session.
- Use `-DryRun` before syncing when the file set is broad.
- Use `tools\orin_status.ps1` to capture the live Orin status and error logs
  before diagnosing a hardware/runtime issue.
- Do not hand-edit source on Orin for long-lived changes.
- After Orin logs show an error, map it back to a local file, edit locally,
  then sync again.

The script refuses remote roots outside `/home/jetson/workspaces/` and excludes
`.git/`, `__pycache__/`, Python bytecode, pytest/mypy caches, datasets, logs,
and large model artifacts.
