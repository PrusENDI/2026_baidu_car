---
name: orin-remote-runner
description: Use when working in this Baidu smartcar project with Jetson Orin, ssh, rsync, remote run copies, deploy scripts, runtime logs, or local-vs-remote source-of-truth decisions.
---

# Orin Remote Runner

## Core Rules

- Treat the local Windows project as the only source of truth.
- Treat the Jetson Orin workspace as a disposable run copy.
- Make all code and config edits locally in this project.
- Never copy source files from Orin back into the local project.
- Pull back only logs or explicitly requested runtime artifacts, and place them under `logs/orin/` or another clearly named artifact directory.
- Do not exclude models, weights, or configuration by default. This is whole-car code, not a standalone model repo.
- Exclude caches, virtual environments, logs, temporary files, and build output from deploys.
- Keep changes reviewable in local git. Push only after the user explicitly confirms the target remote.

## Required Workflow

1. Inspect or edit local files only.
2. Use `scripts/orin_config.ps1` for SSH target, local project root, remote workspace, run command, log paths, and rsync excludes.
3. Run deploys through `scripts/deploy_to_orin.ps1`.
4. Use dry-run first:

   ```powershell
   .\scripts\deploy_to_orin.ps1
   ```

5. Deploy only after the dry-run looks correct:

   ```powershell
   .\scripts\deploy_to_orin.ps1 -Apply
   ```

6. Use remote delete only when the user explicitly asks for it:

   ```powershell
   .\scripts\deploy_to_orin.ps1 -Apply -Delete
   ```

7. Run on Orin through:

   ```powershell
   .\scripts\run_on_orin.ps1
   ```

8. Fetch logs through:

   ```powershell
   .\scripts\fetch_orin_logs.ps1
   ```

9. Review and commit local changes:

   ```powershell
   git status --short
   git diff
   git add <files>
   git commit -m "message"
   ```

## rsync Notes

- The current Orin target is `jetson@192.168.0.155`.
- The current Orin run-copy path is `/home/jetson/workspaces/baidu_car_2026_official_run_copy/`.
- The project currently uses MSYS2 rsync on Windows.
- Use `scripts/deploy_to_orin.ps1` instead of handwritten rsync commands.
- `scripts/deploy_to_orin.ps1` defaults to dry-run.
- `-Apply` copies files.
- `-Delete` is allowed only after explicit confirmation that the remote path is a disposable run copy.
- See `docs/ORIN_RSYNC_WORKFLOW.md` for user-facing commands and troubleshooting context.

## Debug Entrypoints

Use guarded debug entrypoints before running the full task flow:

```powershell
.\scripts\run_on_orin.ps1 -Command "python3 debug/chassis_check.py move --x 0.1 --seconds 0.3"
.\scripts\run_on_orin.ps1 -Command "python3 debug/arm_check.py reset"
```

Add `--apply` inside the remote Python command only when the user intentionally wants hardware motion:

```powershell
.\scripts\run_on_orin.ps1 -Command "python3 debug/chassis_check.py --apply move --x 0.1 --seconds 0.3"
```

## Git Remote Notes

The configured `origin` is currently Gitee, not GitHub:

```text
https://gitee.com/younglet/baidu_smartcar_2026.git
```

Do not push to Gitee unless the user explicitly asks for it. The GitHub remote is configured as:

```text
github  https://github.com/PrusENDI/2026_baidu_car.git
```

Push GitHub with `git push github master` after local verification.

## Hard Stops

Stop and ask before doing any of these:

- Connecting to Orin when SSH target, remote workspace, or run command is unclear.
- Running any command that writes source files on Orin by hand.
- Using `rsync --delete` outside `scripts/deploy_to_orin.ps1`.
- Pulling anything from Orin into local source directories.
- Excluding model, weight, or config directories from deploys.
- Pushing to Gitee or GitHub without explicit user confirmation of the target remote.

## Review Checklist

- Local edits are visible in local `git diff`.
- Remote path is an absolute dedicated workspace, not `/`, `/home`, or a broad parent directory.
- Dry-run output has been reviewed before first real deploy.
- Runtime logs come back under `logs/orin/`.
- Any network access or Orin connection follows the current Codex approval flow.
- Commits are local until the user confirms a push target.
