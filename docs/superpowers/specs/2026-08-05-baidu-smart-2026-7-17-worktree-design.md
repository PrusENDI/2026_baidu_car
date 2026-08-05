# Baidu Smart 2026-7-17 Worktree Design

## Goal

Convert `C:\weizijian\documents\baidu\baidu_smart_2026_7_17` into a linked Git worktree without changing its project-root layout or discarding its existing files.

## Repository layout

The source repository stores this project below `2026/baidu_smartcar_2026`. A subtree branch named `baidu-smart-2026-7-17` therefore exposes that project directly at the new worktree root. The branch starts from the `2026/baidu_smartcar_2026` subtree of `lane-test-telemetry-20260803`, after which the existing directory snapshot is committed as the branch state.

## Orin synchronization

The synchronization scripts retained in the prior `lane-test-telemetry` worktree will use:

- Local root: `C:\weizijian\documents\baidu\baidu_smart_2026_7_17`
- Remote root: `/home/jetson/workspaces/baidu_smart_2026_7_17/`

The default paths in the main sync script, debug-tools wrapper, and Orin status script will stay consistent. Their PowerShell tests will assert the new defaults before the scripts are changed.

## Safety and verification

The conversion preserves existing working-directory files and changes only Git administrative metadata. Verification covers the registered worktree/branch association, the committed snapshot, PowerShell sync tests, and a plan-only synchronization run that performs no remote write.
