# CV + Manual Joint Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在新 GPU 容器上使用 CV 双目标数据与两组十字路口手柄曲率标签联合微调一个 `[speed_demand,kappa_action]` 模型，并分别评估 CV、`lap_002` 和官方手柄参考集。

**Architecture:** 本地保留原始数据，远端建立独立项目、数据和产物目录。训练 manifest 中 CV 使用 `[1,1]`，手柄使用 `[0,1]`；手柄验证只计算曲率。先做环境与数据审计、单元测试和 smoke，再启动正式训练并保留日志、checkpoint 与评估报告。

**Tech Stack:** PowerShell、OpenSSH、Python 3、PaddlePaddle、Pillow、JSONL、NVIDIA GPU。

---

### Task 1: Audit the new container

**Files:**
- Read: remote system, GPU, Python, Paddle and disk state

- [ ] Connect to `root@connect.bjb1.seetacloud.com:43447` and record `nvidia-smi`, `python --version`, Paddle CUDA status, free disk and existing `/root/autodl-tmp` contents.
- [ ] Reuse a compatible environment only if Paddle imports and sees the GPU; otherwise create a dedicated environment without overwriting unrelated environments.

### Task 2: Stage code and immutable source data

**Files:**
- Source: repository worktree
- Source: `lane-sessions0817/cv_and_image_set_lane_backup/cv_lane_tests`
- Source: `lane_sessions2/lane_sessions2/lap_002`
- Source: `offical-line-test/image_set_lane (1).zip`
- Source: `offical-line-test/image_set_lane_eval (1).zip`

- [ ] Sync the current worktree to an isolated remote project directory, excluding `.git`, virtual environments and existing artifacts that are not training inputs.
- [ ] Upload the selected CV train/validation sessions, two crossroad manual sessions, `lap_002`, and the two official zip files to an isolated remote dataset directory.
- [ ] Compare local and remote file counts and SHA256 hashes for JSON and zip inputs.

### Task 3: Validate labels and build manifests

**Files:**
- Run: `tools/lane_training/prepare_action_manifest.py`
- Read: `lane_training/manifest.py`

- [ ] Verify every CV record has `label_semantics=raw_control_with_model_target`, `model_target`, and `target_mask=[1,1]`.
- [ ] Verify every manual record has finite `[vx,vy,wz]`; derive `kappa_action=wz/max(abs(vx),0.12)` exactly once and use `target_mask=[0,1]`.
- [ ] Build the joint train/CV-validation manifest from the chosen CV sessions and two crossroad manual sessions.
- [ ] Build a separate `lap_002` manual-validation manifest so its metrics cannot be averaged into CV validation.
- [ ] Inspect the official zip label format; convert it into a separate, non-destructive validation manifest only if the steering command semantics can be proven.

### Task 4: Run tests and smoke training

**Files:**
- Test: `tests/lane_training/`
- Run: `tools/lane_training/train_action.py`

- [ ] Run focused manifest, dataset, mask and loss tests with the remote Python environment.
- [ ] Run a two-epoch GPU smoke using the joint manifest and the intended initial checkpoint.
- [ ] Confirm the report contains finite loss, nonzero valid speed only for CV samples, and valid curvature for both CV and manual samples.

### Task 5: Run formal training

**Files:**
- Run: `tools/lane_training/train_action.py`
- Create: remote training output and log directories

- [ ] Start from the compatible existing action checkpoint, never from random weights.
- [ ] Train 20 epochs with batch size 64, learning rate `1e-5`, curvature weight `2.0`, GPU device, and a persistent log file.
- [ ] Monitor GPU utilization, loss finiteness, checkpoint creation and disk usage while training runs.

### Task 6: Evaluate and export

**Files:**
- Run: `tools/lane_training/evaluate_action.py`
- Run: `tools/lane_training/export_action.py`

- [ ] Evaluate `best.pdparams` on CV validation and `lap_002` independently.
- [ ] Evaluate official manual data independently only after its semantics pass Task 3.
- [ ] Compare speed MAE only on CV; compare curvature direction, amplitude and sequence behavior on manual reference sets.
- [ ] Export the selected checkpoint, verify dynamic/static maximum difference is at most `1e-5`, and preserve `deployment.json` plus `.tgz` hashes.
- [ ] Do not deploy to Orin until offline reports are reviewed and the user authorizes vehicle deployment.
