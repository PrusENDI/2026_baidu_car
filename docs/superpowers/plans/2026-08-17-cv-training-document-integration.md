# CV 教师与 CNN 训练文档整合实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将训练交接文档改造成 CV 自动采集、CNN 训练和车端部署的唯一当前入口，同时保留 CV 状态文档作为实验历史。

**Architecture:** 以当前代码为字段和公式的唯一事实来源，在训练交接文档中维护稳定接口与操作闭环，在 CV 状态文档中仅维护视觉算法和实验演进。两份文档通过相对链接互相引用，并明确当前结论与历史决策的边界。

**Tech Stack:** Markdown、Git、PowerShell、ripgrep

---

### Task 1: 核对当前代码契约

**Files:**
- Read: `smartcar/whalesbot/tools/lane_collect/ssh_test.py`
- Read: `smartcar/whalesbot/tools/lane_collect/pid_control.py`
- Read: `smartcar/whalesbot/tools/curvature_control.py`
- Read: `lane_training/dataset.py`
- Read: `lane_training/manifest.py`
- Read: `car_wrap_2026.py`
- Read: `config_car.yml`

- [ ] **Step 1: 提取采集记录字段**

Run:

```powershell
rg -n 'state|control|legacy_pid_state|model_target|target_mask|effective_dt_s' smartcar/whalesbot/tools/lane_collect/ssh_test.py
```

Expected: 能定位 `[vx, vy, wz]`、`[speed_demand, kappa_action]`、mask 和真实时间字段。

- [ ] **Step 2: 提取控制公式和 profile**

Run:

```powershell
rg -n 'action_curvature|finite_dt|max_deceleration_mps2|max_acceleration_mps2|legacy_error|kappa_action' smartcar/whalesbot/tools/lane_collect/pid_control.py smartcar/whalesbot/tools/curvature_control.py car_wrap_2026.py config_car.yml
```

Expected: 能定位 `wz = vx * kappa_action` 的实现、真实 `dt` 梯度参数和两种部署 profile。

- [ ] **Step 3: 提取训练标签和 mask 规则**

Run:

```powershell
rg -n 'speed_demand|kappa_action|target_mask|horizontal_flip|teacher' lane_training/dataset.py lane_training/manifest.py
```

Expected: CV 样本使用双目标，手柄样本只监督曲率，水平翻转只反转曲率符号。

### Task 2: 更新训练交接主文档

**Files:**
- Modify: `docs/lane-cnn-training-handoff.md`
- Reference: `docs/cv-lane-development-status-2026-08-15.md`

- [ ] **Step 1: 更新文档顶部状态与权威边界**

写明当前工作树分支为 `orin-main-20260814`，当前训练与部署接口以本文件和代码为准；CV 逐帧实验历史链接到状态文档。

- [ ] **Step 2: 增加当前闭环摘要**

摘要必须按以下顺序描述数据流：

```text
320x240 原图
  -> CV Pure Pursuit 教师
  -> 实际底盘命令 [vx, vy, wz]
  -> model_target [speed_demand, kappa_action]
  -> CNN
  -> CurvatureSpeedController
  -> [vx, 0, vx * kappa_action]
```

- [ ] **Step 3: 写入数据字段和兼容规则**

明确记录：

```text
CV target_mask      = [1.0, 1.0]
手柄 target_mask    = [0.0, 1.0]
水平翻转            = speed_demand 不变，kappa_action 反号
原始 state/control  = [vx, vy, wz]
```

- [ ] **Step 4: 写入真实时间后处理和部署 profile**

记录 `0.194 m/s²` 减速率、`0.129 m/s²` 加速率、`finite_dt` 的回退行为，以及 `legacy_error` 仅用于旧模型、`kappa_action` 用于新模型。

- [ ] **Step 5: 写入离线重标注、训练、导出和验收清单**

清单必须区分可从图片重新生成的 CV 标签与不能凭图片恢复的真实手柄速度；验收至少覆盖输出尺度、方向、时序、闭环完整一圈和实车速度梯度。

- [ ] **Step 6: 标记历史决策章节**

在第 16～17 节前明确说明它们记录方案演进，不代表当前接口；第 18 节和新增闭环章节代表当前实施结论。

### Task 3: 更新 CV 历史文档入口

**Files:**
- Modify: `docs/cv-lane-development-status-2026-08-15.md`

- [ ] **Step 1: 增加训练主文档链接**

在文档顶部声明：视觉算法、帧段和参数演进保留在本文件；当前采集标签、CNN 输出和部署后处理统一维护在 `lane-cnn-training-handoff.md`。

- [ ] **Step 2: 避免维护重复接口**

明确历史章节中的旧 `[vy, yaw_error]`、按帧梯度和状态机结论只用于追溯，不能覆盖训练主文档的当前定义。

### Task 4: 一致性验证与提交

**Files:**
- Verify: `docs/lane-cnn-training-handoff.md`
- Verify: `docs/cv-lane-development-status-2026-08-15.md`

- [ ] **Step 1: 检查当前字段是否齐全**

Run:

```powershell
rg -n 'speed_demand|kappa_action|target_mask|legacy_error|effective_dt_s|0\.194|0\.129' docs/lane-cnn-training-handoff.md
```

Expected: 每个当前接口关键词都出现在稳定结论或检查清单中。

- [ ] **Step 2: 检查两份文档的交叉链接**

Run:

```powershell
rg -n 'cv-lane-development-status-2026-08-15\.md' docs/lane-cnn-training-handoff.md
rg -n 'lane-cnn-training-handoff\.md' docs/cv-lane-development-status-2026-08-15.md
```

Expected: 两条命令均至少返回一处相对链接。

- [ ] **Step 3: 检查格式和未决标记**

Run:

```powershell
git diff --check
rg -n '待补充|尚未决定|后续填写' docs/lane-cnn-training-handoff.md docs/cv-lane-development-status-2026-08-15.md
```

Expected: `git diff --check` 无输出；文档中没有未决标记。

- [ ] **Step 4: 审查并提交文档差异**

Run:

```powershell
git diff -- docs/lane-cnn-training-handoff.md docs/cv-lane-development-status-2026-08-15.md
git add docs/lane-cnn-training-handoff.md docs/cv-lane-development-status-2026-08-15.md
git commit -m "docs: integrate CV teacher with CNN training handoff"
```

Expected: 提交只包含两份目标文档。
