# 2026-07-29 X 轴微动开关机械归零开发与现场调试日志

> 日期：2026-07-29  
> 本地分支：`lane-test-telemetry`  
> 本地工程：`C:\weizijian\documents\baidu\2026_baidu_car\.worktrees\lane-test-telemetry\2026\baidu_smartcar_2026`  
> 正确 Orin 目录：`/home/jetson/workspaces/baidu_car_2026_official_run_copy/`  
> 禁止目录：`/home/jetson/workspaces/baidu_smart_2026_7_17/`，今日未接触  
> 当前状态：设计文档已提交；实现代码仍在脏工作树中，未单独提交；最新 `arm_base.py` 已同步 Orin；最新立即停机逻辑尚未完成现场复测

## 1. 执行摘要

今天完成了 X 轴微动开关机械归零的阶段二设计、实现、任务调用调整和两轮现场问题修正。

核心结果如下。

1. 使用 MC602 专用 `AI1`、现有 `AnalogInput2` / `sensor_analog_a` 通讯路径读取微动开关。
2. 现场已确认：松开约为 `raw=0`，按下约为 `raw=4010–4036`，典型值约 `4023`。
3. `reset_x()` 不再把编码器停滞或机械堵转视为归零成功；只有微动开关稳定高电平才能建立零点。
4. 每次程序启动只机械归零一次，成功后离开开关并停在 `x=0.015 m`；任务中只回到 1.5 cm 安全位置，不再重复撞开关归零。
5. 现场第一次归零测试暴露 MC602 速度命令需要周期刷新；只下发一次速度会在约 2.1 cm 后停止并被判为编码器停滞。代码已增加低电平运动阶段的负速度刷新。
6. 现场随后发现稳定采样期间仍持续顶压开关。代码已改为第一次高电平立即 `x_speed(0)`，静止补采剩余 4 个样本；任一回落都失败且不恢复运动。
7. 最新 `arm_base.py` 已同步到正确 Orin，远端与本地 SHA-256 完全一致。

## 2. 工作树保护与变更边界

开始和各次修改前均检查了 Git 状态、差异和近期提交。工作树原本包含大量用户未提交修改和未跟踪文件，因此今天遵循以下边界：

- 没有清理、回退或覆盖无关改动。
- 设计文档按文件单独提交。
- `arm_base.py`、`arm_cfg.yaml` 和 `car_task_function.py` 中与本项目有关的实现仍与文件内既有用户修改共存，没有把整个文件作为独立实现提交。
- 最后一次 Orin 同步只选择了 `smartcar/whalesbot/vehicle/arm/arm_base.py`，没有使用“同步全部改动”。
- 没有访问或同步到禁止目录。

## 3. 现场电气与传感器结论

### 3.1 硬件与端口

- 控制器：MC602。
- 使用接口：专用模拟输入 `AI1`。
- 上位机对象：`AnalogInput2(1)`。
- 协议设备：`sensor_analog_a`。
- `dev_id=0x08`，`mode=0`。
- 微动开关接法：COM+NC。

### 3.2 已确认原始值

| 状态 | AI1 原始值 |
|---|---:|
| 松开 | `0` |
| 按下 | `4010–4036` |
| 典型按下值 | `4023` |
| 断线 | 按当前接法约为 `4023`，与按下无法由软件区分 |

最终配置使用：

```yaml
homing:
  limit_port: 1
  active: above
  threshold: 2000
  stable_samples: 5
  speed: 0.06
  timeout: 8.0
  safe_position: 0.015
```

### 3.3 已接受的电气限制

COM+NC 两线接法下，按下和断线都可能表现为高电平：

- 启动采样阶段若已经是高电平，归零立即失败且不运动，可以拦截“启动前已经断线”。
- 若运动途中线路恰好从低电平变成高电平，软件无法判断这是按下还是断线，可能在断线位置建立零点。
- 这是今天明确接受的现场限制，没有增加双触发、退离重触发或额外硬件状态。

## 4. 只读诊断脚本

文件：`tools/read_x_limit_switch.py`。

脚本特性：

- 只支持 `AI1`、`AI2`、`AI3`。
- 使用 `AnalogInput2`，不实例化电机，不驱动任何轴。
- 支持 `--threshold`、`--active above/below` 和 `--stable-samples`。
- 每次采样前清空 MC602 设备对象的 `last_data`，避免串口无新响应时复用缓存旧值。
- 未传阈值时持续输出 `state=UNCONFIGURED`，但仍显示原始 `raw`。

基础采样命令：

```bash
python3 tools/read_x_limit_switch.py --port AI1
```

今天现场曾出现该命令“什么也读不到”的临时问题，之后用户确认问题已经解决。由于没有保留下能够区分串口占用、初始化阻塞或采样错误的完整输出，日志不对该临时问题声明根因。

该脚本当前仍是未跟踪文件，Git 没有可用于证明历史版本差异的基线。后续如需可靠追踪，应单独审查并提交该文件。

## 5. 阶段二设计决策

### 5.1 最终目标

外接微动开关是 X 轴唯一可信机械零点。编码器只用于零点建立后的相对位置控制：

- 微动开关稳定触发：允许建立零点。
- 编码器停滞：只能判失败。
- 机械堵转：只能判失败。
- 超时、传感器异常、串口异常：返回 `False`，不得建立零点。
- 普通 `move_x_position()`：不得修改机械零点。

### 5.2 单次启动归零

程序每次启动执行一次完整机械归零：

```text
启动连续低电平确认
  -> 低速负向回收
  -> 第一次高电平立即停止
  -> 静止确认剩余高电平样本
  -> 写入 x_pose_start，建立 x=0
  -> 移动到 x=0.015 m
  -> 连续确认开关低电平释放
```

本次运行后续需要回收时，只调用 `retract_x_safe()` 返回 1.5 cm，不再重新撞击微动开关。

### 5.3 启动已经触发

按用户选择的简化方案，启动时必须先连续读到 5 个低电平样本：

- 任一样本为高电平立即失败。
- 不执行退离。
- 不直接清零。
- 不启动 X 轴电机。

## 6. 实现文件与行为

### 6.1 `smartcar/whalesbot/vehicle/arm/arm_cfg.yaml`

在 `horiz_cfg` 下增加 `homing` 配置：

- `AI1`；
- `active: above`；
- 阈值 `2000`；
- 稳定样本数 `5`；
- 归零速度 `0.06 m/s`；
- 超时 `8 s`；
- 安全位置 `0.015 m`。

初始化时校验端口、方向、有限阈值、样本数、速度范围、超时和安全位置。

### 6.2 `smartcar/whalesbot/vehicle/arm/arm_base.py`

新增或调整的本项目行为：

- 使用 `AnalogInput2` 初始化 AI1。
- 增加 `_read_x_limit_fresh()`，拒绝缓存旧样本、`None`、不可转换值和非有限数值。
- 增加 `_x_limit_is_triggered()`，按配置判断高/低有效。
- 增加进程内状态 `x_zero_valid`，初始化始终为 `False`，不从 YAML 恢复可信零点。
- `reset_x(out_time=None, speed_limit=None)` 保留无参和兼容参数调用。
- `reset_x()` 所有退出路径最终执行 `x_speed(0)`。
- 增加 `retract_x_safe(out_time=6.0)`，只有当前进程已经机械归零时才能移动到 1.5 cm，并在到位后确认开关稳定释放。
- `reset_position(rehome_x=True)` 默认执行一次机械归零和安全回收；`False` 时只复用当前零点做安全回收。
- X 重置失败会由 `reset_position()` 抛出 `RuntimeError`，阻止初始化继续。
- 删除安全回收后再次执行 `self.x = 0` 的行为，避免重新命令 X 轴返回开关。

### 6.3 `smartcar/whalesbot/vehicle/base/controller_wrap.py`

本项目复用了既有 `AnalogInput2` 高层封装。该对象内部在 MC602 路径使用 `Sensor_Analog2_2`。今天没有为 AI 口增加不存在的内部上拉配置接口。

### 6.4 `smartcar/whalesbot/vehicle/base/mc602_ctl2.py`

本项目复用了既有 `sensor_analog_a` 协议定义和 `Sensor_Analog2_2`，没有修改已经确认正确的舵机 angle16 协议。

### 6.5 `car_task_function.py`

任务编排改为“一次机械归零，多次安全回收”：

- `init()` 保留一次无参 `my_car.arm.reset_position()`，作为正常任务入口唯一一次机械 X 归零。
- `get_order()` 改为 `reset_position(rehome_x=False)`，保留 Y 重置，但 X 只回到 1.5 cm。
- 任务中直接 `reset_x()` 调用已消除。
- 当前审计到 12 处 `retract_x_safe()` 调用，覆盖播种、水塔、作物采收和分拣存储等需要回收或旋转保护的位置。
- 安全回收失败时终止当前流程，不继续旋转、下降、抓取或释放。
- 没有把所有业务用途的 `move_x_position(0)` 机械替换为 1.5 cm；放置坐标等业务目标按调用语义保留。

## 7. 今日现场问题与根因修正

### 7.1 问题一：约 2.1 cm 后停滞失败

首次一行归零测试输出：

```text
2026-07-29 16:20:38,747 arm_base.py, line 544, ERROR:X 轴归零期间编码器停滞，未建立零点 actual=-0.020961, raw=0.0
[X_HOME_TEST] FAIL homed=False
```

根因：最初恒速寻零只在进入循环前发送一次负速度命令。MC602 的速度输出存在刷新/看门狗行为，命令未周期刷新时电机在运动途中停止，随后连续编码器无位移被正确判为失败。

修正：

- 开关仍为低电平时，每个运动循环刷新 `self.x_speed(-homing_speed)`。
- 停滞仍然是失败，未恢复旧的“停滞即成功”逻辑。

### 7.2 问题二：开关触发后仍堵转一段时间

第二版代码在连续 5 个高电平样本期间仍然发送负速度。仅 4 个 `sleep(0.05)` 就可能带来：

```text
0.06 m/s × 0.20 s = 0.012 m
```

即首次触发后理论上还会继续向开关方向顶约 12 mm，尚未计入串口调用时间。这解释了即使安装微动开关，电机仍堵转一段时间的现场现象。

最终修正：

```text
低电平：读取 AI1 -> 刷新负速度 -> 更新编码器与停滞检查
首次高电平：立即 x_speed(0) -> 计为第 1 个触发样本 -> 离开运动循环
静止阶段：每 20 ms 补采一个样本，直到连续 5 个高电平
任一低电平回落：保持停止，返回 False，不恢复运动
```

静止确认阶段不执行编码器停滞判断，因为此时停止是主动行为；只有稳定高电平成功路径才能写入 `x_pose_start`。

最新立即停机修正已同步 Orin，但截至本日志写入时尚未进行新的真实归零复测，因此不能声明现场堵转问题已经验证消失。仍可能存在电机惯性和下位机执行停止命令的短暂延迟。

## 8. 今日提交记录

今天新增 5 个文档提交：

| 提交 | 时间 | 内容 |
|---|---|---|
| `8b8278d` | 14:34 | `docs: design X limit switch homing` |
| `81ff58c` | 14:51 | `docs: use one-time X homing and safe retract` |
| `252874d` | 14:58 | `docs: plan X limit switch homing` |
| `b574bb3` | 17:48 | `docs: stop X on first limit trigger` |
| `089e73c` | 17:50 | `docs: plan immediate X stop on trigger` |

相关文档：

- `docs/superpowers/specs/2026-07-29-x-limit-switch-homing-design.md`
- `docs/superpowers/plans/2026-07-29-x-limit-switch-homing.md`

背景设计提交 `b305b3f docs: design X limit switch reader` 创建于前一天，不计入今日 5 个提交，但今天的实现继续遵循该设计中的只读 AI1 诊断边界。

## 9. 验证与未执行项

### 9.1 已执行的非硬件验证

- 审查 Git 状态、近期提交和精确差异。
- `git diff --check` 通过。
- 使用本机缓存 Python 运行：

```powershell
uv --no-cache run --no-project --offline python -m py_compile smartcar/whalesbot/vehicle/arm/arm_base.py
```

命令退出码为 0。

- 静态审计确认最新代码中：
  - 高电平判断发生在负速度刷新之前；
  - 第一次高电平立即执行 `x_speed(0)`；
  - 静止确认循环内没有负速度命令；
  - 信号回落、超时和异常都会保持停止并返回失败；
  - `finally` 再次无条件停止 X 轴。

### 9.2 按用户要求未执行

- 未新增测试。
- 未运行红测、绿测或大范围测试套件。
- 未由本地工具运行真实开关采样。
- 未由本地工具运行电机归零。
- 未运行 `move_x_position(0.10)`。
- 最新立即停机版本同步后，尚未获得新的现场归零输出。

## 10. Orin 同步记录

正确目标目录：

```text
/home/jetson/workspaces/baidu_car_2026_official_run_copy/
```

第一次现场停滞问题修正后，`arm_base.py` 曾同步到正确目录，远端记录约为：

```text
size=39155
mtime=2026-07-29 16:25:46 +0800
```

最终“首次高电平立即停机”版本再次只同步：

```text
smartcar/whalesbot/vehicle/arm/arm_base.py
```

远端校验记录：

```text
path=/home/jetson/workspaces/baidu_car_2026_official_run_copy/smartcar/whalesbot/vehicle/arm/arm_base.py
size=40107
mtime=2026-07-29 17:52:06.210437100 +0800
sha256=d41e5d0ce8bef41dee15acc66cd93c8bbbebd0bd1af1f5a7391620038d2e2dd2
```

本地 SHA-256 与远端一致。

## 11. 当前 Bash 归零测试命令

以下命令会真实驱动 X 轴，仅供现场在机构旁有人看护、可随时断电时执行：

```bash
cd /home/jetson/workspaces/baidu_car_2026_official_run_copy/ && python3 -u -c 'from smartcar.whalesbot.vehicle.arm.arm_base import ArmController; a=ArmController(); ok=a.reset_x(); print("[X_HOME_TEST] {} homed={}".format("PASS" if ok else "FAIL", a.x_zero_valid), flush=True)'
```

`reset_x()` 成功后 X 轴停在开关触发位置，不会自动回到 1.5 cm。完整启动路径由 `reset_position()` 在归零成功后调用 `retract_x_safe()`。

## 12. 当前已知限制与后续事项

### 12.1 与本次现场问题直接相关

1. 需要现场复测最新立即停机版本，确认第一次高电平后不再持续顶压开关。
2. 若仍有明显过冲，需要区分停止命令延迟、电机惯性、机械开关安装位置和 `0.06 m/s` 归零速度，不应重新允许消抖期间持续负向运动。
3. 需要保存下一次测试的完整日志，包括首次高电平、停机、稳定样本数、最终原始值和是否建立零点。

### 12.2 代码审查发现但今日未处理

1. `reset_position()` 当前在 X 归零/安全回收前先发送手腕 UP 和机械臂 RIGHT；若上电时 X 外伸，存在先旋转再回收的碰撞风险。
2. X、Y 重置线程共享 MC602 串口。串口锁覆盖完整事务，不会直接串帧，但 Y 循环可能延迟 X 速度刷新。
3. AI1 已强制新鲜采样；编码器读取仍可能在 MC602 无新响应时复用设备缓存旧值。稳定触发成功前最终编码器读取尚未增加同等级的新鲜值检查。
4. `set_position_start()` 仍能修改 `x_pose_start` 而不清除 `x_zero_valid`；当前仓库没有调用者，但未来调用可能破坏可信零点语义。
5. COM+NC 两线方案无法区分运动途中断线和真实按下，这是已接受的硬件限制。

这些事项没有混入今天的立即停机修正，后续应逐项设计和验证。

## 13. 文件状态汇总

| 文件 | 今日用途 | 当前 Git 状态说明 |
|---|---|---|
| `docs/superpowers/specs/2026-07-29-x-limit-switch-homing-design.md` | 归零与安全回收设计 | 已通过多个文档提交记录 |
| `docs/superpowers/plans/2026-07-29-x-limit-switch-homing.md` | 实施计划及立即停机增量计划 | 已通过两个计划提交记录 |
| `smartcar/whalesbot/vehicle/arm/arm_cfg.yaml` | AI1、阈值、消抖、速度、超时、1.5 cm 配置 | 文件仍有未提交修改 |
| `smartcar/whalesbot/vehicle/arm/arm_base.py` | 机械归零、安全回收、失败传播、立即停机 | 文件仍有未提交修改；最新版本已同步 Orin |
| `car_task_function.py` | 单次启动归零与任务安全回收调用 | 文件仍有大量既有和本项目未提交修改 |
| `tools/read_x_limit_switch.py` | 只读 AI1 诊断 | 未跟踪文件 |
| `docs/debug-daily/2026-07-29-x-limit-switch-homing.md` | 本日志 | 新增文件 |

## 14. 结论

今天已经把 X 轴归零的成功条件从“不可靠的编码器停滞/机械堵转”切换为独立 AI1 微动开关，并将任务流程改造成“每次启动归零一次、后续只回 1.5 cm”。现场反馈进一步确认了两个控制时序问题：MC602 负速度需要周期刷新，而微动开关消抖不能在持续顶压状态下完成。

当前最新代码已经做到第一次高电平立即停机，并在静止状态下确认剩余样本。该版本已准确同步到正确 Orin，但最终现场效果仍需下一次受控归零输出确认；在取得该证据前，不把“堵转问题已解决”写成已验证结论。
