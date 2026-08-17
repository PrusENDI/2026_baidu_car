# CV 自动采样巡线开发状态（2026-08-15）

本文是当前 CV 自动采样模式的上下文恢复入口。后续开始工作前，应先阅读本文，再查看
文末列出的设计、操作和验证文件，避免因对话上下文压缩而重新采用已经否定的方案。

> **文档边界（2026-08-17）：**本文继续维护 CV 视觉算法、赛道帧段、参数变化和实车实验历史。当前采集字段、CNN 输出 `[speed_demand,kappa_action]`、训练 mask 和车端 `kappa_action` profile 统一维护在 [CNN 训练接续文档](lane-cnn-training-handoff.md)。本文历史章节中的 `[vy,yaw_error]`、PID 前 `state`、按帧梯度或旧启动状态描述只用于追溯，不能覆盖训练主文档和当前代码。

当前有效 worktree 为 `C:\weizijian\documents\baidu car\baidu_smartcar_2026\.worktrees\lane-cnn-finetuning`，分支为 `orin-main-20260814`。下方旧路径、提交号和 Git 状态均保留为当时快照。

### 文档维护规则（用户要求，2026-08-16）

从现在开始，每次修改 CV 相关代码、参数、测试、离线回放逻辑或运行开关，都必须在本文
追加或更新进度记录，不能只保留在对话中。每条记录至少写明：

1. 修改日期、判断依据和目标；
2. 明确修改的文件与关键参数；
3. 生效范围，以及保持关闭或明确未修改的功能；
4. 测试、离线回放和实车验证结果；未运行的验证也必须明确注明；
5. 是否已同步 Orin、同步的文件及远端校验结果；
6. 是否已提交或推送 Git，以及对应分支和提交号；
7. 当前仍未解决的问题、已否定方案和下一步建议。

后续对话无论是否发生上下文压缩，都应先完整阅读本文并以最新日期的记录和当前代码为
事实来源。历史章节中的提交号、工作区状态和参数可能已经过时，必须结合最新追加记录及
只读 Git/代码检查确认。除非用户明确要求，不得因为更新本文而自动提交、推送或同步 Orin。

## 1. 仓库状态（历史快照）

```text
仓库：C:\weizijian\documents\baidu car\2026_official_code\baidu_smartcar_2026
分支：orin-main-20260814
当前提交：4cc938c feat(cv): fit IPM lane errors for CNN labels
远程：github/orin-main-20260814
状态：本地领先远程 4 个提交，尚未推送
```

当前未跟踪目录：

```text
2026/
artifacts/
```

这两个目录包含用户文件和离线诊断结果，不得清理、覆盖或加入提交，除非用户明确要求。

最新四个本地提交：

```text
4cc938c feat(cv): fit IPM lane errors for CNN labels
6d8090d fix(cv): delay steering onset by travel distance
72cf82a feat(cv): add steering-based adaptive speed
325fdf8 docs: design adaptive CV lane speed control
```

注意：`6d8090d` 引入的 0.15 m 转向延迟已在 `4cc938c` 中移除。不要仅根据历史提交
误以为当前控制器仍使用距离延迟。

## 2. 当前目标和已经确认的约束

CV 模式只用于固定赛道自动巡线和采集 CNN 训练数据，不替换比赛主程序中的 CNN
巡线入口。

已经确认的约束：

1. 摄像头原图和手柄数据相同，均保存为 320×240；训练时再缩放为 128×128。
2. 底盘是麦克纳姆轮，允许同时使用前进速度 `vx`、横向速度 `vy` 和角速度 `wz`。
3. 直道允许高速，弯道随 PID 前航向标签增大而自动降速。
4. 当前十字路口状态机关闭。实车高速时普通巡线和短时命令继承可以直接通过十字。
5. 不再使用原图坐标中的标准线残差斜率直接判断航向。
6. 不再使用固定的 0.15 m 转向启动延迟。
7. 每张图像必须产生只由当前帧决定、可供 CNN 学习的 `error_y/error_angle`。
8. PID 后的底盘命令必须与 PID 前 CNN 标签分开记录。

## 3. 原 CNN 控制链和标签语义

原工程的实际控制链为：

```text
CNN 输出 error_y / error_angle
  -> lane_pid.get_out(-error_y, -error_angle)
  -> y_speed / angle_speed
  -> set_velocity(forward_speed, y_speed, angle_speed)
```

代码证据位于：

```text
car_wrap_2026.py:1028  lane_base()
car_wrap_2026.py:1044  lane_pid.get_out(-error_y, -error_angle)
config_car.yml:56      lane_pid 参数
```

CV session 当前记录契约：

```text
state[0] = 实际下发的 forward_speed
state[1] = PID 前 CV error_y
state[2] = PID 前 CV error_angle

control[0] = 实际下发的 forward_speed
control[1] = PID 后实际下发的 vy
control[2] = PID 后实际下发的 wz
```

因此：

- CNN 训练继续读取 `state[1]`、`state[2]`；
- 回放和底盘控制审查读取 `control`；
- `state` 与 `control` 不应相等；
- `cv.raw_lateral/raw_heading` 保留为几何诊断量，不直接作为训练标签。

无效帧短时继承时，`state` 和 `control` 都继承上一条有效记录，并设置
`held=true`、`command_source=short_invalid_hold`。

## 4. 当前视觉算法

主要实现：

```text
smartcar/whalesbot/tools/lane_collect/opencv_lane.py
```

处理链：

```text
320×240 BGR 原图
  -> 暗色赛道二值化（threshold=175）
  -> 从近场种子提取连通区域
  -> 提取左右边界
  -> 线段筛选、短断线填补、假双边界剔除
  -> 用 standard_lane.json 对单边线补全当前车道中心
  -> 使用 perspective.json 转换为等效 IPM 坐标
  -> 完整 ROI 二次中心线拟合
  -> 近端横向位置、近端切线和曲率
  -> ErrorMapping 映射到 CNN/PID 标签
```

等效 IPM 定义：

```text
standard_center = (standard_left + standard_right) / 2
standard_width  = standard_right - standard_left

x_ipm = (current_center - standard_center)
        * perspective.wid / standard_width
s_ipm = perspective.arr[row]
```

然后拟合：

```text
x_ipm(s_ipm) = a*s_ipm^2 + b*s_ipm + c
```

输出定义：

```text
raw_lateral = 近端 x_ipm / (perspective.wid / 2)
raw_heading = atan(近端 dx_ipm/ds_ipm)
curvature   = 二次曲线近端曲率

error_y     = -0.10 * raw_lateral
error_angle = -0.40 * raw_heading
```

使用完整 ROI 的原因：近端原图拟合会把车辆横向偏移误认为航向。整圈第 3201 帧是
典型直道，旧算法产生约 `+0.646` 的假航向；等效 IPM 完整 ROI 拟合将其降至约
`+0.019 rad`。

标准参考文件：

```text
standard/standard_lane.json
standard/perspective.json
```

`perspective.json` 来自开源方案 `lane_idea` 的参数，并已插值到当前 72～192 的
120 行 ROI。它不是 OpenCV `warpPerspective` 矩阵，而是逐行的前向坐标和横向尺度参考。

## 5. 当前控制参数

主要实现：

```text
smartcar/whalesbot/tools/lane_collect/pid_control.py
```

当前使用与原 CNN 主程序相同的 PID：

```text
横向 PID：Kp=6.0, Ki=0, Kd=0.1, output_limits=[-0.7, 0.7]
航向 PID：Kp=1.95, Ki=0, Kd=0, output_limits=[-1.5, 1.5]
调用方式：pid_y(-error_y), pid_angle(-error_angle)
```

其他控制参数：

```text
控制周期：0.05 s
直道最高速度：0.20 m/s
急弯最低速度：0.08 m/s
满转向标签参考：0.60
降速曲线指数：1.5
单帧最大降速：0.015 m/s
单帧最大加速：0.010 m/s
单帧最大角速度变化：0.04 rad/s
```

速度计算使用 PID 前 `error_angle`，不是 PID 放大后的角速度。PID 前 EMA、航向死区、
启动直行距离和每次转向方向的距离延迟均已删除。角速度输出限速仍保留，但只影响
`control`，不改变 `state` 标签。

## 6. 实车采集入口和安全行为

入口：

```text
collect_data.py --cv-low-speed
smartcar/whalesbot/tools/lane_collect/ssh_test.py
```

启动命令：

```bash
python3 collect_data.py --cv-low-speed
```

控制命令：

```text
start
stop
status
quit
```

当前安全行为：

- 启动、停止、异常和退出时发送 `[0, 0, 0]`；
- 摄像头帧龄超过 0.25 s 时停车；
- 普通无效帧最多继承上一条有效命令 10 帧；
- 连续第 11 个无效帧停车；
- 十字状态机 `CROSS_STATE_ENABLED=False`；
- 每次 `start` 新建独立 session，不覆盖旧数据。

## 7. 最新整圈离线验证

数据集：

```text
C:\weizijian\documents\baidu car\lane_sessions2\lane_sessions2\lap_001
帧范围：1～3432
```

结果目录：

```text
artifacts/cv_ipm_pid_labels_0001_3432
```

整圈主要指标：

```text
有效帧数：3404
基础无效帧：28
短时继承帧：14
整体角速度相关性：0.5789
手柄转向段相关性：0.5720
方向一致率：87.77%
CV 有转向而手柄接近零：1197 帧
CV 转向有效帧：2137
平均单帧角速度变化：0.00961
大于 0.05 的跳变：2 次
```

末段直道抽查：

| 帧 | raw_heading | error_angle | CV wz | 手柄 |
|---:|---:|---:|---:|---:|
| 3201 | +0.0186 | -0.00745 | -0.0145 | 0 |
| 3226 | +0.0128 | -0.00514 | -0.0100 | 0 |
| 3301 | +0.0910 | -0.0364 | -0.0710 | -0.0168 |
| 3401 | -0.0479 | +0.0192 | +0.0374 | 0 |

结论：原来第 3201 帧的大假航向已消除，说明 IPM 解耦方向正确；但全圈相关性从上一版
约 0.63 降到约 0.58，且手柄为零而 CV 有转向的帧仍然较多。不能仅依据直道改善就
判断当前版本已经可以高速实车闭环。

## 8. 当前最重要的问题

### 8.1 2775～2781 急弯/横线误识别

2772～2780 已经进入方向不可辨识区：2772～2776 只剩很短的左边线，2777～2780
则完全无有效边界。继承机制本身工作正常，但上一条 2776 的方向已经错误：

```text
2776 CV wz：约 +0.488
2777～2780：继承约 +0.488
手柄记录：约 -0.421
```

2755～2776 的共同特征：

- 画面近场几乎被整块暗色区域占满；
- 算法只找到左边界；
- 可用边界从 88 行逐步下降到 10 行；
- `corner_detected=True`；
- 横向橙色边缘被当作继续延伸的赛道边界；
- 二次曲线近端切线在 2772 后迅速翻转。

因此问题不是“无效帧没有继承”，而是“进入无效区之前的少量假有效帧已经输出错误
方向”。从 2772 开始画面主要只剩一小段左边线，随后变成横线；单帧图像本身没有
足够信息判断车辆接下来应左转还是右转。单边线仍可能给出边界位置，但不能可靠给出
航向方向。这是观测上的不可辨识区，不应要求 CNN 从这些图像学习一个不存在的单帧
左右标签。

提速会让横线阶段经历的帧数减少，但不会改变单帧没有左右信息这个事实；如果进入
横线前的一帧已经选错方向，即使只错 2～3 帧，也可能因为角速度较大而把车辆带出
赛道。控制上应在横线起始事件被确认后，短时保持进入事件前的可信转向方向，并按
车辆行驶距离（无里程计时按 `hold_time = hold_distance / forward_speed`）退出，
而不是按固定帧数退出。数据上应将这段标记为 `held=true` 或低置信度，默认不作为
CNN 单帧监督样本。

短时保持只能延续“进入横线前已经正确”的方向，不能修复 2772～2776 已经错误的
方向。因此退化状态的触发点不能放在 2777 这种完全无效帧，至少应在“单边有效行数
快速下降且检测到横线/急弯”时提前进入。对于这条固定赛道且只执行一次的动作，可以
在确认赛道进度后使用预先确定的右转 passthrough 状态；不要简单增加继承帧数，否则
只会更久地继承错误命令。

这里必须利用时序，但时序策略应是“方向连续性锁定”，而不是盲目继承当前瞬时输出。
本圈数据表明：CV 在约 2735 帧从接近零翻成正方向，而手柄到 2747 帧开始明确负方向；
此时一直是 `left_only`，新正方向没有双边线高置信证据。因此可采用以下最小状态：

```text
NORMAL：接受双边线或高置信几何结果
PENDING：低置信单边线产生相反方向时，不立即切换，只累计候选方向
TURN_LOCK：候选方向连续满足高置信/固定赛道事件后才切换
AMBIGUOUS_HOLD：有效行数快速下降或横线出现时，保持 TURN_LOCK 的方向
```

这样可以利用进入弯道前的时序趋势，拒绝 2735～2776 期间由单边线引起的瞬时反向；
但如果进入 `AMBIGUOUS_HOLD` 前的锁定方向本身错误，时序也无法凭空修正，仍需固定
赛道事件或实车标定提供右转先验。

### 8.2 横向控制符号和增益尚未实车确认

当前初始映射为：

```text
error_y = -0.10 * raw_lateral
```

手柄 `lap_001` 的横向标签全部为零，离线数据无法验证麦克纳姆横移的正确方向和恢复
速度。第一次实车验证必须低速制造轻微左偏/右偏，确认 `vy` 将车辆推回赛道中心，
而不是继续推离中心。

### 8.3 整圈趋势相关性仍不够高

当前全圈相关性约 0.58，低于上一版约 0.63。原因不能简单解释为 CV 转得更急，主要还
包括：

- CV 转向持续时间明显长于手柄；
- 急弯/横线附近存在错误方向；
- 手柄包含驾驶者的稀疏操作和回正策略，CV 是连续几何纠偏；
- 当前 `-0.40` 航向映射是初始比例，不是完整标定结果；
- 近端切线位置在稳定直道和急弯响应之间仍有折中。

相关性只是诊断指标。最终标准仍是车辆能否在目标速度下以标准姿态稳定完成整圈，
同时标签方向、范围和图像时序可用于 CNN 学习。

### 8.4 标签虽然是单帧的，但控制仍有历史

`state[1:3]` 只由当前帧产生，满足单帧 CNN 标签要求；`control` 仍受以下历史状态影响：

- PID 的 `Kd`；
- 角速度单帧变化限制；
- 前进速度加减速限制；
- 无效帧命令继承。

这是预期行为。训练应使用 `state`，底盘审查使用 `control`，不能再次把两者合并。

## 9. 下一步建议顺序

### 9.0 正式评测目标：零时延趋势一致

当前最终目标不是让 CV 几何量单独看起来平滑，而是让 CV 输出与手柄在同一赛道位置
出现相同的转向趋势和时序。只有这样，提速后车辆才会在同一位置开始转向、在同一
位置回正，而不是靠更低速度掩盖提前或滞后。

后续每次整圈比较必须同时报告：

```text
zero_lag_correlation          零时延相关性
best_lag_correlation          允许搜索时移后的相关性
best_lag_frames               最佳时移帧数，目标接近 0
turn_direction_accuracy       有效转向帧方向一致率
onset_frame_error             每个弯道开始转向的帧差
release_frame_error           每个弯道回正的帧差
manual_zero_cv_active_count   手柄直行而 CV 转向的帧数
```

当前 `lap_001` 基线：

```text
整圈零时延相关性：约 0.579
整圈最佳时移相关性：约 0.623，最佳时移约 -7 帧
2700～2810 最佳时移超过 +40 帧，方向还存在反向问题
3200～3432 最佳时移约 -34 帧，说明末段仍有明显时序差异
```

### 全局提前量状态机实验结果

曾将 `SharpTurnStateMachine` 以 `enter_heading=0.03` 全赛道启用，规则是提前检测后
锁定方向，单边线不允许反向，双边线低航向才退出。该实验已经离线跑完整圈，但结果
不可接受：

```text
整体相关性：0.320
手柄转向段相关性：0.351
方向一致率：61.6%
第一十字窗口方向一致率：6.25%
第二十字窗口方向一致率：0%
```

原因是低阈值预检测会把普通小曲率和噪声提前锁成转弯，导致状态机在其他赛段误触发。
该全局接入已关闭，不得作为当前实车版本使用。保留的安全规则仅是：当状态机未来在
固定赛道目标弯道中启用后，单边线不能推翻已锁定方向，必须等待 `both/mixed` 双边线
恢复并满足退出条件。

“最佳时移相关性高”不能替代“零时延相关性高”。后续调参优先修正弯道事件的开始、
持续和回正时刻，再调整整体幅值；不能只通过提高或降低比例系数来掩盖时间错位。

### P0：修复急弯横线前的假有效帧

重点分析 `lap_001` 的 2755～2785，并至少检查：

1. `corner_detected=True` 且单边有效行持续快速减少时，是否提前进入退化状态；
2. 近场覆盖率、中心线到达深度和横线水平度是否可作为可靠门限；
3. 是直接继承最后稳定方向，还是使用已有 `SharpTurnStateMachine` 的方向锁定逻辑；
4. 修复后不得破坏第一、第二十字路口的直行通过能力。

验收必须同时看：2776 方向、2777～2780 继承方向、整圈方向一致率和直道假转向数。

### P0：低速实车确认横向符号

先将车辆放在直道，分别轻微左偏和右偏，确认：

```text
raw_lateral 符号
error_y 符号
PID 后 vy 符号
车辆实际横移方向
```

任何一项相反都应立即停车并修正映射符号，不应直接跑整圈。

### P1：重新跑整圈并比较控制趋势

修复急弯保护后重新生成独立 artifacts 目录，比较：

- 全圈和转向段相关性；
- 方向一致率；
- 手柄为零而 CV 转向的帧数；
- 217～278、1161～1181、2755～2785、3201～3432；
- `state` 标签分布与 `control` 命令分布。

### P1：训练侧兼容性复核

远程训练器必须继续：

- 从 `state[1]` 读取第一个输出；
- 从 `state[2]` 读取第二个输出；
- 使用 320×240 原图并在训练时缩放到 128×128；
- 从原 CNN 权重继续微调；
- 按完整 session 划分训练集和验证集，不能随机打散相邻视频帧。

同时确认训练工具没有把新的 `state[1:3]` 再当作 PID 后 `vy/wz` 做二次缩放。

## 10. 当前验证状态

聚焦测试：

```text
tests/test_opencv_lane_analyzer.py
tests/test_cv_lane_pid_control.py
tests/test_turn_state.py

结果：29 passed
```

整个 `tests/` 目录结果：

```text
37 passed, 1 failed
```

唯一失败为已有的 `test_manual_key_gate.py`，它要求 `car_wrap_2026.py` 中存在精确文本
`if key_val == 1:`，与本次 CV 改造无关。

直接运行全仓库 `pytest` 还会收集 Paddle Serving 示例，并因当前环境未安装
`paddle_serving_client` 而停止。CV 聚焦测试应继续使用：

```powershell
& 'C:\weizijian\documents\baidu car\baidu_smartcar_2026\.worktrees\lane-cnn-finetuning\.venv-lane\Scripts\python.exe' `
  -m pytest tests\test_opencv_lane_analyzer.py `
            tests\test_cv_lane_pid_control.py `
            tests\test_turn_state.py -q
```

## 11. 关键文件索引

```text
视觉分析：smartcar/whalesbot/tools/lane_collect/opencv_lane.py
误差映射：smartcar/whalesbot/tools/lane_collect/calibration.py
CV PID：smartcar/whalesbot/tools/lane_collect/pid_control.py
实车入口：smartcar/whalesbot/tools/lane_collect/ssh_test.py
十字/急弯状态：smartcar/whalesbot/tools/lane_collect/turn_state.py
整圈比较：scripts/compare_cv_control.py
采集入口：collect_data.py
主 CNN 控制：car_wrap_2026.py
主 PID 配置：config_car.yml
```

配套文档：

```text
docs/cv-lane-collection-operation-guide.md
docs/opencv-cv-low-speed-ssh-test.md
docs/superpowers/specs/2026-08-15-cv-ipm-cnn-label-design.md
docs/superpowers/plans/2026-08-15-cv-ipm-cnn-label-plan.md
```

## 12. 恢复上下文时的最短检查清单

后续模型或开发者接手时，按以下顺序恢复：

1. 阅读本文；
2. 执行 `git status --short --branch`，确认分支和未跟踪目录；
3. 查看 `4cc938c`，确认当前 `state/control` 语义；
4. 查看 `artifacts/cv_ipm_pid_labels_0001_3432/comparison_summary.json`；
5. 查看 2768、2776、2781、3201 四张原图；
6. 运行 29 个聚焦测试；
7. 从“P0：修复急弯横线前的假有效帧”继续，不要重新引入原图斜率或固定 0.15 m 延迟。

## 13. 无路线限定的边界连续性保护（待实现）

### 13.1 约束

后续不再实现路线限定状态机。禁止使用：

- 固定帧号或固定里程区间；
- “第几个弯道”“第一次/第二次十字路口”等路线次数；
- 针对 2772～2780 或其他已知窗口硬编码方向；
- 只在某一个赛道事件执行一次的 one-shot 状态。

允许保留的历史仅是通用视觉连续性信息，例如最近几帧可相信的边界侧别、航向方向和峰值。
相同规则必须能够在整圈每一次单边线退化中重复触发和自动释放。

现有 `CrossStraightStateMachine` 和低阈值全局 `SharpTurnStateMachine` 均不得重新接入实车控制。
新逻辑应实现为独立的“边界连续性保护/滤波器”，而不是路线事件状态机。

### 13.2 通用视觉规则

边界侧别本身提供急弯方向证据：

```text
left_only  -> 图像航向应为正 -> 底盘角速度为负 -> 右转
right_only -> 图像航向应为负 -> 底盘角速度为正 -> 左转
```

该对应关系只用于判断单边拟合是否发生不可信反向，不能在普通单边线帧中强制制造大转角。

处理顺序：

1. 当单边拟合连续多帧与侧别所代表的方向一致时，记录该方向和最近可信航向峰值。
2. 当前航向仍与记录方向一致时，完全采用当前帧拟合，不修改正常弯道输出。
3. 同一侧单边线持续存在，但当前航向突然反向时：
   - 尚未检测到横线/急弯证据时，不立即放大反向输出；
   - 检测到横线/急弯证据后，认为单边二次拟合已经越过可观测范围，继承此前可信方向；
   - 保护幅值使用通用下限与最近可信峰值的较大者，不使用赛道位置决定幅值。
4. 完全无边界的短暂帧继续继承上一条有效底盘命令，由统一的丢线安全超时负责停车。
5. 可见边界从 `left_only` 切换为 `right_only`，或反向切换时，立即释放原方向保护，让当前几何量执行回正/反打。
6. `both/mixed` 双边线恢复时清除单边历史；下一弯道重新按同一规则判断。

这套逻辑不锁定固定角速度直到双边线恢复。边界侧别切换后必须允许立即回正，否则会把正确的转弯后半段也错误保持成同一方向。

### 13.3 离线原型结果

以下结果来自 `lap_001` 1～3432 的一次性离线原型，尚未写入生产代码，也未接入 SSH 实车：

```text
整体零时延相关性             0.5789 -> 0.6752
手柄转向段相关性             0.5720 -> 0.7142
转向方向一致率               87.77% -> 93.36%
手柄为零但 CV 有转向         1197 -> 1204 帧
整圈最佳时移相关性           0.7423
整圈最佳时移                 约 +9 帧
2700～2920 零时延相关性       0.8769
2700～2920 最佳时移           约 +6 帧
```

这里的正时移表示：要把手柄曲线向后移动，才能与当前 CV 曲线最好对齐，因此原始 CV 仍然提前。
方向错误已经大幅减少，但“手柄为零而 CV 有转向”的帧数没有下降，说明这一版主要修复了退化阶段的错误方向，
并没有解决所有普通弯道的提前转向。

### 13.4 为什么整体相关性仍只有约 0.68

整体相关性仍偏低不是一个单独的增益问题，主要由以下时序差异共同造成：

1. **普通弯道仍提前响应。** CV 对远场轻微几何变化连续输出小转角，而手柄通常先保持零，再以阶梯方式开始转向。
   因此即使方向正确，零时延逐帧相关性仍会被大量“CV 已转、手柄仍为零”的帧拉低。
2. **第一十字附近仍明显提前。** 该窗口 CV 大约从 180 帧开始持续转向，手柄约从 206 帧开始；
   217～278 窗口的最佳时移约为 +41 帧。边界连续性保护只能避免 248 帧附近错误反向，不能自动修正前半段过早起转。
3. **2772～2780 的方向已修复，但不能代表整圈。** 2700～2920 主转向的明显起转约在 2742 帧，手柄约在
   2747 帧，仍提前约 5 帧；该窗口整体最佳时移约 +6 帧，已经接近目标但还不是零。
4. **3200～3432 末段仍严重失配。** 该窗口零时延相关性约 0.032，最佳时移仍约 +34 帧，是整圈相关性的重要短板。
5. **输出形状不同。** 手柄标签包含驾驶员的零保持、阶梯加角和主动反打；CV 是连续几何误差，经 PID 和角速度步进限制后仍比手柄更平滑、更长。
   相同方向和相近峰值并不保证逐帧 Pearson 相关性高。
6. **幅值标定仍不完全一致。** 通用保护下限可以补足退化阶段的转向，但普通弯道的 `heading_scale`、PID 增益和手柄幅值分布仍有差异。

所以当前结论是：新逻辑值得实现，因为它修复了最危险的单边线错误反向；但它不是完整的时序校准。
实现后仍需单独处理普通弯道起转过早和 3200～3432 末段失配，不能继续单纯提高保护幅值来追求相关性。

### 13.5 下一轮验收

实现边界连续性保护后，整圈评测至少同时报告：

```text
zero_lag_correlation
best_lag_correlation
best_lag_frames
turn_direction_accuracy
manual_zero_cv_active_count
217～278、1161～1181、2700～2920、3200～3432 分窗口指标
```

本轮只接受通用视觉条件带来的改善。若某个改动只能改善一个固定窗口，却降低其他弯道或需要知道赛道位置，则不得接入实车版本。

### 13.6 启动阶段的唯一里程保护

启动后的第一个弯道允许一个明确的启动例外：实车控制使用底盘编码器里程，前进
`0.10 m` 内强制 `vy=0、wz=0`，只保持前进，之后立即交还通用视觉控制。

该保护位于 `ssh_test.py` 的实际发送命令层，不能使用图像帧号代替里程，也不能复制到后续弯道。
采集记录仍保存 PID 前的 `state[1:3]` 标签，实际发送的直行覆盖只记录在 `control` 和
`command_source=initial_straight_guard` 中。

## 14. 远场定方向、中近场定时机：第一版实现

新增通用连续滤波器：

```text
smartcar/whalesbot/tools/lane_collect/temporal_filter.py
```

该实现没有固定帧号、里程窗口、弯道次数或有限事件状态。每帧计算：

```text
preview_direction   远场/单边连续方向
arrival_weight      中近场弯道到达度，范围 0～1
lateral_motion_ema  近场横向姿态变化量
coverage_loss       边界覆盖损失
direction_held      单边线反向是否被拒绝
```

方向只由此前可靠远场趋势以及 `left_only/right_only` 的边界拓扑连续性确定。中近场仅通过
横向姿态变化、横线置信度和边界覆盖损失计算 `arrival_weight`，不能推翻方向。

整圈 `lap_001` 1～3432 离线结果：

```text
有效赛段：1～3397（3398 起为终点区，不计分）
整体零时延相关性             0.5802 -> 0.6805
手柄转向段相关性             0.5720 -> 0.7168
方向一致率                   87.77% -> 94.18%
手柄为零但 CV 转向           1167 -> 610 帧
方向反转次数                 34 -> 18
整圈最佳时移                 +7 -> +9 帧
整圈最佳时移相关性           0.6240 -> 0.7346
```

关键时序：

```text
第一主弯：CV 188，手柄 206，仍提前 18 帧
右转 2700～2854：CV 2741，手柄 2747，仍提前 6 帧
左转 2855～2920：CV 与手柄首次非零均为 2855；手柄持续左转约从 2870 开始，
CV 持续左转仍提前约 15 帧；窗口方向一致率 100%
2777～2780：方向一致率 100%，CV 保持右转
3180～3397：有效转向帧方向一致率 100%；该窗口包含多个短动作，不能合并成一个弯道计算起点
3398～3422：车辆已到终点，手柄转向不是有效教师动作，完全排除评分
```

结论：第一版明显减少了无意义的提前小转向，并修复了 2777～2780 的方向；但整圈最佳时移
没有接近零，第一主弯仍提前。因此 `ssh_test.py` 已接好代码但
`PREVIEW_TIMING_ENABLED=False`，当前只允许通过 `compare_cv_control.py --temporal-filter`
离线评测，不得默认用于实车。

离线结果目录：

```text
artifacts/cv_preview_timing_valid_lap_0001_3397
```

## 15. 到达权重提前问题的验证结果（2026-08-16）

逐帧追踪确认，当前单一 `arrival_weight` 同时承担起转、弯中保持和短时断线保持。它采用
`max(当前弯角/横移/覆盖证据, 历史值 × 0.97)`，因此存在结构性冲突：衰减慢会把上一动作带入
下一弯，衰减快又会在弯中几何证据短暂下降时过早卸掉转向。

本轮依次离线验证了三种连续参数方案，均使用有效范围 1～3397：

```text
方案                              整体相关性  转向段相关性  最佳时移  结果
原始第一版                        0.6805      0.7168       +9       当前保留
单边横移需几何支持、取消强制到达  0.6317      0.6540       +11      后段有效小弯被压制
仅在单边模式限制横移              0.6433      0.6663       +10      第一弯仍提前 17 帧
加快衰减并提高弯角门槛            0.5850      0.6028       +12      第一弯持续动作晚到 31 帧
```

因此上述控制行为修改已撤回，保留成绩最好的第一版；`PREVIEW_TIMING_ENABLED` 继续保持
`False`。已保留两项不改变控制的改进：

- 输出弯角、覆盖、横移及横移支持量等诊断字段；
- `render_cv_frame_review.py` 从第 1 帧预热分析器、滤波器、PID 和无效帧保持，只保存请求窗口，
  避免从 3200 等晚段直接启动造成错误的时间状态。

下一版不应继续调整同一个权重的阈值。应将“当前帧起转证据”和“已建立转向后的短时保持”拆成
两个连续量：前者只负责起转，后者只能延续已经发生且方向连续的动作，不能独立开启转向。

## 16. 当前修改清单与工作区状态（2026-08-16）

### 16.1 Git 状态

```text
仓库：C:\weizijian\documents\baidu car\2026_official_code\baidu_smartcar_2026
分支：orin-main-20260814
远程：github/orin-main-20260814
状态：本地领先远程 12 个提交
最新提交：08d7c5a feat(cv): hold launch straight for initial odometry distance
```

`2026/` 和 `artifacts/` 是未跟踪目录，必须保留，不能在整理代码时删除或覆盖。当前远场方向、
中近场时序滤波和审查工具仍是未提交工作；在用户明确要求前不要提交或推送。

### 16.2 已提交且继续生效的修改

- CV 采集保存原始 `320×240` 图像，训练阶段再缩放；
- `state = [forward_speed, error_y, error_angle]`，其中 `state[1:3]` 保持为 CNN 学习并送入
  原工程双 PID 的单帧误差；
- 实际麦克纳姆底盘命令单独保存到 `control = [vx, vy, wz]`，不得拿历史滤波后的底盘命令
  替换单帧 CNN 标签；
- 启动后使用编码器里程强制直行 `0.10 m`，仅这一次允许 `vy=0,wz=0`，不用帧号，也不能
  复用到后续弯道；
- 直道允许高速，随 PID 前航向误差增大自动降速；无效短帧继承最近有效底盘命令；
- 固定赛道参考、透视宽度、左右边界和单边标准线偏差均已接入基础 IPM 寻线。

### 16.3 当前保留但尚未提交的修改

| 文件 | 修改内容 | 当前状态 |
|---|---|---|
| `smartcar/whalesbot/tools/lane_collect/opencv_lane.py` | 输出 `ipm_far_s`、`ipm_far_x`、`preview_offset` 和归一化远场偏移 | 保留 |
| `smartcar/whalesbot/tools/lane_collect/temporal_filter.py` | 新增无固定路线状态的 `PreviewTimingFilter`，远场定方向，中近场到达权重定执行幅值 | 保留，实车关闭 |
| `smartcar/whalesbot/tools/lane_collect/__init__.py` | 导出 `PreviewTimingConfig/PreviewTimingFilter` | 保留 |
| `smartcar/whalesbot/tools/lane_collect/ssh_test.py` | 接入滤波器、启动/停止时复位，并增加安全开关 | `PREVIEW_TIMING_ENABLED=False` |
| `scripts/compare_cv_control.py` | 增加 `--temporal-filter`、最佳时移、方向一致率、误激活和关键窗口统计 | 保留 |
| `scripts/render_cv_frame_review.py` | 生成原图/边界/中心线/控制诊断逐帧图；从第 1 帧预热，只保存请求窗口 | 保留 |
| `tests/test_preview_timing_filter.py` | 覆盖远场方向保持和中近场到达控制 | 保留 |
| `docs/superpowers/plans/2026-08-15-preview-direction-timing-plan.md` | 记录实现、验证、否决方案及恢复过程 | 保留 |

滤波器额外输出以下诊断量，不改变当前最佳控制行为：

```text
corner_arrival_signal
coverage_arrival_signal
lateral_motion_signal
supported_motion_signal
preview_signal_ema
preview_heading
arrival_weight
direction_held
```

### 16.4 已验证后撤回的修改

以下方案不得误认为当前代码目标，也不应在新对话中直接重新应用：

- 每次远场符号变化就清空到达权重：同一右弯内远场符号会在 2775、2781、2807 多次变化，
  会导致弯中错误卸载；
- 所有横移都必须有弯角/覆盖支持：会压掉 3180～3397 的有效小幅转向；
- 只在单边模式限制横移：整体相关性下降且第一主弯仍提前；
- 将 `arrival_decay` 从 `0.97` 调到 `0.90` 并提高弯角门槛：第一弯持续动作反而晚 31 帧；
- 对整圈统一增加 9 帧固定延迟：没有适应速度和不同弯道几何，不采用。

代码已经恢复为本轮评测中成绩最好的第一版控制行为。下一步若继续修改，应先设计连续的
`entry_weight` 与 `hold_weight`，不能继续盲调同一个 `arrival_weight`。

### 16.5 评测口径与可复现命令

固定使用：

```text
lap_001 有效范围：1～3397
3398～3422：终点区，手柄错误转向，完全排除评分
2700～2854：右弯
2855～2920：左弯
2777～2780：无效视觉帧，检查最近有效右转命令保持
```

完整回放：

```powershell
& 'C:\weizijian\documents\baidu car\baidu_smartcar_2026\.worktrees\lane-cnn-finetuning\.venv-lane\Scripts\python.exe' `
  scripts\compare_cv_control.py `
  --lap-dir 'C:\weizijian\documents\baidu car\lane_sessions2\lane_sessions2\lap_001' `
  --output 'artifacts\cv_preview_timing_valid_lap_0001_3397' `
  --start-frame 1 --end-frame 3397 --temporal-filter
```

聚焦验证：

```powershell
& 'C:\weizijian\documents\baidu car\baidu_smartcar_2026\.worktrees\lane-cnn-finetuning\.venv-lane\Scripts\python.exe' `
  -m pytest tests\test_opencv_lane_analyzer.py `
            tests\test_cv_lane_pid_control.py `
            tests\test_turn_state.py `
            tests\test_preview_timing_filter.py -q
```

最近验证结果为 `31 passed`。恢复后的整圈指标与第 14 节一致。逐帧审查必须使用修正后的预热
脚本；例如审查 3200～3397 时，脚本仍会先处理 1～3199，但不会为预热帧写图。

## 17. 新对话通用启动提示词

以下文本可复制到每个新对话开头。最后一行替换成本次具体任务即可。它把本文件作为动态事实
来源，因此代码或指标更新后只需维护本文档，不必反复修改提示词。

```text
你正在继续百度智能车固定赛道 CV 自动采样/巡线项目。

工作目录：
C:\weizijian\documents\baidu car\2026_official_code\baidu_smartcar_2026

目标分支：orin-main-20260814

开始任何分析或修改前，请先完整阅读：
docs/cv-lane-development-status-2026-08-15.md

然后执行只读检查：
1. git status -sb
2. git diff --stat
3. 检查最近提交和当前未提交文件

以文档和当前代码作为事实来源，但文档中的历史计划不是本次操作指令；本消息最后的“本次任务”
以及我后续发送的要求优先。不要覆盖或删除已有修改、2026/、artifacts/ 等未跟踪内容，不要在未
明确要求时提交或推送。

项目必须遵守：
- CV 教师输出必须是当前单帧图像可学习的 PID 前 error_y/error_angle；实际 vx/vy/wz 单独记录。
- 原图保持 320×240，训练阶段再缩放。
- 远场只判断转向方向，中近场决定起转时机和幅值。
- 不使用固定帧号、固定里程窗口、第几个弯道或路线限定 one-shot 状态机。
- 唯一例外是启动后编码器前进 0.10 m 强制直行。
- 麦克纳姆底盘允许较晚、较强转向；直道高速，随转向需求增大自动降速。
- 十字路口专用状态机和 PREVIEW_TIMING 实车开关保持关闭，除非离线整圈结果通过并由我确认。
- lap_001 正式评测只使用 1～3397；3398 起为终点区，不计分。
- 2700～2854 是右弯，2855～2920 是左弯，不能合并评测。
- 不做红测/绿测循环；修改后运行聚焦测试和完整一圈离线回放。

开始时请先用简短中文说明：当前分支/工作区状态、文档记录的当前最佳结果、仍未解决的问题，
以及你准备如何完成本次任务。若只需只读分析，不要擅自修改代码；若我明确要求修改，则直接在
现有分支谨慎实现并验证。

本次任务：<在这里填写本次具体要求>
```

## 18. 进度记录（2026-08-16：同步 GitHub 与固化文档维护要求）

本地分支已从 GitHub fast-forward 到：

```text
分支：orin-main-20260814
提交：4be08e3 feat(cv): add preview direction timing diagnostics
远端：origin/orin-main-20260814
```

本次仅拉取并阅读代码与文档，没有同步 Orin，也没有运行测试或离线回放。拉取前已有的
日志删除项、模型目录、`artifacts/` 和其他未跟踪内容均保留，未清理、覆盖或加入提交。

用户明确要求：以后每次修改都必须在本文写明进度，防止上下文压缩后遗忘。本要求已写入
文首“文档维护规则”，后续修改必须同时维护本文；本次文档修改尚未提交、推送或同步 Orin。

### 18.1 Orin 运行文件同步（2026-08-16）

已按用户确认，仅同步以下 6 个运行文件到
`/home/jetson/workspaces/baidu_smart_2026_8_14/` 对应路径：

```text
smartcar/whalesbot/tools/lane_collect/__init__.py
smartcar/whalesbot/tools/lane_collect/opencv_lane.py
smartcar/whalesbot/tools/lane_collect/pid_control.py
smartcar/whalesbot/tools/lane_collect/ssh_test.py
smartcar/whalesbot/tools/lane_collect/temporal_filter.py
smartcar/whalesbot/tools/lane_collect/turn_state.py
```

未同步文档、离线脚本或测试文件。6 个文件的本地与远端 SHA256 全部一致，
`temporal_filter.py` 已在 Orin 新建。远端关键状态确认：

```text
max_forward_speed=0.20
min_forward_speed=0.08
initial_straight_distance_m=0.10
MAX_INVALID_HOLD_FRAMES=10
CROSS_STATE_ENABLED=False
PREVIEW_TIMING_ENABLED=False
```

本次同步后未运行测试、离线回放或实车验证；没有提交或推送本次本地文档修改。

### 18.2 暂停横向底盘输出（2026-08-16）

由于 `vy` 的实车符号和恢复增益仍未单独确认，用户决定先将横向底盘输出设为 0，避免
横向纠偏方向错误时把车辆快速推离赛道。

本次修改：

```text
smartcar/whalesbot/tools/lane_collect/pid_control.py
  CvLanePidConfig.lateral_limit: 0.70 -> 0.0

tests/test_cv_lane_pid_control.py
  更新断言：保留 PID 前 error_y，但实际 lateral_speed 必须为 0

docs/cv-lane-collection-operation-guide.md
  更新当前横向输出和已知限制说明
```

生效语义：`state[1]` 继续记录当前帧 `error_y`，用于诊断和训练；`control[1]` 以及实际发送
给底盘的 `vy` 恒为 0。航向、自适应速度、启动 0.10 m 直行保护、无效帧 10 帧保持均未改；
`CROSS_STATE_ENABLED=False` 和 `PREVIEW_TIMING_ENABLED=False` 保持不变。

验证与同步：

```text
tests.test_cv_lane_pid_control：13 tests passed
git diff --check：通过
Orin 仅同步 pid_control.py
远端 lateral_limit=0.0
本地/远端 SHA256：924f91e283920397fed3378771c02f8bd244f25027669187b4e0dc37ef78e7fe
```

本次没有运行完整一圈离线回放或实车验证；本地源码、测试和文档修改尚未提交或推送 GitHub。

### 18.3 减弱弯道降速并加快出弯恢复（2026-08-16）

实车反馈当前降速过于激进。旧策略在 `abs(error_angle)=0.40` 时立即给出最低速度
`0.08 m/s`，并允许每帧降速 `0.03 m/s`、但每帧只恢复 `0.005 m/s`，表现为入弯快速
降到低速、出弯长时间恢复不上来。

本次修改：

```text
smartcar/whalesbot/tools/lane_collect/pid_control.py
  full_turn_reference: 0.40 -> 0.60
  max_deceleration_step: 0.03 -> 0.015 m/s
  max_acceleration_step: 0.005 -> 0.010 m/s

tests/test_cv_lane_pid_control.py
  更新减速和恢复步进断言

docs/cv-lane-collection-operation-guide.md
  更新当前控制参数表
```

保持不变：`min_forward_speed=0.08`、`max_forward_speed=0.20`、
`speed_curve_exponent=1.5`、`lateral_limit=0.0`、启动 0.10 m 直行保护、10 帧无效保持；
十字状态机和预览时序滤波继续关闭。

验证与同步：

```text
tests.test_cv_lane_pid_control：13 tests passed
git diff --check：通过
Orin 仅同步 pid_control.py
本地/远端 SHA256：288fb3b4c3a67f04678e177802cd87bbc5d2d44e6ce756abcdfb24df75fc70c7
```

本次尚未运行完整一圈离线回放或实车验证；本地源码、测试和文档修改尚未提交或推送 GitHub。

### 18.4 IPM 中线 Pure Pursuit 第一版（2026-08-16）

判断依据：公开第 18 届镜头组代码采用“单边边线内移重建中线、按物理距离重采样、
固定预瞄点 Pure Pursuit、转向 PD/角速度反馈”的结构。当前项目已经具有逐行单边中线
重建和厘米单位 IPM 坐标，旧控制最后一步却仍由完整远近场拟合的近端切线直接产生转向，
容易把路径预告和当前应转曲率混为一体。本次仅替换 CV 采集入口的转向生成，不接入 CNN
主程序。

修改文件与参数：

```text
smartcar/whalesbot/tools/lane_collect/opencv_lane.py
  pure_pursuit_lookahead_m=0.30
  ipm_units_per_meter=100.0
  在原始 IPM 中线直接插值目标点
  curvature=-2*x/(x*x+s*s)

smartcar/whalesbot/tools/lane_collect/pid_control.py
  steering_mode 保留 heading_pid 回退，CV 入口启用 pure_pursuit
  target_wz=forward_speed*curvature
  full_turn_curvature_m_inv=5.0
  pure_pursuit_gain=1.0
  Pure Pursuit 入弯步进 0.10 rad/s/帧，回正/反向步进 0.04 rad/s/帧
  heading_pid 回退模式仍保持原 0.04 rad/s/帧

smartcar/whalesbot/tools/lane_collect/ssh_test.py
  仅 --cv-low-speed 构造 pure_pursuit 控制器
  session teacher=opencv_ipm_pure_pursuit

scripts/compare_cv_control.py
  新增 --pure-pursuit 和 --lookahead-m
  离线无效帧保持与实车统一为 10 帧

tests/test_opencv_lane_analyzer.py
tests/test_cv_lane_pid_control.py
  增加目标点插值、坐标符号、wz=v*kappa、标签换算及入弯/回正步进测试
```

保持不变：`vy=0`、启动后编码器前 0.10 m 强制直行、十字状态机关闭、
`PREVIEW_TIMING_ENABLED=False`、连续无效 10 帧保持而第 11 帧停车、原图 320x240、CNN
主程序隔离。没有恢复任何普通弯道固定延迟。

验证：聚焦 `unittest` 共 29 项通过，`py_compile` 和 `git diff --check` 通过。整圈
`lap_001` 使用有效帧 1～3397 回放，并扫过 0.12、0.15、0.20、0.25、0.30 m 五个预瞄
距离。0.30 m 综合最好：

```text
                         旧 heading_pid    Pure Pursuit 0.30 m
全局相关                 0.517             0.619
人工转弯帧相关           0.480             0.688
同方向率                 85.6%             89.4%
大于 0.05 的步进次数     91                19
总变化量                 35.7              18.9
最佳相关滞后             +6 帧             +8 帧
```

首次实现回放时发现 IPM 横坐标与底盘 yaw 正方向相反；未取反版本相关为 -0.574，已在
同步实车前修正并由测试锁定。短预瞄并不会自动推迟转向，会因距离平方更小而放大横向纠偏。

仍未解决：第一普通弯窗口的主转向起点指标仍比人工主转向早 56 帧，逐帧数据表明主要是
Pure Pursuit 在纠正约 3 cm 横向位置，而非远场切线直接起转。当前没有擅自增加横向死区或
固定延迟；需要结合最新实车图像判断这是必要的居中控制还是应减弱的误差。2700～2854
右弯和 2855～2920 左弯不得合并评价；已分别保留在回放汇总中。

同步与版本状态：本次尚未同步 Orin，尚未实车验证，尚未提交或推送 Git。离线产物位于
`artifacts/cv_pure_pursuit_*`，未修改模型、数据集和日志；工作区原有删除项与未跟踪内容
均未清理或覆盖。

### 18.5 Pure Pursuit 实车试验与启动直行增加 5 cm（2026-08-16）

用户确认将 Pure Pursuit 第一版同步 Orin 实车测试，并要求发车强制直行在原 0.10 m
基础上增加 5 cm。本次唯一参数变化：

```text
smartcar/whalesbot/tools/lane_collect/pid_control.py
  initial_straight_distance_m: 0.10 -> 0.15 m
```

`ssh_test.py` 同步更新注释；操作指南和测试同步更新。该保护仍只在每次 `start` 后按编码器
累计距离生效一次，不是普通弯道延迟。Pure Pursuit 仍为 0.30 m 预瞄、`wz=v*kappa`、
入弯步进 0.10 rad/s/帧；`vy=0`、十字状态机关闭、`PREVIEW_TIMING_ENABLED=False`、
无效保持 10 帧和第 11 帧停车均未改变。没有接入 CNN 主程序。

验证结果：`py_compile` 通过，聚焦 `unittest` 共 30 项通过，`git diff --check` 通过。

已使用 `tools/sync_to_orin.ps1`，仅同步以下 3 个运行文件到
`/home/jetson/workspaces/baidu_smart_2026_8_14/`：

```text
smartcar/whalesbot/tools/lane_collect/opencv_lane.py
smartcar/whalesbot/tools/lane_collect/pid_control.py
smartcar/whalesbot/tools/lane_collect/ssh_test.py
```

远端参数确认：

```text
pure_pursuit_lookahead_m=0.30
steering_mode="pure_pursuit"
initial_straight_distance_m=0.15
pure_pursuit_entry_step=0.10
CROSS_STATE_ENABLED=False
PREVIEW_TIMING_ENABLED=False
```

本地/Orin SHA256 一致：

```text
opencv_lane.py  3064b20b5e8179b775569d0b02df9918ccb4378f6f3873f555e65bcd90bd546e
pid_control.py  463fcee31b7a16c11b9731e0ec39679d5bed80a7742a582fd3b60c34310fdf01
ssh_test.py     3c8e7842d6fffd8c1bae15aeaa16257c8a48a63aa44882a3118d82ea71cb480b
```

本次尚未取得 Pure Pursuit 实车 session，尚未提交或推送 Git。下一步必须重点观察发车
0.15 m 保护结束瞬间、第一个普通弯横向纠偏是否仍表现为提前转向，以及直角弯的峰值
`pure_pursuit_curvature_m_inv/angular_speed/forward_speed`；不要同时开启十字或预览滤波。

### 18.6 全局缩短 Pure Pursuit 半径（2026-08-16）

实车 session `cv_low_speed_20260816_151003_287664` 共 765 帧、16.23 m。用户已在 Orin
手动将直道/弯道速度改为 0.30/0.12 m/s。三个主要锐角弯在 0.30 m 预瞄下曲率长期卡在
约 3.33 1/m，实际 `wz` 约 0.515 rad/s、等效半径约 0.30 m，车辆转不过去；同帧按
0.15 m 预瞄离线重算，弯中曲率约 6.46～6.65 1/m，可表达约 0.15 m 半径。

用户判断麦轮底盘理论上允许原地转向，且赛道只有两处普通弯、其余主要为锐角和直角弯，
决定不采用双预瞄或锐角状态机，直接全局缩短转弯半径。修改：

```text
opencv_lane.py
  pure_pursuit_lookahead_m: 0.30 -> 0.15 m

pid_control.py
  max_forward_speed: 0.20 -> 0.30 m/s（固化 Orin 手调值）
  min_forward_speed: 0.08 -> 0.12 m/s（固化 Orin 手调值）
```

保持不变：`initial_straight_distance_m=0.15`、`pure_pursuit_gain=1.0`、
`pure_pursuit_entry_step=0.10`、`heading_limit=1.50`、`vy=0`、十字关闭、预览滤波关闭、
无效保持 10 帧。没有增加弯道识别、固定延迟或路线状态机，也没有接入 CNN 主程序。

验证：`py_compile`、`git diff --check` 通过，聚焦 `unittest` 共 30 项通过。使用最新
session 图像按 0.15 m、0.30/0.12 m/s 离线回放，765 帧均得到控制命令（其中 16 帧按
既定无效保持），角速度峰值约 1.04 rad/s，未触发 1.50 rad/s 限幅。该 session 的
`state[2]` 是 Pure Pursuit 等效 PID 输入而非手柄角速度，因此不使用其符号相关系数评价。

已使用 `tools/sync_to_orin.ps1`，仅同步：

```text
smartcar/whalesbot/tools/lane_collect/opencv_lane.py
smartcar/whalesbot/tools/lane_collect/pid_control.py
```

远端确认预瞄 0.15 m、速度 0.30/0.12 m/s、启动直行 0.15 m、入弯步进 0.10 rad/s/帧。
本地/Orin SHA256 一致：

```text
opencv_lane.py  dd24dab954c8821d4f66f24cc7e6cc3e7980fd153b7c219b9758e3b24578eada
pid_control.py  23151880e97d39a63154623fb9055f754450ee1783d9c061c1cc8a3b19714b5f
```

尚未取得 0.15 m 预瞄的新实车结果，尚未提交或推送 Git。

### 18.7 恢复 0.30 m 预瞄并只增强锐角曲率（2026-08-16）

0.15 m 全局预瞄实车反馈确认直线和普通弯退化，因此否定全局缩短预瞄方案。按用户确认，
恢复 0.30 m 预瞄，并采用不含状态机的单帧几何条件：

```text
ratio = abs(pure_pursuit_target_x_m / pure_pursuit_target_s_m)
ratio < 0.80: curvature 保持不变
ratio >= 0.80: curvature *= 2.0
```

目标是保留 0.30 m 对直道和普通弯的稳定性，只在目标点已明显横向展开、符合锐角/直角
几何特征时，将约 3.33 1/m 曲率提高到约 6.67 1/m。没有双预瞄、帧计数、里程窗口、
路线状态机或统一转向延迟。速度继续固化为 0.30/0.12 m/s，启动直行继续为 0.15 m；
`vy=0`、十字和预览滤波继续关闭。

新增诊断字段：`pure_pursuit_raw_curvature_m_inv`、
`pure_pursuit_target_lateral_ratio`、`pure_pursuit_sharp_boosted`。尚未提交或推送 Git。

验证完成：`py_compile`、`git diff --check` 通过，聚焦 `unittest` 共 31 项通过。对最新
765 帧实车 session 回放，锐角增强仅触发 42 帧，其中 41 帧为 `right_only`，主要集中在：

```text
356～369
467～486
597～611
```

三个主要锐角弯的原始曲率约 3.33 1/m，增强后约 6.66 1/m，按 0.12 m/s 最低速度得到
约 0.80 rad/s 稳态角速度。普通帧保持 0.30 m Pure Pursuit 原曲率，没有采用全局 0.15 m。

已仅同步 `smartcar/whalesbot/tools/lane_collect/opencv_lane.py` 到 Orin；远端确认：

```text
pure_pursuit_lookahead_m=0.30
pure_pursuit_sharp_ratio_threshold=0.80
pure_pursuit_sharp_gain=2.0
```

本地/Orin SHA256 一致：

```text
91d6c13f3b530601de22aa674b8c0766e29cf2247bb469b0ef0c49ac38799963
```

尚未取得锐角增益版本的新实车结果，尚未提交或推送 Git。

### 18.8 直角弯反向释放与锐角切线后备（2026-08-16）

分析最新实车 session `cv_low_speed_20260816_153005_153124`（774 帧）。用户标注的
291～359、394～465、537～575 三个直角弯并非转向幅度不足，而是视觉请求已经反向后，
`max_heading_release_step=0.04` 仍让底盘保持旧方向：

```text
弯道  视觉曲率反向帧/里程  底盘角速度反向帧/里程  拖尾
1     322 / 7.485 m         343 / 7.860 m             21 帧 / 0.375 m
2     430 / 9.932 m         451 / 10.329 m             21 帧 / 0.397 m
3     574 / 12.858 m        587 / 13.088 m             13 帧 / 0.231 m
```

682～707 的最后一个锐角弯是独立问题。693 帧已经出现角点；696～705 帧
`corner_score=0.74～0.84`、`corner_heading=-1.45～-1.33 rad`，但 0.30 m Pure Pursuit
目标横向比只有 0.01～0.09，直到 706 帧才跳至 1.10。因此不是角点未识别，而是固定预瞄
目标在宽阔锐角入口没有及时产生横向偏移。

本次最小修改：

1. `pid_control.py` 新增 `max_heading_reverse_step=0.10 rad/s/帧`，仅当请求角速度与当前
   输出反号时使用；普通同方向释放继续保持 0.04，不改变普通弯退出平滑性。
2. `opencv_lane.py` 增加严格限定的角点切线后备：角点分数至少 0.70、方向已知、目标横向比
   小于 0.15、`corner_near_progress<=0.50` 且仍有参考边线时，采用
   `curvature=-2*sin(corner_heading)/0.30`。正常 Pure Pursuit 一旦产生有效横移即继续主导。
3. 新增诊断字段 `pure_pursuit_corner_fallback` 和
   `pure_pursuit_corner_curvature_m_inv`。保留 0.30 m 预瞄、锐角增益 2.0、`vy=0`、
   十字状态机关闭、预览时序关闭、启动直行 0.15 m 和无效保持 10 帧。

使用该 session 已记录的逐帧视觉指标进行等价控制回放：角点后备只在 696～700 帧触发，
曲率约 6.57～6.62 1/m；未在前三个直角弯的退出阶段触发。反号释放预计把三个直角弯的
角速度过零帧分别从 343/451/587 提前至约 332/440/579。Windows 当前无 Python 解释器，
本地测试命令无法启动；`git diff --check` 通过，新增聚焦测试代码但未执行。

已按指定 `tools/sync_to_orin.ps1` 仅同步以下两个运行文件到
`/home/jetson/workspaces/baidu_smart_2026_8_14/`：

```text
smartcar/whalesbot/tools/lane_collect/opencv_lane.py
smartcar/whalesbot/tools/lane_collect/pid_control.py
```

同步脚本确认远端文件存在。按用户要求不再进行远端测试或 SHA256 验证。本地 SHA256：

```text
opencv_lane.py  db1b398da1246560f332fb11f0f5c23d94a2baa32a2387d2c35327f14548232f
pid_control.py  c217b3b59d1b35f1e87edc327803be08a5902def87db6a11a1c6ab68d08235ef
```

### 18.9 最后锐角的弱角点提前路径（2026-08-16）

最新实车 session `cv_low_speed_20260816_155619_154951` 共 805 帧，已下载到
`artifacts/orin_cv_low_speed_20260816_155619_154951`。前三个直角弯的反向拖尾问题已经解决，
但最后锐角仍失败。逐帧结果显示角点检测并不晚：705 帧 / 15.045 m 已得到约
`-1.45 rad` 的稳定角点方向；706～710 帧分数维持 0.59～0.61，Pure Pursuit 横向比接近
零。由于原角点后备要求 `corner_score>=0.70`，直到 724 帧 / 15.510 m 才首次启动，
晚约 18 帧 / 0.446 m，并因分数波动在 726、729 帧中断。

不能全局把强门限从 0.70 降至 0.58：整圈会额外命中起步、普通弯和直角弯退出帧。因此保留
原强角点路径不变，只增加一个严格的弱角点提前路径：

```text
corner_score >= 0.58
reference_tracking_mode == mixed
abs(raw_heading) <= 0.15
pure_pursuit_target_lateral_ratio < 0.15
0.15 <= corner_near_progress <= 0.50
corner_direction_known == True
```

该条件表达的是“左右参考类型混合、IPM 拟合仍近似直行，但角点切线已经稳定”的特定矛盾。
离线扫描最新 805 帧时，新增路径只命中最后锐角的 706～710 帧；扫描上一版 774 帧时，
只命中同一最后锐角的 680～683 帧，没有新增命中前三个直角弯。新增诊断字段
`pure_pursuit_corner_fallback_reason`，值为 `strong_score` 或 `early_mixed_flat`。

修改文件为 `opencv_lane.py` 和对应聚焦测试。Windows 无 Python 环境，测试代码未执行；
`git diff --check` 通过。已使用指定 `tools/sync_to_orin.ps1`，仅同步
`smartcar/whalesbot/tools/lane_collect/opencv_lane.py` 到指定 Orin 项目；按用户要求未进行
远端验证。本地 SHA256 为
`a4b08a2935313843ab65ab991854f0a5b5f46f34587ba98b7881c11df3176f6d`。

### 18.10 全图唯一右转的宽松候选与锁存（2026-08-16）

最新实车 session `cv_low_speed_20260816_160815_207155` 共 679 帧。上一版弱角点路径实际在
627～629 帧触发，强角点路径在 645～649 帧触发，但都把约 `-1.43 rad` 的图像角点切线
转换成了正曲率约 `+6.6 1/m`，角速度分别升至正值 0.27 和 0.58 rad/s。用户确认该弯为
全图唯一右转；真正代表右转的负 Pure Pursuit 曲率直到 654 帧以后才稳定出现，受反号释放
限制，实车角速度到 665 帧才变负。因此问题不仅是门控严格，还包括角点切线后备方向错误且
触发不连续。

按“整圈只有一个右转”增加方向专用控制：

```text
右转候选：
  corner_score >= 0.55
  corner_heading <= -1.00
  reference_tracking_mode == mixed
  target_lateral_ratio < 0.20
  corner_near_progress <= 0.70

确认：连续 2 帧
确认后的曲率：-6.0 1/m
交还 Pure Pursuit：raw_curvature <= -1.5 且 target_lateral_ratio >= 0.20
```

控制器确认后锁存右转，因此角点暂时丢失不会中断；交还后标记本圈已消费，不再重复触发。
`controller.reset()` 会在下一次发车清除候选、锁存和消费状态。新增诊断字段：
`route_right_turn_candidate`、`route_right_turn_active`、`route_right_turn_consumed`，锁存期间
控制原因记录为 `route_right_turn`。

三份最新 session 的离线候选扫描中，非最后右转位置最多只有孤立 1 帧，无法达到连续 2 帧；
最后右转均能确认。对 `cv_low_speed_20260816_160815_207155` 回放时，右转在
626 帧 / 15.129 m 锁存，持续负曲率穿过 630～640 的角点丢失区，在
654 帧 / 15.855 m 交还正常负曲率，比原本可靠负向控制提前约 0.73 m。前三个已解决直角弯
不进入右转锁存。

修改 `opencv_lane.py`、`pid_control.py` 和对应聚焦测试；Windows 无 Python 环境，测试未执行，
`git diff --check` 通过。已使用指定脚本仅同步两个运行文件到 Orin，按用户要求未进行远端
验证。本地 SHA256：

```text
opencv_lane.py  32062439636e55858f1b238e1c0260b2302d0a067d7a3d72a1f1337e68c4fe48
pid_control.py  7ad942ed5e520d5517856bc4117da4c4ad1c159e173cfb365b8b135c3c7ecee1
```

### 18.11 撤销右转状态机，改为单帧锐角右转覆盖（2026-08-16）

用户指出 18.10 的连续 2 帧确认、锁存、交还和整圈消费逻辑本质上是状态机，不符合当前简化
控制方向；同时更正赛道信息：前面还存在一个普通右弯，因此不能按“全图唯一右转”锁存。
18.10 的控制器计数、锁存、交还和消费字段已全部撤销。

最新 session 中普通右弯约在 40～82 帧：主要为 `left_only`，Pure Pursuit 已自然产生负曲率，
角点切线约到 `-1.25 rad`。最后锐角则在 625～629、645～649 帧表现为 `mixed`、角点切线
`-1.40～-1.45 rad`，且目标横向比仍小。由此改成完全无状态的单帧几何覆盖：

```text
corner_score >= 0.55
corner_direction_known == True
corner_heading <= -1.35
reference_tracking_mode == mixed
target_lateral_ratio < 0.20
corner_near_progress <= 0.70

当前帧 curvature = -6.0 1/m
下一帧重新独立判断
```

没有计数、保持、锁存、交还或“本圈已触发”标志。普通右弯不满足 `mixed` 和极陡切线条件，
保持原 Pure Pursuit。最后锐角命中帧直接使用负曲率，避免旧角点后备错误产生正曲率。新增诊断
`route_right_turn_override`；命中帧的 `pure_pursuit_corner_fallback_reason` 为
`right_turn_direct`，控制原因仍可记录为 `route_right_turn`，但该原因完全由当前帧指标决定。

三份 session 离线扫描中，最新 session 只命中最后锐角的 625～629、645～649；较早 session
除最后锐角外仅有极少孤立帧。孤立帧不会延续，且角速度仍受单帧 0.10 步进限制。
修改 `opencv_lane.py`、`pid_control.py` 和对应测试；Windows 无 Python 环境，测试未执行，
`git diff --check` 通过。已使用指定脚本仅同步两个运行文件到 Orin，按用户要求未进行远端
验证。本地 SHA256：

```text
opencv_lane.py  dea0e23127151e3784433bea3072fa4294b11de4adcd7701a96af2f9bc81502e
pid_control.py  8b39dfe9dcd3559aaa705fc6f5e4ea5c2bb76e515cdca1e638e536a1f427462c
```

### 18.12 推迟锐角右转入口并用当前帧几何延续（2026-08-16）

最新局部 session `cv_low_speed_20260816_163020_043507` 共 136 帧，只采集了同一个锐角右拐
附近的过程，不是完整赛道。18.11 的松条件分别在 59、78～84、106～119 命中；78～84 对应
远处阶段，确实过早。每段命中结束后错误正曲率立即恢复，负角速度又被反向释放；106～119
前半段还在抵消旧正角速度，真正有效的右转输出持续不足。

用户明确要求只在上一完整圈 645～649 对应的近场阶段之后开始，并继续输出。保持完全无状态，
改为两个当前帧几何分支：

```text
near_entry:
  corner_score >= 0.70
  corner_heading <= -1.35
  reference_tracking_mode == mixed
  target_lateral_ratio < 0.20
  0.20 <= corner_near_progress <= 0.70

high_score_continuation:
  corner_score >= 0.77
  reference_tracking_mode in (mixed, left_only)
  target_lateral_ratio >= 0.15
  abs(raw_heading) >= 0.60
```

任一分支只覆盖当前帧，下一帧重新判断，没有计数、锁存、消费或交还状态。对同一右锐角几何
`mixed + corner_heading<=-1.35`，在尚未满足近场入口时禁用旧的通用正向角点后备，仅保留原始
Pure Pursuit，避免先向错误方向积累角速度。新增诊断 `route_right_turn_waiting` 和
`route_right_turn_reason`（`near_entry` / `high_score_continuation`）。

右转专用曲率由 `-6.0` 提高为 `-10.0 1/m`，专用建立/反号步进由 0.10 提高为
`0.20 rad/s/帧`；普通转向步进仍保持原值。按最低速度 0.12 m/s，稳态右转角速度约
`-1.2 rad/s`，原方案约 `-0.72 rad/s`。

离线回放本局部 session 时，新条件只命中 116～125：116～119 为近场入口，120～125 为
高分数延续；126 以后视觉无效时，沿用既有最多 10 帧的最后有效命令保持。完整赛道 session
`160815` 只命中最后锐角 645～649，普通右弯不命中；更早完整圈也未命中普通右弯。

修改 `opencv_lane.py`、`pid_control.py` 和对应测试；Windows 无 Python 环境，测试未执行，
`git diff --check` 通过，且运行代码中不存在右转状态字段。待仅同步两个运行文件，按用户要求
不进行远端验证。已使用指定脚本仅同步两个运行文件到 Orin。本地 SHA256：

```text
opencv_lane.py  0b0e67e4af3abedce1777e258026108d76feffb15df64637fbedb569c235e1ea
pid_control.py  c8e0a8b89b727d1373a8944f155bc0b428d1b6f888bf6d854750071aff5a7b95
```

### 18.13 删除右锐角延续分支，近场入口后交回自然负曲率（2026-08-16）

最新局部 session `cv_low_speed_20260816_164022_192486` 共 120 帧。右锐角强制覆盖从 57 持续
至 70，共 14 帧：57～62 为 `near_entry`，角速度已从 -0.19 建立到 -1.19 rad/s；63 帧
跟踪切为 `left_only`，正常 Pure Pursuit 已产生正确负曲率 -2.00 1/m，本应立即交回。但旧的
`high_score_continuation` 又在 63～70 强制 -10.0 1/m，使角速度到 64 帧达到 -1.50 并
长时间维持，导致转向过度；92 帧还发生一次多余延续触发。

最小修正为彻底删除 `high_score_continuation` 分支及其三个配置参数，只保留无状态
`near_entry`：

```text
corner_score >= 0.70
corner_heading <= -1.35
reference_tracking_mode == mixed
target_lateral_ratio < 0.20
0.20 <= corner_near_progress <= 0.70
```

专用曲率 -10.0 1/m 和专用步进 0.20 rad/s/帧暂时保持。入口条件结束后，当前帧立即恢复
普通 Pure Pursuit；由于 63 帧以后自然曲率已经为负，现有同方向 0.04 释放会平滑继续右转，
不需要延续门控或状态。离线扫描：完整圈 `160815` 只覆盖 645～649（5 帧），最新局部
session 只覆盖 57～62（6 帧）；多余 92 帧触发消失。

修改 `opencv_lane.py` 和对应测试；`pid_control.py` 本次控制参数未再变化，但仍需与前次同步
版本保持一致。Windows 无 Python 环境，测试未执行，`git diff --check` 通过。待仅同步明确的
运行文件，按用户要求不进行远端验证。已使用指定脚本仅同步 `opencv_lane.py` 到 Orin；
本地 SHA256 为
`a4f6fbcc0f772082f702838cd271db95cb0c94cc6b1eddfec4d374e566698ce0`。

### 18.14 右锐角剩余轻微过转微调（2026-08-16）

最新局部 session `cv_low_speed_20260816_164422_904541` 共 212 帧。删除延续分支后，强制覆盖
已稳定为 57～62 六帧；角速度依次约 -0.19、-0.39、-0.59、-0.79、-0.99、-1.19 rad/s。
63 帧起正常 Pure Pursuit 已给出 -2.01 1/m 自然负曲率，但同方向 0.04 释放和 67～73 的
自然锐角曲率仍使右转输出保持较长，实车反馈只剩轻微过转。

曾考虑删除 frame 62，但 frame 61/62 的 `corner_near_progress` 实际均为 0.4417，无法用 near
稳定区分；frame 62 ratio=0.109，完整圈有效 frame 648 ratio=0.104，差值仅 0.005，用 ratio
硬切对车位和光照变化过于敏感。因此不再缩短命中帧，而做小幅输出微调：

```text
right_turn_candidate_max_near_progress: 0.70 -> 0.45
right_turn_heading_step: 0.20 -> 0.18 rad/s/帧
right_turn_curvature_m_inv: 保持 -10.0 1/m
```

near 上限用于防止更后的 mixed 帧重新进入；在当前六帧数据中仍命中 57～62。按 0.18 步进
等价回放，六帧角速度约为 -0.17、-0.35、-0.53、-0.71、-0.89、-1.07 rad/s，相比当前峰值
降低约 0.12 rad/s。普通转向、自然 Pure Pursuit、0.04 同方向释放和无效保持 10 帧均不变。

修改 `opencv_lane.py`、`pid_control.py` 和对应测试；Windows 无 Python 环境，测试未执行，
`git diff --check` 通过。已使用指定脚本仅同步两个运行文件到 Orin，按用户要求未进行远端
验证。本地 SHA256：

```text
opencv_lane.py  88469eabd122309c39c0191215268b6b9cc5312ad568c0c686f5dcab1f6f4097
pid_control.py  0a6c349632bd86443682019487db65639dce79a07d3beb874f2c6fbf091e2e7a
```

### 18.15 右锐角改为当前帧米制几何曲率（2026-08-16）

最新完整赛道 session `cv_low_speed_20260816_165026_843936` 中，右锐角覆盖为
724～728，共 5 帧，角速度只按 `right_turn_heading_step=0.18 rad/s/帧` 从约
`-0.16` 累加到 `-0.88 rad/s`；能够单独通过的 `cv_low_speed_20260816_164422_904541`
覆盖为 57～62，共 6 帧，角速度累加到约 `-1.19 rad/s`。两次命中的图像几何接近，
但输出峰值取决于门控命中了多少帧，因此会受帧率、车速和入口姿态影响。帧号只用于离线定位，
不得作为运行时判断或控制标注。

保持现有无状态单帧门控不变，删除固定 `right_turn_curvature_m_inv=-10.0` 和右转专用
`right_turn_heading_step=0.18 rad/s/帧`。命中帧使用 Hough 角点段的近端行
`corner_segment[1]`，从 `perspective.json` 取得当前角点的实际前向距离，再按以下几何式计算：

```text
distance_m = perspective[near_y - roi_top] / ipm_units_per_meter
distance_used_m = max(distance_m, 0.20)
curvature = -2 * sin(abs(corner_heading)) / distance_used_m
curvature = clip(curvature, -10.0, 0.0)
wz = clip(forward_speed * curvature, -heading_limit, heading_limit)
```

右锐角命中时直接采用该有界几何目标，并同步更新控制器的上一角速度；不再经过专用逐帧累加。
门控结束后仍由原 Pure Pursuit 接管并沿用普通方向释放限制。新增诊断字段
`route_right_turn_distance_m` 和 `route_right_turn_curvature_m_inv`。该改动不增加计数、锁存、
固定帧号或路线状态机，也不改变普通弯、十字开关、`vy=0`、启动直行保护和无效帧保持逻辑。

用两份实车记录中的原始 `corner_segment` 离线代入：完整圈 724～728 的距离约
`0.427 -> 0.279 m`、曲率约 `-4.64 -> -7.09 1/m`；局部通过记录 57～62 的距离约
`0.414 -> 0.263 m`、曲率约 `-4.79 -> -7.53 1/m`。两者由相近图像几何得到相近曲率，
不再因为 5 帧或 6 帧覆盖而产生不同的逐帧累计终值。用户随后确认同步，已使用指定的
`tools/sync_to_orin.ps1`，仅将以下两个运行文件同步到
`/home/jetson/workspaces/baidu_smart_2026_8_14/` 对应路径：

```text
smartcar/whalesbot/tools/lane_collect/opencv_lane.py
smartcar/whalesbot/tools/lane_collect/pid_control.py
```

同步脚本执行成功；按既定要求未进行远端 SHA、参数读取或运行验证。Windows 本地仍无 Python
解释器，聚焦测试代码已更新但无法执行；`git diff --check` 通过。本次未修改或覆盖模型、
数据集、日志及其他未提交内容。

本地运行文件 SHA256：

```text
opencv_lane.py  95cd0110b08bb002229fb095735e4dd6670184170f96cff32f1018368d686e2f
pid_control.py  429c7516748374ee42657428eca2b38e113405b676218508d2329e4cf07db592
```

### 18.19 使用 150～165 限幅的全局 Otsu，并微降右锐角后半段增益（2026-08-16）

最新完整实车 session `cv_low_speed_20260816_173313_422375` 的三个标注问题段为：

```text
51～82    十字前右缓弯被分成两段
415～465 第二个直角弯被分成两段
660～696 最后右锐角仍轻微过转
```

逐帧记录和原图确认：415～465 全程视觉有效，但固定阈值 175 在 436～447 连续产生 12 帧
错误负曲率；问题帧 ROI 中约 84%～91% 像素被归入暗色连通域。原始全局 Otsu 阈值随画面在
约 139～166 之间变化。固定阈值 160 与原始 Otsu 都能消除这 12 帧反向，但原始 Otsu 会在
十字等大面积路面画面降到约 139。用户确认采用每帧全局 Otsu，并将实际阈值限制在
`[150, 165]`，不使用局部 `adaptiveThreshold`。

本次修改：

```text
LaneAnalyzerConfig.threshold: 175 -> None
adaptive_threshold_min: 150
adaptive_threshold_max: 165
pure_pursuit_right_turn_sharp_gain: 1.30 -> 1.20
```

`_segment_track()` 先计算整张工作灰度图的 Otsu 阈值，再夹紧到 150～165 后重新二值化；固定
阈值和橙色边界分支仍保留。每帧新增以下诊断，包含无效帧：

```text
segmentation_threshold_mode
segmentation_threshold_otsu
segmentation_threshold_used
segmentation_threshold_min
segmentation_threshold_max
```

修改文件为 `opencv_lane.py`、`ssh_test.py`、`tests/test_opencv_lane_analyzer.py` 和本文档。
未修改 `pid_control.py`、十字状态机、预览时序、速度、`vy=0`、右锐角入口 0.65 米制增益、
无效帧保持、CNN 主程序及数据保存格式。

使用 Orin 当前环境对 871 张原图做只读等效回放（逐帧计算 Otsu、夹紧后使用对应固定阈值，
并代入 1.20 右锐角后半段增益）：

```text
有效帧：856（相对当前 175 版本新增 6，丢失 1）
实际阈值达到下限 150：80 帧
实际阈值达到上限 165：29 帧
415～465：51/51 有效，错误负曲率由 12 帧降为 0
51～82：61～68 仍无效；阈值修改不解决十字几何不可辨识问题
660～696：675～683 使用 1.20 增益，累计增强曲率较当前版本约降低 5%
```

十字前分段原因仍是 57～60 的低置信度单边拟合先把可靠右转释放到约 `-0.12 rad/s`，随后
61～68 才进入无效保持；不能通过阈值修复，后续需单独研究按视觉退化与里程退出的方向连续性
保护。本次没有恢复旧十字路线状态机。

Windows 本地仍无 Python/OpenCV，新增聚焦测试未执行；已完成等效实图回放，且聚焦文件的
`git diff --check` 通过。已使用指定 `tools/sync_to_orin.ps1`，仅同步以下两个运行文件到
`/home/jetson/workspaces/baidu_smart_2026_8_14/` 对应路径：

```text
smartcar/whalesbot/tools/lane_collect/opencv_lane.py
smartcar/whalesbot/tools/lane_collect/ssh_test.py
```

同步脚本执行成功；按既定要求未进行远端 SHA、参数读取或运行验证。本次未提交或推送 Git。
同步时本地 SHA256：

```text
opencv_lane.py  52859371617235040c0d4258cff7826039f40c63e0f12f9afc1da9d77f25c886
ssh_test.py     41cdf1f43c6b379c58c5cbf53d5d35040b864c1a321d838bae6d9d3905678d79
```

### 18.20 小幅降低通用锐角/直角增强（2026-08-16）

18.19 同步后，用户实车反馈直角弯也有轻微转向过度。为保持普通弯和最后右锐角专用控制不变，
本次只将达到 `pure_pursuit_sharp_ratio_threshold=0.80` 后的通用增强降低 10%：

```text
pure_pursuit_sharp_gain: 2.0 -> 1.8
```

最后右锐角的 `pure_pursuit_right_turn_sharp_gain=1.20` 和入口
`right_turn_curvature_gain=0.65` 均不变；普通弯、阈值 150～165 限幅 Otsu、速度、释放限速、
`vy=0`、十字关闭、预览关闭和无效保持均未修改。

使用 Orin 当前环境和完整 session `cv_low_speed_20260816_173313_422375` 做只读等效回放：

```text
291～359：8 帧通用增强，最大绝对曲率 8.26 -> 7.43 1/m
415～465：8 帧通用增强，最大绝对曲率 7.73 -> 6.96 1/m
537～575：6 帧通用增强，最大绝对曲率 6.67 -> 6.00 1/m
660～696：0 帧通用增强，最后右锐角结果保持不变
```

修改 `opencv_lane.py`、`tests/test_opencv_lane_analyzer.py` 和本文档。Windows 本地无
Python/OpenCV，聚焦测试已更新但未执行；`git diff --check` 通过。已使用指定
`tools/sync_to_orin.ps1`，仅同步 `opencv_lane.py` 到
`/home/jetson/workspaces/baidu_smart_2026_8_14/` 对应路径；同步脚本执行成功，按既定要求
未进行远端 SHA、参数读取或运行验证。本次未提交或推送 Git。本地运行文件 SHA256：

```text
opencv_lane.py  0b93bdb7605ebbe5c7973d9ec2883b3f1ec5b0a8b759a36eb4895910aa4ff086
```

### 18.21 将 2026-08-16 CV 修改同步到 GitHub

本次 Git 同步目标为 `origin/orin-main-20260814`。提交范围严格限制为当天的 CV 循迹源码、
离线比较脚本、测试和上下文文档：

```text
docs/cv-lane-collection-operation-guide.md
docs/cv-lane-development-status-2026-08-15.md
scripts/compare_cv_control.py
smartcar/whalesbot/tools/lane_collect/opencv_lane.py
smartcar/whalesbot/tools/lane_collect/pid_control.py
smartcar/whalesbot/tools/lane_collect/ssh_test.py
tests/test_cv_lane_pid_control.py
tests/test_opencv_lane_analyzer.py
```

明确排除 `artifacts/`、实车 session、数据集、模型、日志及日志删除、调试文件和其他未提交工作区
内容。推送前已刷新 GitHub 分支，确认本地与 `origin/orin-main-20260814` 无领先/落后提交。
Windows 本地无 Python/OpenCV，单元测试未运行；实图等效回放结果记录在 18.19 和 18.20，
聚焦文件的 `git diff --check` 通过。

GitHub 同步已完成：CV 源码、测试和当时的文档提交为
`4ccb632 feat(cv): stabilize pure pursuit lane collection`，并已成功推送到
`origin/orin-main-20260814`。本段同步结果记录随后以仅包含本文档的补充提交推送。

### 18.17 仅取消右锐角后半段的通用 2 倍增强（2026-08-16）

`right_turn_curvature_gain=0.65` 已在最新 session
`cv_low_speed_20260816_171541_636535` 中确认生效：右锐角专用入口只覆盖 663～665 三帧，
曲率为 `-2.85/-3.16/-3.72 1/m`，实际角速度为
`-0.89/-0.94/-1.05 rad/s`。因此剩余过转不是同步失败，也不是入口专用增益仍过大。

门控结束后 672～675 和 677～679 进入通用锐角增强：参考模式均为 `left_only`、曲率为负，
原始曲率约 `-3.23～-5.13 1/m`，被全局 `pure_pursuit_sharp_gain=2.0` 放大至约
`-6.46～-10.26 1/m`。角速度再次从约 `-0.77` 建立到 `-1.29 rad/s`，随后 680～682
三帧无效保持继续维持约 `-1.25 rad/s`，这才是 0.65 版本仍过转的主要来源。

同一完整 session 的前三个直角弯锐角增强窗口 328～334、426～432、540～546 均为
`right_only` 且曲率为正；最后右锐角后半段则为 `left_only` 且曲率为负，可以由当前帧边线和
曲率符号无状态地区分。用户确认增加：

```text
pure_pursuit_right_turn_sharp_gain = 1.0

ratio >= 0.80 且 reference_mode == left_only 且 curvature < 0:
  使用 1.0 倍原始曲率
其他锐角：
  继续使用原 2.0 倍曲率
```

新增诊断字段 `pure_pursuit_sharp_gain_applied`。不增加计数、锁存或路线状态机，不改变右锐角
入口 0.65 米制几何曲率、普通右弯、前三个直角弯、速度、角速度释放、`vy=0`、十字关闭、
预览关闭、启动直行保护和无效保持。

用 `171541` 已记录指标等价回放，入口 663～665 完全不变；右锐角后半段预计峰值角速度从
约 `-1.29` 降至 `-0.97 rad/s`，680～682 无效保持从约 `-1.25` 降至 `-0.77 rad/s`。
前三个直角弯代表帧的增强曲率仍约为 `8.41/8.65/6.38 1/m`，保持 2 倍增强。Windows 本地
无 Python，聚焦测试已更新但未执行；`git diff --check` 通过。已使用指定同步脚本，仅同步
本次实际修改的 `opencv_lane.py` 到指定 Orin 项目；脚本执行成功，按既定要求未做远端验证。
本地 SHA256：

```text
opencv_lane.py  dedf7a8a3048499eb4c9e733e314dd42fe2cb91828660c00a8d29490b68e7e6f
```

### 18.18 右锐角后半段恢复小幅增强（2026-08-16）

18.17 将右锐角 `left_only + 负曲率` 后半段的锐角增益从原全局 `2.0` 降为 `1.0` 后，
实车反馈转向不足。用户要求只回加一部分增强，不恢复原来已经确认过转的 2 倍输出。

本次仅修改：

```text
pure_pursuit_right_turn_sharp_gain: 1.0 -> 1.30
```

右锐角入口的米制几何增益仍为 `0.65`；只有入口交还后的 `left_only + 负曲率 + ratio>=0.80`
使用 1.30 倍。前三个直角弯的 `right_only + 正曲率` 继续使用 2.0 倍。没有新增帧数、计数、
锁存、里程窗口或路线状态机，普通弯、速度、释放限制、无效保持、`vy=0`、十字关闭和预览关闭
均不变。

用 `cv_low_speed_20260816_171541_636535` 的已记录指标等价回放，右锐角入口 663～665 仍为
约 `-0.89～-1.05 rad/s`；后半段预计峰值由 1.0 版本的约 `-0.97` 提高到约
`-1.17 rad/s`，仍低于原 2.0 版本的约 `-1.29 rad/s`。Windows 本地无 Python，聚焦测试
已更新但未执行；`git diff --check` 通过。已使用指定同步脚本，仅同步本次实际修改的
`opencv_lane.py` 到指定 Orin 项目；脚本执行成功，按既定要求未做远端验证。本地 SHA256：

```text
opencv_lane.py  7338f232f8caf0ff37b31d1266cd744998532cdbfecbb953345d41f446f22088
```

### 18.16 降低右锐角米制曲率增益（2026-08-16）

同步 18.15 后取得三份新实车记录，其中 `cv_low_speed_20260816_170251_684130` 仍是同步前
旧代码：右转覆盖 728～731，曲率固定 `-10.0 1/m`，角速度按旧步进约从 `-0.17` 增加到
`-0.71 rad/s`。同步后的两份记录已经包含新距离诊断：

```text
cv_low_speed_20260816_170734_283660
  覆盖 637～639，仅 3 帧
  几何曲率 -4.70、-5.28、-6.53 1/m
  实际角速度三帧均达到 -1.50 rad/s

cv_low_speed_20260816_170913_817676
  覆盖 690～692，仅 3 帧
  几何曲率 -4.64、-4.98、-5.73 1/m
  实际角速度 -1.39、-1.42、-1.50 rad/s
```

因此过转不是门控帧数过长，而是把角点切线按完整 Pure Pursuit 曲率立即执行后，入口直接接近
角速度限幅；门控结束时控制器的上一输出也已达到 `-1.50`，随后自然 Pure Pursuit 锐角增强继续
请求负转向。`170913` 到 700 帧仍接近强负转向，到 706 帧仍约 `-1.20 rad/s`，强转向拖过约
0.33 m。

用户确认采用最小幅度修正：新增 `right_turn_curvature_gain=0.65`，只作用于当前帧右锐角
米制几何曲率：

```text
curvature = -2 * sin(abs(corner_heading)) / distance_m * 0.65
wz = forward_speed * curvature
```

门控、距离下限、曲率上限和直接几何输出结构均不变，不恢复逐帧步进，不增加计数、锁存或路线
状态机。普通弯、普通 Pure Pursuit、全局锐角增益、前三个直角弯反号释放、速度、`vy=0`、
十字关闭、预览关闭、启动直行保护和无效保持均未修改。

基于已记录车速和视觉指标的等价回放中，`170913` 入口三帧预计从
`-1.39/-1.42/-1.50` 降为约 `-0.91/-0.92/-1.01 rad/s`，随后自然 Pure Pursuit 段峰值约
`-1.17 rad/s`；位于旧版转向不足与 18.15 明显过转之间。Windows 本地无 Python，测试代码
已更新但未执行；`git diff --check` 通过。已使用指定 `tools/sync_to_orin.ps1`，仅同步
`opencv_lane.py` 和 `pid_control.py` 到指定 Orin 项目；同步脚本执行成功，按既定要求未做
远端 SHA、参数读取或运行验证。本地 SHA256：

```text
opencv_lane.py  1e416af39ad2d88471e52e1c99ceaafc1affe284b07e9be7d9c7935648a26d72
pid_control.py  429c7516748374ee42657428eca2b38e113405b676218508d2329e4cf07db592
```

## 19. CV 教师与 CNN 训练接口整合（2026-08-17）

CV 控制和训练数据接口已在 `orin-main-20260814` 分支整合，控制代码基线提交为
`30d0f0e feat: add CV teacher and curvature CNN control`。本节只记录与 CV 开发历史相关的
结果；完整数据字典、训练缺口、部署 profile 和验收清单见
[CNN 训练接续文档](lane-cnn-training-handoff.md)第 18、20 节。

当前 CV session 不再把 PID 前误差写入普通 `state`：

```text
state/control    = [actual_vx, actual_vy, actual_wz]
legacy_pid_state = [forward_speed, error_y, error_angle]
model_target     = [speed_demand, kappa_action]
target_mask      = [1.0, 1.0]
```

`kappa_action` 在启动直行保护、短时无效帧继承、锐角处理和最终限幅之后，按实际命令
`actual_wz/max(abs(actual_vx),0.12)` 重新计算。这样 CNN 学到当帧实际转向动作，车端只保留
可调速度曲线、真实时间速度梯度和 `wz=actual_vx*kappa_action`，不重复执行 CV 的角速度
建立/释放逻辑。

速度和 CV 角速度梯度已经由按帧步长改为真实 `dt` 速率。速度参数为减速
`0.194 m/s²`、加速 `0.129 m/s²`，无效或缺失 `dt` 回退到 `0.05 s`。当前
`finite_dt()` 没有正数上限钳制，长时间卡顿后的首步变化仍是部署风险；修复前必须依靠
watchdog、控制器重置和低速验证，不能直接宣称丢帧安全。

本次同时确认训练路径尚未全部固化：GitHub 已有 CV/手柄 manifest、双输出标签、mask 和
数据集增强，但尚无已跟踪的 mask 损失训练器、新语义正式训练入口和一键导出工具。本地
未跟踪 `tools/` 不能作为远端可复现流程。后续 CV 视觉修改仍记录在本文；训练接口只在训练
接续文档维护，避免两份文档再次分叉。
