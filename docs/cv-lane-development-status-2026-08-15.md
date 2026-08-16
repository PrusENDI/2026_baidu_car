# CV 自动采样巡线开发状态（2026-08-15）

本文是当前 CV 自动采样模式的上下文恢复入口。后续开始工作前，应先阅读本文，再查看
文末列出的设计、操作和验证文件，避免因对话上下文压缩而重新采用已经否定的方案。

## 1. 仓库状态

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
满转向标签参考：0.40
降速曲线指数：1.5
单帧最大降速：0.03 m/s
单帧最大加速：0.005 m/s
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
