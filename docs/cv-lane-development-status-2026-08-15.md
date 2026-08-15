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

2777～2780 是已知无效帧，继承机制本身工作正常，但上一条 2776 的方向已经错误：

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
方向”。下一步应优先增加急弯横线有效性保护，或在这类退化帧中锁定进入急弯前的
稳定方向。不要简单增加继承帧数，因为那只会更久地继承错误命令。

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
