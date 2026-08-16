# CV 固定赛道巡线与 CNN 数据采集操作指南

## 1. 用途与数据契约

CV 模式用于在固定赛道上自动巡线，并同步采集可继续训练现有车道 CNN 的图像和控制标签。它只从 `collect_data.py` 启动，不替换比赛主程序中的 CNN 巡线入口。

每一条训练记录同时保存 PID 前 CNN 标签和 PID 后底盘命令：

```text
state[0] = forward_speed
state[1] = cv_error_y
state[2] = cv_error_angle

control[0] = forward_speed
control[1] = actual_lateral_speed
control[2] = actual_angular_speed
```

现有 CNN 训练器仍直接读取 `state[1]` 和 `state[2]`，运行时再调用
`lane_pid.get_out(-error_y, -error_angle)`。因此 CV 标签也保存在 `state`，但实际发送给
麦克纳姆底盘的 `vy/wz` 单独保存在 `control`，两者不能再混为一组数据。

图像链路与现有 CNN 保持一致：

```text
前置摄像头 BGR 图像
  -> 以原始 320 x 240 保存为 JPG
  -> 训练时缩放为 128 x 128
  -> 训练时 BGR 转 RGB
  -> pixel / 127.5 - 1.0
  -> CNN
```

## 2. 当前控制参数

当前参数针对直道高速、弯道自动降速的固定赛道采集：

| 参数 | 当前值 |
|---|---:|
| 控制周期基础等待 | 0.05 s |
| 启动强制直行距离 | 0.15 m（仅发车一次） |
| 直道最高前进速度 | 0.30 m/s |
| 急弯最低前进速度 | 0.12 m/s |
| 转向方法 | IPM 中线 Pure Pursuit（仅 `--cv-low-speed`） |
| Pure Pursuit 预瞄距离 | 0.30 m |
| 锐角目标点横前比阈值 | `abs(target_x / target_s) >= 0.80` |
| 锐角曲率增益 | 2.0 |
| 满转向曲率参考 | 5.0 1/m |
| Pure Pursuit 增益 | 1.0 |
| 降速曲线指数 | 1.5 |
| 每帧最大降速 | 0.015 m/s |
| 每帧最大加速 | 0.010 m/s |
| 横向误差映射 | `-0.10 * raw_lateral` |
| 航向误差映射 | 旧航向法回退/诊断仍为 `-0.40 * raw_heading` |
| 横向 PID | `Kp=6, Ki=0, Kd=0.1, limit=0`，实车符号确认前禁用 `vy` |
| 航向 PID | `Kp=1.95, Ki=0, Kd=0, limit=+-1.50` |
| 入弯最大角速度变化 | 0.10 rad/s/帧 |
| 回正/反向最大角速度变化 | 0.04 rad/s/帧 |
| 相机最大允许帧龄 | 0.25 s |
| 普通无效帧命令继承 | 最多 10 帧 |
| 工作图像 | 320 x 240 |
| 保存图像 | 原始 320 x 240 |
| 灰度阈值 | 175，暗色赛道分割 |
| ROI | 顶部裁掉 30%，底部裁掉 20% |
| 中线拟合区 | 完整有效 ROI 的等效 IPM 坐标 |

每行先用标准宽度补全单边线中心，再用 `perspective.json` 将横向偏移和前向距离转换
到等效鸟瞰坐标。在等效 IPM 中线上直接插值得到前方 0.30 m 目标点，以
`curvature = -2*x/(x*x+s*s)` 计算曲率；负号用于把图像向右为正转换成底盘向左转为正。
底盘目标角速度为 `wz = forward_speed * curvature`。当目标点横向/前向距离比达到
0.80 时，仅将该帧曲率放大 2 倍，以缩小锐角和直角弯半径；普通弯和直道保持原曲率。
远场二次拟合的近端截距、切线和
曲率继续作为诊断量，但不再直接产生实际转向。固定转向延迟和 PID 前 EMA 均未恢复。

为保持现有 CNN 接口，`state[2]` 保存 P-only 航向控制器可还原为稳态 Pure Pursuit
目标角速度的等效 `error_angle`；实际经过速度和角速度步进限制后发送的 `wz` 仍只保存
在 `control[2]`。新 session 的 teacher 标记为 `opencv_ipm_pure_pursuit`。

## 3. 固定赛道参考文件

程序启动时必须能读取项目根目录下的：

```text
standard/standard_lane.json
standard/perspective.json
```

辅助审查文件包括：

```text
standard/standard_origin.jpg
standard/standard_binary.jpg
standard/standard_boundaries.jpg
standard/perspective_open_source.json
```

标准线来自固定相机安装姿态下的标准赛道图像。更换摄像头、改变安装高度或俯仰角、调整分辨率后，必须重新生成标准参考，不得继续沿用旧文件。

## 4. 上车前检查

1. 将车辆放在与参考圈相同的起点、方向和标准姿态。
2. 确认前置摄像头为索引 1，在 Linux 上通常对应 `/dev/cam1`。
3. 确认摄像头固定牢靠，画面没有旋转、遮挡或明显过曝。
4. 确认赛道边界、十字路口和地面照明与参考采集时基本一致。
5. 确认急停方式可用，操作人员站在可以立即拿起车辆或切断动力的位置。
6. 第一次实车运行不要直接无人值守，也不要从赛道中途启动。
7. 检查工作区可写空间；每次 `start` 都会新建独立 session，不覆盖旧数据。

当前十字状态机关闭，车辆依靠普通循迹和无效帧短时继承直接通过十字路口。

Pure Pursuit 曲率增大时会按曲率降速并提高 `wz/vx`；不再按里程延迟转向。
`odometry_distance_m` 仍只用于启动前 0.15 m 直行保护和赛段分析，不参与普通弯道转向。

## 5. 启动命令

在车辆项目根目录通过 SSH 执行：

```bash
cd /path/to/baidu_smartcar_2026
python3 collect_data.py --cv-low-speed
```

指定输出根目录：

```bash
python3 collect_data.py \
  --cv-low-speed \
  --cv-output dataset/cv_lane_tests
```

程序初始化完成后会先发送零速度，并显示：

```text
CV low-speed test ready. Commands: start, stop, status, quit
cv>
```

可用命令：

| 命令 | 功能 |
|---|---|
| `start` | 重置控制器和十字状态机，创建新 session 并开始巡线采集 |
| `status` | 显示是否运行以及当前 session 目录 |
| `stop` | 立即停车、写完 `data.json` 并关闭当前 session |
| `quit` / `exit` | 停车、保存并退出进程 |

## 6. 实车运行步骤

1. 启动程序，等待出现 `cv>`，确认车辆仍静止。
2. 再次确认起点和车头姿态正确。
3. 输入 `start`。
4. 观察车辆是否以标准姿态进入第一段赛道。
5. 第一十字前重点观察缓弯是否完成，以及普通循迹是否能平滑回正并直行通过。
6. 第二十字重点观察是否保持直行，不能因边界短暂失效转入侧路。
7. 遇到错误方向、持续偏离、相机卡顿或障碍物时立即输入 `stop`。
8. 完整一圈后立即输入 `stop`，不要让车辆继续进入第二圈。
9. 输入 `status` 确认已停止，记录终端打印的 session 路径和保存帧数。

以下情况会自动停车：

- 相机帧超过 0.25 秒未更新；
- CV 连续失效并超过 10 帧继承窗口；
- 图像保存或控制循环发生异常；
- SSH 挂断、进程收到终止信号或按下 `Ctrl+C`。

自动停车后不会自动恢复。必须检查车辆位置，重新放回标准起点，再输入 `start` 创建新 session。

## 7. 十字路口状态机

`OpenCVLaneSshTest.CROSS_STATE_ENABLED` 当前为 `False`。实车提速后，车辆在
10 帧无效结果继承窗口内可以直接穿过十字，因此不再执行固定帧数或固定里程的
十字动作。十字区域如果短暂无法计算边界，会完整继承上一帧实际速度和转向；
第 11 个连续无效帧会立即停车。

## 8. 采集目录和字段

每次 `start` 创建：

```text
dataset/cv_lane_tests/
  cv_low_speed_YYYYMMDD_HHMMSS_xxxxxx/
    000000.jpg
    000001.jpg
    ...
    data.json
    session.json
```

单帧记录示例：

```json
{
  "img_path": "000123.jpg",
  "state": [0.14, -0.025, -0.12],
  "control": [0.14, -0.15, -0.234],
  "teacher": "opencv_ipm_error",
  "label_semantics": "cnn_pid_input_error",
  "held": false,
  "command_source": "standard",
  "steering_demand": 0.60,
  "target_forward_speed": 0.105,
  "cv": {},
  "timestamp": 0.0
}
```

`command_source` 常见值：

| 值 | 含义 |
|---|---|
| `standard` | 标准左右边界循迹 |
| `short_invalid_hold` | 当前帧失效，实际发送上一条有效命令 |

`held=true` 表示图像分析失效，本帧继承上一帧的标签和实际底盘命令。训练时建议单独
统计，若数量异常或连续时间过长则丢弃该 session。

## 9. 一圈结束后的数据检查

必须完成以下检查后，才能把 session 标记为训练数据：

1. `data.json` 可以解析，记录数大于零。
2. JPG 数量与 JSON 记录数一致。
3. 第一张、中间、两个十字和最后一张图片方向正常。
4. `state` 与 `control` 均有 3 个有限数值，并确认两者语义不同。
5. `angular_speed` 的左右符号与手柄数据一致。
6. 转向趋势覆盖完整弯道，没有长时间错误反向。
7. `held=true` 数量没有异常增多。
8. 整圈没有人工推动、碰撞、脱线后继续采集或第二圈数据。

快速检查 JSON：

```bash
python3 - <<'PY'
import json
from pathlib import Path

session = Path("dataset/cv_lane_tests/替换为本次session")
rows = json.loads((session / "data.json").read_text())
images = list(session.glob("*.jpg"))
held = sum(bool(row.get("held")) for row in rows)
yaw = [float(row["state"][2]) for row in rows]
print("records:", len(rows))
print("images:", len(images))
print("held:", held)
print("yaw range:", min(yaw), max(yaw))
PY
```

## 10. 接入现有 CNN 训练环境

现有远程训练工程位于：

```text
/root/autodl-tmp/lane-cnn/project
```

训练器已确认：

- 从 `state[1]` 读取 CNN 的 `error_y` 标签；
- 从 `state[2]` 读取 CNN 的 `error_angle` 标签；
- 使用 RGB、128 x 128、`/127.5 - 1`；
- 从官方 CNN 权重开始微调；
- 命令归一化尺度为 `vy=0.15`、`yaw=0.94`。

接入步骤：

1. 将审核通过的完整 CV session 放入 collected sessions 归档。
2. 在 `session-classification.csv` 中为每个 `data.json` 增加一行。
3. 设置 `session_type=full_lap`。
4. 分配 `split=train` 或 `split=validation`，并设置 `usable=true`。
5. 新建 `mode=mixed` 的训练 YAML；当前 `official_only` 配置会完全忽略 CV 数据。
6. 固定赛道 mixed 训练应关闭水平翻转；镜像会制造不存在的十字和弯道。
7. 保留官方/手柄数据；CV 已包含非零横向误差，但其初始缩放仍需实车校准。

建议第一轮采样比例：

```yaml
sampling:
  official: 0.50
  full_lap: 0.35
  hard_segment: 0.15

train:
  mode: mixed
```

远程准备工具依据 `session-classification.csv` 决定 session 是否参与训练，不读取本地 `session.json` 的 `usable_for_training`。CSV 的记录数和路径必须与归档完全一致。

## 11. 已知限制

- 当前前进速度在 0.12～0.30 m/s 之间按转向需求变化，最低弯道速度仍需根据实车抓地和赛道曲率继续校准。
- 横向标签继续记录，但实际 `vy` 输出暂时限制为 0；低速确认符号和恢复增益后才能启用。
- 十字状态机当前关闭，连续无效超过 10 帧仍会停车。
- 当前每帧同步写 JPG，并每 10 帧更新一次 JSON；磁盘过慢会降低实际控制频率。
- `state` 是单帧视觉误差标签，`control` 是发送命令；两者都不是编码器实测速度。
- 改变赛道、摄像头姿态、分辨率或照明后，需要重新验证标准参考和阈值。

## 12. 常见问题

### 启动时提示缺少标准参考

确认从项目根目录启动，并存在：

```text
standard/standard_lane.json
standard/perspective.json
```

### 启动后车辆立即停车

检查相机索引、`/dev/cam1`、USB 连接、画面更新时间和 CV 是否能分割暗色赛道。

### 车辆左右方向相反

立即停车，禁止继续采集。检查 `heading_scale` 符号、摄像头是否镜像以及底盘角速度正方向。

### 十字路口无法直接通过

检查十字区域连续无效帧是否超过 10 帧、进入十字前是否已经回正，以及磁盘写入是否
导致控制循环明显低于 20 Hz。当前不会触发任何十字专用动作。

### 训练后横移输出接近零

CV 数据占比过高。降低 `full_lap` 采样比例，并保留足够的官方/手柄 `vy` 标签数据。
