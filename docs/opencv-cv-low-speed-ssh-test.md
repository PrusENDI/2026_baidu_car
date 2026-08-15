# OpenCV 循迹低速 SSH 测试

该模式只从 `collect_data.py` 启动，不导入主程序的 CNN 循迹入口，也不需要蓝牙手柄。

## 启动

在项目根目录通过 SSH 执行：

```bash
python collect_data.py --cv-low-speed
```

程序启动后车辆保持停止，终端显示 `cv>` 提示符。可用命令：

```text
start   开始 OpenCV+PID 自适应速度循迹
stop    立即停车并保存本次测试记录
status  查看运行状态和当前测试目录
quit    停车、保存并退出
```

`Ctrl+C`、SSH 标准输入结束、SSH 挂断、进程终止信号、CV 无效或摄像头帧超过
0.25 秒未更新，都会发送 `[0, 0, 0]` 速度并结束当前测试段。算法失效停车后不会自动恢复，
必须检查车辆位置并重新输入 `start`。

## 当前控制与标签

```text
forward_speed = 0.08～0.20 m/s
state[1] = PID-before cv_error_y
state[2] = PID-before cv_error_angle
control[1] = actual lateral_speed command
control[2] = actual angular_speed command
```

横向和航向使用与主程序相同的两组 PID。误差来自完整 ROI 的等效 IPM 二次中线拟合，
固定转向距离延迟已取消。

## 测试数据

每次 `start` 创建一个独立目录：

```text
dataset/cv_lane_tests/cv_low_speed_YYYYMMDD_HHMMSS_xxxxxx/
```

其中包含：

- 原始 320×240 彩色 JPG，训练时再缩放为 128×128；
- `data.json`：CV 误差、实际底盘控制量和诊断信息；
- `session.json`：本次参数和用途说明。

记录使用现有 CNN 读取的 `state[1:3]` 字段保存 PID 前误差标签，并用 `control`
保留实际底盘命令；同时带有 `held` 和 `command_source` 字段。正式训练是否采用该
session 仍应以整圈审查和训练侧
`session-classification.csv` 为准。
