# OpenCV 循迹低速 SSH 测试

该模式只从 `collect_data.py` 启动，不导入主程序的 CNN 循迹入口，也不需要蓝牙手柄。

## 启动

在项目根目录通过 SSH 执行：

```bash
python collect_data.py --cv-low-speed
```

程序启动后车辆保持停止，终端显示 `cv>` 提示符。可用命令：

```text
start   开始以 0.05 m/s 进行 OpenCV+PID 低速循迹
stop    立即停车并保存本次测试记录
status  查看运行状态和当前测试目录
quit    停车、保存并退出
```

`Ctrl+C`、SSH 标准输入结束、SSH 挂断、进程终止信号、CV 无效或摄像头帧超过
0.25 秒未更新，都会发送 `[0, 0, 0]` 速度并结束当前测试段。算法失效停车后不会自动恢复，
必须检查车辆位置并重新输入 `start`。

## 首轮参数

```text
forward_speed = 0.05 m/s
error_angle = -0.316 * raw_heading
error_y = 0
angular_speed limit = +/-0.35 rad/s
angular speed step = 0.08 rad/s per control update
```

横向 PID 在首轮测试中关闭，只验证角度方向、角速度和急弯连续性。

## 测试数据

每次 `start` 创建一个独立目录：

```text
dataset/cv_lane_tests/cv_low_speed_YYYYMMDD_HHMMSS_xxxxxx/
```

其中包含：

- 与 CNN 输入一致的 128×128 彩色 JPG；
- `data.json`：CV 误差、实际底盘控制量和诊断信息；
- `session.json`：本次参数和用途说明。

首轮记录带有 `usable_for_training: false`，不得直接合并进正式训练集。
