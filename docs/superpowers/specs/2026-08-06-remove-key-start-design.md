# 删除按键启动功能设计

## 目标

完整移除开机按键启动监督器及为其新增的 MC602 掉线恢复链路，使车辆程序恢复为由操作者在项目根目录人工执行 `python car_start_2026.py`。保留采收、机械臂、视觉和车辆参数等与按键启动无关的现有工作区修改。

## 回退策略

采用精确回退，不重置整个工作区。删除只为按键启动监督器存在的文件；对同时包含其他开发内容的文件，只移除按键启动或 MC602 掉线保护相关代码。

不保留“取消按键但仍开机自动运行”的兼容入口，也不保留推理后端监督器。人工启动前的环境准备重新由操作者负责。

## 删除范围

删除以下运行资产及专用测试：

- `scripts/start_with_key.sh`
- `scripts/wait_for_start_key.py`
- `scripts/inference_backend_probe.py`
- `systemd/baidu-smart-key-start.service`
- `tests/test_key_start_helpers.py`
- `smartcar/whalesbot/vehicle/base/controller_health.py`
- `tests/test_mc602_disconnect.py`

删除只描述上述功能的设计和实施计划：

- `docs/superpowers/specs/2026-08-05-key-start-supervisor-design.md`
- `docs/superpowers/plans/2026-08-05-key-start-supervisor.md`
- `docs/superpowers/specs/2026-08-06-mc602-disconnect-recovery-design.md`
- `docs/superpowers/plans/2026-08-06-mc602-disconnect-recovery.md`
- `docs/superpowers/specs/2026-08-06-mc602-disconnect-timeout-design.md`

## 代码还原范围

`car_start_2026.py` 恢复为直接顺序调用现有任务函数，不再捕获 `ControllerDisconnected`、执行专用清理或返回状态码 75。

`smartcar/whalesbot/vehicle/base/serial_wrap.py` 移除 `CommunicationHealth`、`ControllerDisconnected`、连续失败计时、控制器重启等待和为监督器设计的重新枚举行为，恢复仓库基线通信逻辑。

`smartcar/whalesbot/vehicle/driver/mecanum.py` 移除断连异常处理和 `_controller_disconnected` 状态传播，恢复仓库基线里程计线程逻辑。

`car_wrap_2026.py` 只移除 `ControllerDisconnected` 导入、断连标志以及按键线程中的断连捕获。文件内视觉、采收和其他任务相关修改全部保留。

## 保留范围

以下内容不属于本次回退：

- `car_task_function.py` 中的作物采收、播种、水塔和射击调整
- `car_wrap_2026.py` 中的视觉、相机姿态及任务逻辑调整
- `config_car.yml` 的车辆参数调整
- `smartcar/whalesbot/vehicle/arm/arm_cfg.yaml` 的机械臂现场参数
- 与本功能无关的调试资料、测试和未跟踪文件

## 启动流程

回退后不再安装或运行 `baidu-smart-key-start.service`。操作者进入项目根目录后执行：

```bash
python car_start_2026.py
```

程序立即进入原有初始化及任务流程，不等待端口 5 的按键值，也不由 shell 监督器自动重启。

## 测试与验收

先增加一个最小回归测试，声明人工启动契约并断言监督器运行资产不存在；在删除资产前运行时该测试必须因资产仍存在而失败，完成回退后必须通过。

随后执行：

- `car_start_2026.py` 及受影响 Python 文件的编译检查
- 与车辆入口、串口和底盘相关的现有测试
- 全仓库关键词扫描，确认运行代码中不再引用 `start_with_key`、`wait_for_start_key`、`baidu-smart-key-start`、`ControllerDisconnected`、`CommunicationHealth` 或状态码 75
- `git diff --check`，确认补丁不存在空白错误

验收标准是：监督器资产和专用保护代码全部消失，`python car_start_2026.py` 仍是有效入口，无关工作区修改保持不变。
