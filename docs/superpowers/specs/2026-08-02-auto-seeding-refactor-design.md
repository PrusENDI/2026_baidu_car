# 正式版播种任务可读性重构设计

## 目标

重构 `2026/baidu_smartcar_2026/car_task_function.py` 中的 `auto_seeding()`，使播种任务的机械臂位姿、底盘位姿、视觉参数和等待时间集中、明确、易于调整，同时让主业务函数只表达播种流程。

本次是等价重构。当前正式版中的标定值、车辆和机械臂动作顺序、安全检查、异常触发条件、`debug` 行为均保持不变，不从 `8_2/car_task_function_new.py` 引入参数或流程。

## 范围

本次修改范围仅包括：

- `car_task_function.py` 中播种任务使用的模块级配置、私有辅助函数和 `auto_seeding()`。
- 播种任务的静态结构、配置值、安全顺序和动作轨迹回归测试。
- 现有播种测试中已经与当前正式版不一致的断言。

不修改其他任务，不修改底盘、机械臂或视觉控制底层实现，不处理与播种无关的既有测试失败，不修改或提交当前工作区中的 `7_17/` 和 `8_2/` 内容。

## 行为兼容要求

以下行为必须与重构前完全一致：

- 函数签名保持为 `auto_seeding(debug=False)`。
- 播种点顺序保持为 `cylinder_3`、`cylinder_2`、`cylinder_1`。
- 三个理论播种点继续由 `x_length=0.46`、首点距离 `0.55`、间距 `0.15`、投影角 `math.pi / 4` 和目标航向 `0.74` 计算。
- 正常入口巡线距离保持 `0.85`，调试入口距离保持 `0.5`，速度保持 `0.3`。
- 第一轮视觉对齐偏置保持 `delta_x=-0.1`、`delta_y=-0.05`。
- 保存的底盘位置继续沿当前航向前移 `0.05`。
- 右侧识别前底盘偏移保持 `[0.0, -0.02, 0.0]`。
- 右侧机械臂预置保持 `x=0.25`、`y=0.20`，视觉允许范围保持 `(0.24, 0.260)`。
- 抓取、下降、抬升、回收、翻转、恢复保存位置、视觉补偿和释放的调用顺序不变。
- `retract_x_safe()`、`move_x_position()` 和左侧 `adjust_arm_position()` 的现有失败保护不变。
- 结束姿态、提示音、状态输出及调试返回原点行为不变。

## 配置设计

播种配置放在 `auto_seeding()` 及其私有辅助函数之前，并按职责拆分：

### `SEEDING_CHASSIS_POSES`

集中描述底盘和路线参数：

- `geometry`：转角基准、首点距离、播种点间距、投影角和目标航向。
- `task_entry`：调试/正式入口距离和巡线速度。
- `right_detection_offset`：右侧抓取识别前的底盘相对位移。
- `saved_pose_forward_offset`：第一轮对齐后保存位置的前移补偿。

理论播种点仍由 `_build_seeding_cylinder_poses()` 计算，避免手工维护三套相互关联的坐标。

### `SEEDING_ARM_POSES`

集中描述机械臂位姿：

- `initial`：播种入口姿态。
- `right_pickup`：右侧抓取前的高度、水平预置和方向。
- `left_placement`：左侧放置高度、方向和释放高度。
- `finish`：任务结束时的高度、水平位置和手腕方向。

需要读取实时值或安全位置的字段继续在执行时读取，不把运行时状态错误地固化为常量。

### `SEEDING_VISION_PARAMS`

集中描述视觉对齐参数：

- `record_alignment`：第一轮记录播种位置使用的 `delta_x` 和 `delta_y`。
- `right_pickup_alignment`：右侧抓取识别使用的机械臂水平范围。

### `SEEDING_TIMING`

为当前流程中已有的固定等待时间命名。重构只把既有等待值移入配置，不新增、删除或调整等待。

### `SEEDING_CYLINDER_ORDER`

显式保存播种顺序：

```python
SEEDING_CYLINDER_ORDER = (
    "cylinder_3",
    "cylinder_2",
    "cylinder_1",
)
```

## 函数边界

### `_build_seeding_cylinder_poses()`

从 `SEEDING_CHASSIS_POSES["geometry"]` 计算三个理论播种点，返回以圆柱标签为键的字典。该函数不访问硬件，便于直接进行数值测试。

### `_log_seeding_event(event, **fields)`

统一处理播种任务输出。只有该函数直接调用 `print()`；它根据事件名和字段生成阶段日志，并在需要时读取里程计、机械臂 X/Y 和方向。

日志函数只读取状态，不得发送底盘移动、机械臂移动、抓取、释放或停车命令。不新增日志开关，本次仅封装现有输出职责。

### `_prepare_seeding_task(debug, cylinder_poses)`

完成初始机械臂高度调整、安全回收、LEFT/DOWN 姿态设置和入口巡线。它根据 `debug` 选择当前既有入口距离。

### `_record_seeding_poses(cylinder_poses)`

按 `SEEDING_CYLINDER_ORDER` 依次移动到理论点、进行第一轮视觉对齐、应用五厘米前移补偿，并保存 `[car_x, car_y, heading, arm_x]`。返回 `saved_poses` 字典。

### `_pick_and_place_seed(cylinder_key, cylinder_pose, saved_pose)`

完成单个目标的完整动作：

1. 机械臂抬升并安全回收。
2. 翻转到 RIGHT 后伸到预抓取位置。
3. 底盘移动到理论点并应用右侧识别前偏移。
4. 使用当前水平范围进行视觉对齐。
5. 调整吸嘴、吸附、下降并抬升。
6. 安全回收后翻转到 LEFT。
7. 恢复保存的机械臂 X 和底盘位置。
8. 执行左侧视觉补偿，下降并释放。

现有异常继续在相同的失败条件和阶段抛出。错误信息可以增加 `cylinder_key`，但不得改变判断条件或吞掉异常。

### `_finish_seeding_task(debug, cylinder_poses)`

恢复结束姿态，移动到当前既有结束点，输出完成信息和状态；`debug=True` 时继续返回 `[0.0, 0.0, 0.0]`。

### `auto_seeding(debug=False)`

主函数仅保留业务编排：

```python
def auto_seeding(debug=False):
    cylinder_poses = _build_seeding_cylinder_poses()
    _prepare_seeding_task(debug, cylinder_poses)
    saved_poses = _record_seeding_poses(cylinder_poses)

    for cylinder_key in SEEDING_CYLINDER_ORDER:
        _pick_and_place_seed(
            cylinder_key,
            cylinder_poses[cylinder_key],
            saved_poses[cylinder_key],
        )

    _finish_seeding_task(debug, cylinder_poses)
```

主函数中不得直接出现 `print()`、底盘控制或机械臂控制细节。

## 数据流

`_build_seeding_cylinder_poses()` 首先生成理论位置。第一轮由 `_record_seeding_poses()` 生成：

```python
{
    "cylinder_3": [car_x, car_y, heading, arm_x],
    "cylinder_2": [car_x, car_y, heading, arm_x],
    "cylinder_1": [car_x, car_y, heading, arm_x],
}
```

第二轮不重新推导放置位置，而是按检测标签使用对应的 `saved_pose`，保持当前正式版的数据关系。

## 异常处理

本次不增加重试、跳过、降级或异常捕获：

- 初始化水平安全回收失败时终止。
- 右侧抓取前安全回收失败时终止。
- RIGHT 侧水平预伸出失败时终止。
- 抓取后安全回收失败时禁止翻转到 LEFT。
- LEFT 侧恢复保存的水平位置失败时终止。
- 左侧放置前视觉补偿失败时禁止释放。

## 测试设计

新增播种重构专用测试，采用静态结构检查和伪车辆动作轨迹两层验证。

### 静态结构和配置测试

- 四组配置和播种顺序常量存在。
- 所有配置值与重构前正式版一致。
- `_build_seeding_cylinder_poses()` 产生与重构前相同的三组坐标。
- `auto_seeding(debug=False)` 签名保持不变。
- `auto_seeding()` 不直接调用 `print()`。
- 主函数按准备、记录、循环播种、结束的顺序调用辅助函数。

### 伪车辆动作轨迹测试

使用不连接硬件的伪 `my_car` 记录控制调用和参数，分别验证：

- 正常模式与调试模式入口和结束行为。
- 第一轮记录位置的车辆动作、视觉参数和五厘米补偿。
- 单个目标的右侧抓取和左侧放置动作顺序。
- 所有安全检查发生在对应翻转或释放之前。
- 失败返回值触发现有异常，不继续执行危险动作。
- `_log_seeding_event()` 只产生读取和输出，不产生控制调用。

### 既有测试处理

现有播种测试中，右侧偏移仍期望 `-0.06`、安全回收仍查找 `reset_x()`，已与当前正式版的 `-0.02` 和 `retract_x_safe()` 不一致。本次只把这些播种断言更新为当前正式行为。

其他任务已有的测试失败不纳入本次范围；验证结果中需明确区分新增/播种相关测试与无关基线失败。

## 验收标准

- 新增播种重构测试全部通过。
- 播种相关安全测试通过。
- `car_task_function.py` 可由 Python AST 和编译器正常解析。
- 伪车辆记录的重构后动作轨迹与重构前基线完全一致。
- Git diff 不包含任何意外的播种标定值、运动参数或动作顺序变化。
- `auto_seeding()` 主函数可直接读出“计算位置、准备、记录、逐个播种、结束”的业务流程。
- `auto_seeding()` 主函数不包含直接打印、机械臂或底盘控制细节。
- 不提交或修改用户现有的 `7_17/` 和 `8_2/` 工作区内容。
