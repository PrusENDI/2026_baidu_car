# Debug Session: goods-align-none-setpoint

- Status: [OPEN]
- Symptom: `find_goods()` 第三次调用 `move_to_detection_target()` 时，PID 执行 `None - float` 抛出 TypeError。
- Session ID: `goods-align-none-setpoint`

## Runtime evidence supplied

- 订单识别成功：金针菇（地址 2）、番茄（地址 1）。
- 搜索第一件货物 `h_jin_zhen_gu` 时连续出现目标对齐超时。
- 异常位置：`car_wrap_2026.py:1611`，`out_x = -pid_x(dx)`。
- PID 内部异常：`self.setpoint - input_`，其中 `self.setpoint` 为 `None`。
- 调用参数：`label=label, delta_x=None, delta_y=dy`。

## Falsifiable hypotheses

1. 无检测框时控制循环仍调用 X 轴 PID，且 `delta_x=None` 被设置为 PID setpoint。
2. `h_jin_zhen_gu` 在三次尝试中均未检测到，超时分支未在 PID 调用前返回。
3. 第三次搜索前移动底盘导致检测状态为空或滞后。
4. `delta_x=None` 并非自动对齐语义，而是不合法的 X 轴 PID 目标值。

## Evidence conclusion

- 已收到 pre-fix 事件：`label=h_jin_zhen_gu, delta_x=null, delta_y=-0.2`。
- 第一次搜索在返回日志产生前抛错，证明检测到目标后进入了 X 轴 PID。
- `move_to_detection_target()` 将 `delta_x` 直接赋给 PID setpoint，因此假设 4 已确认；假设 1 部分确认。
- 异常发生于第一次搜索，假设 2、3 被否定。

## Minimal fix

将 `find_goods()` 三次调用中的 `delta_x=None` 改为该接口合法的默认对齐目标 `delta_x=0.0`。等待 post-fix 实机验证。
