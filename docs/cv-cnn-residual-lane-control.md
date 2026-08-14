# OpenCV 直道保持与 CNN 弯道残差融合方案

## 1. 目标

在固定室内赛道中，让 OpenCV 负责可解释、稳定的直道保持，让 CNN 负责急弯、复杂边界和人工驾驶经验修正，同时保留现有 PID 与车辆速度控制链路。

本方案优先控制实现复杂度：第一阶段不修改 CNN 模型结构，只做 OpenCV/CNN 绝对误差平滑混合；验证闭环稳定后，再将 CNN 训练为真正的残差模型。

预期效果：

- 直道上主要依赖 OpenCV 中线，减少 CNN 受光照和背景变化影响；
- 进入弯道时平滑增加 CNN 权重，避免硬切换引起方向突变；
- 急弯和复杂路段由 CNN 学习人工驾驶中的提前转向和特殊修正；
- OpenCV 无效时由 CNN 临时兜底，并降低车速；
- 最终控制仍统一经过现有 `lane_pid`，保持限幅和控制接口不变。

## 2. 当前实现

当前运行时车道控制链路位于 `car_wrap_2026.py`：

```text
cap_front.read()
    ↓
get_lane_results()
    ↓
self.crusie(image)              # CNN 推理服务
    ↓
(error_y, error_angle)
    ↓
self.lane_pid.get_out(...)
    ↓
set_velocity(speed, y_speed, angle_speed)
```

相关文件：

- `car_wrap_2026.py`
  - `get_lane_results()`：读取图像并取得 CNN 的两个误差输出；
  - `lane_base()`：将误差交给 `lane_pid`，并根据角度误差降速。
- `smartcar/paddlebaidu/paddle_jetson/base/infer_wrap.py`
  - `LaneInfer`：将输入缩放为 `128 × 128`，输出模型预测结果。
- `smartcar/whalesbot/tools/lane_collect/opencv_lane.py`
  - `OpenCVLaneAnalyzer`：输出横向误差、航向误差、置信度、弯角和边界状态。
- `smartcar/whalesbot/tools/lane_collect/calibration.py`
  - `ErrorMapping`：将 OpenCV 几何量映射到 CNN 误差域。

目前 `lane_collect` 中的 OpenCV 分析器只用于采集和离线分析，尚未接入闭环运行时。

## 3. 总体架构

```text
                         ┌─ OpenCVLaneAnalyzer ─> cv_error
前置摄像头同一帧图像 ────┤
                         └─ LaneInfer/CNN ──────> cnn_output
                                      ↓
                              权重计算与平滑
                                      ↓
                            融合后的 error_y/angle
                                      ↓
                                 现有 lane_pid
                                      ↓
                    set_velocity(speed, y_speed, angle_speed)
```

必须对同一帧图像同时执行 OpenCV 和 CNN，避免两条路径因为车辆运动产生时间错位。

## 4. 分阶段实施

### 4.1 阶段一：绝对误差混合

阶段一继续使用现有 CNN，CNN 仍输出绝对误差：

```text
cnn_error_y
cnn_error_angle
```

融合公式：

```text
error_y     = (1 - w) × cv_error_y     + w × cnn_error_y
error_angle = (1 - w) × cv_error_angle + w × cnn_error_angle
```

其中 `w` 是 CNN 权重：

- `w = 0`：完全使用 OpenCV；
- `w = 1`：完全使用 CNN；
- `0 < w < 1`：进入或离开弯道时平滑混合。

阶段一不属于严格的残差学习，但无需重新训练模型，适合先验证 OpenCV 标定、弯道判断和闭环切换。

### 4.2 阶段二：真正的残差模型

验证阶段一稳定后，将 CNN 重新训练为输出：

```text
delta_y
delta_angle
```

融合公式改为：

```text
error_y     = cv_error_y     + w × delta_y
error_angle = cv_error_angle + w × delta_angle
```

训练标签：

```text
delta_y_label     = expert_error_y     - cv_error_y
delta_angle_label = expert_error_angle - cv_error_angle
```

`expert_error_*` 必须与当前 CNN 训练标签或人工操控数据使用同一含义、方向和量纲。

## 5. 直道、弯道与复杂路段判断

优先使用 `OpenCVLaneAnalyzer.process()` 已有输出：

- `valid`：OpenCV 结果是否可用；
- `confidence`：当前检测置信度；
- `raw_heading`：赛道几何航向；
- `metrics["corner_detected"]`：是否检测到陡峭弯角；
- `metrics["tracking_mode"]`：`both`、`left_only`、`right_only`、`mixed` 或 `none`；
- `cross_status`：是否疑似十字或宽度异常区域。

初始判断规则建议：

```python
def compute_cnn_weight(cv_result):
    if not cv_result.valid:
        return 1.0

    confidence = float(cv_result.confidence)
    heading = abs(float(cv_result.raw_heading or 0.0))
    corner = bool(cv_result.metrics.get("corner_detected", False))
    mode = cv_result.metrics.get("tracking_mode", "none")

    if confidence < 0.50:
        return 1.0

    if corner or heading >= 0.35:
        return 1.0

    if heading <= 0.15 and mode in ("both", "mixed"):
        return 0.0

    return min(max((heading - 0.15) / (0.35 - 0.15), 0.0), 1.0)
```

这些阈值只是初始值，必须根据实车记录调整。不要仅凭单帧 `corner_detected` 硬切换控制源。

### 5.1 权重时间平滑

对目标权重使用低通平滑：

```python
target_weight = compute_cnn_weight(cv_result)
self._cnn_weight = (
    0.80 * self._cnn_weight
    + 0.20 * target_weight
)
```

还可以设置进入弯道快、退出弯道慢：

```python
rate = 0.35 if target_weight > self._cnn_weight else 0.12
self._cnn_weight += rate * (target_weight - self._cnn_weight)
```

这样 CNN 可以及时接管急弯，同时避免刚出弯就立即切回 OpenCV。

## 6. 运行时接入建议

### 6.1 初始化

在车辆封装类初始化阶段创建分析器和状态变量：

```python
from smartcar.whalesbot.tools.lane_collect import (
    LaneAnalyzerConfig,
    OpenCVLaneAnalyzer,
)

self.cv_lane_analyzer = OpenCVLaneAnalyzer(
    LaneAnalyzerConfig(
        work_size=(320, 240),
        segmentation="dark",
    )
)
self._cnn_weight = 0.0
self._last_valid_cv_error = None
```

需要将阈值、ROI、权重和限幅移入 `config_car.yml`，避免后续反复修改代码。

### 6.2 修改 `get_lane_results()`

阶段一伪代码：

```python
def get_lane_results(self):
    image = self.cap_front.read().copy()

    cnn_result = self.crusie(image)
    cnn_y = float(cnn_result[0])
    cnn_angle = float(cnn_result[1])

    cv_result = self.cv_lane_analyzer.process(image)
    target_weight = self._compute_cnn_weight(cv_result)
    self._cnn_weight = self._smooth_weight(target_weight)

    if cv_result.valid:
        cv_y = float(cv_result.error_y)
        cv_angle = float(cv_result.error_angle)
        self._last_valid_cv_error = (cv_y, cv_angle)

        error_y = (1.0 - self._cnn_weight) * cv_y + self._cnn_weight * cnn_y
        error_angle = (
            (1.0 - self._cnn_weight) * cv_angle
            + self._cnn_weight * cnn_angle
        )
    else:
        error_y = cnn_y
        error_angle = cnn_angle

    return error_y, error_angle
```

阶段二只需要将两行融合公式替换为：

```python
error_y = cv_y + self._cnn_weight * cnn_delta_y
error_angle = cv_angle + self._cnn_weight * cnn_delta_angle
```

### 6.3 PID 保持不变

`lane_base()` 继续使用现有逻辑：

```python
error_y, error_angle = self.get_lane_results()
y_speed, angle_speed = self.lane_pid.get_out(-error_y, -error_angle)
self.set_velocity(speed, y_speed, angle_speed)
```

不要直接融合 `y_speed` 或 `angle_speed`，否则两个控制源可能具有不同的动态响应和限幅，调试会明显变复杂。

## 7. 标定要求

OpenCV 与 CNN 的输出必须统一以下定义：

1. 正负方向一致；
2. 横向误差范围一致；
3. 航向误差范围一致；
4. 时间对应关系一致；
5. 输入图像裁剪和方向一致。

`ErrorMapping` 当前默认只保持原始值，它的注释也说明在闭环控制前必须标定。可以使用：

```text
error_y = raw_lateral × lateral_scale + lateral_bias

error_angle = raw_heading × heading_scale
              + raw_lateral × heading_lateral_mix
              + heading_bias
```

建议用人工驾驶记录离线拟合这些参数，先确认符号，再拟合比例和偏置。不能直接将 OpenCV 的弧度值与模型输出或角速度命令相加。

## 8. 残差训练数据

每条训练样本至少保存：

```text
timestamp
image_path
expert_error_y
expert_error_angle
cv_valid
cv_error_y
cv_error_angle
cv_confidence
cv_raw_heading
cv_tracking_mode
cv_corner_detected
delta_y
delta_angle
```

数据组成建议：

- 直道正常样本：残差接近零，用于抑制 CNN 偏置；
- 普通弯道样本：学习提前转向和出弯回正；
- 急弯样本：提高采样权重；
- 单边界、mask 断裂和复杂区域：学习 OpenCV 失效附近的修正；
- 偏左、偏右、车头偏斜：学习恢复行为；
- 多种室内光照：验证分割和控制链路稳定性。

训练集、验证集和测试集应按完整驾驶片段划分，不能把相邻视频帧随机分散到不同集合。

### 8.1 损失函数

第一版可以使用加权 Smooth L1：

```text
loss = sample_weight × (
    λy × SmoothL1(pred_delta_y, target_delta_y)
    + λa × SmoothL1(pred_delta_angle, target_delta_angle)
)
```

急弯、恢复和复杂路段可以设置更高的 `sample_weight`。直道数据仍需保留，但应控制数量，避免大量相似直道帧淹没弯道样本。

## 9. 安全策略

### 9.1 OpenCV 无效

OpenCV 无效时不能将 `(0, 0)` 当作有效直道误差。应：

- 将 CNN 权重升至 `1.0`；
- 降低前进速度；
- 记录失败原因；
- 连续多帧 OpenCV 和 CNN 均不可信时停车。

### 9.2 残差限幅

阶段二必须限制 CNN 残差：

```python
cnn_delta_y = np.clip(cnn_delta_y, -max_delta_y, max_delta_y)
cnn_delta_angle = np.clip(
    cnn_delta_angle,
    -max_delta_angle,
    max_delta_angle,
)
```

第一轮实车测试建议将最大残差限制在完整控制范围的 10%～30%。

### 9.3 输出跳变限制

对最终误差或 PID 输出增加单帧最大变化限制，防止模型异常和权重切换产生瞬时大转向。

### 9.4 速度控制

除现有基于 `error_angle` 的降速外，可以加入：

```text
OpenCV 无效             -> 低速
confidence 低           -> 降速
CNN 权重较高            -> 弯道速度
连续多帧均不可信         -> 停车
```

## 10. 配置建议

建议在 `config_car.yml` 增加独立配置段：

```yaml
lane_fusion:
  enabled: false
  mode: absolute_blend       # absolute_blend / residual

  cv:
    work_size: [320, 240]
    segmentation: dark
    threshold: 175
    min_confidence: 0.50

  gate:
    straight_heading: 0.15
    curve_heading: 0.35
    enter_rate: 0.35
    exit_rate: 0.12

  residual_limits:
    error_y: 0.15
    error_angle: 0.20

  safety:
    invalid_speed: 0.12
    max_invalid_frames: 8
```

默认 `enabled: false`，便于随时回退到当前纯 CNN 控制链路。

## 11. 日志与可视化

每帧至少记录：

```text
cv_valid
cv_error_y
cv_error_angle
cv_confidence
cv_raw_heading
tracking_mode
corner_detected
cnn_y / cnn_delta_y
cnn_angle / cnn_delta_angle
cnn_weight
fused_error_y
fused_error_angle
speed
y_speed
angle_speed
```

视频调试画面建议同时显示：

- OpenCV lane mask；
- 左右边界和拟合中线；
- CNN 权重；
- OpenCV、CNN 与融合后的输出；
- 当前模式：`STRAIGHT`、`TRANSITION`、`CURVE`、`CV_INVALID`。

## 12. 测试计划

### 12.1 离线回放

对已有驾驶数据逐帧运行两条路径，输出 CSV 和叠加视频，检查：

- 直道时 `w` 是否接近 0；
- 入弯前 `w` 是否平滑上升；
- 出弯时是否缓慢下降；
- OpenCV 无效时是否及时切换；
- 融合结果是否存在符号相反或幅值突变。

### 12.2 单元测试

至少覆盖：

- 居中直道；
- 左右偏移直道；
- 普通弯道和急弯；
- 单侧边界；
- 两侧边界均丢失；
- 权重进入和退出速率；
- CNN 残差限幅；
- OpenCV 无效时不误判为直道。

### 12.3 低速实车

按以下顺序测试：

1. 只记录 OpenCV，不参与控制；
2. 直道固定 `w = 0`，弯道固定 `w = 1`；
3. 开启权重平滑；
4. 测试不同光照；
5. 人工制造轻微偏离；
6. 最后启用残差模型。

任何阶段出现不稳定时，使用配置开关立即回退到纯 CNN。

## 13. 验收标准

建议以完整圈和困难片段统计，而不是只看单帧误差：

- 不同光照下连续完整跑圈成功率；
- 直道横向误差均值和最大值；
- 入弯、弯中、出弯的最大偏差；
- 人工制造轻微偏离后的恢复成功率和恢复时间；
- OpenCV 无效次数及持续帧数；
- 控制输出单帧最大跳变；
- 平均推理延迟和最低控制帧率。

## 14. 推荐落地顺序

1. 校准 `ErrorMapping`，确认 OpenCV 和 CNN 输出方向、范围一致；
2. 将 OpenCV 分析器接入 `get_lane_results()`，仅记录不控制；
3. 实现 `absolute_blend`，验证直道/弯道切换；
4. 加入权重平滑、异常降速和配置回退；
5. 离线生成残差标签；
6. 训练并部署 `delta_y/delta_angle` 模型；
7. 切换到 `residual` 模式，先以严格限幅低速验证；
8. 完成多光照、偏离恢复和完整跑圈测试。

该顺序把感知、标定、融合和模型训练拆开验证，可以避免一次同时引入多个不确定因素。
