# 兼容现有双 PID 接口的循迹 CNN 训练设计与操作说明

**日期：** 2026-08-06

**项目：** `baidu_smart_2026_7_17`

**状态：** 已确认设计，待编写实现计划

## 1. 目标

使用当前人工遥控采集的前置摄像头图像和车辆状态训练新的循迹 CNN。新模型必须完全兼容现有运行接口：输入预处理、输出数量、输出顺序、推理服务、双 PID、底盘调用和 `car_task_function.py` 均保持不变，部署时只替换循迹模型文件。

新模型的数据流固定为：

```text
前置摄像头图像
  -> CNN
  -> [error_y, error_angle]
  -> 当前双 PID
  -> [y_speed, angle_speed]
  -> set_velocity(speed, y_speed, angle_speed)
```

本设计中的训练标签来自人工遥控数据：

```text
error_y     := state[1]（人工横移控制量）
error_angle := state[2]（人工转向角速度控制量）
```

这些名称保留现有运行接口的命名。训练和部署必须保持相同的数值顺序与范围，不改变当前控制链。

## 2. 明确不做的事项

- 不改成单输出 `deviation` 模型。
- 不修改 `car_task_function.py` 的任务调用。
- 不修改 `lane_base()` 的双 PID 控制结构。
- 不修改 `config_car.yml` 中的巡线 PID 参数。
- 不修改模型服务端口、模型目录名称或调用协议。
- 第一版不引入十字路口状态机、锐角状态机或时序网络。
- 不把 2025 年传统视觉算法接入车辆运行路径。

## 3. 现有运行接口

当前巡线循环位于 `car_wrap_2026.py`：

```python
error_y, error_angle = self.get_lane_results()
y_speed, angle_speed = self.lane_pid.get_out(-error_y, -error_angle)
self.set_velocity(speed, y_speed, angle_speed)
```

当前模型接口必须保持：

| 项目 | 约束 |
|---|---|
| 输入形状 | `[1, 3, 128, 128]` |
| 输入类型 | `float32` |
| 输入颜色 | BGR 图像在推理预处理中转换为 RGB |
| 归一化 | `pixel / 127.5 - 1.0` |
| 数据布局 | NCHW |
| 输出形状 | `[1, 2]` |
| `output[0]` | `error_y` |
| `output[1]` | `error_angle` |
| 模型文件 | `cnn_lane.pdmodel`、`cnn_lane.pdiparams` |
| 部署目录 | `smartcar/paddlebaidu/models/lane_model/` |

现有离线报告中，新模型在 2234 张验证图像上的输出范围为：

```text
error_y:     -0.0006344 ～ 0.0016188
error_angle: -0.6125198 ～ 0.7163030
```

该范围作为新模型离线检查的参考，不作为训练时的硬裁剪范围。训练标签仍以实际采集控制量为准，并过滤超出底盘安全范围的异常值。

## 4. 赛道与学习任务

赛道包含：

- 一个十字路口，两次经过均为直行；
- 一个锐角右拐；
- 其他主要弯道为左拐直角。

这些位置不存在同一画面需要执行不同路线选择的问题，因此第一版可以使用单帧 CNN。训练数据必须体现唯一驾驶策略：

- 十字路口始终直行，只允许保持车道所需的小幅修正；
- 左直角使用平滑、可重复的左转操作；
- 锐角右拐提前减速并平滑右转，避免突然打满手柄；
- 同一位置不得混入方向相反或明显不同的驾驶动作。

如果以后同一视觉场景需要执行不同路线，单帧 CNN 将无法仅凭图像区分任务意图，届时应增加任务状态输入；这不属于本次范围。

## 5. 数据采集设计

### 5.1 当前数据格式

采集器为每一帧保存图像和车辆状态：

```json
{
  "img_path": "0001.jpg",
  "state": [0.15, 0.0, -0.21]
}
```

字段含义：

```text
state[0] = vx，前进速度
state[1] = vy，横向速度
state[2] = wz，旋转角速度
```

训练时只使用 `state[1]` 和 `state[2]` 作为两个输出标签；`state[0]` 用于数据检查和分析，不作为模型输出。

### 5.2 采集数量

第一版目标：

| 数据 | 建议数量 |
|---|---:|
| 完整正常圈 | 15～20 圈 |
| 锐角右拐强化 | 额外 5～10 次 |
| 每种明显不同光照 | 3～5 圈 |
| 预计有效图像 | 10000～20000 张 |

训练效果主要取决于动作一致性和场景覆盖，不以盲目增加连续重复帧为目标。

### 5.3 会话隔离

当前采集器每次启动从 `0000.jpg` 开始，并重新创建内存中的 `data.json`。多次采集不能直接反复写入同一目录，否则可能覆盖先前图像。

每次完整采集后必须立即把目录保存为独立会话：

```text
dataset/lane_sessions/
  lap_001/
    0000.jpg
    0001.jpg
    data.json
  lap_002/
  lap_003/
  sharp_right_extra_001/
```

训练工具按会话读取数据，不把所有图像改名后混入单一目录，从而保留圈次边界并支持无泄漏划分。

### 5.4 驾驶规范

- 车辆尽量保持赛道中心，驾驶动作连续、平滑。
- 普通直道不做无必要的左右摇摆。
- 左直角采用相似的入弯点和转向幅度。
- 十字路口保持直行，不故意探索左右出口。
- 锐角右拐应覆盖轻微左偏、居中和轻微右偏三种入弯位置，但最终路线必须正确。
- 出界、碰撞、错误方向、摄像头遮挡、严重模糊、程序停车和车辆被人工搬动的帧不进入训练集。
- 应至少保留一整圈从未参与训练的数据作为最终离线验收集。

## 6. 数据检查与划分

训练前的检查工具必须报告：

- 会话数、图像数和标签数；
- 缺失图像、损坏图像、重复文件名；
- `state` 长度错误、布尔值、字符串、NaN 和无穷值；
- `vx`、`vy`、`wz` 的最小值、最大值、均值和分位数；
- 停车帧比例；
- 直行帧与转向帧比例；
- 每个会话的样本数和标签分布。

默认过滤规则：

- 图像不存在或 OpenCV 无法解码：剔除并报告；
- `state` 不是三个有限数值：剔除并报告；
- `abs(vy) > 0.7` 或 `abs(wz) > 1.5`：作为异常值剔除并报告；
- `vx == 0` 且 `vy == 0` 且 `wz == 0`：默认剔除；
- 车辆正常前进但转向为零的帧必须保留，它们是直道和十字直行的重要样本。

数据按会话划分：

```text
训练集：约 80% 会话
验证集：约 20% 会话
最终验收集：至少 1 个完整独立会话
```

禁止逐帧随机划分，因为相邻视频帧高度相似，会造成训练集与验证集泄漏。

## 7. 数据平衡与增强

直道帧通常远多于弯道帧。训练加载器按 `abs(error_angle)` 对样本分桶并加权采样，保证每个批次包含足够转向帧。第一版使用以下分桶定义：

```text
直行：abs(error_angle) < 0.05
轻转：0.05 <= abs(error_angle) < 0.30
明显转向：abs(error_angle) >= 0.30
```

三个分桶的采样权重由工具根据样本数量自动反比计算，并限制单个样本最大权重，避免少量异常帧被反复过采样。

允许的图像增强：

- 随机亮度；
- 随机对比度；
- 随机 Gamma；
- 轻微高斯模糊；
- 轻微传感器噪声。

第一版禁用：

- 水平翻转，因为它会制造赛道中不存在的锐角左拐；
- 大角度旋转、透视变换和大幅随机裁剪，因为标签没有随几何变化同步修正；
- 改变宽高比。

## 8. 模型设计

使用轻量卷积回归网络，保持 128×128 输入并满足 Orin 实时推理需求：

```text
Input: 3×128×128
Conv 3→24, kernel=5, stride=2 + ReLU
Conv 24→36, kernel=5, stride=2 + ReLU
Conv 36→48, kernel=5, stride=2 + ReLU
Conv 48→64, kernel=3, stride=2 + ReLU
Conv 64→64, kernel=3, stride=2 + ReLU
AdaptiveAvgPool 4×4
Flatten
Linear 1024→256 + ReLU + Dropout(0.3)
Linear 256→64 + ReLU
Linear 64→2
```

最后一层为线性层，不使用 Softmax。输出不在运行时额外缩放，确保导出的两个数直接对应训练标签范围。

## 9. 训练设计

默认训练参数：

| 参数 | 默认值 |
|---|---:|
| 优化器 | AdamW |
| 学习率 | `1e-3` |
| 权重衰减 | `1e-4` |
| Batch size | 64 |
| 最大 epoch | 100 |
| Early stopping patience | 12 |
| 随机种子 | 2026 |

每个样本的标签为：

```python
target = [record["state"][1], record["state"][2]]
```

损失函数采用逐输出 Smooth L1：

```python
loss_y = smooth_l1(pred[:, 0], target[:, 0])
loss_angle = smooth_l1(pred[:, 1], target[:, 1])
loss = loss_y + loss_angle
```

模型选择不仅查看总损失，还必须分别保存：

- `MAE(error_y)`；
- `MAE(error_angle)`；
- 转向帧上的角度 MAE；
- 明显转向帧的方向符号正确率；
- 输出最小值、最大值和百分位数；
- 验证集预测与标签的 Pearson 相关系数。

最终选择验证集综合指标最佳且输出范围正常的 checkpoint，而不是机械选择最后一个 epoch。

## 10. 导出设计

训练 checkpoint 导出为 Paddle 静态推理模型：

```text
cnn_lane.pdmodel
cnn_lane.pdiparams
```

导出前后必须使用同一批固定测试图像做数值一致性检查：

```text
动态图输出与静态模型输出的最大绝对差 <= 1e-5
```

静态模型还必须通过以下契约检查：

- 只需要一个图像输入；
- 输入接受 `[1, 3, 128, 128] float32`；
- 输出包含两个有限数值；
- 输出顺序为 `[error_y, error_angle]`；
- CPU 和 Orin GPU 均能成功执行一次推理。

## 11. 离线评估

新旧模型在同一独立验收集上逐帧比较。评估输出包括：

```text
per_frame.csv
summary.json
```

每帧至少记录：

```text
img_path
label_error_y
label_error_angle
old_error_y
old_error_angle
new_error_y
new_error_angle
old_latency_ms
new_latency_ms
```

必须人工查看以下关键帧：

- 十字入口、十字中心和十字出口；
- 锐角右拐入弯、弯心和出弯；
- 每个左直角；
- 最亮、最暗和阴影明显的路段；
- 新旧模型预测方向不一致的帧。

离线验收条件：

- 所有输出均为有限数值；
- 输出形状始终为两个数；
- 明显左右转向的符号与人工标签一致；
- 新模型转向帧 MAE 不高于旧模型；
- 新模型平均推理延迟不明显高于旧模型；
- 十字直行帧没有持续的大幅转向输出。

## 12. 部署与回滚

部署前先把当前模型目录完整复制为带时间戳的备份：

```text
smartcar/paddlebaidu/models/lane_model_backup_YYYYMMDD_HHMMSS/
```

然后只替换：

```text
smartcar/paddlebaidu/models/lane_model/cnn_lane.pdmodel
smartcar/paddlebaidu/models/lane_model/cnn_lane.pdiparams
```

不得同时修改 PID、相机分辨率、推理预处理和巡线速度，否则无法判断问题来自模型还是控制参数。

回滚方式是停止推理服务，把备份的两个模型文件恢复到 `lane_model`，再重新启动推理服务。回滚不需要修改 Python 代码。

## 13. 装车测试

装车测试按风险递增：

1. 零运动推理：车辆轮子不接触地面，只检查输出与方向。
2. 架空轮测试：确认左弯画面产生正确方向的轮速响应。
3. `0.05 m/s`、3 秒直道测试。
4. `0.05 m/s`、只经过一次普通左直角。
5. `0.05 m/s`、只经过一次十字直行。
6. `0.05 m/s`、只经过一次锐角右拐。
7. 低速完整一圈。
8. 连续两圈，确认第二次十字直行同样稳定。
9. 分阶段恢复任务使用速度。

每次测试必须有人在车旁准备停止车辆。出现以下任一情况立即停止并回滚：

- 左右方向相反；
- 模型输出 NaN、无穷值或数量不是两个；
- 十字路口持续向侧路转向；
- 锐角右拐出现不可恢复的外切；
- 角速度长期打满 `±1.5`；
- 推理超时或服务异常退出。

## 14. 操作说明

### 14.1 当前即可执行：采集数据

在车辆运行环境中，从项目根目录启动：

```text
python collect_data.py
```

使用蓝牙手柄正常驾驶。按住采集按键 3 时，前置摄像头图像和车辆状态写入：

```text
dataset/image_set_lane/
```

完成一圈并正常退出后，确认目录中同时存在 JPG 图像和 `data.json`。随后立即把整个 `image_set_lane` 保存成新的会话目录，例如：

```text
dataset/lane_sessions/lap_001/
```

下一次采集前必须确认新的 `image_set_lane` 为空或不存在，防止覆盖旧会话。不要使用采集器的“清空全部数据”功能处理需要保留的会话。

### 14.2 当前即可执行：人工数据检查

每个会话完成后：

1. 检查 `data.json` 能正常打开且记录数大于零；
2. 抽查开头、中间和结尾的图片；
3. 确认图片方向、曝光和清晰度正常；
4. 确认记录中的 `img_path` 文件实际存在；
5. 记录该会话的圈次、光照、速度和异常路段；
6. 如果整圈驾驶明显失败，整圈标记为不参与训练；
7. 局部错误则删除对应连续帧及其 JSON 记录，保留原始会话备份。

### 14.3 实现训练工具后的标准命令接口

后续实现应提供以下平台无关的 Python CLI；训练环境确认后仅调整依赖安装方式，不改变命令含义。

检查数据：

```text
python tools/lane_training/inspect_dataset.py \
  --sessions dataset/lane_sessions \
  --report artifacts/lane_dataset_report.json
```

生成按会话划分的清单：

```text
python tools/lane_training/build_manifest.py \
  --sessions dataset/lane_sessions \
  --output artifacts/lane_manifest \
  --seed 2026
```

训练：

```text
python tools/lane_training/train_lane_cnn.py \
  --manifest artifacts/lane_manifest \
  --output runs/lane_cnn_001 \
  --epochs 100 \
  --batch-size 64 \
  --learning-rate 0.001 \
  --seed 2026
```

导出：

```text
python tools/lane_training/export_lane_cnn.py \
  --checkpoint runs/lane_cnn_001/best.pdparams \
  --output runs/lane_cnn_001/inference
```

接口验证：

```text
python tools/lane_training/validate_export.py \
  --checkpoint runs/lane_cnn_001/best.pdparams \
  --model runs/lane_cnn_001/inference \
  --manifest artifacts/lane_manifest/test.json
```

新旧模型离线比较：

```text
python tools/lane_training/evaluate_models.py \
  --dataset artifacts/lane_manifest/test.json \
  --old-model smartcar/paddlebaidu/models/lane_model \
  --new-model runs/lane_cnn_001/inference \
  --output runs/lane_cnn_001/evaluation
```

这些训练工具属于后续实现范围；在对应脚本实现并通过测试前，不应把上述命令当作当前仓库已经可运行的命令。

## 15. 失败处理

| 失败 | 处理 |
|---|---|
| 图片缺失或损坏 | 剔除样本并在报告中列出 |
| JSON 损坏 | 停止使用该会话，优先从备份恢复 |
| 左右方向学反 | 禁止装车，检查手柄符号和输出顺序 |
| 直道持续摆动 | 检查转向标签延迟、直行样本质量和输出范围 |
| 十字转入侧路 | 增加十字直行样本，删除十字处错误修正数据 |
| 锐角右拐不足 | 增加正确锐角样本并提高明显转向采样权重 |
| 锐角右拐过度 | 删除打满手柄和出界样本，增加平滑出弯数据 |
| 训练好、装车差 | 首先核对 RGB/BGR、128×128 和归一化一致性 |
| 推理输出数量错误 | 导出失败，禁止替换当前模型 |
| 推理延迟过高 | 减少网络通道数，保持输出接口不变 |

## 16. 完成标准

本方案完成必须同时满足：

- 数据检查、划分、训练、导出、接口验证和新旧评估工具均有自动化测试；
- 新模型输入输出契约与现有 `LaneInfer` 完全一致；
- 不修改 `car_task_function.py`、巡线 PID 和运行控制链；
- 新旧模型离线报告可复现；
- 装车依次通过直道、左直角、十字直行、锐角右拐和完整圈测试；
- 当前模型有明确、已验证的回滚副本；
- 操作记录能够追溯训练数据会话、训练参数、checkpoint 和部署模型。
