# 百度智能车巡航 CNN 训练接续文档

更新时间：2026-08-17
用途：会话压缩或更换对话后，先阅读本文件，再查看所链接的脚本和原始报告。本文是当前 CV 采集标签、CNN 训练接口和车端部署语义的主入口；[CV 自动采样巡线开发状态](cv-lane-development-status-2026-08-15.md)保留视觉算法、赛道帧段和参数演进历史。历史章节中的旧接口不得覆盖本文顶部、第 18 节和第 20 节的当前结论。

## 当前最终模型接口（2026-08-17）

巡线 CNN 保持两个输出，但语义改为：

```text
[speed_demand, kappa_action]
```

`speed_demand` 学习 CV 的 `steering_demand`，只负责驱动车端可调速度曲线；
`kappa_action` 学习最终动作曲率，负责普通巡线、偏移归位、启动保护、断线保持和
锐角动作。车端执行 `target_vx=speed_curve(speed_demand)`、基于真实 `dt` 的速度
梯度以及 `wz=actual_vx*kappa_action`，当前固定 `vy=0`。本文后续历史章节若仍写
`[error_y,error_angle]` 或仅输出 `kappa_action`，均以本节为准。

CV 与手柄数据的公共原始字段固定为 `state/control=[vx,vy,wz]`。模型标签使用独立
字段 `model_target={speed_demand,kappa_action}` 与 `target_mask`：CV mask 为
`[1,1]`，手柄数据默认只监督曲率，mask 为 `[0,1]`。训练数据集必须启用 mask
返回并对两个输出分别计算损失，禁止把手柄速度需求占位值作为有效标签。

## 1. 仓库与操作边界

- Worktree：`C:\weizijian\documents\baidu car\baidu_smartcar_2026\.worktrees\lane-cnn-finetuning`
- 分支：`orin-main-20260814`，跟踪 `github/orin-main-20260814`。
- 当前 CV/CNN 控制整合基线：`30d0f0e feat: add CV teacher and curvature CNN control`；后续文档提交号以 `git log -1` 为准。
- 保留当前 worktree，不清理本地未跟踪的虚拟环境、训练实验脚本和 `artifacts/`。
- 不修改 `car_task_function.py`。
- 不运行或上传官方 locked test。
- 新 CV 模型的两维输出是 `[speed_demand, kappa_action]`；旧 D4/R7 的 `[vy, yaw_error]` 只在 `legacy_error` profile 下继续兼容。
- 现有 epoch 5/20 车端包均为用户明确接受风险后的导出，不代表行为门禁通过，也不能仅凭离线指标宣称一定可上车。
- 后续若改变输入分辨率、模型结构、解冻范围、损失或数据角色，应先说明预期收益和退化风险；本文件不构成自动授权。

当前工作树包含大量未跟踪的训练脚本、测试和 `artifacts/`，均视为有效工作成果，不要清理或覆盖。

## 2. 运行环境

GPU 服务器：

```powershell
ssh -i "C:\Users\fjcy\.ssh\autodl_bjb2_28297" -p 42832 root@connect.bjb2.seetacloud.com
```

- 本地私钥：`C:\Users\fjcy\.ssh\autodl_bjb2_28297`
- 远端 Python：`/root/autodl-tmp/envs/lane-d5/bin/python`
- 远端项目：`/root/autodl-tmp/lane-cnn/project`
- 数据根目录：`/root/autodl-tmp/lane-cnn/datasets/lane_sessions2`
- 旧 Paddle 模型启动必须设置：`FLAGS_enable_pir_api=0`

## 3. 模型输入、输出与车端控制

- 官方网络结构固定输入为 `[N, 3, 128, 128]`，仍保留两个输出节点；输出节点的语义由部署 profile 决定。
- 训练端和车端都将图像直接双线性缩放为 `128×128`，归一化为 `pixel / 127.5 - 1.0`，再转成 RGB/CHW。
- `128×128` 是官方模型和车端部署接口的一部分，不只是训练参数。只改训练端分辨率会造成模型与车端预处理不一致。
- `legacy_error`：旧模型输出 `[vy, yaw_error]`，继续通过旧 PID 和 `CROP` 降速链；旧模型第二维不是角速度，不能用 `yaw/speed` 推导曲率。
- `kappa_action`：新模型输出 `[speed_demand, kappa_action]`，其中 `speed_demand∈[0,1]` 是弯道速度需求强度，`kappa_action` 是最终动作曲率。
- 新模型车端执行 `target_vx=speed_curve(speed_demand)`、真实 `dt` 速度梯度、`wz=actual_vx*kappa_action` 和角速度安全限幅，当前 `vy=0`。
- profile 在 `config_car.yml` 的 `lane_control.profile` 中选择；部署新模型前必须显式改为 `kappa_action`，默认值 `legacy_error` 是为防止旧模型被误解释。
- 正确车端入口以当前 worktree 的 `car_wrap_2026.py` 为准，不再引用工作树外的旧副本或固定 SHA256。

相关实现：

- `lane_training/dataset.py`
- `smartcar/paddlebaidu/paddle_jetson/base/infer_wrap.py`
- `config_car.yml`

## 4. 当前现场数据

原始归档：

```text
C:\weizijian\documents\baidu car\lane_sessions2.tar.gz
```

远端解压结构：

```text
/root/autodl-tmp/lane-cnn/datasets/lane_sessions2/
  lap_001/data.json
  lap_001/<images>
  lap_002/data.json
  lap_002/<images>
  lap_003/data.json
  lap_003/<images>
```

数据含义：

- `lap_001`：完整一圈现场采集；当前 R7 只把人工确认的两个直角作为曲线监督，其余帧进入 D4 保持池。
- `lap_002`：完整一圈现场采集；用于过诊断和评测，但未进入当前 R7 训练。
- `lap_003`：重复采集，覆盖大部分弯道和特殊路段；当前 R7 使用人工确认的弯道、直行十字路口及其余保持帧。
- 三个 lap 都是真实赛道光照，不是 `lane_imageset` 那种人工提高曝光度的抗干扰增强集。
- 采集驾驶方式、速度与官方驾驶员不同，因此标签不是绝对真值；当前方法在弯道上允许模型向现场标签靠近，同时用 D4 教师限制整体漂移。

当前 R7 不使用以下数据：

- 官方 locked test；
- `offical-line-test`；
- `lane_imageset`；
- `image_set_lane`；
- 官方原始训练数据回放。

## 5. 人工确认的物理路段

不要再用 `abs(yaw) >= 0.05` 自动检测结果代替物理路段。下面的范围由用户按赛道逐段确认，帧号按加载后的记录索引、首尾均包含。

### lap_003

| 范围 | 物理含义 | R7 角色 |
|---|---|---|
| 197–277 | 十字路口前缓弯 | curve |
| 266–472 | 十字路口，要求直行 | cross，覆盖重叠的 266–277 |
| 487–685 | 两边线倒圆的直角 | curve |
| 703–860 | 两边线倒圆的直角 | curve |
| 834–1148 | 两边线倒圆的直角 | curve；R7 合并为 487–1148 |
| 1148–1219 | 另一个十字路口，要求直行 | cross，覆盖 1148 |
| 1260–1390 | 直角 | curve |
| 1559–1630 | 十字路口前缓弯 | curve |
| 2567–2734 | 直角 | curve |
| 3997–4124 | 锐角 | curve |
| 4124–4208 | 锐角后直角 | curve |
| 4209–4344 | 后续直角 | curve；R7 合并为 3997–4344 |

### lap_001

| 范围 | 物理含义 | R7 角色 |
|---|---|---|
| 1666–1842 | 直角 | curve |
| 2164–2307 | 直角 | curve |

脚本中的角色覆盖顺序很重要：先标 curve，再标 cross，因此十字路口与弯道重叠的帧最终按 cross 处理。

## 6. 当前基底 D4

真正的 D4：

```text
artifacts/lane_hard_turn_d4_full/best.pdparams
SHA256: 3398fc6b2d2be73d1ec7eed6f54f51f0be6e2b6c5814f255485d23442167cb49
```

注意：历史对话中曾把“官方模型经 lap_003 训练 5 轮”的一组结果误称为 D4。凡是没有上述 SHA256 的 checkpoint，都不能当作真正 D4。

D4 的实际特点：

- 实车急弯转向较强，但在新现场存在稳定性问题；
- 人工物理段上有的弯幅值过强，有的入口起转偏晚；
- 不能根据自动 yaw 阈值段推断“急弯完全没识别”的概率；
- 当前 R7 从真正 D4 初始化，不再从官方原始模型初始化。

## 7. R7 长训方法

训练入口：

```text
tools/lane_training/d4_physical_segments_long_finetune.py
```

输出目录：

```text
/root/autodl-tmp/lane-cnn/project/artifacts/d4_physical_long_r7
```

核心配置：

| 参数 | 值 |
|---|---:|
| epochs | 100 |
| batch size | 64 |
| 回归头基础学习率 | `1e-5` |
| 最后四个卷积层基础学习率 | `3e-7, 6e-7, 1.5e-6, 3e-6` |
| optimizer | Adam |
| seed | 20260812 |
| 光照扰动 severity | 3，概率 1.0 |
| 解冻范围 | 最后四个卷积层和回归头 |
| vy | D4 教师保持，不向现场标签学习 |
| saturation loss | 关闭 |

每个 epoch 约 66 个 batch：curve 33、cross 13、preserve 20。curve 以连续物理段组成 batch；cross 是两个直行十字路口；preserve 从 lap_001/lap_003 的其他帧抽样。

### 分阶段教师混合与学习率

曲线目标为：

```text
target_yaw = label_fraction * 现场驾驶标签
           + (1 - label_fraction) * D4教师输出
```

| epoch | 现场标签占比 | 学习率倍率 |
|---|---:|---:|
| 1–20 | 60% | 1.0 |
| 21–60 | 75% | 0.3 |
| 61–100 | 85% | 0.1 |

其他角色：

- cross：目标 yaw 强制为 0，确保两个十字路口直行；
- preserve：目标 yaw 为 D4 教师输出，抑制非目标区域漂移；
- 所有角色的目标 vy 都是 D4 教师输出。

### 损失

`yaw_curve_loss` 权重：

```text
point=1.0
slope=0.5
curvature=0.2
integral=1.0
phase=0.5
saturation=0.0
```

总损失：

```text
total = role_weight * yaw_curve_loss
      + 4.0 * direction_loss
      + 3.0 * light_consistency_loss
      + vy_teacher_loss
```

其中 role weight：curve=2、cross=3、preserve=2。方向损失只作用于曲线目标中 `abs(target_yaw) >= 0.05` 的帧；光照一致性要求同一现场图像及其扰动版本给出接近的 yaw。

设计意图：

- 不只匹配单帧峰值，而是同时匹配 yaw 点值、斜率、曲率、整体冲量和相位；
- 弯道逐步向现场驾驶标签靠近；
- 用 D4 教师保存非目标区域和 vy；
- 用真实现场图像加合成光照扰动，提高同一场景在亮度变化下的输出稳定性；
- 关闭 saturation loss，避免训练机制系统性压低强转弯。

## 8. R7 评测方法与结果

评测脚本：

```text
artifacts/evaluate_r7_checkpoints.py
```

它评测 D4 和每 5 个 epoch 的 checkpoint，不运行 locked test。指标包括：

- 人工曲线段 yaw MAE；
- yaw 绝对冲量与现场标签的误差；
- 有效转弯帧方向一致性；
- `abs(yaw) >= 0.05` 的起转帧偏移；
- 同一图像加入光照扰动前后的 yaw 差；
- `abs(yaw) >= 0.5` 的 PID 饱和率；
- 两个直行十字路口的平均绝对 yaw 和误转率。

汇总结果：

| 模型 | 曲线 MAE ↓ | 冲量误差 ↓ | 方向一致性 ↑ | 起转绝对偏移 ↓ | 光照 yaw 差 ↓ | 饱和率 ↓ | 十字误转率 ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|
| D4 | 0.1454 | 0.3709 | 0.9142 | 16.63 | 0.0366 | 0.2261 | 0.0266 |
| epoch 5 | 0.1359 | 0.2981 | 0.9223 | 17.75 | 0.0330 | 0.2063 | 0.0169 |
| epoch 20 | 0.1220 | 0.2103 | 0.9284 | 17.50 | 0.0288 | 0.1356 | 0.0145 |
| epoch 100 | 0.1119 | 0.1727 | 0.9318 | 19.75 | 0.0264 | 0.0412 | 0.0145 |

结论：训练轮数增加时，大部分标签拟合和光照稳定性指标持续改善，但并非所有行为都单调改善。后期会进一步削弱 D4 的强转向，并使部分入口时序恶化；例如 `lap_003:1260–1390` 的起转偏移后期约从 `+12` 帧恶化到 `+30` 帧。因此没有选择 epoch 100。

当前两个有价值的折中点：

- epoch 5：更接近 D4，弯道总体转向强度约比 D4 低 7%，适合优先保留 D4 强转向；
- epoch 20：更接近现场标签，稳定性和整体曲线误差更好，但比 epoch 5 更温和。

原始报告：

- `artifacts/d4_physical_long_r7_training_report.json`
- `artifacts/d4_physical_long_r7_evaluation_every_5.json`

## 9. 已导出的车端包

### epoch 5：当前偏强转向版本

```text
checkpoint:
artifacts/d4_physical_long_r7_epoch_5.pdparams
SHA256: 535b72ac6e6abd96a3d818bbd1e41b6c9bab0c3feceab1df8679673ec1db9c02

车端包:
artifacts/d4_physical_long_r7_car_export_epoch5/lane_model_risk_accepted.tgz
SHA256: a444392e856f481b6d1071ba2a98c03d4dfe79864942d959d1fedb9873f6ca5e

cnn_lane.pdiparams SHA256:
4ac3823268e2388f8fbd2ae1e07157f820c0dbc66516b8d0859f3713e22dd8c5
```

导出一致性：JIT batch 1=`0`，JIT batch 32=`6.98e-10`，CPU=`2.61e-7`，GPU=`0`。

### epoch 20：当前偏稳定版本

```text
checkpoint SHA256:
08b8c41d996712227c6e0c27731e89de8740cfe80ca5cc71df4fe53fbc16cbd2

车端包:
artifacts/d4_physical_long_r7_car_export/lane_model_risk_accepted.tgz
SHA256: 1b301ac3dcffb68de982c72ab6eab7a7d644aa14bcc17b7c349c90de489c02b9
```

导出一致性：JIT batch 1=`0`，JIT batch 32=`9.31e-10`，CPU=`3.18e-7`，GPU=`0`。

两个包的 `deployment.json` 均明确记录：

```text
user_risk_acceptance=true
behavior_gates_passed=false
selected=false
locked_test_performed=false
```

## 10. 后续会话必须避免的旧结论

- 不再使用旧的绝对 A/B 指标作为当前新赛道训练门禁；用户已明确丢弃该指标。
- 不再把 `abs(yaw) >= 0.05` 自动片段称为真实急弯；物理路段以第 5 节人工范围为准。
- 不再把 CNN 第二维当作角速度，也不做 `yaw/speed` 曲率换算。
- 不把“更多 epoch”理解成必然更好；它会更靠近 lap 标签，也可能降低 D4 转向强度或推迟入口。
- 不把 lap 驾驶标签当作与官方标签同一控制策略下的绝对真值。
- 不把离线单帧峰值当作完整转向质量；至少同时查看冲量、方向、时序和整段趋势。
- 不把导出一致性测试等同于实车行为验证。

## 11. lap_001 基线验证（2026-08-16）

为评估当前模型在另一种现场光照条件下的表现，使用服务器上的 D4 基线模型对 `lap_001` 进行了全量推理。该测试未加入任何 OpenCV 光照预处理，因此结果可作为后续预处理强度对照的基线。

### 测试环境与方法

- 服务器：`connect.bjb1.seetacloud.com:40932`
- 远端 Python：`/root/autodl-tmp/envs/lane-d5/bin/python`
- 远端项目：`/root/autodl-tmp/lane-cnn/project`
- checkpoint：`artifacts/lane_hard_turn_d4_full/best.pdparams`
- 数据：`lane_sessions2/lap_001/data.json` 及其图像，共 3433 帧
- 输入：双线性缩放到 `128×128`，RGB 转 CHW，归一化为 `pixel / 127.5 - 1.0`
- Paddle 运行前设置：`FLAGS_enable_pir_api=0`
- 推理设备：GPU
- 评估方式：加载 `data.json` 中的 `state[1]`（vy）和 `state[2]`（yaw），逐帧推理后计算回归误差、方向一致率、误转率和时序抖动

### 结果

| 指标 | lap_001 基线 |
|---|---:|
| `vy MAE` | 0.000178 |
| `yaw MAE` | 0.088979 |
| `yaw error P95` | 0.399365 |
| 转向方向一致率 | 0.913306（91.33%） |
| 误转率 | 0.107055（10.71%） |
| `yaw jitter P95` | 0.098375 |
| 综合分数 | 0.202904 |

### 结论与后续对照

当前模型在 `lap_001` 上能够大体保持正确的转向方向，但角度误差和误转率仍有优化空间。后续测试应保持模型、数据划分和指标不变，仅改变输入图像处理方式，至少比较：

1. 原图基线；
2. 轻度亮度/对比度归一化或 CLAHE；
3. 中度光照处理。

所有候选预处理必须同时应用于训练/微调输入和推理输入，避免产生输入分布不一致。优先以 `yaw MAE`、误转率、方向一致率和 `yaw jitter P95` 的跨光照表现选择方案。

## 12. 光照预处理筛选与转向时序复测（2026-08-16）

在未微调的 `lane_hard_turn_d4_full/best.pdparams` 上，使用 `lap_001` 做了输入端光照预处理筛选。为模拟车端流程，处理均在原始分辨率图像上执行，再缩放到 `128×128`、转 RGB/CHW 并按 `pixel / 127.5 - 1.0` 归一化。该实验是推理筛选，不等同于加入预处理后的模型微调结果。

### 全圈结果

| 处理方式 | `yaw MAE` | 误转率 | 方向一致率 | `yaw jitter P95` | 综合分数 |
|---|---:|---:|---:|---:|---:|
| 原图 | 0.08898 | 10.71% | 91.33% | 0.09837 | 0.20290 |
| 轻度 CLAHE（clipLimit=1.5） | 0.08482 | 7.83% | 90.73% | 0.11611 | 0.16969 |
| 强 CLAHE（clipLimit=3.0） | 0.08625 | 7.14% | 89.92% | 0.13167 | 0.16426 |
| 轻度 Gamma（γ=0.90） | 0.09009 | 10.95% | 91.53% | 0.09883 | 0.20658 |
| 强 Gamma（γ=0.75） | 0.09215 | 11.44% | 91.73% | 0.09391 | 0.21373 |

CLAHE 能降低整体角度误差和误转率，但会增加输出抖动；Gamma 校正没有带来收益。强 CLAHE 的综合分数最低，但方向一致率和时序抖动退化，因此不能仅按综合分数直接选用。

### 人工确认直角的时序结果

以下窗口严格使用第 5 节人工确认的物理区间，窗口内的 `onset_delta` 仅用于测量起转时间偏移，正值表示预测晚于标签；`peak_delta` 为预测峰值帧减去标签峰值帧。

| 处理方式 | 1666–1842 起转偏移 | 1666–1842 峰值偏移 | 2164–2307 起转偏移 | 2164–2307 峰值偏移 |
|---|---:|---:|---:|---:|
| 原图 | +13 帧 | −7 帧 | +14 帧 | −18 帧 |
| 轻度 CLAHE | +15 帧 | −7 帧 | +13 帧 | −18 帧 |
| 强 CLAHE | +6 帧 | −7 帧 | +13 帧 | −18 帧 |
| 轻度 Gamma | +14 帧 | −7 帧 | +13 帧 | −18 帧 |
| 强 Gamma | +16 帧 | −32 帧 | +14 帧 | −18 帧 |

当前主要时序风险仍然是第二个直角（`2164–2307`）：所有预处理的起转仍晚约 13–14 帧，峰值也提前约 18 帧。说明该问题不能归因于简单亮度偏差，后续微调必须对该物理窗口加入时序/曲线约束，不能只优化全圈平均光照鲁棒性。第一个直角的强 CLAHE 结果（起转 +6 帧）值得保留为候选，但必须同时验证抖动和方向一致率。

### 其他方法复测

在相同模型和相同 `lap_001` 上进一步测试了以下方法：LAB 亮度百分位归一化、HSV V 通道时间平滑、灰世界白平衡、低频照明场校正和弱 Retinex。

| 处理方式 | `yaw MAE` | 误转率 | 方向一致率 | `yaw jitter P95` | 综合分数 |
|---|---:|---:|---:|---:|---:|
| HSV V 时间平滑（轻度） | 0.08835 | 10.50% | 91.53% | 0.10451 | 0.20016 |
| HSV V 时间平滑（强度较高） | 0.08934 | 10.87% | 91.33% | 0.10392 | 0.20490 |
| 灰世界白平衡 | 0.08976 | 10.99% | 91.73% | 0.10547 | 0.20661 |
| 低频照明场校正（σ=31） | 0.08615 | 10.13% | 93.15% | 0.09981 | 0.19414 |
| 低频照明场校正（σ=61） | 0.08599 | 9.64% | 92.54% | 0.09912 | 0.18904 |
| LAB 百分位归一化（2%–98%） | 0.10643 | 10.01% | 90.32% | 0.13459 | 0.21516 |
| LAB 百分位归一化（5%–95%） | 0.11247 | 11.28% | 90.52% | 0.15379 | 0.23456 |
| 弱 Retinex（σ=31） | 0.09767 | 8.86% | 90.93% | 0.12166 | 0.19404 |

低频照明场校正是本轮中最平衡的新增候选：比原图降低了角度误差，同时方向一致率和抖动没有像强 CLAHE 那样明显退化。但它在两个直角上的起转偏移仍为 `+13/+14` 帧，峰值偏移仍为 `−7/−18` 帧，因此不能解决已确认的晚起转问题。LAB 百分位拉伸不建议继续使用；Retinex 虽降低误转率，但误差和抖动变差。后续若进入微调，优先考虑“低频照明场校正（σ=61）”和“轻度 CLAHE”两条分支，并分别加入第二个直角的时序约束。

### 色相与颜色增益复测

训练集与 `lap_001` 的颜色统计存在一定差异：`lap_001` 的 RGB 均值约为 `[145.8, 151.9, 149.9]`，训练集约为 `[154.9, 153.6, 148.1]`；`lap_001` 的平均饱和度也更高（约 30.6 对 19.4）。在原始分辨率上测试了轻度 RGB 颜色增益、训练集匹配增益、饱和度调整和小范围 Hue 偏移。

| 处理方式 | `yaw MAE` | 误转率 | 方向一致率 | `yaw jitter P95` | 综合分数 |
|---|---:|---:|---:|---:|---:|
| 原图 | 0.08898 | 10.71% | 91.33% | 0.09837 | 0.20290 |
| 轻度 RGB 增益 `[1.03,1.01,0.995]` | 0.08890 | 10.87% | 91.73% | 0.10119 | 0.20446 |
| 训练集匹配增益 `[1.062,1.011,0.988]` | 0.08929 | 10.71% | 92.14% | 0.10352 | 0.20323 |
| 饱和度 ×0.80 | 0.09047 | 11.89% | 93.35% | 0.10253 | 0.21642 |
| Hue −8（约 −16°） | 0.09386 | 11.24% | 87.70% | 0.11685 | 0.21347 |
| Hue +8（约 +16°） | 0.08655 | 10.75% | 94.15% | 0.09671 | 0.20074 |
| Hue −8 + 饱和度 ×0.80 | 0.09448 | 11.57% | 90.32% | 0.11336 | 0.21743 |

Hue `+8` 是本轮全圈指标最好的颜色处理，但它在第一个直角的峰值提前约 33 帧，不能直接用于车端。颜色增益基本没有改善，负方向 Hue 和降饱和度会使角度误差变差。当前结论是：`lap_001` 的色相差异确实会影响模型输出，但应在后训练中使用小范围 Hue 扰动/颜色校正增强，而不是固定地把所有比赛图像旋转 Hue `+8`。后训练时应同时保留原图分支，并对两个人工直角增加峰值位置和起转时序检查。

### 固定标准化管线组合复测

为减少部署时的分支和参数选择，进一步测试了固定颜色增益 `[1.062, 1.011, 0.988]` 与亮度校正的组合。所有处理仍在原始分辨率执行，再缩放到模型输入尺寸。

| 固定管线 | `yaw MAE` | 误转率 | 方向一致率 | `yaw jitter P95` | 综合分数 |
|---|---:|---:|---:|---:|---:|
| 原图 | 0.08898 | 10.71% | 91.33% | 0.09837 | 0.20290 |
| 固定颜色增益 | 0.08929 | 10.71% | 92.14% | 0.10352 | 0.20323 |
| 颜色增益 + 照明场校正（σ=31） | 0.08596 | 10.13% | 93.35% | 0.10003 | 0.19390 |
| 颜色增益 + 照明场校正（σ=61） | 0.08585 | 9.72% | 92.74% | 0.10248 | 0.18968 |
| 颜色增益 + 轻度 CLAHE | 0.08297 | 7.63% | 91.13% | 0.11825 | 0.16564 |

固定颜色增益本身收益很小；与照明场校正组合后能稳定改善误差，σ=61 比 σ=31 更好。颜色增益 + 轻度 CLAHE 的全圈误差最低，但抖动增加、方向一致率下降，且第一个直角起转偏移为 `+15` 帧。因此后训练候选优先级为：先测试“颜色增益 + 照明场校正（σ=61）”，再测试“颜色增益 + 轻度 CLAHE”；两者都必须通过两个直角的时序检查后才能进入车端。

## 13. 光照后训练初测（2026-08-16）

从 `lane_hard_turn_d4_full/best.pdparams` 初始化，进行了一轮受控的光照后训练实验。训练只使用官方 `image_set_lane` 的 4049 帧，`lap_001` 未参与训练，仅用于跨光照验证，避免直接拟合现场验证标签。

### 训练设置

- 每个样本同时生成原图和固定标准化图：RGB 增益 `[1.062, 1.011, 0.988]` + 低频照明场校正 `σ=61`
- 输入仍为 `128×128`、RGB/CHW、`pixel / 127.5 - 1.0`
- 从 D4 checkpoint 初始化
- 仅解冻最后两层卷积和回归头
- 卷积层学习率 `2e-6`，回归头学习率 `1e-5`
- batch size `64`，训练 `8` epochs
- 损失包含原图/处理图监督、两种输入的一致性约束和 D4 教师保持项
- 输出目录：`/root/autodl-tmp/lane-cnn/project/artifacts/light_posttrain_illum61_test`

### lap_001 结果

| 模型与输入 | `yaw MAE` | 误转率 | 方向一致率 | `yaw jitter P95` | 综合分数 |
|---|---:|---:|---:|---:|---:|
| D4 + 原图 | 0.08898 | 10.71% | 91.33% | 0.09837 | 0.20290 |
| D4 + 标准化图 | 0.08585 | 9.72% | 92.74% | 0.10248 | 0.18968 |
| 后训练 epoch 8 + 原图 | 0.08207 | 8.74% | 91.73% | 0.09153 | 0.17554 |
| 后训练 epoch 8 + 标准化图 | 0.07949 | 8.00% | 93.35% | 0.09089 | 0.16539 |

后训练同时改善了原图和标准化图输入，说明模型确实学到了更稳的光照特征，而不是只记住某一种处理伪影。但两个人工直角的时序没有变化：`1666–1842` 起转仍为 `+13` 帧，`2164–2307` 仍为 `+14` 帧；峰值偏移仍约为 `−7/−18` 帧。因此该实验不能作为解决晚起转的最终模型。

下一步如果继续训练，应在保持光照一致性损失的同时，对两个人工物理窗口加入起转位置、yaw 斜率和峰值相位约束，并继续同时报告原图与标准化图结果。

## 14. 新增实验：只用 lap_001 从零训练

实验入口：

```text
tools/lane_training/scratch_lap1_train.py
```

评测入口：

```text
artifacts/evaluate_scratch_lap2.py
```

目的：验证固定赛道条件下，保持官方 CNN 架构但随机初始化、只用 `lap_001` 监督，是否比 D4/R7 更适合 `lap_002`。

配置：随机初始化 `CnnModel`；只使用约 4505 帧的 `lap_001`，`lap_002` 约 3134 帧严格只作评测；100 epochs、batch 64、Adam、学习率 `1e-4`；yaw 额外权重 2；Smooth L1 输出损失；severity=3 光照扰动一致性；vy 也直接学习 lap_001 标签，因此不具备 D4/R7 的教师 vy 保持约束。

本地结果：

```text
artifacts/scratch_lap1_r1/evaluation_every_5.json
artifacts/scratch_lap1_r1/training_report.json
artifacts/scratch_lap1_r1/epoch_45.pdparams
SHA256: 2756e7a02712a837e47bab805911019b598ea97f745edcd6f6fecc2dd1f9af03
```

`lap_002` 汇总：

| 模型 | yaw MAE ↓ | 曲线归一化误差 ↓ | 方向一致性 ↑ | 直道误转率 ↓ | PID 饱和率 ↓ | vy MAE ↓ |
|---|---:|---:|---:|---:|---:|---:|
| 官方 | 0.0849 | 0.6665 | 0.9057 | 0.1432 | 0.0740 | 0.0006 |
| D4 | 0.0898 | 0.7537 | 0.9113 | 0.1205 | 0.0996 | 0.0002 |
| R7 epoch 5 | 0.0838 | 0.7075 | 0.9000 | 0.1095 | 0.0865 | 0.0002 |
| R7 epoch 20 | 0.0766 | 0.6328 | 0.8972 | 0.0993 | 0.0558 | 0.0002 |
| scratch epoch 45 | **0.0663** | **0.4798** | 0.9736 | 0.1393 | **0.0258** | 0.0081 |

结论：scratch epoch 45 的 lap_002 yaw 拟合明显优于 D4/R7，但直道误转仍高于 R7 epoch 20，vy 误差显著更大。epoch 间波动也很明显，例如 epoch 20 方向一致性高但直道误转率达到 0.2782，epoch 100 的 yaw MAE 回升到 0.0824。因此不能按训练轮数或训练集 loss 直接选车端版本；scratch epoch 45 目前只是离线研究 checkpoint，未导出车端包。

用户已取消“官方初始化、同配置 lap_001 对照组”，该实验已停止，不纳入比较。

## 15. 推荐的接续读取顺序

新会话按以下顺序恢复上下文：

1. 阅读本文件。
2. 阅读 `tools/lane_training/d4_physical_segments_long_finetune.py`，确认训练代码未变化。
3. 阅读 `artifacts/d4_physical_long_r7_evaluation_every_5.json` 中拟比较的具体 epoch。
4. 检查目标 checkpoint 和车端包 SHA256。
5. 在提出新训练前，明确是要“保留 D4 强度”还是“进一步靠近现场标签”，并逐段检查人工物理路段，不要先按全局平均值下结论。

## 16. CV 教师标签与 CNN 底盘后处理选择（2026-08-16）

> **历史决策记录：**第 16、17 节记录从 `[vy, yaw_error]` 过渡到动作曲率接口之前的分析，便于追溯为什么放弃旧方案。这里出现的 `state=[forward_speed,error_y,error_angle]`、按帧梯度和“尚未修改代码”等描述已经失效；当前实现以第 18、20 节和代码为准。

新的 OpenCV Pure Pursuit 逻辑已经由用户在真实赛道测试，整体行驶没有明显问题。后续目标是
让 CNN 学习该教师，同时保留实车阶段调整转向幅值、弯道速度和起转/回正时序的能力。

### 16.1 CV session 同时保存两层数据

CV 采集的每一帧同时记录：

```text
state   = [forward_speed, error_y, error_angle]
control = [forward_speed, actual_vy, actual_wz]
```

两者不能混为同一种监督目标：

- `state[1:3]` 是 CV 底盘控制器之前的单帧教师意图；
- `control` 是经过曲率降速、速度梯度、角速度梯度、锐角覆盖和断线保持后，实际发送到底盘的命令；
- 当前训练加载器只读取 `state[1]` 和 `state[2]`，不会自动读取 `control`。

对应的两条合法链路是：

```text
意图克隆：图像 -> CNN state -> CV 底盘控制器 -> vx/vy/wz
动作克隆：图像 -> CNN control -> 仅安全限幅 -> vx/vy/wz
```

不能先训练最终 `control`，再完整执行一次 CV 的降速和角速度梯度，否则会产生双重后处理。

### 16.2 直接输出 vx/wz 的优缺点

在当前 `vy=0` 的前提下，可以把两输出模型改成 `[vx, wz]`，直接监督
`control[0]`、`control[2]`。它能在离线数据上直接拟合右锐角覆盖、弯道降速和断线保持后的
实际命令，但不保证闭环实车性能更好。

原因是最终 `control` 不只取决于当前图像，还包含：

```text
上一帧速度和角速度
速度加减速限制
入弯、回正和反向释放限制
断线继承
启动里程保护
```

相似图像可能因历史状态不同而对应不同的 `vx/wz`。固定赛道能减轻该问题，但车辆一旦偏离
训练轨迹，直接动作模型缺少明确几何误差和独立控制器帮助恢复。其输出已经包含教师后处理，
继续调整死区、限速或滤波会同时改变起转、峰值、冲量和回正位置，调试含义不清晰。

### 16.3 当前推荐：保留 vy/yaw_error 并复用 CV 控制器

当前推荐继续保持官方两输出接口：

```text
CNN output[0] = error_y
CNN output[1] = error_angle
```

训练监督继续读取 CV session 的 `state[1]`、`state[2]`。车端不再使用旧的固定速度加
`Kp=3` 简单 PID，而是将 CNN 输出接入与 CV 模式共享的底盘控制层：

```text
CNN error_angle
  -> 还原 CV 转向需求/曲率
  -> CV 曲率降速（0.30 -> 0.12 m/s）
  -> 速度减速/恢复梯度
  -> actual_speed * curvature
  -> CV 入弯/回正/反向角速度梯度
  -> set_velocity(vx, 0, wz)
```

CV 当前保存标签的关系为：

```text
error_angle = -(target_forward_speed * curvature) / 1.95
```

因此车端需要使用与标签生成一致的映射反解转向需求，不能直接沿用旧车端
`Kp=3`、最低速度 `0.18 m/s` 的处理。实现时应复用 CV 的控制类或抽取共享控制核心，不应在
CNN 入口复制一套容易漂移的参数和公式。

### 16.4 为什么保留后处理更适合实车调试

保留 PID 前语义后，可以分别调整：

- 小误差死区或起转门限：抑制过早的小幅转向；
- 转向映射或增益：调整弯道半径和峰值；
- 入弯角速度梯度：调整转向建立速度；
- 同方向回正梯度：调整出弯释放；
- 反向释放梯度：调整连续反向弯；
- 曲率降速、最低速度和速度恢复梯度：调整弯中速度；
- 训练标签相位和连续曲线损失：修正模型本身的提前或滞后。

后处理只能调整已经出现的模型信号。如果 CNN 在弯道到达前完全没有输出转向趋势，降低门限
或提高增益也不能凭空生成正确方向，必须通过远场特征、标签相位或起转段损失修正训练。反之，
若模型很早出现小幅正确方向，可以通过死区、门限和非线性映射推迟底盘实际执行。

### 16.5 当前决定和后续验证口径

当前不优先采用 `[vx, wz]` 直接动作模型。优先路线为：

```text
训练：CV state[1:3] -> CNN [error_y, error_angle]
推理：CNN [error_y, error_angle] -> 共享 CV 底盘控制器 -> [vx, 0, wz]
```

后续验证不能只比较 `yaw MAE`，必须将 CNN 预测送入共享底盘控制器，再与 CV session 的
`control[0]`、`control[2]` 比较：

- 直道和弯道速度趋势；
- 转向方向与峰值；
- 起转和回正位置，优先按编码器距离评价；
- 角速度冲量和整段曲线；
- 右锐角和断线保持窗口；
- 完整 session 的闭环低速实车结果。

本节只记录架构分析和当前推荐，尚未修改训练加载器、模型输出、CNN 车端入口或 CV 控制器，
也未运行新的训练、离线评测或实车验证。

## 17. 推荐方案的边界与此前忽视的问题（2026-08-16）

> **历史风险分析：**本节中的单帧歧义、闭环分布偏移、推理延迟和教师版本混训风险仍然成立，但旧的 `state[1:3]` 输出建议、按帧梯度描述和启动保护标签关系已被当前动作曲率数据格式替代。

本节补充采用“CNN 预测 `state[1:3]`、再复用 CV 底盘控制器”时必须单独处理的问题。两输出
标签能够表达主要视觉转向趋势，但不自动包含 CV 控制器的全部离散状态、历史状态和运行时序。

### 17.1 右锐角 override 没有完全编码在两维标签中

CV 右锐角命中时，`route_right_turn_override=True`，当前帧会绕过普通角速度建立限制，
直接使用几何目标。`state[2]` 只保存稳态 `error_angle`，没有保存“本帧需要绕过梯度”的标志。

因此即使 CNN 完美预测 `error_angle`，共享控制器也可能在右锐角入口建立得比 CV 慢。第一版
可以根据解码后的大负曲率使用连续的快速建立规则；如果仍有明显差异，应增加辅助输出，例如：

```text
[error_y, error_angle, override_probability]
```

若必须保持两输出，则只能接受右锐角动态的近似，不能声称所有 CV 分支都被精确编码。

### 17.2 断线保持具有历史依赖

断线时 `state` 和 `control` 会继承上一条有效命令。此时标签取决于上一帧，而不只由当前
图像决定。固定赛道和固定方向能降低歧义，但若左右弯都出现相似横线/无效画面，单帧 CNN
可能输出平均方向。

训练时应至少：

- 单独标记 `held=true`；
- 降低长时间重复保持帧的监督权重；
- 统计相似无效画面是否存在相反标签；
- 必要时增加有效性辅助输出或连续帧输入。

不能因为记录中存在保持命令，就推断该命令一定是当前单帧可辨识的。

### 17.3 启动里程保护不在普通 `state` 语义中

启动前 `0.15 m` 的控制层保护会把实际 `control[2]` 置零，但保存的 `state[2]` 仍可能包含
弯道意图。因此 CNN 推理端必须继续保留编码器启动保护，否则模型会在发车后提前转向。

该保护只能按里程执行，不能改成固定帧数或复制成普通弯道延迟。

### 17.4 当前梯度参数是按帧而不是按真实时间

CV 控制循环约为 20 Hz，但 `entry_step`、`release_step` 和 `reverse_step` 当前以“每次调用”
限制角速度。若 CNN 推理运行在不同频率，相同参数会产生不同的每秒建立/释放速度：

```text
10 Hz、20 Hz、30 Hz 对应完全不同的实际角速度变化率
```

共享控制器必须固定循环频率，或将步进改为基于实际 `dt` 的速率限制。否则即使 CNN 预测完全
相同，轨迹也会因推理频率不同而改变。

### 17.5 CNN 推理延迟会改变空间起转位置

OpenCV 和 CNN 的图像采集、推理及下发延迟不同。延迟 `0.10 s` 在 `0.30 m/s` 时就相当于
约 `0.03 m` 的空间偏移。起转和回正验收不能只看帧号，应同时记录：

```text
图像时间、推理完成时间、命令下发时间、当前速度、编码器距离
```

如果延迟稳定，可以在训练标签相位或运行管线中补偿；如果延迟波动，应先稳定循环周期。

### 17.6 不同 CV 参数版本的数据不能直接混训

近期 CV 曾改变预瞄距离、锐角增益、右锐角入口增益、Otsu 阈值、速度范围、启动距离和
角速度释放规则。不同版本可能让同一视觉几何对应不同 `state/control` 标签。

训练前必须按 `session.json` 筛选最终版本，并核对：

```text
teacher、lookahead、speed range、sharp gain、right-turn gain、Kp、梯度参数、阈值模式
```

否则 CNN 会学习不同教师策略的平均值。

### 17.7 旧模型与 CV 模型的数值尺度并不天然相同

旧 D4/R7 模型在旧车端 `Kp=3` 控制链下训练和评估；CV 标签按
`error_angle = -steady_wz / 1.95` 生成。虽然两者都是 `[vy, yaw_error]`，数值控制尺度并不
完全相同。

部署包必须标记控制配置，例如：

```json
{"output_semantics": "vy_yaw_error", "control_profile": "legacy_pid"}
```

或：

```json
{"output_semantics": "vy_yaw_error", "control_profile": "cv_pure_pursuit"}
```

不能让旧模型自动使用新的 CV 后处理并假设行为不变。

### 17.8 `error_y` 当前对实际轨迹没有作用

CV 当前将 `lateral_limit` 设为 0，实际 `vy` 固定为 0，但仍保存 `error_y`。继续高权重训练第一维
会占用模型容量，却不会改善当前轨迹。第一阶段应保持输出接口，但降低 `error_y` 损失权重；
等麦克纳姆横移方向和恢复增益完成实车确认后，再恢复横向控制监督。

### 17.9 闭环分布偏移仍然存在

CNN 只在 CV 已经跑正确的轨迹上学习。实车一旦因预测误差偏离几厘米，图像可能进入训练集
没有覆盖的姿态。固定赛道只能减轻，不能消除该问题。

后续需要多圈、不同初始姿态、不同光照和少量左右偏置恢复数据；必要时用 CNN 实车失败/恢复
片段继续微调。评测必须按完整 session 划分，不能随机打散相邻视频帧。

## 18. 最终统一方案：速度条件下的 CV/CNN 曲率训练

本节是第 16、17 节讨论后的当前实现。旧的 `[vy,yaw_error]` 方案只供历史
模型在 `legacy_error` profile 下兼容，不再作为新 CV 数据的训练接口。

### 18.1 统一数据格式与标签

CV session 每帧保存以下三层数据：

```text
state/control    = [actual_vx, actual_vy, actual_wz]
legacy_pid_state = [forward_speed, error_y, error_angle]
model_target     = {speed_demand, kappa_action}
target_mask      = [1.0, 1.0]
```

`state` 与 `control` 使用和手柄 session 相同的原始底盘命令格式。CV 的
`legacy_pid_state` 只用于诊断旧接口，不是新模型标签。新标签定义为：

```text
speed_demand = command.steering_demand
kappa_action = actual_wz / max(abs(actual_vx), 0.12)
```

`speed_demand` 不是绝对速度，范围为 `[0,1]`：`0` 表示直道最高速度需求，
`1` 表示最强弯道降速需求。`kappa_action` 在短时断线继承、右锐角处理、启动
直行保护和最终限幅之后重新按实际命令计算，因此监督的是当帧最终转向动作。
记录同时包含 `held`、`command_source`、`control_reason`、编码器距离、
`timestamp_monotonic` 和 `effective_dt_s`。

手柄 session 继续保存 `state=[vx,vy,wz]`，加载时派生：

```text
speed_demand = 0.0                   # 占位，不参与损失
kappa_action = wz / max(abs(vx), 0.12)
target_mask  = [0.0, 1.0]
```

因此手柄数据可以补充转向和偏移恢复，但不能用其油门直接监督 CV 速度曲线。
水平翻转时 `speed_demand` 不变，`kappa_action` 反号。

### 18.2 模型与控制流程

```text
320×240 原图（训练/推理入口再缩放为 128×128）
    ↓
CNN 输出 [speed_demand, kappa_action]
    ↓
speed_curve(speed_demand) 计算 target_vx
    ↓
按真实 dt 执行速度梯度得到 actual_vx
    ↓
wz = clip(actual_vx × kappa_action, ±1.50)
    ↓
安全限幅与 watchdog
    ↓
输出 vx, 0, wz
```

CV 教师端仍可使用 Pure Pursuit、真实时间角速度建立/释放、断线保持和启动保护，
但这些行为已经编码进最终 `kappa_action`。CNN 后处理只移植速度曲线与真实时间
速度梯度，不再次执行 CV 的入弯/回正角速度梯度或路线 override，避免双重处理。

训练直接对两个输出计算带 mask 的稳健回归损失，概念形式为：

```text
loss_speed = mask[0] × SmoothL1(speed_pred, speed_label)
loss_kappa = mask[1] × SmoothL1(kappa_pred, kappa_label)
loss       = normalized(loss_speed + lambda_kappa × loss_kappa)
```

mask 的归一化必须避免手柄样本的 `speed_demand=0` 占位值参与速度损失。本方案
当前固定 `vy=0`；偏移恢复先由 `kappa_action` 学习，不增加独立横移网络。

### 18.3 真实时间速度梯度

CV 教师和 CNN 控制器都使用单调时钟计算真实 `dt`。当前速度参数为：

```text
减速率 = 0.194 m/s²
加速率 = 0.129 m/s²
首帧或无效 dt 回退 = 0.05 s
vx_next = move_towards(vx_previous, target_vx, rate_per_second × effective_dt_s)
```

`finite_dt()` 会把缺失、非数值、非有限或小于等于零的输入回退为 `0.05 s`，并把
最小值限制为 `1e-4 s`。当前实现**没有正数 `dt` 上限**：如果控制循环长时间卡顿，
恢复后的第一步可能允许较大的速度或角速度变化。部署前应增加 `dt_max` 或在 watchdog
恢复时重置控制器；在该问题修复前，不得把“丢帧时已限制 dt”写入验收结论。

### 18.4 旧图片离线重标注

旧手柄或旧采集图片可以按完整序列重新交给 CV 教师处理：

```text
旧图片 → 按原顺序回放 CV → 生成 control/model_target/target_mask → 训练 CNN
```

每圈开始时必须重置控制器状态，不能逐张独立处理；应使用原始单调时间戳，
没有时间戳时使用固定标称周期并写入 `session.json`。重标注结果必须保存到
独立的 `cv_relabelled_*` session，不得覆盖原始手柄数据。

重标注必须固定 ROI、透视、曲率、速度范围、速度梯度、有效 `dt`、锐角规则、
断线保持、启动处理和代码版本。线段质量差、检测失败或动作明显不适合当前
图像的样本应剔除或单独归入恢复数据集。离线重标注可以扩充图片和统一标签，
但不能替代新控制策略下的真实速度、底盘姿态和执行延迟采集。

当前 GitHub 分支尚未包含“一条命令完成整圈序列重标注”的正式工具；不能逐张调用
单帧分析器冒充序列重标注。实现该工具时必须复用 session 级控制器状态和时间顺序。

### 18.5 当前训练工具完成度

已经提交并经过测试的部分：

- `lane_training/manifest.py`：加载 CV 动作 session，并把手柄 `[vx,vy,wz]` 转为曲率标签；
- `lane_training/dataset.py`：返回 `[speed_demand,kappa_action]`，支持 `target_mask` 和水平翻转；
- `tests/lane_training/`：验证 CV/手柄 mask、曲率换算和数据增强；
- `car_wrap_2026.py`：支持 `legacy_error` 与 `kappa_action` 两种车端 profile。

2026-08-17 已在当前工作树补齐并同步到远端训练环境、完成 GPU smoke 验证的部分：

- `tools/lane_training/train_action.py` 与 `lane_training/action_training.py`：正式双输出训练入口，强制 `return_target_mask=True`；
- `lane_training/action_loss.py`：对两个输出执行 mask 归一化 Smooth L1，手柄速度占位值不参与损失；
- `tools/lane_training/evaluate_action.py`：按有效 mask 统计速度需求和曲率误差；
- `tools/lane_training/export_action.py`：导出静态模型、`deployment.json` 和 `.tgz` 车端包，记录输出语义及 `kappa_action` profile；
- `tools/lane_training/prepare_action_manifest.py`：合并 CV 与手柄 session 并保留 split/mask。

远端 Paddle 3.3.1 默认生成 `cnn_lane.json` 与 `cnn_lane.pdiparams`；已用
`paddle.inference.Config` 验证该组合可加载。Orin 部署前仍需核对车端 Paddle 版本是否支持 PIR
JSON；如果只支持旧 `cnn_lane.pdmodel`，需要在兼容版本环境重新导出。以上新增文件当前尚未
提交或推送 GitHub，不能把远端临时同步副本当作最终版本库交付。

### 18.6 标准验收要求

验证必须按完整 session/lap 划分，至少比较：

1. `wz` 符号、幅值和最终底盘命令误差；
2. 曲率误差；
3. 入弯、回正位置和转向时序；
4. 右锐角、断线保持和启动窗口；
5. 速度趋势、实际 `dt` 和速度梯度；
6. 推理频率、相机到命令延迟；
7. 完整闭环轨迹回放和低速实车结果。

## 19. 旧方案最低验收要求（历史记录）

> 本节针对旧的“预测 yaw 后再送入共享 CV 控制器”方案，仅保留其有价值的时序验收维度。当前 `[speed_demand,kappa_action]` 模型应按第 18.6 和第 20 节验收。

新模型不能只报告 `yaw MAE`。必须将预测送入共享 CV 控制器后，比较：

```text
最终 vx/wz 与 CV control 的误差
直道/弯道速度趋势
起转和回正的编码器位置
角速度峰值、冲量和整段曲线
右锐角入口建立速度
断线保持窗口
推理周期和端到端延迟
完整 session 的低速闭环实车结果
```

上述清单仍可作为补充诊断，但其中“预测送入共享 CV 控制器”的链路不是当前部署实现。

## 20. CV 教师—CNN 训练—车端部署操作闭环（2026-08-17）

本节给出接续开发时的最短正确路径。CV 视觉细节、赛道窗口和参数演进见
[CV 自动采样巡线开发状态](cv-lane-development-status-2026-08-15.md)。

### 20.1 采集与 session 检查

1. 通过 `collect_data.py` 进入 CV 实车采集模式，保存 320×240 原图；训练和推理阶段再缩放为 128×128。
2. 每次启动都要创建新 session，并在开始时重置 PID、时间滤波、断线保持、速度状态和里程原点。
3. 检查 `session.json` 中至少存在 `model_output_fields`、`controller`、`speed_control` 和 `launch_guard`。
4. 检查 `data.json` 每帧都包含：

```json
{
  "state": [0.20, 0.0, -0.40],
  "control": [0.20, 0.0, -0.40],
  "legacy_pid_state": [0.20, 0.01, 0.12],
  "model_target": {
    "speed_demand": 0.65,
    "kappa_action": -2.0
  },
  "target_mask": [1.0, 1.0],
  "held": false,
  "command_source": "standard",
  "effective_dt_s": 0.05
}
```

示例数值只说明字段关系，不是赛道固定标签。必须验证 `state==control` 且
`kappa_action≈control[2]/max(abs(control[0]),0.12)`；启动保护和无效帧继承也按
最终实际命令重新计算曲率。

### 20.2 生成训练 manifest

- CV session 使用 `load_cv_action_session()`，要求 `label_semantics=raw_control_with_model_target`。
- 手柄 session 使用 `load_manual_action_session()`，从原始 `[vx,vy,wz]` 派生曲率。
- 不覆盖原始 `data.json`；合并结果保存为新的 manifest，并保留 session/lap 边界。
- 训练、验证和测试必须按完整 session 划分，禁止把同一圈的相邻帧随机拆到不同集合。
- 不同 CV 参数或代码版本先分组统计，确认标签策略一致后才能混训。

### 20.3 训练入口必须满足的契约

正式训练器接入前必须同时满足：

```text
LaneDataset(..., return_target_mask=True)
model output shape = [N, 2]
output semantics   = [speed_demand, kappa_action]
masked loss        = 分输出乘 target_mask 后按有效权重归一化
horizontal flip    = [speed_demand, -kappa_action]
```

只修改 manifest 而继续使用旧 `[vy,yaw_error]` 损失会静默训练出语义错误的模型。
手柄样本的第一维 mask 为零，训练器若忽略 mask，会把大量 `speed_demand=0` 占位值
错误学习成“所有手柄画面都应直道高速”。

### 20.4 导出与车端部署

导出包必须记录：

```json
{
  "output_semantics": ["speed_demand", "kappa_action"],
  "control_profile": "kappa_action",
  "input_size": [128, 128],
  "source_image_size": [320, 240]
}
```

部署新模型时把 `config_car.yml` 的 `lane_control.profile` 改为 `kappa_action`，并核对
最高/最低速度、曲率满量程、速度指数、加减速率和角速度限幅。旧 D4/R7 模型必须继续
使用 `legacy_error`，不能仅替换模型文件而沿用错误 profile。

### 20.5 验收清单

离线检查：

- 两个输出范围、符号和 mask 生效；
- CV 与 CNN 的速度需求、曲率、最终 `vx/wz` 趋势；
- 入弯、回正、连续反向弯、右锐角、断线保持和启动窗口；
- 相机时间、推理完成时间、命令时间、编码器位置和 `effective_dt_s`；
- 完整一圈闭环回放，不只比较随机帧 MAE。

实车检查：

- 先以低速 profile 跑完整一圈，再逐步恢复目标速度；
- 确认直道加速、入弯减速和出弯恢复符合真实时间，而不是帧率；
- 检查卡顿或 watchdog 恢复后的首个 `dt`，当前无 `dt_max` 时不得直接高速测试；
- 记录偏离赛道后的恢复样本，用于补充闭环分布，而不是只重复标准轨迹。

### 20.6 关键文件索引

- CV session 写入：`smartcar/whalesbot/tools/lane_collect/ssh_test.py`
- CV Pure Pursuit 与真实时间梯度：`smartcar/whalesbot/tools/lane_collect/pid_control.py`
- CNN 曲率速度后处理：`smartcar/whalesbot/tools/curvature_control.py`
- 车端 profile 入口：`car_wrap_2026.py`、`config_car.yml`
- CV/手柄 manifest：`lane_training/manifest.py`
- 标签、mask 和增强：`lane_training/dataset.py`
- 数据接口测试：`tests/lane_training/test_manifest.py`、`tests/lane_training/test_dataset.py`
- CV 控制与采集契约测试：`tests/test_cv_lane_pid_control.py`

### 20.7 第一版固定赛道长训方案（2026-08-17）

### 20.8 最简 CV+手柄联合训练方案（2026-08-17）

当手柄数据用于纠正 CV 自动采集在十字路口等局部路段的转向时，第一版采用分输出监督，
不把手柄采集时的低速命令误当成速度策略：

```text
CV 样本：      speed_demand + kappa_action，target_mask=[1,1]
手柄样本：     仅 kappa_action，target_mask=[0,1]
```

手柄 `data.json` 的 `state=[vx_command, vy_command, wz_command]` 是手柄映射后发送给底盘的
命令，不是底盘实测反馈。其曲率标签统一计算为：

```text
kappa_action = wz_command / max(abs(vx_command), 0.12)
```

不得再次乘手柄映射比例，也不得把手柄 `vx_command` 直接作为 `speed_demand` 标签。手柄
样本的 `speed_demand=0` 仅为占位值，必须由 mask 排除速度损失。

联合训练仍输出 `[speed_demand, kappa_action]`。手柄样本只通过曲率损失纠正十字路口转向，
而车端仍完整执行速度后处理：

```text
CNN speed_demand -> speed_curve -> target_vx
                 -> real-dt slew -> actual_vx
CNN kappa_action -> wz = actual_vx * kappa_action
```

因此，手柄数据不直接提供新的速度策略，但不会绕过速度计算；十字路口最终速度仍由模型
预测的 `speed_demand` 和车端真实时间速度梯度决定。

第一轮最简数据组合为：CV 训练/验证数据加上两组十字路口手柄数据作为训练样本。训练前应
从 D4 或现有 action checkpoint 初始化，保持网络输出和 `kappa_action` 车端 profile 不变。
验收至少检查普通 CV 验证集是否退化、十字路口误转是否减少、`wz` 符号/幅值是否正确，
以及十字路口前后的减速和出弯恢复是否仍正常。若需要让手柄图片直接监督速度，必须先对
同一批手柄图片生成可信的 `speed_demand` 标签，不能从其他 CV 图片按帧复制速度标签。

第一版只执行一条训练路线，不做随机初始化或其他对照实验；手柄数据仅按本节的曲率纠正规则加入。赛道完全固定，
允许模型针对当前场地过拟合；训练目标是复现当前 CV 教师在固定赛道上的完整速度和曲率
动作，而不是追求跨赛道泛化。

#### 数据角色与划分

用户计划提供至少三圈完整 CV session，以及单独采集的弯道专项 session。第一轮训练配置为：

```text
初始化              D4 checkpoint
训练数据            新采集 CV 数据
旧 D4/官方训练图片   不加入
D4 teacher loss     关闭
手柄数据            两组十字路口手柄数据，仅监督 kappa_action
训练 mask           CV 全部为 [1.0, 1.0]
模型输出            [speed_demand, kappa_action]
```

训练方案选择阶段按完整 session 划分：

```text
训练集：完整圈 1 + 完整圈 2 + 全部弯道专项 session
验证集：完整圈 3
```

### 20.9 手柄参考验证集与 CV 问题集划分（2026-08-17）

当前十字路口 CV 验证数据来自转向仍有问题的 CV 自动采集版本，不能单独作为“正确答案”。
训练完成后必须把验证拆成以下角色：

```text
cv_val                 当前 CV 验证集；检查原有 CV 速度/普通路线是否退化
manual_lap002          lane_sessions2/lane_sessions2/lap_002；手柄正确转向参考
manual_official        offical-line-test；官方手柄正确转向参考
cross_cv_problem       当前 CV 十字路口问题集；只用于诊断误转改善，不作为唯一金标准
```

`manual_lap002` 使用原始 `data.json`，其手柄 `vx=0.1` 仍只用于派生
`kappa_action=wz/max(abs(vx),0.12)`，验证时只统计曲率指标，不统计速度误差。
`manual_official` 当前实际目录名为 `offical-line-test`（单个连字符），目录中是
`image_set_lane (1).zip` 和 `image_set_lane_eval (1).zip`，不是可直接加载的 action
session。必须先解压到独立目录并确认/转换为包含 `img_path`、`state=[vx,vy,wz]` 的
统一 `data.json`；不得覆盖原始 zip 或把只有图片的旧标签直接当作手柄速度标签。

第一轮训练 manifest 仍只把 CV 训练集和两组十字路口手柄数据标为 `train`，CV 验证集标为
`val`。`lap_002` 和官方手柄数据不参与训练，分别生成独立验证 manifest（或使用独立 split）
后调用同一个 action evaluator，避免把不同验证来源混成一个平均数。

放行必须同时满足：普通 `cv_val` 速度策略无明显退化；`manual_lap002` 和
`manual_official` 的转向方向正确、起转时序和持续时间合理；`cross_cv_problem` 的错误
转向减少；并在车端后处理后检查 `actual_vx`、`wz`、十字路口前减速和出弯恢复。手柄验证
只约束 `kappa_action`，不能因为手柄低速命令而改变 CNN 的速度策略。

禁止把第三圈相邻帧随机拆回训练集。这里保留完整验证圈不是为了测跨赛道泛化，而是为了
选择 checkpoint，并检查模型是否学到可重复的固定赛道动作，而不是某一圈的控制抖动。
选定训练轮数和参数后，最终模型从 D4 重新初始化，使用三圈完整数据和全部弯道专项数据
按选定轮数重训一次。

弯道专项 session 必须包含入弯前、完整弯中和出弯回正后的连续帧，不能只截取曲率峰值。
专项数据还必须使用与完整圈相同的标签语义、CV 控制版本、ROI、速度曲线和曲率计算方式。

#### 采样比例

训练 batch 不按原始帧数直接均匀抽样，初始比例为：

```text
完整圈普通帧      60%
弯道专项帧        40%
```

如果目标锐角训练后仍明显偏弱，可将弯道专项提高到 50%，但第一版不超过 60%，避免弯道
样本同时通过重复采样和大损失权重被双重放大，从而导致直道持续转向或提前降速。

#### 分阶段长训

以约 3000～5000 帧、batch size 32 为预期规模。正式训练按下列阶段执行：

```text
阶段 1：只训练全连接输出头
  epochs = 10
  learning rate = 1e-4

阶段 2：解冻最后两层卷积和输出头
  epochs = 30
  head learning rate = 2e-5
  convolution learning rate = 2e-6

阶段 3：全网络长训
  epochs = 160
  learning rate = 5e-6，余弦衰减到 1e-7
```

初始总长度为 200 epochs。若训练集约 3500 帧、batch size 32，则约为 22000 次优化器更新。
每 5 epochs 保存一个 checkpoint，不能只保留最低全局平均 loss 的单一版本。最终轮数由完整
验证圈的关键路段行为选择；如果 200 epochs 结束时仍稳定改善，可以继续训练并保持相同
checkpoint 间隔。

#### 双输出损失

`speed_demand` 范围为 `[0,1]`，`kappa_action` 典型满量程约为 `5 m^-1`。正式训练前应先
归一化曲率头，避免未归一化的曲率数值天然压过速度头：

```text
speed_scaled = speed_demand
kappa_scaled = kappa_action / 5.0

loss_speed = SmoothL1(speed_prediction, speed_demand)
loss_kappa = SmoothL1(kappa_prediction / 5.0, kappa_action / 5.0)
loss = loss_speed + 2.0 * loss_kappa
```

两个分量仍分别乘 `target_mask` 并按有效权重归一化。第一版全为 CV 样本，mask 均为
`[1,1]`，但训练器必须继续保持 mask 契约，不能创建仅对当前数据有效的旁路实现。

#### 数据增强

固定赛道第一版以拟合现场图像为优先：

```text
水平翻转        关闭
Hue/颜色旋转    关闭
模糊和噪声      关闭
亮度增强        关闭或只保留经现场确认的轻微范围
```

水平翻转虽然能正确反转 `kappa_action`，但会构造固定赛道中不存在的镜像路线，因此不用于
第一版。任何增强都必须同时保持入弯和回正时序，不得只根据随机帧 MAE 决定是否启用。

#### checkpoint 选择

固定验证圈至少逐段比较：

- 入弯和回正位置；
- `kappa_action` 的方向、峰值、持续时间和整段冲量；
- 最后锐角及其他专项弯道；
- 直道曲率偏置和误转；
- `speed_demand` 的入弯升高、弯中保持和出弯释放；
- 相邻帧输出抖动；
- 经车端速度曲线与真实 `dt` 后处理还原的整圈 `vx/wz`。

不能仅按训练 loss 或整圈平均 MAE 选择模型。完成离线选择后，仍须使用低速
`kappa_action` profile 进行完整一圈实车验收，再逐步恢复目标速度。

#### 训练器实施门槛

当前 action 入口已支持双输出、mask、`kappa_action/5.0` 归一化、固定赛道关闭水平翻转、
评测和导出，但仍是全网络统一学习率和均匀采样的基础版本。正式长训开始前必须补齐并测试：

1. 三阶段冻结/解冻；
2. 输出头与卷积层分层学习率；
3. 完整圈与弯道专项的 60/40 分组采样；
4. `kappa_action / 5.0` 的归一化 masked loss（已完成）；
5. 每 5 epochs checkpoint 及完整验证圈逐段报告。

在这些能力完成前，只能运行 smoke 验证，不能直接把当前基础入口用于 200-epoch 正式训练。

### 20.10 首轮 20-epoch CV+手柄联合训练结果（2026-08-17）

新容器为 `connect.bjb1.seetacloud.com:43447`，GPU 为 RTX 4080 SUPER 32 GB，使用
`/root/autodl-tmp/envs/lane-d5/bin/python`、Paddle 3.3.1。远端独立目录为：

```text
/root/autodl-tmp/lane-cnn/run-20260817-cross-manual
```

正式训练没有使用最初过小的 863 CV + 646 手柄组合。最终训练集为：

```text
CV 训练集1                                      863
CV 可用圈 0～867                                868
CV 弯道                                          490
CV 锐角连续弯                                    220
CV 弯道                                          381
CV 合计                                         2822
十字路口手柄 1+2                                 646
训练合计                                        3468
CV 独立验证                                      868
```

其中 CV 训练数据有 2045 帧满足 `abs(kappa_action)>=0.05`；手柄样本占训练样本约
18.6%。`或许可以用`、`先不要用` 和两个 `不可用` session 均未加入。手柄 mask 为
`[0,1]`，CV mask 为 `[1,1]`。训练前按 TDD 修正了两项基础入口：曲率先除以 5.0 再计算
Smooth L1；action 固定赛道训练显式设置水平翻转概率为 0。聚焦测试远端 16 项全部通过。

本轮从真实 D4（SHA256 `3398fc6b2d2be73d1ec7eed6f54f51f0be6e2b6c5814f255485d23442167cb49`）
初始化，训练 20 epochs、batch 64、学习率 `1e-5`、曲率权重 2。CV 验证从 epoch 1 的
`speed_mae=0.2161/kappa_mae=1.0040` 改善到 epoch 20 的
`speed_mae=0.0666/kappa_mae=0.2652`。

独立手柄验证显示 checkpoint 存在权衡：epoch 10 在 `lap_002`、官方训练参考集、官方 eval
上的曲率 MAE 分别约为 `0.562/0.479/0.478`，优于 epoch 20 的
`0.628/0.624/0.638`；epoch 20 的 CV 指标与部分方向一致率更好，但零转向帧误转率更高。
因此没有仅按 CV 平均 MAE宣布唯一最终模型，而是导出两个候选：

```text
epoch 10 平衡版
  artifacts/cross_manual_joint_20260817_remote/export_epoch10_balanced.tgz
  SHA256 51b2232c72409dc66fbd01ab678ddfb577a0c95766f121a3572da8ac54295fe8

epoch 20 CV 最佳版
  artifacts/cross_manual_joint_20260817_remote/export_epoch20_cv_best.tgz
  SHA256 19b030ae83716614d2bec831d30fee887e5257227a02da1aaca3da924fecb10f
```

两者动态/静态最大差异分别为 `3.58e-7` 和 `2.38e-7`，均使用 `kappa_action` profile。
本轮未部署 Orin；下一步必须低速闭环对比十字路口误转、普通弯道、锐角、入弯减速和出弯恢复。
本轮是基础入口的 20-epoch 候选训练，不替代上文仍待实现的 200-epoch 分层学习率/分组采样长训。

### 20.11 固定赛道 200-epoch 长训实测结果（2026-08-18）

本轮从原始 D4 重新开始，仅使用 CV 2822 帧与十字路口手柄 646 帧。manifest SHA256 为
`fab8953b90515befa811574c37251efe1e3bf7a19422a70f2203b86780876758`，D4 SHA256 为
`3398fc6b2d2be73d1ec7eed6f54f51f0be6e2b6c5814f255485d23442167cb49`。训练采用 CV 70%、
手柄有效转向 15%、手柄零转向上下文 15%，batch 64、63 batches/epoch、曲率除以 5 的
masked loss、曲率权重 2。

阶段边界实测无误：epoch 1～20 为 head/clean，epoch 21～160 为 rear/mild photometric，
epoch 161～200 为 full/clean；epoch 161 增强命中数为 0。epoch 200 CV 验证
`speed_mae=0.0445`、`kappa_mae=0.3045`，完整 40 个评测报告和训练报告保存在
`artifacts/cross_manual_long_20260817_remote/`。

三角色候选为：`crossroad_best=epoch 55`（十字路口方向/召回 `1.000/1.000`，曲率 MAE
`0.4494`）、`balanced=epoch 135`（方向/召回 `0.9865/0.9459`，误转率 `0.5385`）、
`cv_preservation_best=epoch 200`（CV speed/kappa MAE `0.0445/0.3045`，十字路口曲率
MAE `0.1806`）。候选均为 `selected=false`、`vehicle_test_performed=false`，未部署 Orin。

导出归档 SHA256：

```text
epoch_0055_crossroad_best.tgz       1e9d5d4bc3636cc4bb237c730ba3a6e6545844e100d746a670fe15746a441d6c
epoch_0135_balanced.tgz             626c0f090c028d666f3fe4e72d7ac8ea434cc0582168c5fadb3e95b913ab1b33
epoch_0200_cv_preservation_best.tgz a882834db36193937a738900c51794a608c994014df4dcf2e3ce6b57089d8a52
```

三个候选动态/静态最大差异不超过 `4.77e-7`。最终放行仍需低速实车比较十字路口误转、
普通弯道/锐角、入弯减速和出弯恢复；在此之前不宣布唯一最终模型，也不修改 `config_car.yml`。

### 20.12 epoch 160 补充导出及 D4/epoch 200 对比（2026-08-18）

逐 checkpoint 复核发现自动标记的 `balanced=epoch 135` 不是实际最优折中点。epoch 160 与
epoch 135 的十字路口方向正确率和有效转向召回相同（`0.9865/0.9459`），但十字路口误转率、
十字路口曲率 MAE、CV speed MAE 和 CV kappa MAE 分别由
`0.5385/0.2587/0.0498/0.3366` 改善为 `0.4965/0.2028/0.0463/0.3155`。因此补充导出
epoch 160，角色仅记为 `balanced_replacement`，不修改原 shortlist，也不自动选中：

```text
artifacts/cross_manual_long_20260817_remote/exports_extra/epoch_0160_balanced_replacement.tgz
checkpoint SHA256  1f4ec8a5ccf624b4d96914fb3b10cb41e95bb0e199111ff51389fcd7a32b7056
archive SHA256     d147fe0538c800e4a5ad0a6addd5ba8a1aa85366de5d67075946c0d61d352366
dynamic/static     3.5762786865234375e-7
selected           false
vehicle test       false
```

随后使用真正 D4（SHA256 `3398fc6b2d2be73d1ec7eed6f54f51f0be6e2b6c5814f255485d23442167cb49`）
与 epoch 200（SHA256 `0061c3546b0c35dcd873873bb3e3cb42a9397f67dd9b93a71a3f9fb9e40a44b4`）
对同一 CV 验证圈和 `lap_002` 重新推理。D4 输出是 `[vy,yaw_error]`，必须走
`legacy_error + PID + CROP`；epoch 200 输出是 `[speed_demand,kappa_action]`，必须走
速度曲线、真实 `dt` 速度梯度和 `wz=actual_vx*kappa_action`。两者第二维单位不同，禁止把
D4 原始第二维直接当曲率，也禁止把 D4 原始 MAE 与 epoch 200 曲率 MAE 直接比较。

CV 验证圈共 868 帧。本次从原始 `data.json` 读取每帧 `effective_dt_s`，修正了旧 action
manifest 未保留该字段而导致 868 帧全部使用 `0.05 s` fallback 的限制。经过各自正确车端
后处理后，结果为：

| CV 物理指标 | D4 legacy，Kp=1.95 | D4 历史 Kp=3 | epoch 200 |
|---|---:|---:|---:|
| 实际速度 MAE ↓ | 0.0365 | 0.0365 | **0.0171** |
| 实际 `wz` MAE ↓ | 0.2495 | 0.3615 | **0.0675** |
| 转向方向正确率 ↑ | 0.9025 | 0.9025 | **0.9465** |
| 有效转向召回 ↑ | 0.5889 | 0.6119 | **0.8738** |
| 非转向帧误转率 ↓ | 0.3884 | 0.4029 | **0.0928** |
| `wz` 序列相关性 ↑ | 0.7293 | 0.7106 | **0.9003** |
| 转向冲量比（理想 1） | 1.2559 | 1.7258 | **0.9388** |
| 起转中位偏移 | 晚 3 帧 | 晚 3 帧 | **0 帧** |
| 最低速度 m/s | 0.18 | 0.18 | 0.12 |
| 最大绝对 `wz` rad/s | 1.50 | 1.50 | 0.9535 |

epoch 200 在新 CV 验证圈的速度、角速度、方向、召回、误转和时序均明显优于 D4；D4 在
Kp=1.95 和历史 Kp=3 两种解释下结论相同。epoch 200 的 action 空间曲率 MAE、方向、召回、
误转率、相关性和冲量比分别为 `0.3045/0.9164/0.8607/0.2016/0.9401/0.9733`。

`lap_002` 共 3134 帧，只以 `kappa=wz/max(abs(vx),0.12)` 比较曲率，不评速度。D4 先经过
legacy PID/CROP 后再换算曲率；以当前 Kp=1.95 为主结果：

| `lap_002` 曲率指标 | D4 legacy，Kp=1.95 | epoch 200 |
|---|---:|---:|
| 曲率 MAE ↓ | 0.9555 | **0.6701** |
| 方向正确率 ↑ | 0.7998 | **0.8988** |
| 有效转向召回 ↑ | 0.6685 | **0.8733** |
| 零转向误转率 ↓ | **0.1842** | 0.7043 |
| 序列相关性 ↑ | 0.7275 | **0.7417** |
| 转向冲量比（理想 1） | 1.4499 | **0.9744** |
| 起转中位偏移 | 晚 7 帧 | 提前 16 帧 |
| 回正中位偏移 | 提前 8 帧 | **0 帧** |
| 峰值曲率 | 8.3333 | **6.2212** |
| P95 相邻曲率变化 ↓ | 0.9441 | **0.4566** |

因此 epoch 200 在 `lap_002` 上的方向、召回、幅值、冲量和回正更好，但存在明显提前转向和
零转向上下文残余曲率；D4 更克制，却容易晚起转和漏掉弱弯。epoch 200 的 `0.7043` 误转率
使用很低的 `abs(kappa)>=0.05 m^-1` 阈值，不能直接解释为 70% 的直道发生大幅转弯；按
`0.25 m/s` 换算只相当于约 `0.0125 rad/s`。实车仍必须检查这些低幅输出是否累积成可见偏航。

epoch 200 在十字路口 1 的 38 个有效转向帧中漏检 6 帧，具体为：

```text
frame       target kappa     predicted kappa
0088.jpg       -0.1402          -0.0350
0089.jpg       -0.4205          -0.0165
0274.jpg       -0.2804          -0.0043
0279.jpg       -0.1402          -0.0050
0280.jpg       -1.2616          -0.0082
0281.jpg       -1.4018          -0.0005
```

这些帧预测符号仍为负，因此方向正确率保持 100%，但幅值低于 `0.05 m^-1`，有效转向召回
下降。`0280～0281` 的目标曲率已经达到 `-1.26/-1.40 m^-1` 而模型几乎输出零，是本轮最
重要的转向不足风险；`0088～0089` 次之，`0274` 是前后很快回零的孤立有效帧。

#### 低速实车测试检查表

首次只使用低速 `kappa_action` profile，对 epoch 200、epoch 160 和 epoch 55 做同一路线
闭环比较，不先提高目标速度。每个候选至少完整跑一圈，并逐项记录：

1. 起步直道及长直道是否持续轻微偏转，尤其检查 `lap_002` 离线误转所代表的低幅残余曲率；
2. 普通弯道是否比正确轨迹提前约 16 帧切入，是否压内线或在入弯前摆动；
3. 十字路口 1 对应 `0088～0089` 和 `0279～0281` 视觉阶段是否转向不足，重点确认
   `0280～0281` 是否出现直行、晚转或需要人工接管；
4. 十字路口 2 是否保持正确方向、正常起转和回正，避免为修正路口 1 引入过转；
5. 普通弯道、锐角和最后右锐角的方向、峰值和持续时间，确认不存在 D4 式晚起转或 PID 限幅；
6. `actual_vx` 是否在入弯前平滑降至安全范围、弯中不突变、出弯后及时恢复；
7. `wz` 是否出现接近 `1.5 rad/s` 限幅、连续跳变或左右符号抖动；
8. 出弯和十字路口结束后是否完全回正，不能只检查成功进入弯道；
9. 若发生偏航，保留原始图像、模型两维输出、`effective_dt_s`、后处理 `target_vx/actual_vx/wz`
   和人工接管帧号，区分模型曲率、速度梯度和控制限幅问题；
10. epoch 200 只有在两个十字路口、普通弯道、锐角、直道回正和速度恢复均通过后才可由用户
    明确选中；在此之前继续保持 `selected=false`，不得自动部署或修改 `config_car.yml`。
