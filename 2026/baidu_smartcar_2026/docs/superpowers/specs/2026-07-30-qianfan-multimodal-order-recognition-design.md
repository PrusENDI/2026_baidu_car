# 千帆多模态订单识别统一调用设计

> 日期：2026-07-30
> 分支：`lane-test-telemetry`
> 本地工程：`C:\weizijian\documents\baidu\2026_baidu_car\.worktrees\lane-test-telemetry\2026\baidu_smartcar_2026`
> 正确 Orin 目录：`/home/jetson/workspaces/baidu_car_2026_official_run_copy/`
> 禁止目录：`/home/jetson/workspaces/baidu_smart_2026_7_17/`

## 1. 背景

当前工程存在两条不同的大模型识别链路：

- 动物识别由 `target_shooting_detection()` 调用 `animal_image_analysis()`，将目标检测框裁剪成图片，再交给多模态模型判断有害或有益。
- 订单识别由 `get_order()` 先调用本地 Paddle OCR，随后把 OCR 文本交给文本大模型解析为姓名、货物和楼号。

两条链路重复承担裁图、编码、请求和 JSON 解析职责。动物多模态接口还把提示词、返回字段和模型写死在 `get_image_res()` 中，无法直接复用于订单。订单链路则叠加了 OCR 和文本模型两层识别误差，并增加约三秒的 OCR 稳定采样时间。

千帆平台提供 OpenAI 兼容接口：

```text
base_url = https://qianfan.baidubce.com/v2
model    = amv-4093c27f3796
```

完整 Bearer API Key 属于秘密，只能从运行环境读取，不得写入源码、设计文档、Git 跟踪配置或日志。

## 2. 目标

1. 动物图片和订单图片共用一个底层多模态调用方法。
2. `get_order()` 直接把订单检测框图片交给千帆多模态模型，并获得结构化订单 JSON。
3. 从订单主链路移除 Paddle OCR 和后续文本大模型解析。
4. 暂时保留 `find_name()` / `get_det_ocr()` 的姓名 OCR 行为，避免把配送阶段混入本次变更。
5. 对检测、裁图、网络、JSON 和业务字段失败进行显式处理，失败时不得继续射击或取货。
6. 保留现有动物识别调用兼容性，不改变车辆距离、机械臂位姿、射击坐标或取货动作参数。

## 3. 非目标

- 不在本次设计中重构取货搜索或放货动作。
- 不把收货人姓名识别改成多模态模型。
- 不调整 X 轴机械归零、安全回收或运动参数。
- 不访问或同步到禁止 Orin 目录。
- 不在配置文件中保存 API Key。
- 不引入新的大模型 SDK；继续复用工程已有的 `openai` Python 客户端。

## 4. 方案比较

### 4.1 方案 A：按任务复制两个多模态方法

分别实现动物和订单图片方法。调用直观，但裁图、Base64、请求、超时和 JSON 解析会重复，后续修复容易只落到其中一条链路。

### 4.2 方案 B：通用方法接收任意 prompt 和 schema

调用方直接传提示词和校验规则。底层复用最好，但业务规则散落在调用点，容易传错任务提示词或遗漏校验。

### 4.3 方案 C：任务规格表驱动的统一方法（采用）

对外提供一个 `analyze_task_image(task, label, ...)` 方法。调用方只声明 `animal` 或 `order`；提示词、必需字段和校验器集中在任务规格表中。该方案同时满足复用、调用简单和业务约束集中管理。

## 5. 配置与鉴权

在 `config_car.yml` 中只增加非敏感配置：

```yaml
qianfan:
  base_url: "https://qianfan.baidubce.com/v2"
  multimodal_model: "amv-4093c27f3796"
  request_timeout: 10.0
```

API Key 从 `QIANFAN_API_KEY` 环境变量读取。初始化时执行以下校验：

- 环境变量存在且去除首尾空白后非空；
- `base_url` 为非空 HTTPS 地址；
- 模型标识非空；
- 超时为有限正数。

任一校验失败时，大模型客户端初始化失败并给出不含密钥内容的错误。日志不得输出请求头、完整环境变量、API Key 或包含 API Key 的异常对象。

## 6. 组件设计

### 6.1 多模态任务规格

集中定义两个任务规格：

```text
animal:
  检测标签：animal
  返回字段：result, analysis
  result：整数，只允许 0 或 1
  analysis：非空字符串

order:
  检测标签：order
  返回字段：name, goods, address
  name：非空字符串
  goods：必须属于现有 GOODS_LABEL_DICT 的中文货物集合
  address：整数，只允许 1 或 2
```

提示词要求模型只返回一个 JSON 对象，不输出 Markdown、解释前缀或额外字段。即使提示词提出该要求，本地仍必须执行 JSON 和业务字段校验，不能信任模型自行遵守。

### 6.2 检测框裁图

`MyCar` 内增加单一裁图辅助方法，职责如下：

1. 获取最新检测结果并按任务标签筛选。
2. 按既有排序规则选择一个检测框。
3. 将归一化中心坐标和宽高转换为像素边界。
4. 根据任务规格施加少量边缘扩展，避免订单文字或动物轮廓被截断。
5. 将四个边界限制在图像实际尺寸内。
6. 拒绝空图、零尺寸图和编码失败。
7. 编码为 JPEG，并在请求中使用与内容一致的 `data:image/jpeg;base64,...`。

该辅助方法只读取相机帧和检测结果，不控制车辆或机械臂。

### 6.3 通用千帆多模态请求

`ErnieBotWrap` 增加一个不含业务硬编码的 JSON 多模态方法，输入为：

- JPEG Base64 数据；
- 任务提示词；
- 请求超时。

请求使用现有 `OpenAI` 客户端：

```text
POST /v2/chat/completions
model = amv-4093c27f3796
content = [text prompt, image_url]
```

方法返回解析后的 JSON 对象。解析同时兼容纯 JSON 和被 Markdown JSON 代码块包裹的结果，但不接受无法唯一解析为一个对象的内容。

### 6.4 统一业务入口

`MyCar.analyze_task_image(task, label=None)` 负责串联任务规格、裁图、请求和校验：

```text
读取任务规格
  -> 获取并裁剪对应检测框
  -> JPEG/Base64
  -> 千帆多模态请求
  -> JSON 解析
  -> 任务字段校验
  -> 返回结构化字典
```

调用方不传 prompt、schema 或模型名称，避免业务代码绕过统一约束。

### 6.5 动物识别兼容层

保留 `animal_image_analysis()` 方法名，内部改为调用：

```python
data = self.analyze_task_image(task="animal", label="animal")
```

成功后继续按现有调用约定返回 `(result, analysis)`，使 `target_shooting_detection()` 无需同时承担接口迁移。

识别失败必须产生显式未知状态，不能继续沿用 `animal_list` 的默认 `0`。`target_shooting()` 只对经过校验的整数 `0` 执行射击；`None` 或其他值一律跳过并记录错误。

### 6.6 订单识别调用

`get_order()` 在完成随机订单和固定订单的视觉对齐后，分别调用：

```python
order = my_car.analyze_task_image(task="order", label="order")
```

删除这两个位置的：

- `my_car.get_ocr(label="order")`；
- `my_car.order_analysis.get_res_json(text)`；
- `text_list` 中间文本列表及对应 `None` 分支。

两张订单均验证成功后才允许排序和进入货架取货。任一订单失败或业务字段不合法时，抛出明确异常并停止流程。设计不额外禁止两张订单出现相同楼号、姓名或货物；是否允许重复由比赛订单规则和后续业务校验决定。

`get_ocr()`、`get_det_ocr()` 和 OCR 推理服务暂不删除，因为配送阶段的 `find_name()` 仍依赖它们。

## 7. 重试与超时

每张图片最多执行两次千帆请求：

1. 第一次使用当前对齐后的最新帧。
2. 若发生网络错误、超时、空响应、JSON 解析失败或字段校验失败，则重新读取最新相机帧、重新裁图并重试一次。
3. 第二次仍失败时终止当前任务。

重试前不移动车辆和机械臂，避免在未知状态下改变视角。检测不到目标或裁图本身无效也可以重新取一帧一次，但不得在方法内部启动视觉对齐或机械运动。

## 8. 错误处理与安全边界

以下情况均视为失败：

- 找不到指定标签；
- 检测框越界后为空；
- JPEG 编码失败；
- 缺少 API Key；
- 千帆鉴权失败、限流、超时或服务异常；
- 响应为空或不是单一 JSON 对象；
- 缺少字段、字段类型错误或出现额外字段；
- 动物结果不属于 `0/1`；
- 订单货物不在白名单或楼号不属于 `1/2`。

错误信息可以记录任务名、尝试次数、模型名、耗时、HTTP 状态类别和校验原因，但不得记录 API Key。订单失败发生在吸泵、抓取和车辆继续前进之前；动物失败不得被解释为有害动物。

## 9. 测试与验证

### 9.1 非硬件自动化测试

使用模拟检测结果、相机帧和 OpenAI 客户端响应验证：

- 动物和订单调用共用同一个请求方法；
- 检测框转换和四边界限制正确；
- JPEG 内容与 `image/jpeg` MIME 一致；
- 纯 JSON 与 Markdown JSON 均可解析；
- 动物和订单 schema 校验；
- 第一次失败后使用新帧重试一次；
- 第二次失败后停止；
- 日志和异常不包含测试 API Key；
- 订单失败不会进入排序和取货动作；
- 动物失败不会产生默认射击目标。

### 9.2 静态验证

- 对修改的 Python 文件执行 `py_compile`；
- 执行相关单元测试；
- 执行 `git diff --check`；
- 审查 `get_order()` 已不再调用订单 OCR；
- 审查 `find_name()` 仍保留姓名 OCR。

### 9.3 Orin 分阶段验证

1. 在正确 Orin 目录配置 `QIANFAN_API_KEY`。
2. 先用静态测试图片验证千帆请求和 JSON，不初始化电机。
3. 再用侧摄像头只读采集验证订单裁图，不执行取货。
4. 确认两张订单均稳定解析后，才进入有人看护的取货联调。

任何阶段都不得访问 `/home/jetson/workspaces/baidu_smart_2026_7_17/`。

## 10. 兼容与迁移结果

实施完成后的调用关系为：

```text
target_shooting_detection
  -> animal_image_analysis（兼容层）
  -> analyze_task_image(animal)
  -> 千帆多模态模型

get_order
  -> analyze_task_image(order)
  -> 千帆多模态模型

order_delivery / find_name
  -> get_det_ocr
  -> 现有本地 OCR（本次不变）
```

该边界实现动物和订单图片的一处通用多模态调用，同时避免把配送姓名识别和机械动作重构混入本次工作。
