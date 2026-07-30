# Qianfan Multimodal Order Recognition Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Route animal and order images through one Qianfan multimodal JSON method, remove OCR from order parsing, and fail safely before shooting or pickup when recognition is invalid.

**Architecture:** `ErnieBotWrap` becomes a Qianfan OpenAI-compatible client configured by non-secret YAML values and `QIANFAN_API_KEY`. `MyCar` owns task prompts, detection cropping, retry, and domain validation through `analyze_task_image()`. Existing animal and order task functions become thin consumers; delivery name OCR remains unchanged.

**Tech Stack:** Python 3, OpenCV, PyYAML, OpenAI Python client, existing task detector and camera pipeline.

**User testing constraint:** Do not add tests and do not perform a red/green TDD cycle. Implement directly, then run static and non-hardware verification only.

---

### Task 1: Configure the Qianfan OpenAI-compatible client

**Files:**
- Modify: `config_car.yml:164-168`
- Modify: `smartcar/paddlebaidu/ernie_bot/base/ernie_bot_wrap.py:1-449`

- [ ] **Step 1: Replace the tracked AI Studio token setting with non-secret Qianfan settings**

Replace the `ernie_access_token` section with:

```yaml
# ============================================================
# 千帆 OpenAI 兼容多模态接口配置；API Key 从 QIANFAN_API_KEY 环境变量读取。
# ============================================================
qianfan:
  base_url: "https://qianfan.baidubce.com/v2"
  multimodal_model: "amv-4093c27f3796"
  request_timeout: 10.0
```

- [ ] **Step 2: Initialize `ErnieBotWrap` from the Qianfan block and environment**

Remove the active AI Studio SDK setup and initialize the existing `OpenAI` client with validated values:

```python
qianfan_cfg = config.get("qianfan", {})
api_key = os.environ.get("QIANFAN_API_KEY", "").strip()
base_url = str(qianfan_cfg.get("base_url", "")).strip()
model = str(qianfan_cfg.get("multimodal_model", "")).strip()
request_timeout = float(qianfan_cfg.get("request_timeout", 10.0))

if not api_key:
    raise RuntimeError("缺少 QIANFAN_API_KEY 环境变量")
if not base_url.startswith("https://"):
    raise ValueError("千帆 base_url 必须是 HTTPS 地址")
if not model:
    raise ValueError("千帆 multimodal_model 不能为空")
if not math.isfinite(request_timeout) or request_timeout <= 0:
    raise ValueError("千帆 request_timeout 必须是有限正数")

self.client = OpenAI(api_key=api_key, base_url=base_url)
self.image_model = model
self.request_timeout = request_timeout
```

Do not log `api_key`, request headers, or the complete client exception.

- [ ] **Step 3: Add a strict JSON response parser**

Implement a parser that accepts raw JSON or one fenced JSON block and always returns a dictionary:

```python
@staticmethod
def parse_json_object(content):
    if not isinstance(content, str) or not content.strip():
        raise ValueError("千帆响应为空")
    text = content.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()
    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("千帆响应必须是单个 JSON 对象")
    return data
```

- [ ] **Step 4: Replace the animal-specific image request with one generic multimodal method**

Implement:

```python
def get_multimodal_json(self, base64_image, prompt, request_timeout=None):
    if not isinstance(base64_image, str) or not base64_image:
        raise ValueError("多模态图片 Base64 不能为空")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("多模态提示词不能为空")
    timeout = self.request_timeout if request_timeout is None else request_timeout
    response = self.client.chat.completions.create(
        model=self.image_model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        },
                    },
                ],
            }
        ],
        top_p=0.1,
        timeout=timeout,
    )
    return self.parse_json_object(response.choices[0].message.content)
```

Update legacy `get_res()` to call `self.client.chat.completions.create()` with `self.image_model`, a system message containing `self.prompt_str`, and a user text message. Keep its `(state, content)` return shape and catch request exceptions without printing them. Make `get_res_json()` pass successful content through `parse_json_object()` and return `None` on request or parse failure. Remove the `erniebot` import plus active `erniebot.api_type` and `erniebot.access_token` assignments.

- [ ] **Step 5: Compile the wrapper without making a network request**

Run:

```powershell
python -m py_compile smartcar/paddlebaidu/ernie_bot/base/ernie_bot_wrap.py
```

Expected: exit code `0` and no output.

### Task 2: Add one task-driven image analysis entry point

**Files:**
- Modify: `car_wrap_2026.py:1-527`

- [ ] **Step 1: Define task prompts and validation constraints**

Add task specifications for:

```python
MULTIMODAL_TASKS = {
    "animal": {
        "label": "animal",
        "crop_scale": 1.1,
        "prompt": (
            "识别图片中的动物，并判断它对农田有害还是有益。"
            "只返回一个 JSON 对象，字段必须且只能是 result 和 analysis。"
            "result 必须是整数：有害返回 0，有益返回 1；"
            "analysis 必须是非空中文字符串。不要返回 Markdown。"
        ),
    },
    "order": {
        "label": "order",
        "crop_scale": 1.15,
        "prompt": (
            "读取图片中的订单，提取收货人姓名、所需货物和楼号。"
            "只返回一个 JSON 对象，字段必须且只能是 name、goods、address。"
            "goods 只能是青椒、蘑菇、芹菜、番茄、油菜、豆角、西兰花、"
            "土豆、金针菇之一；address 只能是整数 1 或 2。"
            "不要返回 Markdown 或解释文字。"
        ),
    },
}
```

The order prompt must list the nine accepted goods names and addresses `1/2`; the animal prompt must define `0` as harmful and `1` as beneficial.

- [ ] **Step 2: Use one model client in `ernie_bot_init()`**

Replace the two-client setup with:

```python
def ernie_bot_init(self):
    self.image_analysis = ErnieBotWrap()
```

Remove `self.order_analysis` initialization because order images will no longer use the text analysis path.

- [ ] **Step 3: Add bounded detection crop and JPEG encoding**

Implement the helper with the following behavior:

```python
def _encode_detection_crop(self, label, crop_scale):
    detections = [
        item for item in self.get_detection_results() if item[2] == label
    ]
    if not detections:
        raise ValueError(f"未检测到多模态目标: label={label}")
    image = self.side_image.copy()
    if image.size == 0:
        raise ValueError("侧摄像头图像为空")
    x_c, y_c, width, height = detections[0][4:8]
    img_h, img_w = image.shape[:2]
    center_x = (x_c + 1.0) * img_w / 2.0
    center_y = (y_c + 1.0) * img_h / 2.0
    box_w = width * img_w * crop_scale / 2.0
    box_h = height * img_h * crop_scale / 2.0
    x1 = max(0, int(center_x - box_w / 2.0))
    y1 = max(0, int(center_y - box_h / 2.0))
    x2 = min(img_w, int(center_x + box_w / 2.0))
    y2 = min(img_h, int(center_y + box_h / 2.0))
    if x2 <= x1 or y2 <= y1:
        raise ValueError("多模态目标裁剪区域为空")
    encoded_ok, encoded = cv2.imencode(".jpg", image[y1:y2, x1:x2])
    if not encoded_ok:
        raise ValueError("多模态目标 JPEG 编码失败")
    return base64.b64encode(encoded.tobytes()).decode("ascii")
```

The helper must not issue chassis or arm commands.

- [ ] **Step 4: Add domain validation**

Implement exact-key validation in `_validate_task_image_result(task, data)`. Normalize returned strings with `.strip()` and return a new validated dictionary rather than mutating the model response:

```python
animal keys == {"result", "analysis"}
order keys == {"name", "goods", "address"}
```

Reject booleans where integers are expected. Animal `result` must be `0` or `1`; `analysis` must be a non-empty string. Order `name` must be non-empty, `goods` must be one of `青椒/蘑菇/芹菜/番茄/油菜/豆角/西兰花/土豆/金针菇`, and `address` must be integer `1` or `2`. Unknown task names raise `ValueError` before accessing the camera.

- [ ] **Step 5: Add `analyze_task_image()` with one fresh-frame retry**

Implement:

```python
def analyze_task_image(self, task, label=None):
    spec = MULTIMODAL_TASKS[task]
    detection_label = spec["label"] if label is None else label
    for attempt in range(1, 3):
        try:
            image = self._encode_detection_crop(
                detection_label, spec["crop_scale"]
            )
            data = self.image_analysis.get_multimodal_json(
                image, spec["prompt"]
            )
            return self._validate_task_image_result(task, data)
        except Exception as exc:
            logger.error(
                "多模态识别失败 "
                f"task={task} attempt={attempt} "
                f"error_type={type(exc).__name__}"
            )
            if attempt == 1:
                time.sleep(0.05)
    raise RuntimeError(f"多模态识别失败，已停止任务: task={task}")
```

Do not interpolate `exc` into logs because client exceptions can contain request metadata.

- [ ] **Step 6: Preserve the animal compatibility method**

Replace the duplicated animal crop/request code with:

```python
def animal_image_analysis(self):
    data = self.analyze_task_image(task="animal", label="animal")
    return data["result"], data["analysis"]
```

- [ ] **Step 7: Compile `car_wrap_2026.py`**

Run:

```powershell
python -m py_compile car_wrap_2026.py
```

Expected: exit code `0` and no output.

### Task 3: Switch animal and order task consumers

**Files:**
- Modify: `car_task_function.py:313-330`
- Modify: `car_task_function.py:792-836`
- Modify: `car_task_function.py:1704-1760`

- [ ] **Step 1: Represent failed animal recognition as unknown**

Initialize detection results with `None` instead of `0`, store only validated model results, and keep `target_shooting()` restricted to the explicit integer value `0`. Use `None` as the default list content so an unrecognized animal cannot become a shooting target.

- [ ] **Step 2: Replace random-order OCR and text parsing**

After visual alignment, call:

```python
order_list.append(
    my_car.analyze_task_image(task="order", label="order")
)
```

Keep the beep after successful recognition. Do not call `get_ocr(label="order")` or `order_analysis.get_res_json()`.

- [ ] **Step 3: Replace fixed-order OCR and text parsing**

Use the same `analyze_task_image(task="order", label="order")` call after the fixed-order alignment. Remove `text_list` and the loop that converts OCR strings to order dictionaries.

- [ ] **Step 4: Keep delivery name OCR unchanged**

Confirm `find_name()` still calls `get_det_ocr(det)` and no delivery name behavior was changed.

- [ ] **Step 5: Compile task code**

Run:

```powershell
python -m py_compile car_task_function.py
```

Expected: exit code `0` and no output.

### Task 4: Verify the direct implementation without red/green testing

**Files:**
- Verify: `config_car.yml`
- Verify: `smartcar/paddlebaidu/ernie_bot/base/ernie_bot_wrap.py`
- Verify: `car_wrap_2026.py`
- Verify: `car_task_function.py`

- [ ] **Step 1: Run combined compilation**

Run:

```powershell
python -m py_compile smartcar/paddlebaidu/ernie_bot/base/ernie_bot_wrap.py car_wrap_2026.py car_task_function.py
```

Expected: exit code `0`.

- [ ] **Step 2: Audit call sites**

Run:

```powershell
rg -n "analyze_task_image|order_analysis|get_ocr\(label=\"order\"\)|get_det_ocr" car_wrap_2026.py car_task_function.py
```

Expected: animal and both order calls use `analyze_task_image`; no active `order_analysis` or order OCR call remains; `find_name()` still uses `get_det_ocr`.

- [ ] **Step 3: Audit secrets and provider settings**

Run searches confirming the tracked files contain the Qianfan base URL and model but contain no actual `bce-v3/` credential and no active AI Studio token configuration.

- [ ] **Step 4: Check patch formatting**

Run:

```powershell
git diff --check
```

Expected: exit code `0`.

- [ ] **Step 5: Review exact diffs without committing mixed user changes**

Review only the relevant hunks in the four modified files. Do not clean, revert, or stage unrelated existing changes. Do not run camera, motor, homing, shooting, grasping, or live network API calls during local verification.
