# 项目快速分析（自动生成）

说明：本文件概述 `code_vehicle_wbt` 仓库中主流程、各模块职责与关键调用链，方便快速定位与测试。

**主入口**
- 文件： [car_start.py](car_start.py)
- 作用：任务编排脚本，按按钮输入选择并依次运行若干任务（汉诺塔、BMI、营地、投掷、食材识别、答题、食物配送、救援等）。

**关键类与文件（一行职责）**
- [car_wrap.py](car_wrap.py)：`MyCar`，集成摄像头、推理客户端（`ClintInterface`）、PID、运动控制、定位与 OCR；为上层任务提供高阶接口。
- [task_func.py](task_func.py)：`MyTask`，机械臂与任务级动作（抓取、放置、弹射、食材/答题相关动作）。
- [brakeMgr.py](brakeMgr.py)：行进中视觉检测并紧急制动（`targetBrake`）。
- [infer_cs/base/infer_front.py](infer_cs/base/infer_front.py)：`ClintInterface`（ZMQ 客户端）与 `Bbox`，负责与独立推理后端通信（lane/task/front/ocr 等）。
- [vehicle/driver/vehicle_base.py](vehicle/driver/vehicle_base.py)：底盘与里程计基础（`CarBase`、`ChassisBase`、多种底盘模型），提供 `set_velocity`、里程计更新与速度->轮速转换。
- [vehicle/arm/arm_base.py](vehicle/arm/arm_base.py)：机械臂低层控制（步进电机、舵机、PID），提供 `set`/`set_offset`/`switch_side`/`grap` 等接口。
- [tools/base/tools_class.py](tools/base/tools_class.py)：通用工具（PID、CountRecord、get_yaml、IndexWrap 等）。
- [cvTest.py](cvTest.py)：`AnswerGet`，分格 OCR 读取答题区并返回四个选项识别结果。
- [myBtn.py](myBtn.py)：按键包装，返回按键值。
- [numProcess.py](numProcess.py)：将 AI 返回文本解析为整数或浮点数（`extract_first_integer`、`extract_first_number`）。
- [log_info/base/log_wrap.py](log_info/base/log_wrap.py)：日志初始化与轮换策略。

**主流程与关键调用链（简要）**
1. `car_start.py` 创建 `MyCar()`（`car_wrap.MyCar`），并创建 `brakeMgr`。
2. 等待按键（`myBtn.myButton.retKey()`），按键值用于 `task_answer`。
3. 按顺序执行任务函数（示例顺序）：
   - `hanoi_tower_func()` -> 使用 `lane_*` 系列定位、`task.pick_up_cylinder` 抓取、放置（依赖 `ClintInterface('task')` 和 `MyTask`）。
   - `bmi_cal()` -> 定位、`get_ocr()` 抓取文字并（可选）调用 `ernieInteract` 计算 BMI，调用 `MyTask.bmi_set` 放置结果。
   - `camp_fun()` -> 基于里程计与角度计算目标姿态，使用 `lane_sensor`/`set_vel_time` 行进与校正。
   - `mystery_task()` -> 一组机械臂抓放动作与短距前进。
   - `send_fun()` -> 执行 `MyTask.eject(1)` 弹射。
   - `task_ingredients()` -> OCR 识别食材描述、调用 `ernieInteract` 得到编号，使用 `ingredientsEncoder` 通过侧摄定位并调用 `MyTask.pick_ingredients` 抓取。
   - `task_answer(btnIn)` -> 定位到答题区、机械臂按键（当前实现直接使用外部按键输入作为答案）。
   - `task_fun2()` -> 调整位置并 `MyTask.eject(2)`。
   - `task_food(ing_1, ing_2)` -> 识别菜品、调用 `ernieInteract('food')` 判断能否制作并调用 `MyTask.set_food` 放置。
   - `task_help()` -> 救援/帮助动作序列。
4. 任务依赖：视觉推理（`ClintInterface`）、机械臂（`MyTask`->`ArmBase`）、底盘运动（`MyCar`->`CarBase`）、AI 交互（`ernieInteract`）。

**重要实现细节与建议**
- 视觉推理：`ClintInterface` 使用 ZMQ 将 JPEG bytes 发给后端 infer 服务，后端由 `infer.yaml` 指定端口与模型。运行前确保后端进程运行。
- AI 交互：`MyCar.ernieInteract` 使用 `erniebot`，函数封装了超时与默认返回。注意：代码中有明文 access token，应移动到环境变量或配置文件。 
- 稳定判断：频繁使用 `CountRecord` 保证检测/定位稳定后再执行下一步动作。
- 异常处理：部分 `except:` 未打印异常堆栈，建议改为 `logger.exception(e)` 以便调试。

**测试与运行建议（快速命令）**
- 运行主程序：

```bash
python car_start.py
```

- 启动推理后端（示例，实际后端脚本位置参见 `infer_cs` 下的后端实现）：

```bash
python infer_cs/base/infer_back_end.py
```

- 单模块调试：
  - OCR 答题测试： `python cvTest.py`
  - 机械臂动作测试： `python task_func.py` 中的测试函数或 `task_func.pick_ingredients_test()` 等。

**下一步我可以帮你做的事（选项）**
- 生成更详细的逐文件逐函数调用图（时序/伪代码）。
- 将 `ernieInteract` 的 token 抽离到配置并生成环境变量示例 `env.example`。 
- 在 CI/本地建立模拟模式（无硬件时模拟摄像头与电机）以便脱机测试。

---
文档自动生成于仓库根。若你同意，我接下来可以：
- 把 `ANALYSIS.md` 标记为完成并在 README 中添加链接，或
- 继续实现 token 配置化（移除源码中明文 token）。

请告知你希望的下一步，或直接让我继续我建议的第一项（将 token 抽离）。