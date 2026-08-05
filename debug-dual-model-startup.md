# Debug Session: dual-model-startup
- **Status**: [OPEN]
- **Issue**: 清理另一个高内存推理后端后，同时启用 task2026 与 shucaimodel/model 仍启动异常慢。
- **Debug Server**: http://127.0.0.1:7777/event
- **Log File**: .dbg/trae-debug-log-dual-model-startup.ndjson

## Reproduction Steps
1. 配置 lane、task、goods 三个推理服务。
2. 确认旧推理后端退出。
3. 启动车辆程序，由客户端拉起 infer_back_end.py。
4. 观察从后端启动到客户端连接完成的耗时。

## Hypotheses & Verification
| ID | Hypothesis | Likelihood | Effort | Evidence |
|----|------------|------------|--------|----------|
| A | goods Predictor 构造或图优化本身耗时异常 | High | Low | Pending |
| B | goods 的三次预热而非加载阶段占据主要耗时 | High | Low | Pending |
| C | 内存或 Swap 在第二个 YOLO 加载时形成非线性压力 | Medium | Low | Pending |
| D | 后端初始化异常或线程故障被吞，客户端无超时持续等待 | Medium | Low | Pending |
| E | 仍有旧后端或端口占用导致当前后端未按预期启动 | Medium | Low | Pending |

## Log Evidence
- 第一次复现：日志为空；旧 PID 53154 已运行约 1 小时 15 分，仅约 61 MB，客户端误复用未加载 goods 的旧后端。
- 第二次复现：PID 139884 数分钟后仍只有 1 个线程、约 66 MB、无推理端口，且与主车进程同时持有 /dev/ttyUSB1；卡在 smartcar 包导入阶段。
- 干净停止主车和后端后冷启动：PID 160147 正常完成全部阶段，总耗时 12556 ms。
- 导入 infer_front 约 3498 ms；paddle_jetson 导入约 3 ms。
- 模型构造：lane 2669 ms、task 1652 ms、goods 2217 ms。
- 首轮预热：lane 4878 ms、task 539 ms、goods 259 ms；第二、三轮总计仅约 295 ms。
- 启动完成时后端 RSS 约 3.19 GB、MemAvailable 约 651 MB，SwapFree 从约 3.11 GB 降至约 2.90 GB；存在较高内存压力，但没有阻止本轮启动。

## Verification Conclusion
| ID | Status | Evidence |
|----|--------|----------|
| A | Rejected | goods 构造约 2.2 秒，正常完成，不是永久卡点。 |
| B | Rejected | goods 首轮预热约 259 ms，后续约 85–91 ms；主要首轮开销反而是 lane 的 4.88 秒。 |
| C | Partially confirmed | 双 YOLO 后端常驻约 3.1 GB，可用内存降至约 0.65–0.75 GB并使用 Swap，但干净冷启动仍在 12.6 秒完成。 |
| D | Confirmed as risk | 旧卡死进程停在 smartcar 包导入阶段，并持有硬件串口；后台错误隐藏及 ZMQ 无超时会把故障表现为一直等待。 |
| E | Confirmed root cause for first hang | 客户端复用旧后端；后续脏重启又产生导入阶段卡死。完整停止相关进程后冷启动正常。 |

本轮对比证明：异常的“无限慢”不是 goods Middle 的构造或预热，而是旧/半初始化后端残留与 smartcar 导入硬件副作用造成的脏启动。操作性修复（彻底停止主车与后端后冷启动）已使系统正常运行；尚未实施永久代码修复。
