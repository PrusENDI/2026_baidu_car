# Debug Session: paddle-sigabrt
- **Status**: [OPEN]
- **Issue**: 运行过程中 Paddle/推理相关进程收到 SIGABRT 并生成 core dump；当前日志只有 FatalError 摘要，尚无原始堆栈。
- **Debug Server**: http://127.0.0.1:7777/event
- **Log File**: .dbg/trae-debug-log-paddle-sigabrt.ndjson

## Reproduction Steps
1. 待确认启动命令与触发任务。
2. 运行至终端出现 `FatalError: Process abort signal`。

## Hypotheses & Verification
| ID | Hypothesis | Likelihood | Effort | Evidence |
|----|------------|------------|--------|----------|
| A | Paddle GPU/CUDA/TensorRT 运行时异常主动 abort | High | Med | Pending |
| B | 多个推理请求或进程争用 GPU/模型资源 | Med | Med | Pending |
| C | 相机、ZMQ 或推理后端进程生命周期异常触发原生库 abort | Med | Med | Pending |
| D | 内存或显存压力导致底层运行时失败 | Med | Low | Pending |
| E | 机械臂/串口等非推理原生扩展触发 abort | Low | Med | Pending |

## Log Evidence
当前仅有 SIGABRT 摘要：PID 16545 自身向自身发送 SIGABRT；没有 Python traceback、C++ backtrace、执行命令及崩溃前日志，暂时无法定位模块。

## Verification Conclusion
等待收集 pre-fix 运行证据。
