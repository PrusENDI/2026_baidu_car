# Debug Session: order-api-timeout

Status: [OPEN]

## Symptom

`get_order()` 调用 `analyze_task_image(task="order", label="order")` 时连续两次发生 `APITimeoutError`，随后抛出 `RuntimeError`，主进程退出并出现 Paddle 推理进程 `SIGABRT`。

## Hypotheses

1. 多模态接口网络不可达或服务响应超过客户端读取超时。
2. `analyze_task_image` 或底层客户端配置的超时时间过短。
3. 订单图片体积/服务端推理耗时导致连续两次请求超时。
4. 多模态异常未在任务入口隔离，主流程退出时后台推理线程未正常关闭，继发 `SIGABRT`。

## Existing Evidence

- 本地 lane、task、goods 推理服务均连接成功。
- 车辆移动与两次视觉对齐已经执行。
- 多模态订单分析 attempt 1、attempt 2 均为 `APITimeoutError`。
- 根异常为 `httpcore.ReadTimeout` / `httpx.ReadTimeout`。
- `car_wrap_2026.py:602` 将最终失败包装为 `RuntimeError`。
- `car_task_function.py:1442` 未捕获该异常，导致主流程退出。
- Paddle 的 `SIGABRT` 出现在 Python 异常退出和流媒体停止之后，更像继发清理问题。

## Next Evidence

检查多模态客户端超时、重试配置及 `analyze_task_image` 的实现；在修改业务逻辑前增加最小网络调试上报。
