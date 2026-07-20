# Orin 2026-07-17 副本工作树设计

## 目标

基于 Git 提交 `c1edb8b` 创建独立分支和 worktree，将已通过 rsync checksum
验证的 Orin 副本 `20260720-baidu_smart_2026_7_17` 覆盖到项目目录，同时保留
Git 基线中仅本地存在的调试工具、测试和文档。

## 隔离边界

- 新分支：`orin-20260717`
- 新 worktree：`C:/weizijian/documents/baidu/2026_baidu_car/.worktrees/orin-20260717`
- 基线提交：`c1edb8b`
- 副本来源：`C:/weizijian/documents/baidu/orin-backups/20260720-baidu_smart_2026_7_17`
- 副本覆盖目标：新 worktree 下的 `2026/baidu_smartcar_2026`
- 当前 `lane-test-telemetry` worktree 的已修改和未跟踪文件不复制、不覆盖、不删除。

## 合并规则

使用覆盖式复制，不删除副本中缺少的 Git 基线文件。因此，Orin 副本中的同名文件
成为新 worktree 的工作版本，而本地独有的 `tools/`、`tests/`、`.codex/` 和调试
文档继续保留。模型、日志和缓存可以物理存在于新 worktree，但遵循现有忽略规则，
不加入 Git 提交。

## Orin 目标

新 worktree 的状态检查、同步和调试工具默认远端目录统一为：

```text
/home/jetson/workspaces/baidu_smart_2026_7_17
```

需要同步更新项目技能说明、PowerShell 工具、相关测试和部署文档，避免任何工具继续
默认指向 `baidu_car_2026_official_run_copy`。

## 验证

1. 确认 `.worktrees` 已被 Git 忽略，且新 worktree 分支、路径与基线正确。
2. 对副本中的非缓存文件逐项比较新 worktree，确认覆盖结果一致。
3. 运行 `tools/test_orin_status.ps1` 和 `tools/test_sync_to_orin.ps1`。
4. 对新远端目录执行只读 SSH 目录检查和 rsync dry-run。
5. 确认当前 `lane-test-telemetry` worktree 的修改和未跟踪文件保持不变。

本流程不向 Orin 写入文件，不提交模型、日志、缓存或用户现有未跟踪文档。
