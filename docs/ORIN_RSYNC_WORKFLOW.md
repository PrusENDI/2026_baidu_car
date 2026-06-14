# Orin rsync run-copy workflow

本文档记录本项目的本地开发、Orin 运行副本同步、日志拉回、Git 版本管理流程。

## 核心约定

- Windows 本地仓库是唯一主版本。
- Jetson Orin 工作区只是运行副本。
- 所有代码修改都必须发生在本地仓库。
- 禁止从 Orin 反向覆盖本地源码。
- 日志和运行产物可以从 Orin 拉回本地，例如 `logs/orin/`。
- 不默认排除模型、权重、配置文件，因为这是整车代码。
- 同步默认排除 `.git/`、`.codex/`、`logs/`、缓存、虚拟环境、临时产物、构建产物。

## 当前路径

本地项目根目录：

```text
C:\weizijian\documents\baidu car\2026_official_code\baidu_smartcar_2026
```

Orin 运行副本目录：

```text
/home/jetson/workspaces/baidu_car_2026_official_run_copy/
```

Orin SSH 目标：

```text
jetson@192.168.0.155
```

## 本地准备

Windows PowerShell 需要能找到 `rsync` 和 `ssh`。

```powershell
Get-Command rsync
Get-Command ssh
```

本项目当前使用 MSYS2 rsync。脚本里通过 `scripts/orin_config.ps1` 指定：

```powershell
$Script:RsyncWindowsPathStyle = "msys"
$Script:RsyncSshCommand = "ssh -i /c/tmp/codex_orin_ed25519 -o StrictHostKeyChecking=accept-new"
```

`C:\tmp\codex_orin_ed25519` 是 Codex/rsync 专用 SSH key。不要把私钥提交进仓库。

## 首次同步检查

先进入本地项目根目录：

```powershell
cd "C:\weizijian\documents\baidu car\2026_official_code\baidu_smartcar_2026"
```

执行 dry-run。该命令只列出将同步的文件，不复制、不删除。

```powershell
.\scripts\deploy_to_orin.ps1
```

确认 dry-run 清单合理后，再正式同步。

```powershell
.\scripts\deploy_to_orin.ps1 -Apply
```

默认不使用 `--delete`。只有确认远端目录完全是运行副本时，才允许显式使用：

```powershell
.\scripts\deploy_to_orin.ps1 -Apply -Delete
```

## 远程运行

完整比赛流程目前不适合作为首次远程运行入口。优先使用安全调试脚本。

底盘 dry-run 示例：

```powershell
.\scripts\run_on_orin.ps1 -Command "python3 debug/chassis_check.py move --x 0.1 --seconds 0.3"
```

真正执行底盘小动作时，显式加 `--apply`：

```powershell
.\scripts\run_on_orin.ps1 -Command "python3 debug/chassis_check.py --apply move --x 0.1 --seconds 0.3"
```

机械臂 dry-run 示例：

```powershell
.\scripts\run_on_orin.ps1 -Command "python3 debug/arm_check.py reset"
```

真正执行机械臂动作时，显式加 `--apply`：

```powershell
.\scripts\run_on_orin.ps1 -Command "python3 debug/arm_check.py --apply reset"
```

## 拉回日志

默认从远端 `logs/` 拉回到本地 `logs/orin/`。

```powershell
.\scripts\fetch_orin_logs.ps1
```

不要把 Orin 上的源码目录反向同步回本地。

## 一键调试

dry-run：

```powershell
.\scripts\orin_debug.ps1
```

正式同步、运行、拉日志：

```powershell
.\scripts\orin_debug.ps1 -Apply -Command "python3 debug/chassis_check.py --apply move --x 0.1 --seconds 0.3"
```

首次调试不建议使用 `-Delete`。

## Git 和远端仓库

每次本地修改完成后：

```powershell
git status --short
git diff
```

验证通过后提交：

```powershell
git add <files>
git commit -m "message"
```

推送到已确认的远端。当前不要默认推送，因为现有 `origin` 是 Gitee，需要用户明确确认后才允许上传：

```powershell
git push origin master
```

当前仓库的 `origin` 是 Gitee：

```text
https://gitee.com/younglet/baidu_smartcar_2026.git
```

如果需要推送 GitHub，需要先添加 GitHub remote，例如：

```powershell
git remote add github <github-repo-url>
git push github master
```

## 检查清单

- 本地 `git diff` 可审查所有代码修改。
- Orin 目录只作为运行副本。
- 同步前先 dry-run。
- 正式同步不默认 `--delete`。
- 日志只拉到 `logs/orin/` 等明确产物目录。
- 不从 Orin 拉源码覆盖本地。
- 提交后只在用户明确确认目标远端时推送。
