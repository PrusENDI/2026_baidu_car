# Orin rsync run-copy workflow

本文档记录本项目的本地开发、Jetson Orin 运行副本同步、远程运行、日志拉回、Git 版本管理和 Codex 免密登录方式。

## 核心约定

- Windows 本地仓库是唯一主版本。
- Jetson Orin 工作区只是运行副本，可以重建。
- 所有代码修改必须发生在本地仓库，方便用 `git diff` 或 VS Code 审查。
- 禁止从 Orin 反向覆盖本地源码。
- 日志和明确指定的运行产物可以从 Orin 拉回本地，例如 `logs/orin/`。
- 不默认排除模型、权重、配置文件，因为这是整车代码，不是单独模型仓库。
- 同步默认排除 `.git/`、`.codex/`、`logs/`、缓存、虚拟环境、临时产物、构建产物。
- 不要使用 `rsync --delete`，除非已经明确确认远程目录是可丢弃运行副本。

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

Windows PowerShell 需要能找到 `rsync`，当前建议使用 MSYS2 的 rsync：

```powershell
Get-Command rsync
Get-Command C:\msys64\usr\bin\ssh.exe
```

如果 `rsync` 找不到，把 MSYS2 工具目录加入 Windows PATH：

```powershell
[Environment]::SetEnvironmentVariable(
    "Path",
    [Environment]::GetEnvironmentVariable("Path", "User") + ";C:\msys64\usr\bin",
    "User"
)
```

执行后需要重新打开 PowerShell，新的 PATH 才会生效。

本项目脚本通过 `scripts/orin_config.ps1` 固定 Orin 连接参数：

```powershell
$Script:OrinSshTarget = "jetson@192.168.0.155"
$Script:RemoteWorkspace = "/home/jetson/workspaces/baidu_car_2026_official_run_copy/"
$Script:OrinSshExe = "C:\msys64\usr\bin\ssh.exe"
$Script:OrinSshArgs = @("-i", "/c/tmp/codex_orin_ed25519", "-o", "StrictHostKeyChecking=accept-new")
$Script:RsyncSshCommand = "ssh -i /c/tmp/codex_orin_ed25519 -o StrictHostKeyChecking=accept-new"
```

## Codex 免密登录 Orin

当前使用一把专门给 Codex/rsync 的 SSH key：

```text
C:\tmp\codex_orin_ed25519
C:\tmp\codex_orin_ed25519.pub
```

私钥放在仓库外的 `C:\tmp\`，不要提交进 git，也不要复制到项目目录。仓库里只保存脚本如何引用这把 key。

如果需要重新生成这把 key，在 PowerShell 里执行：

```powershell
C:\WINDOWS\System32\OpenSSH\ssh-keygen.exe --% -t ed25519 -N "" -f C:\tmp\codex_orin_ed25519 -C codex-orin-rsync
```

把公钥安装到 Orin：

```powershell
Get-Content -LiteralPath "C:\tmp\codex_orin_ed25519.pub" |
    C:\WINDOWS\System32\OpenSSH\ssh.exe jetson@192.168.0.155 "mkdir -p ~/.ssh && cat >> ~/.ssh/authorized_keys && chmod 700 ~/.ssh && chmod 600 ~/.ssh/authorized_keys"
```

上面这一步会要求输入一次 Orin 的 `jetson` 用户密码。成功后，用 MSYS2 SSH 测试免密登录：

```powershell
C:\msys64\usr\bin\ssh.exe -i /c/tmp/codex_orin_ed25519 -o StrictHostKeyChecking=accept-new jetson@192.168.0.155 "whoami"
```

预期输出：

```text
jetson
```

注意，下面这种命令含义是“登录后在 Orin 上执行名为 `jetson` 的命令”，不是输入密码：

```powershell
ssh jetson@192.168.0.155 "jetson"
```

如果 Orin 上没有 `jetson` 这个可执行命令，就会报：

```text
bash: jetson: command not found
```

正确测试登录身份应使用：

```powershell
ssh jetson@192.168.0.155 "whoami"
```

或者使用本项目脚本实际采用的 key：

```powershell
C:\msys64\usr\bin\ssh.exe -i /c/tmp/codex_orin_ed25519 -o StrictHostKeyChecking=accept-new jetson@192.168.0.155 "whoami"
```

## 首次同步检查

进入本地项目根目录：

```powershell
cd "C:\weizijian\documents\baidu car\2026_official_code\baidu_smartcar_2026"
```

先执行 dry-run。该命令只列出将同步的文件，不复制、不删除：

```powershell
.\scripts\deploy_to_orin.ps1
```

确认 dry-run 清单合理后，再正式同步：

```powershell
.\scripts\deploy_to_orin.ps1 -Apply
```

默认不使用 `--delete`。只有确认远程目录完全是运行副本时，才允许显式使用：

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

默认从远端 `logs/` 拉回到本地 `logs/orin/`：

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

当前 `origin` 是 Gitee：

```text
https://gitee.com/younglet/baidu_smartcar_2026.git
```

不要默认推送到 Gitee，除非用户明确要求。

当前 GitHub remote 配置为：

```text
github  https://github.com/PrusENDI/2026_baidu_car.git
```

推送 GitHub：

```powershell
git push github master
```

## 检查清单

- 本地 `git diff` 可以审查所有代码修改。
- Orin 目录只作为运行副本。
- 同步前先 dry-run。
- 正式同步默认不带 `--delete`。
- 日志只拉到 `logs/orin/` 等明确产物目录。
- 不从 Orin 拉源码覆盖本地。
- 私钥不进仓库。
- 不推送 Gitee，除非用户明确要求。
