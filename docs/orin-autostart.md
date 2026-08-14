# Orin 开机自启动

`baidu-smart.service` 会在 Orin 开机后以 `jetson` 用户启动
`car_start_2026.py`。程序首先创建 `MyCar`，完成推理模型预热、摄像头、底盘和机械臂初始化，
然后等待车辆物理按键 1。按键 1 只负责开始任务，不再负责启动或预热程序。

## 安装

项目必须部署在：

```text
/home/jetson/workspaces/baidu_smart_2026_7_17
```

在 Orin 上运行：

```bash
cd /home/jetson/workspaces/baidu_smart_2026_7_17
chmod +x scripts/install_orin_service.sh
sudo ./scripts/install_orin_service.sh
```

安装脚本会复制 unit、刷新 systemd 配置、启用开机启动，并立即启动服务。

## 检查和维护

```bash
# 查看状态
systemctl status baidu-smart.service

# 持续查看模型预热和车辆初始化日志
journalctl -u baidu-smart.service -f

# 重启
sudo systemctl restart baidu-smart.service

# 停止
sudo systemctl stop baidu-smart.service

# 取消开机自启动
sudo systemctl disable --now baidu-smart.service
```

更新 Python 代码或模型后不需要重新安装 unit，只需重启服务：

```bash
sudo systemctl restart baidu-smart.service
```

## 暂停开机自启动，改回 SSH 手动启动

先停止当前服务，并取消下次开机自启动：

```bash
sudo systemctl disable --now baidu-smart.service
```

确认服务已经停止并取消启用：

```bash
systemctl is-active baidu-smart.service
systemctl is-enabled baidu-smart.service
```

正常应分别显示 `inactive` 和 `disabled`。然后可以通过 SSH 手动启动主程序：

```bash
cd /home/jetson/workspaces/baidu_smart_2026_7_17
/usr/bin/python3 -u car_start_2026.py
```

需要退出 SSH 手动运行的主程序时，在当前终端按 `Ctrl+C`。

不要在 systemd 服务运行时再次通过 SSH 启动 `car_start_2026.py`，否则两个进程会同时占用
串口、摄像头和推理端口。手动运行前应先确认：

```bash
systemctl is-active baidu-smart.service
```

输出为 `inactive` 后再手动启动。

如果只想取消下次开机自启动、暂时保留当前正在运行的服务，可执行：

```bash
sudo systemctl disable baidu-smart.service
```

## 恢复开机自启动

结束 SSH 中手动运行的主程序后，执行：

```bash
sudo systemctl enable --now baidu-smart.service
```

该命令会恢复下次开机自启动，并立即启动服务。检查结果：

```bash
systemctl is-enabled baidu-smart.service
systemctl is-active baidu-smart.service
```

正常应分别显示 `enabled` 和 `active`。如需验证真实开机流程，可以重启 Orin：

```bash
sudo reboot
```

重启后通过 SSH 检查本次开机日志：

```bash
systemctl status baidu-smart.service
journalctl -u baidu-smart.service -b --no-pager
```

服务异常退出时会在 5 秒后重启。停止服务时，systemd 会同时终止
`car_start_2026.py` 拉起的推理后端，避免遗留旧模型进程。
