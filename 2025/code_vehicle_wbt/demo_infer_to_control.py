#!/usr/bin/env python3
"""
演示：模拟从摄像头到模型到控制量的完整调用链（数值示例化）。
不依赖实际模型或硬件，读取配置并按 `car_wrap.py` 的控制逻辑计算输出。

用法：
  python demo_infer_to_control.py

输出：显示示例的推理结果、PID 输出（y_speed, angle_speed）以及轮速数组（mecanum 举例）。
"""
import yaml
import math
import numpy as np

# 1) 载入车用 PID 配置（与 MyCar 使用的一致）
with open('config_car.yml', 'r', encoding='utf-8') as f:
    cfg = yaml.safe_load(f)

# lane pid 配置（cfg 中的 Kp, Ki, Kd，主流程里对 error 取负数后输入 PID）
kp_y = cfg['lane_pid']['cfg_pid_y']['Kp']
kp_angle = cfg['lane_pid']['cfg_pid_angle']['Kp']

# 2) 载入底盘配置（以 vehicle/driver/cfg_vehicle.yaml 中的 Mecanum 为例）
with open('vehicle/driver/cfg_vehicle.yaml', 'r', encoding='utf-8') as f:
    vcfg = yaml.safe_load(f)

chassis_type = vcfg['vehicle_cfg']['chassis_type']
# 只做 mecanum 示范（默认配置文件为 MecanumChassis）
track = vcfg['vehicle_cfg']['MecanumChassis']['size']['track']
wheel_base = vcfg['vehicle_cfg']['MecanumChassis']['size']['wheel_base']

# 3) 模拟模型输出（来自 LaneInfer）——这是后端推理返回的示例
# 假设 lane model 返回两个值：[error_y, error_angle]
# 其中 error_y 表示横向偏差（像素/归一化），error_angle 表示航向误差
model_output = [0.05, 0.02]  # 示例：车辆偏右 0.05，角度偏差 0.02
error_y, error_angle = model_output

# 4) 根据 car_wrap 中的调用逻辑：
#    在 lane_base 中是: error_y, error_angle = self.cruise(image)
#    然后 y_speed, angle_speed = self.lane_pid.get_out(-error_y, -error_angle)
# 因为 PID 中 Ki=Kd=0（本配置），PID 输出近似为 Kp * input

input_y = -error_y
input_angle = -error_angle

y_speed = kp_y * input_y
angle_speed = kp_angle * input_angle

# 5) 组合期望底盘速度并转换为车轮线速度（Mecanum 逆变换）
#    set_velocity(linear_vx, linear_vy, angular_v) => chassis.get_velocity(...) => wheel_linear
linear_vx = 0.30  # 前向速度示例 (m/s)
linear_vy = y_speed
angular_v = angle_speed

# 复现 MecanumChassis 中的 transform_inverse 计算
# 参考 paddle_jetson/base/infer_wrap.py 中 MecanumChassis.params_init 的逻辑
rad_rolls = math.pi/4 * 1.052
t = math.tan(rad_rolls)
rx = track / 2.0
ry = wheel_base / 2.0
r = rx * t + ry

# transform_inverse 的列向量（每列对应一个轮子）
col0 = [1.0, t, r]
col1 = [-1.0, t, r]
col2 = [-1.0, -t, r]
col3 = [1.0, -t, r]

# 将 car_vel 与 transform_inverse 相乘（car_vel dot transform_inverse）
car_vel = np.array([linear_vx, linear_vy, angular_v])
transform_inverse = np.array([col0, col1, col2, col3]).T  # 3x4 -> 与 car_vel (3,) 相乘得到 (4,)
wheel_linear = car_vel.dot(transform_inverse)  # 结果为 4 轴线速度（单位与 linear_vx 相同）

# 6) 打印演示结果
print('=== 演示：摄像头->模型->PID->轮速 ===')
print(f'Model (LaneInfer) output: error_y={error_y:.4f}, error_angle={error_angle:.4f}')
print(f'PID gains: kp_y={kp_y}, kp_angle={kp_angle}')
print(f'PID inputs (negated): input_y={input_y:.4f}, input_angle={input_angle:.4f}')
print(f'PID outputs: y_speed={y_speed:.4f} m/s, angle_speed={angle_speed:.4f} rad/s')
print(f'Set velocity command: linear_vx={linear_vx:.4f}, linear_vy={linear_vy:.4f}, angular_v={angular_v:.4f}')
print('\nMecanum parameters:')
print(f'  rad_rolls={rad_rolls:.6f}, t=tan(rad_rolls)={t:.6f}, rx={rx:.3f}, ry={ry:.3f}, r={r:.6f}')
print('\nWheel linear velocities (4 wheels):')
for i, w in enumerate(wheel_linear):
    print(f'  wheel_{i+1}: {w:.6f} (m/s)')

print('\n说明：这些 wheel_linear 值会被传给底层电机控制模块，作为速度命令（或进一步映射为电机转速）。')

# 7) 若需调整示例，请直接编辑 model_output 或 linear_vx

if __name__ == '__main__':
    pass
