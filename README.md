# 2026 Baidu Car

This private repository archives the code for the Baidu Smart Car projects.

## Repository Layout

```text
2025/
  code_vehicle_wbt/        2025 vehicle code and hardware-control stack
  starter_pack/            2025 starter-pack metadata that remains after excluding PDFs

2026/
  baidu_smartcar_2026/     2026 vehicle code, task flow, hardware wrappers, and inference wrappers
```

## Included

- 2025 vehicle-control source code.
- 2026 full vehicle project source code.
- Hardware wrappers, chassis logic, arm control, camera tools, collection tools, inference wrappers, and configuration files.
- Model configuration metadata such as `infer_cfg.yml` where present.

## Excluded

The following local assets are intentionally excluded from this GitHub code repository:

- `online/`: AI Studio / online-contest training, submission, virtual environments, temporary checks, and datasets.
- Model weights: `*.pdmodel`, `*.pdiparams`.
- Large compressed archives: `*.tar`, `*.zip`, `*.rar`, `*.7z`.
- Collected images and image assets: `*.jpg`, `*.jpeg`, `*.png`.
- PDF starter-pack documents.
- Python caches, logs, and local runtime output.

Keep those assets in local storage, object storage, GitHub Releases, Git LFS, or a separate dataset/model archive.

## 2026 Entry Points

The main 2026 project is under:

```text
2026/baidu_smartcar_2026/
```

Important files:

```text
car_start_2026.py        Main task runner
car_task_function.py     Competition task logic
car_wrap_2026.py         MyCar high-level vehicle wrapper
config_car.yml           Camera, PID, inference, and token configuration
smartcar/                Hardware, vision, inference, and utility packages
```

Run from the 2026 project root on the vehicle:

```bash
cd 2026/baidu_smartcar_2026
python3 car_start_2026.py
```

Before running the full task flow on a self-built vehicle, calibrate the hardware layers first:

1. Controller and serial connection.
2. Wheel port order and wheel direction.
3. Mecanum chassis kinematics and odometry.
4. Camera device mapping.
5. Arm zero points, limits, and grasp positions.
6. Inference service startup and model paths.
7. Individual task functions.
8. Full competition task sequence.

## Notes

This repository is code-first. It is not a complete runnable artifact until the excluded model files, datasets, and hardware-specific assets are restored to the expected paths.
