# Lane-only Safe Test Design

## Goal

Run the 2026 lane model on Orin without importing the project-wide `smartcar` package, then permit a bounded low-speed lane-following loop only after an explicit inference preflight succeeds.

## Design

Add `tools/lane_only_infer_server.py`, a local ZeroMQ REP service on port 5001.  It loads only Paddle Inference, OpenCV, NumPy, and the lane model paths from `config_car.yml`; it must not import `smartcar` or `infer_wrap.py`.  It answers `ATATA` readiness probes and image requests with the model's two numeric outputs.

Extend `tools/safe_lane_test.py` to start that server as a child process, wait for a bounded readiness interval, make one camera-frame inference preflight, and validate exactly two finite bounded values.  A failed preflight exits before the control loop.  The control loop remains capped at speed 0.10; its default duration remains 15 seconds while the explicit duration limit is 300 seconds. Its `finally` block retains `car.stop()`.

## Safety and Verification

No official files, configuration, or weight files change.  The service logs to a user-visible file under `logs/`, returns a deterministic error on startup failure, and is terminated on script exit.  Tests cover response validation and bounded waiting without opening camera, serial, or issuing a velocity command.  Orin verification proceeds service import/start, readiness, camera plus inference, then only with explicit later authorization can the vehicle be raised and motion tested.
