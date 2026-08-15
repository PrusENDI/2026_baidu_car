# CV IPM Lane Errors and CNN-Compatible Labels

## Goal

Make the fixed-track OpenCV teacher produce per-frame lateral and heading
errors that the existing two-output CNN can learn directly, while recording
the actual post-PID chassis command separately.

## Geometry

For every valid ROI row, recover the current lane center from both boundaries.
When only one boundary is visible, recover the center with the standard lane
width at that row. Convert it to the open-source equivalent bird's-eye frame:

```text
x_ipm = (current_center - standard_center) * perspective.wid / standard_width
s_ipm = perspective.arr[row]
```

Fit `x_ipm(s_ipm)` with a quadratic over the complete valid ROI. The near-field
value gives lateral displacement, the near-field derivative gives heading, and
the second derivative gives curvature. This separates a shifted straight lane
from a genuinely rotated or curved lane and removes the fixed steering-onset
delay.

The geometric lateral value is normalized by half of `perspective.wid`.
`ErrorMapping` maps geometry to the CNN/PID label range with initial fixed-track
calibration values `lateral_scale=-0.10` and `heading_scale=-0.40`. The signs
convert image-right geometry into the existing chassis/hand-label convention.

## Control and recording

The CV path uses the same PID gains and limits as `config_car.yml`:

```text
y:     Kp=6, Ki=0, Kd=0.1, output=[-0.7, 0.7]
angle: Kp=1.95, Ki=0, Kd=0, output=[-1.5, 1.5]
```

The PID is called with `(-error_y, -error_angle)`, matching `lane_base()`.
Output slew limiting remains a control-only stabilizer; it never changes the
per-frame label. Forward speed remains adaptive and is calculated from the
PID-before heading label, as in the CNN runtime.

Each sample records:

```text
state   = [actual_forward_speed, cv_error_y, cv_error_angle]
control = [actual_forward_speed, actual_vy, actual_wz]
```

`state[1:3]` is therefore single-frame, PID-before CNN training data;
`control[1:3]` is the command actually sent to the mecanum chassis.

## Validation

Run the focused analyzer/controller/state-machine tests and replay all of
`lap_001`. Report whole-lap correlation, turning-only correlation, straight
false-steering counts, invalid-frame inheritance, and frames 3201, 3226, 3301,
and 3401. Offline comparison is diagnostic; real-car completion remains the
final safety and calibration check.
