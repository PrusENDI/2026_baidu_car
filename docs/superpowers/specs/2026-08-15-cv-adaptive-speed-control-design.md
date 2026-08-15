# CV Adaptive Speed Control Design

## Goal

Make the OpenCV collection teacher run quickly on straights and slow down as
steering demand increases, while reducing the current tendency to steer before
the vehicle reaches a bend. The saved training label remains the command
actually sent to `car.set_velocity()` for the current camera frame.

## Scope

- Keep saved camera images at their native `320x240` size. Training performs
  any required resize.
- Keep the crossing state machine disabled. Crossings are handled by ordinary
  lane tracking and the existing 10-frame invalid-result hold.
- Change only the OpenCV collection-time analyzer/controller path. Do not alter
  the manual collection path or CNN runtime controller.

## Steering Signal

The standard-lane residual fit continues to provide the steering error, but
the active steering fit uses a configurable near-field interval. Far-field
geometry must not directly increase the angular command merely because a bend
has appeared near the top of the image.

The steering pipeline remains:

1. Map the near-field heading residual with `heading_scale`.
2. Apply the existing EMA and deadband.
3. Run the heading PID.
4. Apply angular-output and frame-to-frame slew limits.

The initial implementation will preserve the current sign convention and
lateral output of zero. It will expose the fit interval and steering limits as
configuration rather than embedding course-specific constants in control
logic.

## Adaptive Forward Speed

Forward speed is calculated from steering demand instead of being constant.
The speed demand uses the absolute pre-slew heading-controller request, so a
sharp bend can reduce speed before the angular command has finished ramping.
The demand is normalized against a configurable full-turn reference and
clamped to `[0, 1]`.

The target speed follows a smooth curve:

```text
target = min_speed + (max_speed - min_speed) * (1 - demand) ^ exponent
```

The first safe defaults will keep `max_speed` at `0.20 m/s`, introduce a
conservative configurable bend speed, and use separate acceleration and
deceleration slew limits:

- entering a bend may reduce speed quickly;
- leaving a bend restores speed more slowly;
- speed always remains between the configured minimum and maximum.

Startup straight holding affects only angular steering. Adaptive speed may
still react to visible bend demand during startup, preventing a forced-straight
period from also forcing maximum speed.

## Collection Records

For every valid or held frame, `state` and `control` continue to store:

```text
[applied_forward_speed, applied_lateral_speed, applied_angular_speed]
```

The session metadata records all adaptive-speed parameters. Per-frame records
add the normalized steering demand and target speed so a training session can
be audited without reconstructing controller state. Held invalid frames retain
the previously applied complete command, including its forward speed, and
remain marked `short_invalid_hold`.

## Failure and Safety Behavior

- Invalid analysis without a prior command stops the vehicle.
- Up to 10 consecutive invalid frames reuse the last applied command; the next
  invalid frame stops the vehicle.
- Stale camera frames, exceptions, explicit stop, and process exit retain the
  existing immediate-stop behavior.
- No crossing-specific acceleration or steering override is introduced.

## Verification

Focused tests will verify:

- straight demand produces maximum speed;
- increasing steering demand monotonically reduces target speed;
- speed never leaves its configured bounds;
- deceleration is faster than acceleration;
- angular sign, EMA, deadband, and slew behavior remain compatible;
- saved labels equal the exact command sent to the vehicle;
- invalid-frame command holding preserves both steering and speed.

After focused tests pass, the existing recorded lap can be processed offline
to compare steering onset, bend-entry speed, bend-exit acceleration, and the
two crossing windows. Offline replay is diagnostic only; real-vehicle tuning
remains necessary for minimum bend speed and angular authority.
