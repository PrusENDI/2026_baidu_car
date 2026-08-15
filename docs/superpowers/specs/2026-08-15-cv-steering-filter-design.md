# CV Steering Filter Design

## Goal

Reduce frame-to-frame OpenCV steering jitter without changing the fixed-track cross state machine, then compare the new controller against the existing full-lap baseline and manual steering record.

## Design

Boundary processing gains a conservative false-double-boundary rejection derived from the open-source `delete_line()` idea. It only acts when both cleaned traces exist, their near-field separation is suspiciously narrow, and the near-field pixels indicate that the two traces likely belong to one curved boundary. The less plausible trace is removed; ambiguous cases remain unchanged.

The controller applies filtering in this order: map raw heading to control error, apply an EMA with `alpha=0.35`, apply a `0.03` deadband, run the existing proportional controller, limit output to `+-0.60`, and limit each frame-to-frame change to `0.04`. Explicit reset clears the EMA and slew state. An invalid result returns an invalid command but preserves filter state so the existing five-frame command hold can resume without a steering discontinuity; disarming still resets all state.

## Verification

Run the focused controller/analyzer/state-machine tests without red/green TDD. Then process frames 1 through 3432 of `lap_001` into a new artifact directory and compare against the saved pre-filter baseline using correlation, direction agreement, output range, saturation count, total variation, large frame-to-frame changes, and sign reversals. Inspect the two cross windows and the known invalid-frame window separately.
