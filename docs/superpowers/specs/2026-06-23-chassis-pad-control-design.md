# Chassis Pad Control Design

## Goal

Create a standalone Bluetooth gamepad tool for chassis-only testing, derived from the existing `collect_control.py` driving behavior.

## Scope

The tool initializes only the Bluetooth gamepad and mecanum chassis driver. It does not initialize cameras, streamers, data collectors, arm controllers, servos, display, inference, or Ernie.

## Behavior

The control mapping matches the normal driving mode in `smartcar/whalesbot/tools/collect_control.py`:

- left stick Y controls forward/backward chassis x speed
- left stick X controls lateral y speed with sign inverted
- right stick X controls yaw speed with sign inverted and multiplied by pi
- L1 plus L2 exits safely
- disconnected pad input stops the chassis and waits for reconnection
- Ctrl+C, exceptions, and normal exit all command zero velocity

## Testing

Local tests cover the pure pad-to-velocity mapping and exit/disconnect detection without importing hardware classes. Hardware execution remains on Orin after syncing the local file.
