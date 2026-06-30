# Camera Debug Control Design

## Goal

Create a standalone camera debugging tool derived from the camera and streamer parts of `collect_control.py`.

## Scope

The tool initializes cameras and the MJPEG streamer only. It does not initialize chassis control, Bluetooth gamepad, data collectors, arm control, display, or buzzer.

## Defaults

The default camera setup matches `collect_control.py`:

- cam1 index 1 at 320x240
- cam2 index 2 at 640x480
- streamer port 5000

CLI options allow single-camera mode and per-camera resolution overrides.

## Testing

Local tests cover argument parsing, resolution parsing, and camera selection without importing or opening real cameras.
