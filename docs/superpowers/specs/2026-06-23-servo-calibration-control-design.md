# Servo Calibration Control Design

## Goal

Create a standalone calibration tool for both PWM servos and RS485 bus servos on the MC601/MC602 control stack.

## Scope

The tool provides interactive commands for selecting a servo, setting angles, stepping angles, changing speed, sweeping a range, centering one servo, and centering all configured servos. It does not write calibration values back into source or YAML files.

## Defaults

The `home-all` command uses current project defaults:

- PWM port 1 -> -42
- PWM port 2 -> -37
- bus port 2 -> 0

The defaults can be overridden at runtime:

```bash
python -m smartcar.whalesbot.tools.servo_calibration_control --home-pwm 1:-42,2:-37 --home-bus 2:0
```

## Safety

Hardware classes are imported only when the tool is run, not during local parsing tests. The tool clamps PWM logical angles to the existing `ServoPwm` range expectation and keeps bus angles as signed integers because the existing bus servo code supports signed angles.
