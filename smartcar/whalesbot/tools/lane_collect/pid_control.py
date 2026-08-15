"""Low-speed PID control used only by the OpenCV collection-time teacher."""

from dataclasses import dataclass
from typing import Dict

import numpy as np

from ..tools_class import PID
from .opencv_lane import LaneAnalysisResult


@dataclass(frozen=True)
class CvLanePidConfig:
    max_forward_speed: float = 0.20
    min_forward_speed: float = 0.08
    # The launch segment is the only route-position exception: keep the
    # chassis straight for a short odometry distance before the first bend.
    # The generic lane logic must not reuse this as a per-turn delay.
    initial_straight_distance_m: float = 0.10
    full_turn_reference: float = 0.40
    speed_curve_exponent: float = 1.5
    max_deceleration_step: float = 0.03
    max_acceleration_step: float = 0.005
    control_period_s: float = 0.05
    lateral_kp: float = 6.0
    lateral_ki: float = 0.0
    lateral_kd: float = 0.1
    heading_kp: float = 1.95
    heading_ki: float = 0.0
    heading_kd: float = 0.0
    lateral_limit: float = 0.70
    heading_limit: float = 1.50
    max_heading_step: float = 0.04


@dataclass(frozen=True)
class CvLaneControlCommand:
    valid: bool
    forward_speed: float
    lateral_speed: float
    angular_speed: float
    error_y: float
    error_angle: float
    reason: str = ""
    steering_demand: float = 0.0
    target_forward_speed: float = 0.0

    def to_record(self) -> Dict[str, float]:
        return {
            "valid": bool(self.valid),
            "forward_speed": float(self.forward_speed),
            "lateral_speed": float(self.lateral_speed),
            "angular_speed": float(self.angular_speed),
            "error_y": float(self.error_y),
            "error_angle": float(self.error_angle),
            "reason": self.reason,
            "steering_demand": float(self.steering_demand),
            "target_forward_speed": float(self.target_forward_speed),
        }


class CvLanePidController:
    """Apply the main runtime's two-error/two-PID control structure safely."""

    def __init__(self, config: CvLanePidConfig = None) -> None:
        self.config = config or CvLanePidConfig()
        self.pid_y = PID(
            Kp=self.config.lateral_kp, Ki=self.config.lateral_ki,
            Kd=self.config.lateral_kd, setpoint=0.0,
            sample_time=None,
            output_limits=(-self.config.lateral_limit,
                           self.config.lateral_limit),
        )
        self.pid_angle = PID(
            Kp=self.config.heading_kp, Ki=self.config.heading_ki,
            Kd=self.config.heading_kd, setpoint=0.0,
            sample_time=None,
            output_limits=(-self.config.heading_limit,
                           self.config.heading_limit),
        )
        self._last_heading_output = 0.0
        self._last_forward_speed = None

    def reset(self) -> None:
        self.pid_y.reset()
        self.pid_angle.reset()
        self._last_heading_output = 0.0
        self._last_forward_speed = None

    def compute(self, result: LaneAnalysisResult,
                distance_m=None) -> CvLaneControlCommand:
        del distance_m
        if (not result.valid or result.error_y is None or
                result.error_angle is None or
                not np.isfinite(result.error_y) or
                not np.isfinite(result.error_angle)):
            return CvLaneControlCommand(
                False, 0.0, 0.0, 0.0, 0.0, 0.0,
                result.reason or "invalid OpenCV lane result")

        # These are the single-frame values saved to state[1:3] and learned
        # by the CNN.  All temporal smoothing is kept after this boundary.
        error_y = float(result.error_y)
        error_angle = float(result.error_angle)
        steering_demand, target_forward_speed = self._speed_target(error_angle)
        forward_speed = self._apply_speed_slew(target_forward_speed)

        # Keep the same sign convention and control sequence as lane_base():
        # controller.get_out(-error_y, -error_angle).
        dt = max(float(self.config.control_period_s), 1e-6)
        lateral_speed = float(self.pid_y(-error_y, dt=dt))
        requested_heading = float(self.pid_angle(-error_angle, dt=dt))
        step = max(float(self.config.max_heading_step), 0.0)
        if step > 0.0:
            requested_heading = float(np.clip(
                requested_heading,
                self._last_heading_output - step,
                self._last_heading_output + step,
            ))
        self._last_heading_output = requested_heading
        return CvLaneControlCommand(
            True, forward_speed, lateral_speed,
            requested_heading, error_y, error_angle,
            "ipm_lane_pid", steering_demand, target_forward_speed)

    def _speed_target(self, error_angle):
        """Map current steering demand to a bounded forward-speed target."""
        minimum = max(float(self.config.min_forward_speed), 0.0)
        maximum = max(float(self.config.max_forward_speed), minimum)
        reference = max(float(self.config.full_turn_reference), 1e-6)
        # Match lane_base(): speed is based on the model/PID-before heading
        # value, not on the amplified PID output.
        demand = float(np.clip(abs(float(error_angle)) / reference, 0.0, 1.0))
        exponent = max(float(self.config.speed_curve_exponent), 1e-6)
        target = minimum + (maximum - minimum) * ((1.0 - demand) ** exponent)
        return demand, float(np.clip(target, minimum, maximum))

    def _apply_speed_slew(self, target):
        """Reduce bend-entry speed faster than speed is restored after a bend."""
        target = float(target)
        if self._last_forward_speed is None:
            self._last_forward_speed = target
            return target
        current = float(self._last_forward_speed)
        if target < current:
            step = max(float(self.config.max_deceleration_step), 0.0)
        else:
            step = max(float(self.config.max_acceleration_step), 0.0)
        if step > 0.0:
            target = float(np.clip(target, current - step, current + step))
        self._last_forward_speed = target
        return target
