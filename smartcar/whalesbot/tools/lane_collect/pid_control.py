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
    full_turn_reference: float = 0.40
    speed_curve_exponent: float = 1.5
    max_deceleration_step: float = 0.03
    max_acceleration_step: float = 0.005
    lateral_scale: float = 0.0
    heading_scale: float = -0.40
    lateral_kp: float = 0.0
    heading_kp: float = 1.0
    lateral_limit: float = 0.04
    heading_limit: float = 0.60
    max_heading_step: float = 0.04
    heading_ema_alpha: float = 0.35
    heading_deadband: float = 0.03
    startup_straight_distance_m: float = 0.10
    heading_onset_delay_m: float = 0.0
    perspective_kb_enabled: bool = False
    perspective_k_gain: float = 35.0
    perspective_b_gain: float = 35.0 / 600.0


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
            Kp=self.config.lateral_kp, Ki=0.0, Kd=0.0, setpoint=0.0,
            sample_time=None,
            output_limits=(-self.config.lateral_limit,
                           self.config.lateral_limit),
        )
        self.pid_angle = PID(
            Kp=self.config.heading_kp, Ki=0.0, Kd=0.0, setpoint=0.0,
            sample_time=None,
            output_limits=(-self.config.heading_limit,
                           self.config.heading_limit),
        )
        self._last_heading_output = 0.0
        self._filtered_heading_error = None
        self._heading_active_sign = 0
        self._pending_heading_sign = 0
        self._pending_heading_distance_m = None
        self._last_forward_speed = None

    def reset(self) -> None:
        self.pid_y.reset()
        self.pid_angle.reset()
        self._last_heading_output = 0.0
        self._filtered_heading_error = None
        self._heading_active_sign = 0
        self._pending_heading_sign = 0
        self._pending_heading_distance_m = None
        self._last_forward_speed = None

    def compute(self, result: LaneAnalysisResult,
                distance_m=None) -> CvLaneControlCommand:
        if (not result.valid or result.raw_lateral is None or
                result.raw_heading is None or
                not np.isfinite(result.raw_lateral) or
                not np.isfinite(result.raw_heading)):
            return CvLaneControlCommand(
                False, 0.0, 0.0, 0.0, 0.0, 0.0,
                result.reason or "invalid OpenCV lane result")

        error_y = float(result.raw_lateral * self.config.lateral_scale)
        control_heading = float(result.raw_heading)
        control_reason = ""
        if self.config.perspective_kb_enabled:
            metrics = result.metrics or {}
            perspective_k = metrics.get("perspective_k")
            perspective_b = metrics.get("perspective_b")
            if (perspective_k is not None and perspective_b is not None and
                    np.isfinite(perspective_k) and np.isfinite(perspective_b)):
                control_heading = float(
                    float(perspective_k) * self.config.perspective_k_gain +
                    float(perspective_b) * self.config.perspective_b_gain)
                control_reason = "perspective_kb"
        raw_error_angle = float(control_heading * self.config.heading_scale)
        alpha = float(np.clip(self.config.heading_ema_alpha, 0.0, 1.0))
        if self._filtered_heading_error is None:
            self._filtered_heading_error = raw_error_angle
        else:
            self._filtered_heading_error = float(
                alpha * raw_error_angle +
                (1.0 - alpha) * self._filtered_heading_error)
        error_angle = self._filtered_heading_error
        deadband = max(float(self.config.heading_deadband), 0.0)
        if abs(error_angle) < deadband:
            error_angle = 0.0
        steering_demand, target_forward_speed = self._speed_target(error_angle)
        forward_speed = self._apply_speed_slew(target_forward_speed)
        error_angle, startup_held = self._apply_startup_straight(
            error_angle, distance_m)
        if startup_held:
            control_reason = "startup_straight"
        error_angle = self._apply_heading_onset_delay(error_angle, distance_m)

        # Keep the same sign convention and control sequence as lane_base():
        # controller.get_out(-error_y, -error_angle).
        lateral_speed = float(self.pid_y(-error_y))
        requested_heading = float(self.pid_angle(-error_angle))
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
            control_reason, steering_demand, target_forward_speed)

    def _speed_target(self, error_angle):
        """Map current steering demand to a bounded forward-speed target."""
        minimum = max(float(self.config.min_forward_speed), 0.0)
        maximum = max(float(self.config.max_forward_speed), minimum)
        reference = max(float(self.config.full_turn_reference), 1e-6)
        proportional_request = abs(
            float(error_angle) * float(self.config.heading_kp))
        demand = float(np.clip(proportional_request / reference, 0.0, 1.0))
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

    def _apply_startup_straight(self, error_angle, distance_m):
        """Hold only the beginning of a collection session straight."""
        distance = max(float(self.config.startup_straight_distance_m), 0.0)
        if (distance == 0.0 or distance_m is None or
                not np.isfinite(distance_m)):
            return error_angle, False
        if max(float(distance_m), 0.0) < distance:
            return 0.0, True
        return error_angle, False

    def _apply_heading_onset_delay(self, error_angle, distance_m):
        """Delay each new steering direction by a measured travel distance."""
        sign = 1 if error_angle > 0.0 else (-1 if error_angle < 0.0 else 0)
        if sign == 0:
            self._heading_active_sign = 0
            self._pending_heading_sign = 0
            self._pending_heading_distance_m = None
            return 0.0
        delay = max(float(self.config.heading_onset_delay_m), 0.0)
        if delay == 0.0 or distance_m is None or not np.isfinite(distance_m):
            self._heading_active_sign = sign
            return error_angle
        distance_m = max(float(distance_m), 0.0)
        if self._heading_active_sign == sign:
            return error_angle
        if self._pending_heading_sign != sign:
            self._heading_active_sign = 0
            self._pending_heading_sign = sign
            self._pending_heading_distance_m = distance_m
            return 0.0
        traveled = max(distance_m - self._pending_heading_distance_m, 0.0)
        if traveled < delay:
            return 0.0
        self._heading_active_sign = sign
        self._pending_heading_sign = 0
        self._pending_heading_distance_m = None
        return error_angle
