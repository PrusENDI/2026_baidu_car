"""Low-speed PID control used only by the OpenCV collection-time teacher."""

from dataclasses import dataclass
from typing import Dict

import numpy as np

from ..tools_class import PID
from .opencv_lane import LaneAnalysisResult


@dataclass(frozen=True)
class CvLanePidConfig:
    steering_mode: str = "heading_pid"
    max_forward_speed: float = 0.30
    min_forward_speed: float = 0.12
    # The launch segment is the only route-position exception: keep the
    # chassis straight for a short odometry distance before the first bend.
    # The generic lane logic must not reuse this as a per-turn delay.
    initial_straight_distance_m: float = 0.15
    full_turn_reference: float = 0.60
    full_turn_curvature_m_inv: float = 5.0
    speed_curve_exponent: float = 1.5
    max_deceleration_step: float = 0.015
    max_acceleration_step: float = 0.010
    control_period_s: float = 0.05
    lateral_kp: float = 6.0
    lateral_ki: float = 0.0
    lateral_kd: float = 0.1
    heading_kp: float = 1.95
    heading_ki: float = 0.0
    heading_kd: float = 0.0
    # Keep CV collection translation disabled until the real-car vy sign and
    # recovery gain have been verified independently on a straight section.
    lateral_limit: float = 0.0
    heading_limit: float = 1.50
    pure_pursuit_gain: float = 1.0
    max_heading_step: float = 0.04
    pure_pursuit_entry_step: float = 0.10
    max_heading_release_step: float = 0.04
    # Release an obsolete turn direction promptly without changing the
    # smoother same-direction decay used by ordinary bends.
    max_heading_reverse_step: float = 0.10


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
        if self.config.steering_mode == "pure_pursuit":
            return self._compute_pure_pursuit(result, error_y)
        if self.config.steering_mode != "heading_pid":
            return CvLaneControlCommand(
                False, 0.0, 0.0, 0.0, 0.0, 0.0,
                f"unknown steering mode: {self.config.steering_mode}")
        error_angle = float(result.error_angle)
        steering_demand, target_forward_speed = self._speed_target(error_angle)
        forward_speed = self._apply_speed_slew(target_forward_speed)

        # Keep the same sign convention and control sequence as lane_base():
        # controller.get_out(-error_y, -error_angle).
        dt = max(float(self.config.control_period_s), 1e-6)
        lateral_speed = float(self.pid_y(-error_y, dt=dt))
        requested_heading = float(self.pid_angle(-error_angle, dt=dt))
        requested_heading = self._apply_heading_slew(requested_heading)
        return CvLaneControlCommand(
            True, forward_speed, lateral_speed,
            requested_heading, error_y, error_angle,
            "ipm_lane_pid", steering_demand, target_forward_speed)

    def _compute_pure_pursuit(self, result, error_y):
        curvature = result.metrics.get("pure_pursuit_curvature_m_inv")
        if curvature is None or not np.isfinite(curvature):
            return CvLaneControlCommand(
                False, 0.0, 0.0, 0.0, 0.0, 0.0,
                "pure pursuit curvature unavailable")
        curvature = float(curvature) * float(self.config.pure_pursuit_gain)
        right_turn_override = bool(result.metrics.get(
            "route_right_turn_override", False))
        reason = ("route_right_turn" if right_turn_override
                  else "ipm_pure_pursuit")
        reference = max(float(self.config.full_turn_curvature_m_inv), 1e-6)
        demand = float(np.clip(abs(curvature) / reference, 0.0, 1.0))
        target_forward_speed = self._speed_target_from_demand(demand)
        forward_speed = self._apply_speed_slew(target_forward_speed)
        requested_heading = float(np.clip(
            forward_speed * curvature,
            -float(self.config.heading_limit),
            float(self.config.heading_limit)))
        if right_turn_override:
            # The stateless geometric override is already bounded by both
            # curvature and yaw-rate limits.  Apply it directly so its turn
            # radius does not depend on how many video frames hit the gate.
            self._last_heading_output = requested_heading
        else:
            requested_heading = self._apply_heading_slew(
                requested_heading,
                entry_step=self.config.pure_pursuit_entry_step)

        # Preserve the existing CNN/PID contract.  This label is the
        # single-frame equivalent input which the P-only heading controller
        # converts back to the steady-state pure-pursuit yaw-rate request.
        heading_kp = max(abs(float(self.config.heading_kp)), 1e-6)
        label_wz = float(np.clip(
            target_forward_speed * curvature,
            -float(self.config.heading_limit),
            float(self.config.heading_limit)))
        error_angle = -label_wz / heading_kp
        dt = max(float(self.config.control_period_s), 1e-6)
        lateral_speed = float(self.pid_y(-error_y, dt=dt))
        return CvLaneControlCommand(
            True, forward_speed, lateral_speed, requested_heading,
            error_y, error_angle, reason, demand,
            target_forward_speed)

    def _speed_target(self, error_angle):
        """Map current steering demand to a bounded forward-speed target."""
        minimum = max(float(self.config.min_forward_speed), 0.0)
        maximum = max(float(self.config.max_forward_speed), minimum)
        reference = max(float(self.config.full_turn_reference), 1e-6)
        # Match lane_base(): speed is based on the model/PID-before heading
        # value, not on the amplified PID output.
        demand = float(np.clip(abs(float(error_angle)) / reference, 0.0, 1.0))
        target = self._speed_target_from_demand(demand)
        return demand, float(np.clip(target, minimum, maximum))

    def _speed_target_from_demand(self, demand):
        minimum = max(float(self.config.min_forward_speed), 0.0)
        maximum = max(float(self.config.max_forward_speed), minimum)
        demand = float(np.clip(demand, 0.0, 1.0))
        exponent = max(float(self.config.speed_curve_exponent), 1e-6)
        target = minimum + (maximum - minimum) * ((1.0 - demand) ** exponent)
        return float(np.clip(target, minimum, maximum))

    def _apply_heading_slew(self, requested, entry_step=None,
                            reverse_step=None):
        requested = float(requested)
        previous = float(self._last_heading_output)
        same_direction = requested == 0.0 or previous == 0.0 or \
            np.sign(requested) == np.sign(previous)
        reversing = not same_direction
        building_turn = same_direction and abs(requested) > abs(previous)
        if reversing:
            if reverse_step is None:
                reverse_step = self.config.max_heading_reverse_step
            step = max(float(reverse_step), 0.0)
        elif building_turn:
            if entry_step is None:
                entry_step = self.config.max_heading_step
            step = max(float(entry_step), 0.0)
        else:
            step = max(float(self.config.max_heading_release_step), 0.0)
        if step > 0.0:
            requested = float(np.clip(
                requested, previous - step, previous + step))
        self._last_heading_output = requested
        return requested

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
