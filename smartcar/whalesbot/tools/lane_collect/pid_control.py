"""Low-speed PID control used only by the OpenCV collection-time teacher."""

from dataclasses import dataclass
from typing import Dict

import numpy as np

from ..tools_class import PID
from .opencv_lane import LaneAnalysisResult


@dataclass(frozen=True)
class CvLanePidConfig:
    forward_speed: float = 0.05
    lateral_scale: float = 0.0
    heading_scale: float = -0.316
    lateral_kp: float = 0.0
    heading_kp: float = 1.0
    lateral_limit: float = 0.04
    heading_limit: float = 0.60
    max_heading_step: float = 0.04
    heading_ema_alpha: float = 0.35
    heading_deadband: float = 0.03


@dataclass(frozen=True)
class CvLaneControlCommand:
    valid: bool
    forward_speed: float
    lateral_speed: float
    angular_speed: float
    error_y: float
    error_angle: float
    reason: str = ""

    def to_record(self) -> Dict[str, float]:
        return {
            "valid": bool(self.valid),
            "forward_speed": float(self.forward_speed),
            "lateral_speed": float(self.lateral_speed),
            "angular_speed": float(self.angular_speed),
            "error_y": float(self.error_y),
            "error_angle": float(self.error_angle),
            "reason": self.reason,
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

    def reset(self) -> None:
        self.pid_y.reset()
        self.pid_angle.reset()
        self._last_heading_output = 0.0
        self._filtered_heading_error = None

    def compute(self, result: LaneAnalysisResult) -> CvLaneControlCommand:
        if (not result.valid or result.raw_lateral is None or
                result.raw_heading is None or
                not np.isfinite(result.raw_lateral) or
                not np.isfinite(result.raw_heading)):
            return CvLaneControlCommand(
                False, 0.0, 0.0, 0.0, 0.0, 0.0,
                result.reason or "invalid OpenCV lane result")

        error_y = float(result.raw_lateral * self.config.lateral_scale)
        raw_error_angle = float(result.raw_heading * self.config.heading_scale)
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
            True, float(self.config.forward_speed), lateral_speed,
            requested_heading, error_y, error_angle)
