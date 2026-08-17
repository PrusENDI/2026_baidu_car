"""Shared speed/curvature post-processing for CV teachers and CNN students."""

from __future__ import annotations

import math
from typing import Optional


def finite_dt(dt_s: Optional[float], fallback_s: float = 0.05) -> float:
    """Return a safe wall-clock interval for control-rate calculations."""
    try:
        value = float(dt_s) if dt_s is not None else float(fallback_s)
    except (TypeError, ValueError):
        value = float(fallback_s)
    if not math.isfinite(value) or value <= 0.0:
        value = float(fallback_s)
    return max(value, 1e-4)


class CurvatureSpeedController:
    """Convert a curvature action to bounded vx/wz with real-time slew."""

    def __init__(self, *, max_speed=0.30, min_speed=0.12,
                 full_turn_curvature=5.0, speed_exponent=1.5,
                 deceleration_rate=0.194, acceleration_rate=0.129,
                 max_angular_speed=1.50, fallback_dt=0.05):
        self.max_speed = max(float(max_speed), 0.0)
        self.min_speed = max(float(min_speed), 0.0)
        self.full_turn_curvature = max(abs(float(full_turn_curvature)), 1e-6)
        self.speed_exponent = max(float(speed_exponent), 1e-6)
        self.deceleration_rate = max(float(deceleration_rate), 0.0)
        self.acceleration_rate = max(float(acceleration_rate), 0.0)
        self.max_angular_speed = max(float(max_angular_speed), 0.0)
        self.fallback_dt = max(float(fallback_dt), 1e-4)
        self._last_speed = None

    def reset(self):
        self._last_speed = None

    def target_speed(self, curvature):
        demand = min(abs(float(curvature)) / self.full_turn_curvature, 1.0)
        return self.target_speed_from_demand(demand)

    def target_speed_from_demand(self, demand):
        """Map a normalized teacher speed demand to a tunable target vx."""
        demand = max(0.0, min(1.0, float(demand)))
        target = self.min_speed + (self.max_speed - self.min_speed) * (
            (1.0 - demand) ** self.speed_exponent)
        return max(self.min_speed, min(self.max_speed, target))

    def slew_speed(self, target, dt_s=None):
        target = float(target)
        if self._last_speed is None:
            self._last_speed = target
            return target
        dt = finite_dt(dt_s, self.fallback_dt)
        rate = (self.deceleration_rate if target < self._last_speed
                else self.acceleration_rate)
        step = rate * dt
        self._last_speed = max(self._last_speed - step,
                               min(self._last_speed + step, target))
        return self._last_speed

    def command(self, curvature, dt_s=None, speed_demand=None):
        curvature = float(curvature)
        target = (self.target_speed(curvature) if speed_demand is None else
                  self.target_speed_from_demand(speed_demand))
        vx = self.slew_speed(target, dt_s)
        wz = max(-self.max_angular_speed,
                 min(self.max_angular_speed, vx * curvature))
        return vx, 0.0, wz, target
