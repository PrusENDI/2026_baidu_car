"""Calibration helpers for the OpenCV data-collection teacher."""

from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class ErrorMapping:
    """Map geometric OpenCV measurements into the CNN output domain.

    The defaults intentionally preserve the raw values.  They are useful for
    offline inspection only and must be calibrated before closed-loop driving.
    Keeping this mapping explicit prevents the old single ``dev`` value from
    being mistaken for the new model's two-output contract.
    """

    lateral_scale: float = 1.0
    lateral_bias: float = 0.0
    heading_scale: float = 1.0
    heading_lateral_mix: float = 0.0
    heading_bias: float = 0.0

    def map(self, raw_lateral: float, raw_heading: float) -> Dict[str, float]:
        error_y = raw_lateral * self.lateral_scale + self.lateral_bias
        error_angle = (
            raw_heading * self.heading_scale
            + raw_lateral * self.heading_lateral_mix
            + self.heading_bias
        )
        return {
            "error_y": float(error_y),
            "error_angle": float(error_angle),
        }
