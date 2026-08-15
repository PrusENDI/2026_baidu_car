"""Traditional-vision lane teacher used only by data-collection tools.

This package deliberately does not export through :mod:`smartcar`.  Runtime
lane following must continue to use the CNN inference path.
"""

from .calibration import ErrorMapping
from .opencv_lane import (
    LaneAnalysisResult,
    LaneAnalyzerConfig,
    OpenCVLaneAnalyzer,
    StandardLaneReference,
)
from .pid_control import CvLaneControlCommand, CvLanePidConfig, CvLanePidController
from .turn_state import (
    CrossStraightConfig,
    CrossStraightDecision,
    CrossStraightState,
    CrossStraightStateMachine,
    SharpTurnStateMachine,
    TurnCommand,
    TurnState,
    TurnStateConfig,
)

__all__ = [
    "ErrorMapping",
    "LaneAnalysisResult",
    "LaneAnalyzerConfig",
    "OpenCVLaneAnalyzer",
    "StandardLaneReference",
    "CvLaneControlCommand",
    "CvLanePidConfig",
    "CvLanePidController",
    "SharpTurnStateMachine",
    "TurnCommand",
    "TurnState",
    "TurnStateConfig",
    "CrossStraightConfig",
    "CrossStraightDecision",
    "CrossStraightState",
    "CrossStraightStateMachine",
]
