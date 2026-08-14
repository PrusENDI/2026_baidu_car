"""Temporal sharp-turn handling for the data-collection OpenCV teacher."""

from dataclasses import dataclass
from enum import Enum
from typing import Optional

from .calibration import ErrorMapping
from .opencv_lane import LaneAnalysisResult


class TurnState(str, Enum):
    NORMAL = "normal"
    CANDIDATE_LEFT = "candidate_left"
    CANDIDATE_RIGHT = "candidate_right"
    TURNING_LEFT = "turning_left"
    TURNING_RIGHT = "turning_right"
    RECOVERING = "recovering"


@dataclass(frozen=True)
class TurnStateConfig:
    enter_heading: float = 0.48
    exit_heading: float = 0.22
    held_heading: float = 0.65
    prior_direction_heading: float = 0.16
    corner_min_score: float = 0.55
    confirm_frames: int = 2
    exit_confirm_frames: int = 4
    recovery_frames: int = 3
    max_missing_frames: int = 8


@dataclass(frozen=True)
class TurnCommand:
    valid: bool
    error_y: Optional[float]
    error_angle: Optional[float]
    raw_lateral: Optional[float]
    raw_heading: Optional[float]
    state: str
    source: str
    reason: Optional[str] = None


class SharpTurnStateMachine:
    """Confirm a turn across frames and bridge short complete boundary loss.

    Positive image heading is named ``right`` and negative heading ``left``.
    Final vehicle sign and scale still belong to :class:`ErrorMapping`.
    """

    def __init__(self, config: Optional[TurnStateConfig] = None,
                 error_mapping: Optional[ErrorMapping] = None) -> None:
        self.config = config or TurnStateConfig()
        self.error_mapping = error_mapping or ErrorMapping()
        self.reset()

    def reset(self) -> None:
        self.state = TurnState.NORMAL
        self._candidate_direction = 0
        self._candidate_frames = 0
        self._turn_direction = 0
        self._exit_frames = 0
        self._recovery_frames = 0
        self._missing_frames = 0
        self._opposite_frames = 0
        self._last_lateral = 0.0
        self._last_heading = 0.0

    def update(self, result: LaneAnalysisResult) -> TurnCommand:
        direction = self._observed_direction(result)
        if result.valid:
            self._last_lateral = float(result.raw_lateral)
            self._last_heading = float(result.raw_heading)

        if self.state == TurnState.NORMAL:
            if direction:
                self._start_candidate(direction)
            return self._from_result(result, "opencv")

        if self.state in (TurnState.CANDIDATE_LEFT, TurnState.CANDIDATE_RIGHT):
            if direction == self._candidate_direction:
                self._candidate_frames += 1
                if self._candidate_frames >= max(1, self.config.confirm_frames):
                    self._enter_turn(direction)
                    if result.valid:
                        return self._from_result(result, "confirmed_opencv")
                    return self._held_turn("confirmed corner while boundaries are missing")
            elif direction:
                self._start_candidate(direction)
            else:
                self.state = TurnState.NORMAL
                self._candidate_direction = 0
                self._candidate_frames = 0
            return self._from_result(result, "candidate")

        if self.state in (TurnState.TURNING_LEFT, TurnState.TURNING_RIGHT):
            mode = result.metrics.get("tracking_mode", "none")
            recovered = (result.valid and mode in ("both", "mixed") and
                         abs(float(result.raw_heading)) <= self.config.exit_heading)
            self._exit_frames = self._exit_frames + 1 if recovered else 0
            if self._exit_frames >= max(1, self.config.exit_confirm_frames):
                self.state = TurnState.RECOVERING
                self._recovery_frames = 0
                return self._from_result(result, "turn_exit")

            if direction == self._turn_direction:
                self._missing_frames = 0
                self._opposite_frames = 0
                if result.valid:
                    return self._from_result(result, "turn_opencv")
                return self._held_turn("corner observation sustains confirmed turn")

            if direction == -self._turn_direction:
                self._opposite_frames += 1
                if self._opposite_frames >= max(1, self.config.confirm_frames):
                    self._enter_turn(direction)
                    if result.valid:
                        return self._from_result(result, "corrected_turn_opencv")
                    return self._held_turn("confirmed correction of turn direction")
                if result.valid:
                    return self._from_result(result, "opposite_candidate_opencv")
            else:
                self._opposite_frames = 0

            heading_agrees = (result.valid and result.raw_heading is not None and
                              float(result.raw_heading) * self._turn_direction >= 0)
            if heading_agrees:
                self._missing_frames = 0
                return self._from_result(result, "turn_opencv")

            self._missing_frames += 1
            if self._missing_frames <= max(0, self.config.max_missing_frames):
                return self._held_turn("temporarily holding confirmed turn direction")
            self.reset()
            return TurnCommand(False, None, None, None, None, self.state.value,
                               "lost", "turn boundaries missing for too many frames")

        # RECOVERING
        if direction:
            self._start_candidate(direction)
            return self._from_result(result, "recovery_candidate")
        if result.valid:
            self._recovery_frames += 1
            if self._recovery_frames >= max(1, self.config.recovery_frames):
                self.state = TurnState.NORMAL
        return self._from_result(result, "recovery")

    def _observed_direction(self, result: LaneAnalysisResult) -> int:
        if result.valid and result.raw_heading is not None and \
                abs(float(result.raw_heading)) >= self.config.enter_heading:
            return 1 if result.raw_heading > 0 else -1

        metrics = result.metrics
        if not metrics.get("corner_detected", False) or \
                float(metrics.get("corner_score", 0.0)) < self.config.corner_min_score:
            return 0
        if metrics.get("corner_direction_known", False):
            heading = float(metrics.get("corner_heading", 0.0))
            if abs(heading) >= self.config.enter_heading:
                return 1 if heading > 0 else -1
        if abs(self._last_heading) >= self.config.prior_direction_heading:
            return 1 if self._last_heading > 0 else -1
        return 0

    def _start_candidate(self, direction: int) -> None:
        self._candidate_direction = 1 if direction > 0 else -1
        self._candidate_frames = 1
        self.state = (TurnState.CANDIDATE_RIGHT if direction > 0
                      else TurnState.CANDIDATE_LEFT)

    def _enter_turn(self, direction: int) -> None:
        self._turn_direction = 1 if direction > 0 else -1
        self._missing_frames = 0
        self._opposite_frames = 0
        self._exit_frames = 0
        self.state = (TurnState.TURNING_RIGHT if direction > 0
                      else TurnState.TURNING_LEFT)

    def _held_turn(self, reason: str) -> TurnCommand:
        raw_heading = self._turn_direction * abs(self.config.held_heading)
        mapped = self.error_mapping.map(self._last_lateral, raw_heading)
        return TurnCommand(True, mapped["error_y"], mapped["error_angle"],
                           self._last_lateral, raw_heading, self.state.value,
                           "turn_hold", reason)

    def _from_result(self, result: LaneAnalysisResult, source: str) -> TurnCommand:
        return TurnCommand(result.valid, result.error_y, result.error_angle,
                           result.raw_lateral, result.raw_heading,
                           self.state.value, source, result.reason)
