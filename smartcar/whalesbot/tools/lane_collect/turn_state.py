"""Temporal sharp-turn handling for the data-collection OpenCV teacher."""

from dataclasses import dataclass, replace
from enum import Enum
from typing import Optional

import numpy as np

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
            two_sided = mode in ("both", "mixed")
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

            # A single visible boundary cannot disprove the locked direction.
            # Only a two-sided observation may confirm an opposite direction.
            if direction == -self._turn_direction and two_sided:
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
            if heading_agrees or not two_sided:
                self._missing_frames = 0
                if heading_agrees:
                    return self._from_result(result, "turn_opencv")
                return self._held_turn("single boundary holds locked turn direction")

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


class CrossStraightState(str, Enum):
    NORMAL = "normal"
    CURVE_CONTINUE = "curve_continue"
    CROSS_STRAIGHT = "cross_straight"
    WAIT_SECOND_CROSS = "wait_second_cross"
    SECOND_CROSS_STRAIGHT = "second_cross_straight"
    DONE = "done"


@dataclass(frozen=True)
class CrossStraightConfig:
    """Route distances for the two known crossings, measured from ``start``."""

    corner_min_score: float = 0.52
    anomaly_heading: float = 0.75
    curve_continue_distance_m: float = 0.20
    cross_straight_distance_m: float = 0.14
    heading_decay_distance_m: float = 0.05
    cross_speed_scale: float = 0.65
    history_size: int = 8
    entry_lateral_limit: float = 0.35
    entry_heading_limit: float = 0.85
    first_trigger_distance_m: float = 3.50
    second_min_trigger_frame: int = 1100
    second_cross_straight_frames: int = 20
    second_heading_decay_frames: int = 8


@dataclass(frozen=True)
class CrossStraightDecision:
    result: LaneAnalysisResult
    state: str
    speed_scale: float
    source: str


class CrossStraightStateMachine:
    """One-shot bend continuation followed by straight-through crossing.

    This is intentionally route-specific: after the first suspicious
    horizontal boundary, it continues the prior bend for a bounded number of
    distance, then gradually removes steering and holds the entry lateral pose.
    Once the crossing window is complete it never triggers again.
    """

    def __init__(self, config: CrossStraightConfig = None) -> None:
        self.config = config or CrossStraightConfig()
        self.reset()

    def reset(self) -> None:
        self.state = CrossStraightState.NORMAL
        self._history = []
        self._entry_lateral = 0.0
        self._entry_heading = 0.0
        self._curve_frames = 0
        self._cross_frames = 0
        self._second_cross_frames = 0
        self._frame_count = 0
        self._state_entry_distance_m = None
        self._distance_m = None

    def update(self, result: LaneAnalysisResult,
               distance_m=None) -> CrossStraightDecision:
        self._frame_count += 1
        if distance_m is not None and np.isfinite(distance_m):
            self._distance_m = max(0.0, float(distance_m))
        self._remember(result)
        if self.state == CrossStraightState.DONE:
            return CrossStraightDecision(result, self.state.value, 1.0, "normal_after_cross")

        if self.state == CrossStraightState.WAIT_SECOND_CROSS:
            if (self._frame_count >= self.config.second_min_trigger_frame and
                    not result.valid):
                self._capture_entry_pose()
                self.state = CrossStraightState.SECOND_CROSS_STRAIGHT
                self._second_cross_frames = 0
                return self._held_result(
                    result, self._entry_lateral, self._entry_heading,
                    self.state.value, "second_cross_straight")
            return CrossStraightDecision(result, self.state.value, 1.0,
                                         "standard_wait_second_cross")

        if self.state == CrossStraightState.NORMAL:
            if self._is_first_cross_approach(result):
                self._capture_entry_pose()
                self.state = CrossStraightState.CURVE_CONTINUE
                self._curve_frames = 0
                self._mark_state_entry_distance()
                return self._held_result(result, self._entry_lateral,
                                         self._entry_heading,
                                         self.state.value, "curve_continue_entry")
            return CrossStraightDecision(result, self.state.value, 1.0, "standard")

        if self.state == CrossStraightState.CURVE_CONTINUE:
            self._curve_frames += 1
            if self._state_distance() >= self.config.curve_continue_distance_m:
                self.state = CrossStraightState.CROSS_STRAIGHT
                self._cross_frames = 0
                self._mark_state_entry_distance()
            else:
                return self._held_result(result, self._entry_lateral,
                                         self._entry_heading,
                                         self.state.value, "curve_continue_hold")

        if self.state == CrossStraightState.CROSS_STRAIGHT:
            self._cross_frames += 1
            distance = self._state_distance()
            decay = min(distance /
                        max(self.config.heading_decay_distance_m, 1e-6), 1.0)
            heading = self._entry_heading * (1.0 - decay)
            if distance >= self.config.cross_straight_distance_m:
                self.state = CrossStraightState.WAIT_SECOND_CROSS
            return self._held_result(result, self._entry_lateral, heading,
                                     self.state.value, "cross_straight")

        if self.state == CrossStraightState.SECOND_CROSS_STRAIGHT:
            self._second_cross_frames += 1
            decay = min(self._second_cross_frames /
                        max(1, self.config.second_heading_decay_frames), 1.0)
            heading = self._entry_heading * (1.0 - decay)
            if self._second_cross_frames >= max(
                    1, self.config.second_cross_straight_frames):
                self.state = CrossStraightState.DONE
            return self._held_result(
                result, self._entry_lateral, heading, self.state.value,
                "second_cross_straight")

        return CrossStraightDecision(result, self.state.value, 1.0, "standard")

    def _remember(self, result: LaneAnalysisResult) -> None:
        if not result.valid or result.raw_lateral is None or result.raw_heading is None:
            return
        self._history.append((float(result.raw_lateral), float(result.raw_heading)))
        del self._history[:-max(1, self.config.history_size)]

    def _is_first_cross_approach(self, result: LaneAnalysisResult) -> bool:
        if not result.valid or len(self._history) < 3:
            return False
        if not self._distance_at_least(self.config.first_trigger_distance_m):
            return False
        metrics = result.metrics
        if metrics.get("tracking_mode") not in ("left_only", "right_only"):
            return False
        if not metrics.get("corner_detected", False):
            return False
        if float(metrics.get("corner_score", 0.0)) < self.config.corner_min_score:
            return False
        return abs(float(result.raw_heading or 0.0)) >= self.config.anomaly_heading

    def _distance_at_least(self, threshold) -> bool:
        return (self._distance_m is not None and
                self._distance_m >= max(float(threshold), 0.0))

    def _mark_state_entry_distance(self) -> None:
        self._state_entry_distance_m = self._distance_m

    def _state_distance(self) -> float:
        if self._distance_m is None or self._state_entry_distance_m is None:
            return 0.0
        return max(0.0, self._distance_m - self._state_entry_distance_m)

    def _capture_entry_pose(self) -> None:
        values = self._history[:-1] if len(self._history) > 1 else self._history
        lateral = [item[0] for item in values]
        heading = [item[1] for item in values]
        self._entry_lateral = float(np.median(lateral))
        self._entry_heading = float(np.clip(
            np.median(heading), -self.config.entry_heading_limit,
            self.config.entry_heading_limit))
        self._entry_lateral = float(np.clip(
            self._entry_lateral, -self.config.entry_lateral_limit,
            self.config.entry_lateral_limit))

    def _held_result(self, result, lateral, heading, state, source):
        held = replace(result, valid=True, reason=None,
                       raw_lateral=float(lateral), raw_heading=float(heading))
        return CrossStraightDecision(held, state,
                                     (self.config.cross_speed_scale
                                      if "cross_straight" in source else 0.85),
                                     source)
