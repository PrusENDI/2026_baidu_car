import unittest

from smartcar.whalesbot.tools.lane_collect import (
    LaneAnalysisResult,
    SharpTurnStateMachine,
    TurnState,
    TurnStateConfig,
)


def observation(valid=True, heading=0.0, lateral=0.0, mode="both", **metrics):
    values = {"tracking_mode": mode, **metrics}
    return LaneAnalysisResult(
        valid=valid,
        error_y=lateral if valid else None,
        error_angle=heading if valid else None,
        raw_lateral=lateral if valid else None,
        raw_heading=heading if valid else None,
        confidence=1.0 if valid else 0.0,
        cross_status="normal",
        cross_score=0.0,
        reason=None if valid else "missing",
        work_size=(320, 240),
        metrics=values,
    )


class SharpTurnStateMachineTests(unittest.TestCase):
    def setUp(self):
        self.machine = SharpTurnStateMachine(TurnStateConfig(
            confirm_frames=2, exit_confirm_frames=2,
            recovery_frames=2, max_missing_frames=3))

    def test_confirmed_corner_holds_direction_during_boundary_loss(self):
        corner = observation(
            valid=False, corner_detected=True, corner_score=0.9,
            corner_direction_known=True, corner_heading=1.2)
        first = self.machine.update(corner)
        second = self.machine.update(corner)
        third = self.machine.update(observation(valid=False))

        self.assertFalse(first.valid)
        self.assertEqual(first.state, TurnState.CANDIDATE_RIGHT.value)
        self.assertTrue(second.valid)
        self.assertEqual(second.state, TurnState.TURNING_RIGHT.value)
        self.assertGreater(second.raw_heading, 0.0)
        self.assertEqual(third.source, "turn_hold")
        self.assertGreater(third.raw_heading, 0.0)

    def test_exact_horizontal_line_without_history_does_not_guess_direction(self):
        ambiguous = observation(
            valid=False, corner_detected=True, corner_score=0.9,
            corner_direction_known=False, corner_heading=0.0)
        command = self.machine.update(ambiguous)
        self.assertFalse(command.valid)
        self.assertEqual(command.state, TurnState.NORMAL.value)

    def test_prior_trend_resolves_ambiguous_horizontal_line(self):
        self.machine.update(observation(valid=True, heading=0.25, mode="left_only"))
        ambiguous = observation(
            valid=False, corner_detected=True, corner_score=0.9,
            corner_direction_known=False, corner_heading=0.0)
        self.machine.update(ambiguous)
        command = self.machine.update(ambiguous)
        self.assertTrue(command.valid)
        self.assertEqual(command.state, TurnState.TURNING_RIGHT.value)

    def test_turn_exits_only_after_stable_two_side_observations(self):
        sharp = observation(valid=True, heading=-0.9, mode="right_only")
        self.machine.update(sharp)
        self.machine.update(sharp)
        self.assertEqual(self.machine.state, TurnState.TURNING_LEFT)
        self.machine.update(observation(valid=True, heading=0.05, mode="both"))
        command = self.machine.update(observation(valid=True, heading=0.04, mode="both"))
        self.assertEqual(command.state, TurnState.RECOVERING.value)


if __name__ == "__main__":
    unittest.main()
