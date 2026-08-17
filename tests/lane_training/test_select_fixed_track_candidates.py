from tools.lane_training.select_fixed_track_candidates import shortlist_candidates


def _report(epoch, direction, false_steer, cross_mae, speed_mae, cv_kappa):
    return {
        "epoch": epoch,
        "checkpoint": f"epoch_{epoch:04d}",
        "checkpoint_sha256": str(epoch),
        "sets": {
            "cross_train": {
                "kappa_mae": cross_mae,
                "sequence": {
                    "direction_accuracy": direction,
                    "active_recall": direction,
                    "false_steer_rate": false_steer,
                },
            },
            "cv_validation": {
                "speed_mae": speed_mae,
                "kappa_mae": cv_kappa,
            },
        },
    }


def test_shortlist_keeps_three_roles_without_selecting_a_deployment():
    reports = [
        _report(10, 0.95, 0.10, 0.30, 0.20, 0.40),
        _report(100, 0.99, 0.08, 0.20, 0.15, 0.35),
        _report(180, 0.97, 0.09, 0.22, 0.05, 0.18),
    ]
    result = shortlist_candidates(reports)
    assert result["roles"]["crossroad_best"]["epoch"] == 100
    assert result["roles"]["cv_preservation_best"]["epoch"] == 180
    assert result["selected"] is False
    assert result["vehicle_test_performed"] is False
    assert 1 <= len(result["candidates"]) <= 3
