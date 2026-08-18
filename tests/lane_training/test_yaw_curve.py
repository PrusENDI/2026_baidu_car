import numpy as np
import paddle

from lane_training.yaw_curve import build_curve_windows, discover_turns, yaw_curve_loss


def test_discover_turns_joins_short_gaps_and_keeps_signs_separate():
    yaw = np.array([0.0, 0.06, 0.07, 0.01, 0.06, 0.07, 0.0, -0.06, -0.07, 0.0])

    assert discover_turns(yaw, max_gap=2, min_active=2) == [(1, 5, 1), (7, 8, -1)]


def test_build_curve_windows_adds_context_around_turn():
    rows = [
        {"yaw": value, "frame": f"{index:04d}"}
        for index, value in enumerate([0.0, 0.0, 0.06, 0.10, 0.12, 0.10, 0.06, 0.0])
    ]

    windows = build_curve_windows(rows, context=1)

    assert [row["frame"] for row in windows[0]] == [
        "0001", "0002", "0003", "0004", "0005", "0006", "0007"
    ]


def test_yaw_curve_loss_provides_only_second_output_gradient():
    prediction = paddle.to_tensor([[0.1, 0.05], [0.1, 0.20], [0.1, 0.10]], dtype="float32")
    prediction.stop_gradient = False
    target = paddle.to_tensor([[0.0, 0.10], [0.0, 0.20], [0.0, 0.15]], dtype="float32")

    total, _ = yaw_curve_loss(prediction, target, [0, 0, 0])
    total.backward()

    gradient = prediction.grad.numpy()
    assert np.allclose(gradient[:, 0], 0.0)
    assert np.any(np.abs(gradient[:, 1]) > 0.0)
