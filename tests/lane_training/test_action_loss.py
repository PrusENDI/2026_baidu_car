import paddle

from lane_training.action_loss import masked_smooth_l1_loss


def test_manual_mask_removes_speed_target_from_loss():
    prediction = paddle.to_tensor([[100.0, 2.0]], dtype="float32")
    target = paddle.to_tensor([[0.0, 0.0]], dtype="float32")
    mask = paddle.to_tensor([[0.0, 1.0]], dtype="float32")

    loss, parts = masked_smooth_l1_loss(prediction, target, mask)

    assert float(loss) == float(parts["kappa"])
    assert float(parts["valid_speed"]) == 0.0
    assert float(parts["valid_kappa"]) == 1.0


def test_cv_mask_trains_both_outputs():
    prediction = paddle.to_tensor([[1.0, 1.0]], dtype="float32")
    target = paddle.zeros([1, 2], dtype="float32")
    mask = paddle.ones([1, 2], dtype="float32")

    loss, parts = masked_smooth_l1_loss(prediction, target, mask)

    assert float(loss) > 0.0
    assert float(parts["valid_speed"]) == 1.0
    assert float(parts["valid_kappa"]) == 1.0
