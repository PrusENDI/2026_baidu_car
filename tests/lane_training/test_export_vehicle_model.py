from tools.lane_training.export_vehicle_model import deployment_metadata


def test_deployment_metadata_records_manual_risk_acceptance():
    metadata = deployment_metadata(
        source_checkpoint="new_track_aggressive_r2/epoch_2.pdparams",
        checkpoint_sha256="abc",
        jit_differences={"batch_1": 0.0, "batch_32": 1e-8},
        inference_differences={"cpu_batch_1": 1e-6, "gpu_batch_1": 0.0},
        file_hashes={"cnn_lane.pdmodel": "def"},
    )

    assert metadata["input_shape"] == [None, 3, 128, 128]
    assert metadata["source_checkpoint"] == "new_track_aggressive_r2/epoch_2.pdparams"
    assert metadata["user_risk_acceptance"] is True
    assert metadata["selected"] is False
    assert metadata["locked_test_performed"] is False
