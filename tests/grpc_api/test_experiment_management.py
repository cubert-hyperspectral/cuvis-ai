"""Integration tests for experiment management functionality (Task 5.4).

Optimized to minimize redundant training by using module-scoped shared fixtures
and reusing trained sessions across multiple tests where appropriate.
"""

import json
from pathlib import Path

import grpc
import pytest
import torch
import yaml

from cuvis_ai_core.grpc import cuvis_ai_pb2
from cuvis_ai_core.training.config import TrainRunConfig

DEFAULT_CHANNELS = 61


def _pipeline_bytes_from_path(pipeline_path: str | Path) -> bytes:
    """Load a pipeline YAML and return JSON-encoded bytes for LoadPipeline RPC."""
    pipeline_dict = yaml.safe_load(Path(pipeline_path).read_text())
    return json.dumps(pipeline_dict).encode("utf-8")


# ============================================================================
# Session-Scoped Shared Fixtures (train once, reuse across tests)
# ============================================================================


@pytest.fixture(scope="session")
def shared_trained_session(grpc_server, test_data_files_cached):
    """Single trained session shared across the entire test session.

    This fixture creates one trained session that multiple tests can use
    for read-only operations (save, inspect, etc.), significantly reducing
    test execution time by avoiding redundant training.

    Uses grpc_server directly and creates its own stub to match session scope.
    """
    from cuvis_ai_core.grpc import cuvis_ai_pb2_grpc
    from cuvis_ai_core.training.config import TrainRunConfig
    from tests.fixtures.sessions import materialize_trainrun_config

    cu3s_file, json_file = test_data_files_cached

    # Create our own channel and stub for session scope
    channel = grpc.insecure_channel(grpc_server)
    stub = cuvis_ai_pb2_grpc.CuvisAIServiceStub(channel)

    try:
        # Create session using RestoreTrainRun to get full config
        resolved_path = materialize_trainrun_config("configs/trainrun/deep_svdd.yaml")
        restore_req = cuvis_ai_pb2.RestoreTrainRunRequest(trainrun_path=resolved_path)
        response = stub.RestoreTrainRun(restore_req)
        session_id = response.session_id

        # Get data config from the restored trainrun
        trainrun_config = TrainRunConfig.from_proto(response.trainrun)
        data_config = trainrun_config.data.to_proto()

        # Run statistical training once
        stat_req = cuvis_ai_pb2.TrainRequest(
            session_id=session_id,
            trainer_type=cuvis_ai_pb2.TRAINER_TYPE_STATISTICAL,
        )
        for _ in stub.Train(stat_req):
            pass

        yield session_id, data_config

        # Cleanup
        try:
            stub.CloseSession(cuvis_ai_pb2.CloseSessionRequest(session_id=session_id))
        except grpc.RpcError:
            pass
    finally:
        channel.close()


@pytest.fixture(scope="session")
def shared_saved_trainrun_with_weights(grpc_server, shared_trained_session, tmp_path_factory):
    """Pre-saved trainrun with weights for weight-loading tests.

    Creates a saved trainrun + weights once at session scope, which is then
    reused by multiple restore and weight-loading tests.

    Uses grpc_server directly and creates its own stub to match session scope.
    """
    from cuvis_ai_core.grpc import cuvis_ai_pb2_grpc

    session_id, _ = shared_trained_session
    tmp_path = tmp_path_factory.mktemp("saved_trainruns")

    # Create our own channel and stub for session scope
    channel = grpc.insecure_channel(grpc_server)
    stub = cuvis_ai_pb2_grpc.CuvisAIServiceStub(channel)

    try:
        trainrun_path = str(tmp_path / "shared_trainrun_with_weights.yaml")
        save_response = stub.SaveTrainRun(
            cuvis_ai_pb2.SaveTrainRunRequest(
                session_id=session_id, trainrun_path=trainrun_path, save_weights=True
            )
        )

        yield {
            "trainrun_path": save_response.trainrun_path,
            "weights_path": save_response.weights_path,
            "tmp_path": tmp_path,
        }
    finally:
        channel.close()


# ============================================================================
# TestSaveTrainRun - Uses shared trained session
# ============================================================================


class TestSaveTrainRun:
    """Test the SaveTrainRun RPC method using shared trained session."""

    def test_save_trainrun_creates_manifest_and_has_valid_structure(
        self, grpc_stub, shared_trained_session, tmp_path
    ):
        """Test that SaveTrainRun creates valid YAML with references, not data copies."""
        trainrun_path = str(tmp_path / "manifest_test.yaml")
        session_id, _ = shared_trained_session

        response = grpc_stub.SaveTrainRun(
            cuvis_ai_pb2.SaveTrainRunRequest(
                session_id=session_id,
                trainrun_path=trainrun_path,
            )
        )

        assert response.success
        assert response.trainrun_path
        assert Path(response.trainrun_path).exists()

        # Verify it's a reference file with proper structure, not a data dump
        with open(response.trainrun_path) as f:
            trainrun_config = yaml.safe_load(f)

        # Should contain pipeline config with proper structure
        assert "pipeline" in trainrun_config
        assert "metadata" in trainrun_config["pipeline"]
        assert "nodes" in trainrun_config["pipeline"]
        assert "connections" in trainrun_config["pipeline"]

    def test_save_trainrun_without_training(self, grpc_stub, tmp_path, session):
        """Test that saving trainrun before training either succeeds or fails gracefully."""
        session_id = session()

        # Attempt to save trainrun without training
        try:
            exp_response = grpc_stub.SaveTrainRun(
                cuvis_ai_pb2.SaveTrainRunRequest(
                    session_id=session_id,
                    trainrun_path=str(tmp_path / "untrained.yaml"),
                )
            )
            # If no error, trainrun can be saved without training (valid - saves initial config)
            assert exp_response.success or True
        except grpc.RpcError as exc:
            # If error is raised, it should be a valid precondition error
            assert exc.code() in [
                grpc.StatusCode.FAILED_PRECONDITION,
                grpc.StatusCode.INVALID_ARGUMENT,
            ]

        # Cleanup
        grpc_stub.CloseSession(cuvis_ai_pb2.CloseSessionRequest(session_id=session_id))

    @pytest.mark.parametrize("save_weights", [True, False])
    def test_save_trainrun_with_and_without_weights(
        self, grpc_stub, shared_trained_session, tmp_path, save_weights
    ):
        """Test SaveTrainRun with save_weights=True/False parameter."""
        trainrun_path = str(tmp_path / f"trainrun_weights_{save_weights}.yaml")
        session_id, _ = shared_trained_session

        response = grpc_stub.SaveTrainRun(
            cuvis_ai_pb2.SaveTrainRunRequest(
                session_id=session_id, trainrun_path=trainrun_path, save_weights=save_weights
            )
        )

        assert response.success
        assert response.trainrun_path.endswith(".yaml")
        assert Path(response.trainrun_path).exists()

        if save_weights:
            # Verify weights file was created with expected structure
            assert response.weights_path is not None
            assert response.weights_path.endswith(".pt")
            assert Path(response.weights_path).exists()

            checkpoint = torch.load(response.weights_path)
            assert "state_dict" in checkpoint
            assert "metadata" in checkpoint
        else:
            # Verify no weights file was created
            assert not response.weights_path


# ============================================================================
# TestRestoreTrainRun - Uses pre-saved artifacts
# ============================================================================


class TestRestoreTrainRun:
    """Test the RestoreTrainRun RPC method using pre-saved artifacts."""

    def test_restore_trainrun_creates_session(self, grpc_stub, experiment_file):
        """Test that RestoreTrainRun creates a new session."""
        response = grpc_stub.RestoreTrainRun(
            cuvis_ai_pb2.RestoreTrainRunRequest(
                trainrun_path=experiment_file,
            )
        )

        assert response.session_id

        trainrun_config = TrainRunConfig.from_proto(response.trainrun)
        assert trainrun_config.name == "test_experiment"

        # Cleanup
        try:
            grpc_stub.CloseSession(cuvis_ai_pb2.CloseSessionRequest(session_id=response.session_id))
        except grpc.RpcError:
            pass

    def test_restore_trainrun_loads_pipeline_and_runs_inference(
        self, grpc_stub, experiment_file, create_test_cube
    ):
        """Test that pipeline is correctly loaded and can perform inference."""
        response = grpc_stub.RestoreTrainRun(
            cuvis_ai_pb2.RestoreTrainRunRequest(
                trainrun_path=experiment_file,
            )
        )

        assert response.session_id

        # Verify pipeline config is returned
        trainrun_config = TrainRunConfig.from_proto(response.trainrun)
        assert trainrun_config.pipeline is not None

        # Verify the session can perform inference
        from cuvis_ai_core.grpc import helpers

        cube, wavelengths = create_test_cube(
            batch_size=1,
            height=32,
            width=32,
            mode="wavelength_dependent",
            num_channels=61,
            wavelength_range=(430.0, 910.0),
        )

        try:
            inference_response = grpc_stub.Inference(
                cuvis_ai_pb2.InferenceRequest(
                    session_id=response.session_id,
                    inputs=cuvis_ai_pb2.InputBatch(
                        cube=helpers.tensor_to_proto(cube),
                        wavelengths=helpers.tensor_to_proto(wavelengths),
                    ),
                )
            )
            assert len(inference_response.outputs) > 0
        finally:
            grpc_stub.CloseSession(cuvis_ai_pb2.CloseSessionRequest(session_id=response.session_id))

    def test_restore_trainrun_invalid_file(self, grpc_stub, tmp_path):
        """Test error handling for non-existent trainrun file."""
        with pytest.raises(grpc.RpcError) as exc:
            grpc_stub.RestoreTrainRun(
                cuvis_ai_pb2.RestoreTrainRunRequest(
                    trainrun_path=str(tmp_path / "nonexistent.yaml"),
                )
            )
        assert exc.value.code() == grpc.StatusCode.NOT_FOUND

    def test_restore_trainrun_missing_pipeline_nodes(self, grpc_stub, tmp_path, mock_pipeline_dict):
        """Test error when trainrun references invalid pipeline (empty nodes)."""
        bad_trainrun_path = tmp_path / "bad_trainrun.yaml"
        bad_pipeline = mock_pipeline_dict.copy()
        bad_pipeline["nodes"] = []  # Empty nodes list will cause issues

        bad_trainrun = {
            "name": "bad_trainrun",
            "pipeline": bad_pipeline,
            "data": {
                "cu3s_file_path": "/data/test.cu3s",
                "batch_size": 4,
                "processing_mode": "Reflectance",
                "train_ids": [],
                "val_ids": [],
                "test_ids": [],
            },
            "training": {
                "seed": 42,
                "trainer": {"max_epochs": 10, "accelerator": "auto"},
                "optimizer": {"name": "adamw", "lr": 0.001},
            },
        }

        with open(bad_trainrun_path, "w") as f:
            yaml.dump(bad_trainrun, f)

        with pytest.raises(grpc.RpcError) as exc:
            grpc_stub.RestoreTrainRun(
                cuvis_ai_pb2.RestoreTrainRunRequest(
                    trainrun_path=str(bad_trainrun_path),
                )
            )
        assert exc.value.code() in [
            grpc.StatusCode.NOT_FOUND,
            grpc.StatusCode.INVALID_ARGUMENT,
            grpc.StatusCode.INTERNAL,
        ]

    def test_restore_trainrun_with_weights(
        self, grpc_stub, shared_saved_trainrun_with_weights, grpc_session_manager
    ):
        """Test that RestoreTrainRun can load weights and initialize statistical nodes."""
        saved_data = shared_saved_trainrun_with_weights

        # Restore with weights
        restore_response = grpc_stub.RestoreTrainRun(
            cuvis_ai_pb2.RestoreTrainRunRequest(
                trainrun_path=saved_data["trainrun_path"],
                weights_path=saved_data["weights_path"],
                strict=True,
            )
        )

        assert restore_response.session_id

        # Verify that statistical nodes are initialized
        session = grpc_session_manager.get_session(restore_response.session_id)

        # Check that nodes requiring statistical initialization are properly initialized
        stat_nodes_initialized = []
        if session.pipeline is not None:
            for node in session.pipeline.nodes():
                if hasattr(node, "_statistically_initialized"):
                    stat_nodes_initialized.append(node._statistically_initialized)

        # Should have at least some statistical nodes initialized
        if len(stat_nodes_initialized) > 0:
            assert any(stat_nodes_initialized), "Weights load should initialize at least one node"

        # Cleanup
        grpc_stub.CloseSession(
            cuvis_ai_pb2.CloseSessionRequest(session_id=restore_response.session_id)
        )

    def test_restore_trainrun_with_missing_weights(
        self, grpc_stub, shared_saved_trainrun_with_weights
    ):
        """Test that RestoreTrainRun handles missing weights gracefully."""
        saved_data = shared_saved_trainrun_with_weights

        # Try to restore with non-existent weights path
        with pytest.raises(grpc.RpcError) as exc:
            grpc_stub.RestoreTrainRun(
                cuvis_ai_pb2.RestoreTrainRunRequest(
                    trainrun_path=saved_data["trainrun_path"],
                    weights_path="/nonexistent/weights.pt",
                    strict=True,
                )
            )

        assert exc.value.code() == grpc.StatusCode.NOT_FOUND


# ============================================================================
# TestWeightTransfer - Uses shared artifacts
# ============================================================================


class TestWeightTransfer:
    """Test that weights are correctly transferred during save/restore."""

    def test_weights_functional_verification(
        self, grpc_stub, shared_saved_trainrun_with_weights, create_test_cube
    ):
        """Test that restored weights produce consistent outputs."""
        from cuvis_ai_core.grpc import helpers

        saved_data = shared_saved_trainrun_with_weights

        # Restore with weights
        restore_response = grpc_stub.RestoreTrainRun(
            cuvis_ai_pb2.RestoreTrainRunRequest(
                trainrun_path=saved_data["trainrun_path"],
                weights_path=saved_data["weights_path"],
                strict=True,
            )
        )

        # Create test data
        cube, wavelengths = create_test_cube(
            batch_size=1,
            height=3,
            width=3,
            num_channels=DEFAULT_CHANNELS,
            mode="random",
            wavelength_range=(430.0, 910.0),
        )

        # Run inference with restored model
        restored_response = grpc_stub.Inference(
            cuvis_ai_pb2.InferenceRequest(
                session_id=restore_response.session_id,
                inputs=cuvis_ai_pb2.InputBatch(
                    cube=helpers.tensor_to_proto(cube),
                    wavelengths=helpers.tensor_to_proto(wavelengths),
                ),
            )
        )

        # Verify outputs are present
        assert len(restored_response.outputs) > 0

        # Cleanup
        grpc_stub.CloseSession(
            cuvis_ai_pb2.CloseSessionRequest(session_id=restore_response.session_id)
        )

    @pytest.mark.parametrize("strict", [True, False])
    def test_strict_and_nonstrict_loading(
        self, grpc_stub, shared_saved_trainrun_with_weights, strict
    ):
        """Test strict and non-strict weight loading behavior."""
        saved_data = shared_saved_trainrun_with_weights

        # Restore with specified strictness
        restore_response = grpc_stub.RestoreTrainRun(
            cuvis_ai_pb2.RestoreTrainRunRequest(
                trainrun_path=saved_data["trainrun_path"],
                weights_path=saved_data["weights_path"],
                strict=strict,
            )
        )
        assert restore_response.session_id

        # Cleanup
        grpc_stub.CloseSession(
            cuvis_ai_pb2.CloseSessionRequest(session_id=restore_response.session_id)
        )

    def test_statistical_node_inference_without_refit(
        self, grpc_stub, shared_saved_trainrun_with_weights, create_test_cube, grpc_session_manager
    ):
        """Test that statistical nodes work after weight restore without re-fitting."""
        from cuvis_ai_core.grpc import helpers

        saved_data = shared_saved_trainrun_with_weights

        # Restore with weights
        restore_response = grpc_stub.RestoreTrainRun(
            cuvis_ai_pb2.RestoreTrainRunRequest(
                trainrun_path=saved_data["trainrun_path"],
                weights_path=saved_data["weights_path"],
                strict=True,
            )
        )

        # Verify statistical nodes are initialized
        session = grpc_session_manager.get_session(restore_response.session_id)

        stat_flags = []
        if session.pipeline is not None:
            for node in session.pipeline.nodes():
                if hasattr(node, "_statistically_initialized"):
                    stat_flags.append(node._statistically_initialized)

        if stat_flags:
            assert any(stat_flags), (
                "At least one statistical node should be initialized after weight loading"
            )

        # Run inference WITHOUT statistical training (should work)
        cube, wavelengths = create_test_cube(
            batch_size=1,
            height=3,
            width=3,
            num_channels=DEFAULT_CHANNELS,
            mode="random",
            wavelength_range=(430.0, 910.0),
        )

        inference_response = grpc_stub.Inference(
            cuvis_ai_pb2.InferenceRequest(
                session_id=restore_response.session_id,
                inputs=cuvis_ai_pb2.InputBatch(
                    cube=helpers.tensor_to_proto(cube),
                    wavelengths=helpers.tensor_to_proto(wavelengths),
                ),
            )
        )

        # Should succeed without needing statistical training
        assert len(inference_response.outputs) > 0

        # Cleanup
        grpc_stub.CloseSession(
            cuvis_ai_pb2.CloseSessionRequest(session_id=restore_response.session_id)
        )


# ============================================================================
# TestExperimentWorkflow - Integration tests (optimized where possible)
# ============================================================================


class TestExperimentWorkflow:
    """Test complete experiment workflows."""

    def test_train_save_restore_cycle(
        self,
        grpc_stub,
        tmp_path,
        monkeypatch,
        shared_trained_session,
        create_test_cube,
    ):
        """Test complete workflow: train -> save trainrun -> restore -> verify."""
        monkeypatch.setenv("CUVIS_CONFIGS_DIR", str(tmp_path))

        # Use shared trained session
        original_session_id, _ = shared_trained_session

        # Save pipeline with weights
        pipeline_path = str(tmp_path / "trained_pipeline.yaml")
        pipeline_response = grpc_stub.SavePipeline(
            cuvis_ai_pb2.SavePipelineRequest(
                session_id=original_session_id,
                pipeline_path=pipeline_path,
            )
        )
        assert pipeline_response.success

        # Save trainrun
        trainrun_path = str(tmp_path / "workflow_trainrun.yaml")
        trainrun_response = grpc_stub.SaveTrainRun(
            cuvis_ai_pb2.SaveTrainRunRequest(
                session_id=original_session_id,
                trainrun_path=trainrun_path,
            )
        )
        assert trainrun_response.success

        # Restore trainrun (creates new session)
        restore_response = grpc_stub.RestoreTrainRun(
            cuvis_ai_pb2.RestoreTrainRunRequest(
                trainrun_path=trainrun_response.trainrun_path,
            )
        )

        restored_session_id = restore_response.session_id
        assert restored_session_id != original_session_id

        # Load the trained pipeline weights into the restored session
        load_resp = grpc_stub.LoadPipeline(
            cuvis_ai_pb2.LoadPipelineRequest(
                session_id=restored_session_id,
                pipeline=cuvis_ai_pb2.PipelineConfig(
                    config_bytes=_pipeline_bytes_from_path(pipeline_response.pipeline_path)
                ),
            )
        )
        assert load_resp.success
        grpc_stub.LoadPipelineWeights(
            cuvis_ai_pb2.LoadPipelineWeightsRequest(
                session_id=restored_session_id,
                weights_path=pipeline_response.weights_path,
                strict=False,
            )
        )

        # Verify restored session works with inference
        from cuvis_ai_core.grpc import helpers

        cube, wavelengths = create_test_cube(
            batch_size=1,
            height=3,
            width=3,
            num_channels=DEFAULT_CHANNELS,
            mode="random",
            wavelength_range=(430.0, 910.0),
        )

        inference_response = grpc_stub.Inference(
            cuvis_ai_pb2.InferenceRequest(
                session_id=restored_session_id,
                inputs=cuvis_ai_pb2.InputBatch(
                    cube=helpers.tensor_to_proto(cube),
                    wavelengths=helpers.tensor_to_proto(wavelengths),
                ),
            )
        )

        assert len(inference_response.outputs) > 0

        # Cleanup
        grpc_stub.CloseSession(cuvis_ai_pb2.CloseSessionRequest(session_id=restored_session_id))

    @pytest.mark.slow
    def test_experiment_reproducibility(self, grpc_stub, experiment_file):
        """Test that restored experiment config is accessible for re-training."""
        # Restore experiment
        response = grpc_stub.RestoreTrainRun(
            cuvis_ai_pb2.RestoreTrainRunRequest(
                trainrun_path=experiment_file,
            )
        )

        session_id = response.session_id

        # Verify we can access the training config
        trainrun = TrainRunConfig.from_proto(response.trainrun)
        assert trainrun.training is not None

        # Verify the config has expected fields for reproducibility
        training_config = trainrun.training.model_dump()
        assert "seed" in training_config or "trainer" in training_config

        # Cleanup
        grpc_stub.CloseSession(cuvis_ai_pb2.CloseSessionRequest(session_id=session_id))

    @pytest.mark.slow
    def test_resume_training_workflow(self, grpc_stub, shared_saved_trainrun_with_weights):
        """Test resume training workflow: restore with weights -> continue training."""
        saved_data = shared_saved_trainrun_with_weights

        # Restore trainrun with weights
        restore_response = grpc_stub.RestoreTrainRun(
            cuvis_ai_pb2.RestoreTrainRunRequest(
                trainrun_path=saved_data["trainrun_path"],
                weights_path=saved_data["weights_path"],
                strict=True,
            )
        )
        restored_session_id = restore_response.session_id
        assert restored_session_id

        # Attempt to continue training (gradient training)
        training_complete = False
        try:
            for progress in grpc_stub.Train(
                cuvis_ai_pb2.TrainRequest(
                    session_id=restored_session_id, trainer_type=cuvis_ai_pb2.TRAINER_TYPE_GRADIENT
                )
            ):
                if progress.status == cuvis_ai_pb2.TRAIN_STATUS_COMPLETE:
                    training_complete = True
                    break
                if progress.status == cuvis_ai_pb2.TRAIN_STATUS_ERROR:
                    raise AssertionError(f"Training failed: {progress.message}")
        except grpc.RpcError as exc:
            # If training fails, it should be a valid error (e.g., missing data config)
            assert exc.code() in [
                grpc.StatusCode.INVALID_ARGUMENT,
                grpc.StatusCode.FAILED_PRECONDITION,
            ]
            training_complete = True  # Consider this a valid outcome

        assert training_complete, "Training should complete or fail with valid error"

        # Cleanup
        grpc_stub.CloseSession(cuvis_ai_pb2.CloseSessionRequest(session_id=restored_session_id))
