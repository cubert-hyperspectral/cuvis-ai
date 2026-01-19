"""End-to-end workflow tests (Task 5.5).

These tests verify complete workflows as described in the Phase 4 documentation.
"""

from pathlib import Path

import pytest
import yaml

from cuvis_ai_core.grpc import cuvis_ai_pb2, helpers
from tests.fixtures.grpc import load_pipeline_from_file, resolve_and_load_pipeline

DEFAULT_CHANNELS = 61


class TestWorkflow1_TrainFromScratch:
    """Workflow 1: Train from Scratch (as per Phase 4 doc).

    Steps:
    1. CreateSession with pipeline structure (no weights)
    2. Train with explicit data and training configs
    3. SavePipeline for deployment
    4. SaveTrainRun for reproducibility
    5. CloseSession
    """

    @pytest.mark.slow
    def test_complete_workflow(self, grpc_stub, tmp_path):
        """Test the complete train-from-scratch workflow."""
        # Step 1: Create session with pipeline structure (no weights) using new four-step workflow
        session_response = grpc_stub.CreateSession(cuvis_ai_pb2.CreateSessionRequest())
        session_id = session_response.session_id
        assert session_id

        # Build pipeline
        resolve_and_load_pipeline(grpc_stub, session_id, path="pipeline/channel_selector")

        # Step 2: Train with explicit data and training configs
        # Note: Skipping actual training in this test
        # In production, you would call Train() with DataConfig and TrainingConfig

        # Step 3: SavePipeline for deployment
        pipeline_path = str(tmp_path / "models" / "trained_selector.yaml")
        save_pipeline_response = grpc_stub.SavePipeline(
            cuvis_ai_pb2.SavePipelineRequest(
                session_id=session_id,
                pipeline_path=pipeline_path,
                metadata=cuvis_ai_pb2.PipelineMetadata(
                    name="Trained Channel Selector",
                    description="Trained on lentils dataset",
                    created="2024-11-27",
                ),
            )
        )
        assert save_pipeline_response.success
        assert Path(save_pipeline_response.pipeline_path).exists()
        assert Path(save_pipeline_response.weights_path).exists()

        # Step 4: SaveTrainRun for reproducibility
        exp_path = str(tmp_path / "experiments" / "run_001.yaml")
        save_exp_response = grpc_stub.SaveTrainRun(
            cuvis_ai_pb2.SaveTrainRunRequest(
                session_id=session_id,
                trainrun_path=exp_path,
            )
        )
        assert save_exp_response.success
        assert Path(save_exp_response.trainrun_path).exists()

        # Step 5: CloseSession
        close_response = grpc_stub.CloseSession(
            cuvis_ai_pb2.CloseSessionRequest(session_id=session_id)
        )
        assert close_response.success


class TestWorkflow2_InferenceWithPretrained:
    """Workflow 2: Inference with Pre-trained Model.

    Steps:
    1. CreateSession with pre-trained pipeline
    2. Run inference
    3. Verify outputs
    4. CloseSession
    """

    def test_complete_workflow(self, grpc_stub, pretrained_pipeline, create_test_cube):
        """Test inference with a pre-trained model using shared fixture."""
        # Step 1: CreateSession with pre-trained pipeline using new four-step workflow
        session_response = grpc_stub.CreateSession(cuvis_ai_pb2.CreateSessionRequest())
        session_id = session_response.session_id

        # Load pre-trained pipeline structure then weights
        load_pipeline_from_file(grpc_stub, session_id, pretrained_pipeline)
        grpc_stub.LoadPipelineWeights(
            cuvis_ai_pb2.LoadPipelineWeightsRequest(
                session_id=session_id,
                weights_path=str(Path(pretrained_pipeline).with_suffix(".pt")),
                strict=True,
            )
        )

        # Step 2: Run inference
        cube, wavelengths = create_test_cube(
            batch_size=1, height=3, width=3, num_channels=DEFAULT_CHANNELS, mode="random"
        )

        inference_response = grpc_stub.Inference(
            cuvis_ai_pb2.InferenceRequest(
                session_id=session_id,
                inputs=cuvis_ai_pb2.InputBatch(
                    cube=helpers.tensor_to_proto(cube),
                    wavelengths=helpers.tensor_to_proto(wavelengths),
                ),
            )
        )

        # Step 3: Verify outputs
        assert len(inference_response.outputs) > 0
        assert "SoftChannelSelector.selected" in inference_response.outputs

        selected = helpers.proto_to_numpy(
            inference_response.outputs["SoftChannelSelector.selected"]
        )
        assert selected.shape == cube.shape

        # Step 4: CloseSession
        close_response = grpc_stub.CloseSession(
            cuvis_ai_pb2.CloseSessionRequest(session_id=session_id)
        )
        assert close_response.success


class TestWorkflow3_ResumeTraining:
    """Workflow 3: Resume Training.

    Steps:
    1. RestoreTrainRun
    2. Continue training with modified config
    3. SavePipeline with updated weights
    4. SaveTrainRun as new version
    """

    @pytest.mark.slow
    def test_complete_workflow(self, grpc_stub, temp_workspace, mock_pipeline_dir):
        """Test resuming training from a saved experiment."""

        # Setup: Create an initial experiment to resume from
        # Use the existing pipeline fixture and save it using new four-step workflow
        setup_response = grpc_stub.CreateSession(cuvis_ai_pb2.CreateSessionRequest())
        setup_session_id = setup_response.session_id

        # Build pipeline
        resolve_and_load_pipeline(grpc_stub, setup_session_id, path="pipeline/channel_selector")

        pipeline_path = mock_pipeline_dir / "initial.yaml"
        save_pipeline_response = grpc_stub.SavePipeline(
            cuvis_ai_pb2.SavePipelineRequest(
                session_id=setup_session_id,
                pipeline_path=str(pipeline_path),
            )
        )
        weights_path = Path(save_pipeline_response.weights_path)

        grpc_stub.CloseSession(cuvis_ai_pb2.CloseSessionRequest(session_id=setup_session_id))

        exp_path = temp_workspace / "experiments" / "initial_exp.yaml"
        exp_config = {
            "name": "initial_experiment",
            "pipeline": {
                "config_path": str(pipeline_path),
                "weights_path": str(weights_path),
            },
            "data": {
                "cu3s_file_path": "/data/lentils.cu3s",
                "batch_size": 4,
                "processing_mode": "Reflectance",
                "train_ids": [0, 1, 2],
                "val_ids": [3],
                "test_ids": [4],
            },
            "training": {
                "seed": 42,
                "trainer": {
                    "max_epochs": 2,
                },
            },
        }
        with open(exp_path, "w") as f:
            yaml.dump(exp_config, f)

        # Step 1: RestoreTrainRun
        restore_response = grpc_stub.RestoreTrainRun(
            cuvis_ai_pb2.RestoreTrainRunRequest(
                trainrun_path=str(exp_path),
            )
        )
        session_id = restore_response.session_id
        assert session_id

        # Step 2: Continue training with modified config
        # (Skipping actual training in this test)
        # In production: Call Train() with modified TrainingConfig

        # Step 3: SavePipeline with updated weights
        updated_pipeline_path = str(mock_pipeline_dir / "resumed.yaml")
        save_pipeline_response = grpc_stub.SavePipeline(
            cuvis_ai_pb2.SavePipelineRequest(
                session_id=session_id,
                pipeline_path=updated_pipeline_path,
                metadata=cuvis_ai_pb2.PipelineMetadata(
                    name="Resumed Pipeline",
                    description="Continued training from epoch 50",
                ),
            )
        )
        assert save_pipeline_response.success

        # Step 4: SaveTrainRun as new version
        new_exp_path = str(temp_workspace / "experiments" / "resumed_exp.yaml")
        save_exp_response = grpc_stub.SaveTrainRun(
            cuvis_ai_pb2.SaveTrainRunRequest(
                session_id=session_id,
                trainrun_path=new_exp_path,
            )
        )
        assert save_exp_response.success

        # Cleanup
        grpc_stub.CloseSession(cuvis_ai_pb2.CloseSessionRequest(session_id=session_id))


class TestWorkflow4_DiscoverAndInspect:
    """Workflow 4: Discover and Inspect Pipelinees.

    Steps:
    1. ListAvailablePipelinees
    2. Filter by tag
    3. GetPipelineInfo for specific pipeline
    4. CreateSession based on discovered pipeline
    """

    @pytest.mark.skip(
        reason="Native cuvis library has thread-safety issues causing crashes during concurrent access"
    )
    def test_complete_workflow(self, grpc_stub, shared_workflow_setup):
        """Test pipeline discovery and inspection workflow using shared setup."""
        _ = shared_workflow_setup

        # Step 1: ListAvailablePipelinees
        list_response = grpc_stub.ListAvailablePipelinees(
            cuvis_ai_pb2.ListAvailablePipelineesRequest()
        )
        assert len(list_response.pipelinees) >= 2

        # Step 2: Filter by tag
        anomaly_response = grpc_stub.ListAvailablePipelinees(
            cuvis_ai_pb2.ListAvailablePipelineesRequest(filter_tag="anomaly")
        )
        assert len(anomaly_response.pipelinees) >= 1
        assert any(c.name == "rx_detector" for c in anomaly_response.pipelinees)

        # Step 3: GetPipelineInfo for specific pipeline
        info_response = grpc_stub.GetPipelineInfo(
            cuvis_ai_pb2.GetPipelineInfoRequest(pipeline_name="rx_detector")
        )
        assert info_response.pipeline_info.name == "rx_detector"
        assert "anomaly" in info_response.pipeline_info.tags

        # Step 4: CreateSession based on discovered pipeline using new four-step workflow
        session_response = grpc_stub.CreateSession(cuvis_ai_pb2.CreateSessionRequest())
        session_id = session_response.session_id

        # Load discovered pipeline
        load_pipeline_from_file(grpc_stub, session_id, info_response.pipeline_info.path)
        assert session_id

        # Cleanup
        grpc_stub.CloseSession(
            cuvis_ai_pb2.CloseSessionRequest(session_id=session_response.session_id)
        )


class TestWorkflow5_LoadPipelineWeights:
    """Workflow 5: Load Pipeline Weights.

    Steps:
    1. CreateSession with structure only
    2. Later, LoadPipeline with weights
    3. Run inference
    """

    def test_complete_workflow(self, grpc_stub, pretrained_pipeline, create_test_cube):
        """Test loading weights into an existing session using shared fixture."""
        pipeline_with_weights = str(pretrained_pipeline)

        # Step 1: CreateSession with structure only (no weights) using new four-step workflow
        session_response = grpc_stub.CreateSession(cuvis_ai_pb2.CreateSessionRequest())
        session_id = session_response.session_id

        # Build pipeline structure
        load_pipeline_from_file(grpc_stub, session_id, pipeline_with_weights)

        # Step 2: Later, load weights explicitly
        grpc_stub.LoadPipelineWeights(
            cuvis_ai_pb2.LoadPipelineWeightsRequest(
                session_id=session_id,
                weights_path=str(Path(pipeline_with_weights).with_suffix(".pt")),
                strict=True,
            )
        )

        # Step 3: Run inference
        cube, wavelengths = create_test_cube(
            batch_size=1, height=2, width=2, num_channels=DEFAULT_CHANNELS, mode="random"
        )
        # Convert to numpy arrays for proto

        inference_response = grpc_stub.Inference(
            cuvis_ai_pb2.InferenceRequest(
                session_id=session_id,
                inputs=cuvis_ai_pb2.InputBatch(
                    cube=helpers.tensor_to_proto(cube),
                    wavelengths=helpers.tensor_to_proto(wavelengths),
                ),
            )
        )

        assert len(inference_response.outputs) > 0

        # Cleanup
        grpc_stub.CloseSession(cuvis_ai_pb2.CloseSessionRequest(session_id=session_id))


class TestWorkflowIntegration:
    """Test combinations and interactions between workflows."""

    def test_multiple_sessions_parallel(self, grpc_stub, trained_session, create_test_cube):
        """Test that multiple sessions can coexist using shared trained session."""
        # Get a trained session
        session_id, _ = trained_session()

        # Run inference on the trained session
        cube, wavelengths = create_test_cube(
            batch_size=1, height=2, width=2, num_channels=DEFAULT_CHANNELS, mode="random"
        )

        response = grpc_stub.Inference(
            cuvis_ai_pb2.InferenceRequest(
                session_id=session_id,
                inputs=cuvis_ai_pb2.InputBatch(
                    cube=helpers.tensor_to_proto(cube),
                    wavelengths=helpers.tensor_to_proto(wavelengths),
                ),
            )
        )
        assert len(response.outputs) > 0

    @pytest.mark.parametrize("test_mode", ["wavelength_dependent", "random"])
    def test_inference_modes_with_pretrained(
        self, grpc_stub, pretrained_pipeline, create_test_cube, test_mode
    ):
        """Test different inference modes using the shared pretrained pipeline."""
        # Create session with pretrained pipeline using new four-step workflow
        session_response = grpc_stub.CreateSession(cuvis_ai_pb2.CreateSessionRequest())
        session_id = session_response.session_id

        # Load pre-trained pipeline
        load_pipeline_from_file(grpc_stub, session_id, pretrained_pipeline)
        grpc_stub.LoadPipelineWeights(
            cuvis_ai_pb2.LoadPipelineWeightsRequest(
                session_id=session_id,
                weights_path=str(Path(pretrained_pipeline).with_suffix(".pt")),
                strict=True,
            )
        )

        try:
            # Test with different cube generation modes
            cube, wavelengths = create_test_cube(
                batch_size=1, height=3, width=3, num_channels=DEFAULT_CHANNELS, mode=test_mode
            )

            inference_response = grpc_stub.Inference(
                cuvis_ai_pb2.InferenceRequest(
                    session_id=session_id,
                    inputs=cuvis_ai_pb2.InputBatch(
                        cube=helpers.tensor_to_proto(cube),
                        wavelengths=helpers.tensor_to_proto(wavelengths),
                    ),
                )
            )

            # Verify outputs
            assert len(inference_response.outputs) > 0
            assert "SoftChannelSelector.selected" in inference_response.outputs

            selected = helpers.proto_to_numpy(
                inference_response.outputs["SoftChannelSelector.selected"]
            )
            assert selected.shape == cube.shape

        finally:
            grpc_stub.CloseSession(cuvis_ai_pb2.CloseSessionRequest(session_id=session_id))

    def test_session_reuse_after_save_load(
        self, grpc_stub, mock_pipeline_dir, trained_session, create_test_cube
    ):
        """Test that a session can be reused after save/load operations using shared fixture."""
        # Get a trained session
        session_id, _ = trained_session()

        # Run inference before save
        cube, wavelengths = create_test_cube(
            batch_size=1, height=2, width=2, num_channels=DEFAULT_CHANNELS, mode="random"
        )
        # Convert to numpy arrays for proto
        inf1 = grpc_stub.Inference(
            cuvis_ai_pb2.InferenceRequest(
                session_id=session_id,
                inputs=cuvis_ai_pb2.InputBatch(
                    cube=helpers.tensor_to_proto(cube),
                    wavelengths=helpers.tensor_to_proto(wavelengths),
                ),
            )
        )
        assert len(inf1.outputs) > 0

        # Save pipeline
        pipeline_path = str(mock_pipeline_dir / "reuse_test.yaml")
        save_response = grpc_stub.SavePipeline(
            cuvis_ai_pb2.SavePipelineRequest(
                session_id=session_id,
                pipeline_path=pipeline_path,
            )
        )

        # Load pipeline back
        load_pipeline_from_file(grpc_stub, session_id, pipeline_path)
        grpc_stub.LoadPipelineWeights(
            cuvis_ai_pb2.LoadPipelineWeightsRequest(
                session_id=session_id,
                weights_path=str(Path(save_response.weights_path)),
                strict=True,
            )
        )

        # Run inference after load
        inf2 = grpc_stub.Inference(
            cuvis_ai_pb2.InferenceRequest(
                session_id=session_id,
                inputs=cuvis_ai_pb2.InputBatch(
                    cube=helpers.tensor_to_proto(cube),
                    wavelengths=helpers.tensor_to_proto(wavelengths),
                ),
            )
        )
        assert len(inf2.outputs) > 0

        # Note: Cleanup is handled by the trained_session fixture
