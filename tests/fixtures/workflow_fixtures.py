"""Shared fixtures for workflow testing to avoid redundant operations."""

from collections.abc import Generator
from pathlib import Path

import pytest

from cuvis_ai_core.grpc import cuvis_ai_pb2

# Cache for pretrained pipelines to avoid recomputation
_pretrained_pipeline_cache = {}


@pytest.fixture(scope="function")
def pretrained_pipeline(grpc_stub, tmp_path, data_config_factory) -> Generator[Path, None, None]:
    """Session-scoped fixture that creates and caches a pretrained pipeline.

    This fixture creates a channel_selector pipeline, performs statistical training,
    and saves it to a temporary directory. The same pipeline is reused across
    all tests in the session to avoid redundant training operations.

    Yields:
        Path: Path to the saved pretrained pipeline YAML file
    """
    cache_key = "pretrained_channel_selector"

    # Check if we already have a cached pipeline
    if cache_key in _pretrained_pipeline_cache:
        yield _pretrained_pipeline_cache[cache_key]
        return

    # Create temporary directory for this session
    temp_dir = tmp_path / "pretrained_pipelines"
    temp_dir.mkdir(exist_ok=True)
    pipeline_path = temp_dir / "pretrained_channel_selector.yaml"

    # Create session and setup pipeline using the new four-step workflow
    session_response = grpc_stub.CreateSession(cuvis_ai_pb2.CreateSessionRequest())
    session_id = session_response.session_id

    # Resolve and load pipeline structure
    config_response = grpc_stub.ResolveConfig(
        cuvis_ai_pb2.ResolveConfigRequest(
            session_id=session_id,
            config_type="pipeline",
            path="pipeline/channel_selector",
        )
    )
    load_response = grpc_stub.LoadPipeline(
        cuvis_ai_pb2.LoadPipelineRequest(
            session_id=session_id,
            pipeline=cuvis_ai_pb2.PipelineConfig(config_bytes=config_response.config_bytes),
        )
    )
    if not load_response.success:
        raise RuntimeError("Failed to load pipeline")

    try:
        # Perform statistical training to initialize nodes
        stat_train_request = cuvis_ai_pb2.TrainRequest(
            session_id=session_id,
            trainer_type=cuvis_ai_pb2.TRAINER_TYPE_STATISTICAL,
            data=data_config_factory(),
        )

        # Consume training progress messages
        for _ in grpc_stub.Train(stat_train_request):
            pass

        # Save the pretrained pipeline
        save_response = grpc_stub.SavePipeline(
            cuvis_ai_pb2.SavePipelineRequest(
                session_id=session_id,
                pipeline_path=str(pipeline_path),
                metadata=cuvis_ai_pb2.PipelineMetadata(
                    name="Pretrained Channel Selector",
                    description="Pretrained on test data for workflow testing",
                    tags=["pretrained", "channel_selector"],
                ),
            )
        )

        if not save_response.success:
            raise RuntimeError("Failed to save pretrained pipeline")

        # Cache the pipeline path for reuse
        _pretrained_pipeline_cache[cache_key] = pipeline_path

        # Yield the pipeline path for test usage
        yield pipeline_path
    finally:
        # Clean up the session
        grpc_stub.CloseSession(cuvis_ai_pb2.CloseSessionRequest(session_id=session_id))


@pytest.fixture
def shared_workflow_setup(
    grpc_stub, mock_pipeline_dir, data_config_factory
) -> Generator[dict, None, None]:
    """Shared setup for workflow tests that need multiple pipeline types.

    Creates and saves multiple pipeline types that can be reused across tests.
    Returns a dictionary with paths to different pipeline types.

    Yields:
        dict: Dictionary containing paths to different pipeline types
    """
    setup_data = {}

    # Create and save an anomaly detection pipeline using the new four-step workflow
    rx_session = grpc_stub.CreateSession(cuvis_ai_pb2.CreateSessionRequest())
    config_response = grpc_stub.ResolveConfig(
        cuvis_ai_pb2.ResolveConfigRequest(
            session_id=rx_session.session_id,
            config_type="pipeline",
            path="pipeline/channel_selector",
        )
    )
    load_response = grpc_stub.LoadPipeline(
        cuvis_ai_pb2.LoadPipelineRequest(
            session_id=rx_session.session_id,
            pipeline=cuvis_ai_pb2.PipelineConfig(config_bytes=config_response.config_bytes),
        )
    )
    if not load_response.success:
        raise RuntimeError("Failed to load pipeline")

    try:
        # Perform statistical training
        stat_train_request = cuvis_ai_pb2.TrainRequest(
            session_id=rx_session.session_id,
            trainer_type=cuvis_ai_pb2.TRAINER_TYPE_STATISTICAL,
            data=data_config_factory(),
        )

        for _ in grpc_stub.Train(stat_train_request):
            pass

        anomaly_pipeline = mock_pipeline_dir / "rx_detector.yaml"
        grpc_stub.SavePipeline(
            cuvis_ai_pb2.SavePipelineRequest(
                session_id=rx_session.session_id,
                pipeline_path=str(anomaly_pipeline),
                metadata=cuvis_ai_pb2.PipelineMetadata(
                    name="RX Anomaly Detector",
                    description="Statistical anomaly detection",
                    tags=["anomaly", "statistical"],
                ),
            )
        )
        setup_data["anomaly_pipeline"] = anomaly_pipeline

        # Create and save a segmentation pipeline using the new four-step workflow
        seg_session = grpc_stub.CreateSession(cuvis_ai_pb2.CreateSessionRequest())
        config_response = grpc_stub.ResolveConfig(
            cuvis_ai_pb2.ResolveConfigRequest(
                session_id=seg_session.session_id,
                config_type="pipeline",
                path="pipeline/channel_selector",
            )
        )
        load_response = grpc_stub.LoadPipeline(
            cuvis_ai_pb2.LoadPipelineRequest(
                session_id=seg_session.session_id,
                pipeline=cuvis_ai_pb2.PipelineConfig(config_bytes=config_response.config_bytes),
            )
        )
        if not load_response.success:
            raise RuntimeError("Failed to load pipeline")

        try:
            # Perform statistical training
            stat_train_request = cuvis_ai_pb2.TrainRequest(
                session_id=seg_session.session_id,
                trainer_type=cuvis_ai_pb2.TRAINER_TYPE_STATISTICAL,
                data=data_config_factory(),
            )

            for _ in grpc_stub.Train(stat_train_request):
                pass

            segmentation_pipeline = mock_pipeline_dir / "segmenter.yaml"
            grpc_stub.SavePipeline(
                cuvis_ai_pb2.SavePipelineRequest(
                    session_id=seg_session.session_id,
                    pipeline_path=str(segmentation_pipeline),
                    metadata=cuvis_ai_pb2.PipelineMetadata(
                        name="Segmentation Pipeline",
                        description="Image segmentation",
                        tags=["segmentation", "deep_learning"],
                    ),
                )
            )
            setup_data["segmentation_pipeline"] = segmentation_pipeline

        finally:
            grpc_stub.CloseSession(
                cuvis_ai_pb2.CloseSessionRequest(session_id=seg_session.session_id)
            )

        yield setup_data

    finally:
        grpc_stub.CloseSession(cuvis_ai_pb2.CloseSessionRequest(session_id=rx_session.session_id))
