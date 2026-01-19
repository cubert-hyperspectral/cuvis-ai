from cuvis_ai_core.grpc.v1 import cuvis_ai_pb2


def test_config_messages_use_bytes():
    """Verify all config messages use bytes config_bytes field."""
    configs = [
        cuvis_ai_pb2.PipelineConfig,
        cuvis_ai_pb2.DataConfig,
        cuvis_ai_pb2.TrainingConfig,
        cuvis_ai_pb2.TrainRunConfig,
    ]

    for config_cls in configs:
        instance = config_cls()
        assert hasattr(instance, "config_bytes")


def test_trainrun_naming():
    """Verify Experiment renamed to TrainRun."""
    assert hasattr(cuvis_ai_pb2, "TrainRunConfig")
    assert hasattr(cuvis_ai_pb2, "SaveTrainRunRequest")
    assert hasattr(cuvis_ai_pb2, "RestoreTrainRunRequest")
    assert not hasattr(cuvis_ai_pb2, "ExperimentConfig")
    assert not hasattr(cuvis_ai_pb2, "SaveExperimentRequest")


def test_new_session_rpcs_exist():
    """Verify new session management RPCs defined."""
    assert hasattr(cuvis_ai_pb2, "CreateSessionRequest")
    assert hasattr(cuvis_ai_pb2, "SetSessionSearchPathsRequest")
    assert hasattr(cuvis_ai_pb2, "CloseSessionRequest")


def test_new_config_rpcs_exist():
    """Verify new config resolution RPCs defined."""
    assert hasattr(cuvis_ai_pb2, "ResolveConfigRequest")
    assert hasattr(cuvis_ai_pb2, "GetParameterSchemaRequest")
    assert hasattr(cuvis_ai_pb2, "ValidateConfigRequest")


def test_new_pipeline_rpcs_exist():
    """Verify new pipeline building RPCs defined."""
    assert hasattr(cuvis_ai_pb2, "LoadPipelineRequest")
    assert hasattr(cuvis_ai_pb2, "LoadPipelineWeightsRequest")
    assert hasattr(cuvis_ai_pb2, "SetTrainRunConfigRequest")
    assert not hasattr(cuvis_ai_pb2, "BuildPipelineRequest")


def test_load_pipeline_uses_wrapper():
    """LoadPipelineRequest should expose typed PipelineConfig wrapper."""
    request = cuvis_ai_pb2.LoadPipelineRequest()
    assert hasattr(request, "pipeline")
    assert isinstance(request.pipeline, cuvis_ai_pb2.PipelineConfig)
    assert not hasattr(request, "pipeline_bytes")


def test_load_weights_oneof():
    """Verify LoadPipelineWeightsRequest supports both weight sources."""
    request = cuvis_ai_pb2.LoadPipelineWeightsRequest()
    assert hasattr(request, "weights_path")
    assert hasattr(request, "weights_bytes")
