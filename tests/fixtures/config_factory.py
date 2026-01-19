"""Configuration factory fixtures for creating pipeline and experiment configs."""

import json

import pytest
import yaml

from cuvis_ai_core.grpc.v1 import cuvis_ai_pb2


def create_pipeline_config_proto(
    pipeline_identifier: str,
    load_weights: bool = False,
) -> cuvis_ai_pb2.PipelineConfig:
    """Create PipelineConfig proto from pipeline identifier.

    This helper function creates a PipelineConfig by loading the YAML structure
    and optionally the weights. For tests, we typically want structure only.

    Args:
        pipeline_identifier: Pipeline path or short name
        load_weights: Whether to load weights (default False for tests)

    Returns:
        PipelineConfig proto with config_bytes field populated

    Examples:
        >>> config = create_pipeline_config_proto("channel_selector")
        >>> config = create_pipeline_config_proto("channel_selector", load_weights=True)
    """
    from cuvis_ai_core.grpc.helpers import resolve_pipeline_path
    from cuvis_ai_core.training.config import PipelineConfig

    # When loading weights, provide the identifier directly without validation
    # This allows the service to handle path resolution and error handling
    if load_weights:
        return cuvis_ai_pb2.PipelineConfig(config_bytes=pipeline_identifier.encode("utf-8"))

    # For load_weights=False, we try to resolve and embed the full YAML content.
    # If resolution fails (e.g., non-existent pipeline), fall back to passing the
    # identifier through so the service can surface a NOT_FOUND error.
    try:
        pipeline_path = resolve_pipeline_path(pipeline_identifier)
    except FileNotFoundError:
        return cuvis_ai_pb2.PipelineConfig(config_bytes=pipeline_identifier.encode("utf-8"))

    # Load the YAML config
    with open(pipeline_path) as f:
        config_dict = yaml.safe_load(f)

    # Convert to PipelineConfig and serialize to JSON
    pipeline_config = PipelineConfig.from_dict(config_dict)

    # For tests without weights, we pass the full JSON config
    # This bypasses the .pt file loading in the service
    config_json = json.dumps(pipeline_config.to_dict())

    return cuvis_ai_pb2.PipelineConfig(config_bytes=config_json.encode("utf-8"))


@pytest.fixture
def create_pipeline_config():
    """Fixture that provides the helper function for creating pipeline configs."""
    return create_pipeline_config_proto


@pytest.fixture
def pipeline_factory(tmp_path):
    """Factory for creating temporary pipeline directories with test configs.

    Returns a callable that writes YAML (and optional weights) files into a
    temporary directory for discovery and loading tests.

    Returns:
        Callable: Function that creates pipeline files and returns the directory path
    """

    def _create_pipeline_dir(configs: list[tuple[str, dict | str, bool]] | None = None):
        pipeline_dir = tmp_path / "pipeline"
        pipeline_dir.mkdir(exist_ok=True)

        if configs:
            for name, yaml_content, has_weights in configs:
                yaml_path = pipeline_dir / f"{name}.yaml"
                if isinstance(yaml_content, dict):
                    yaml_path.write_text(yaml.dump(yaml_content))
                else:
                    yaml_path.write_text(yaml_content)

                if has_weights:
                    weights_path = pipeline_dir / f"{name}.pt"
                    weights_path.write_bytes(b"fake_weights_data")

        return pipeline_dir

    return _create_pipeline_dir


@pytest.fixture
def mock_pipeline_dict():
    """Create a mock pipeline configuration dict for testing.

    Returns a minimal but valid pipeline structure with nodes and connections
    suitable for testing experiment management functionality.

    Returns:
        dict: Pipeline configuration dictionary
    """
    return {
        "metadata": {
            "name": "Test_Pipeline",
            "description": "Pipeline for testing",
            "created": "2024-12-04",
            "tags": ["test"],
            "author": "test",
        },
        "nodes": [
            {
                "name": "LentilsAnomalyDataNode",
                "class": "cuvis_ai.node.data.LentilsAnomalyDataNode",
                "params": {"normal_class_ids": [0]},
            },
            {
                "name": "MinMaxNormalizer",
                "class": "cuvis_ai.node.normalization.MinMaxNormalizer",
                "params": {"eps": 1e-06, "use_running_stats": True},
            },
            {
                "name": "SoftChannelSelector",
                "class": "cuvis_ai.node.selector.SoftChannelSelector",
                "params": {
                    "n_select": 3,
                    "input_channels": 61,
                    "init_method": "uniform",
                    "temperature_init": 5.0,
                    "temperature_min": 0.1,
                    "temperature_decay": 0.9,
                    "hard": False,
                    "eps": 1e-06,
                },
            },
            {
                "name": "TopKIndices",
                "class": "cuvis_ai.node.selector.TopKIndices",
                "params": {"k": 3},
            },
        ],
        "connections": [
            {"from": "LentilsAnomalyDataNode.outputs.cube", "to": "MinMaxNormalizer.inputs.data"},
            {
                "from": "MinMaxNormalizer.outputs.normalized",
                "to": "SoftChannelSelector.inputs.data",
            },
            {"from": "SoftChannelSelector.outputs.weights", "to": "TopKIndices.inputs.weights"},
        ],
    }


@pytest.fixture
def minimal_pipeline_dict():
    """Minimal valid pipeline configuration dictionary for tests."""
    return {
        "version": "1.0.0",
        "metadata": {
            "name": "Test Pipeline",
            "description": "Minimal test pipeline",
            "created": "2024-12-04",
            "tags": ["test"],
        },
        "nodes": [
            {
                "name": "data",
                "class": "cuvis_ai.node.data.LentilsAnomalyDataNode",
                "params": {"normal_class_ids": [0]},
            },
            {
                "name": "normalizer",
                "class": "cuvis_ai.node.normalization.MinMaxNormalizer",
                "params": {"eps": 1e-6, "use_running_stats": False},
            },
            {
                "name": "selector",
                "class": "cuvis_ai.node.selector.SoftChannelSelector",
                "params": {
                    "n_select": 3,
                    "input_channels": 61,
                    "init_method": "uniform",
                    "temperature_init": 5.0,
                    "temperature_min": 0.1,
                    "temperature_decay": 0.9,
                    "hard": False,
                },
            },
        ],
        "connections": [
            {"from": "data.outputs.cube", "to": "normalizer.inputs.data"},
            {"from": "normalizer.outputs.normalized", "to": "selector.inputs.data"},
        ],
    }


@pytest.fixture
def mock_experiment_dict(mock_pipeline_dict):
    """Create a mock experiment configuration dict for testing.

    Returns a complete experiment structure with pipeline, data, and training configs
    suitable for testing experiment management functionality.

    Args:
        mock_pipeline_dict: Pipeline configuration fixture

    Returns:
        dict: Experiment configuration dictionary
    """
    return {
        "name": "test_experiment",
        "pipeline": mock_pipeline_dict,
        "data": {
            "cu3s_file_path": "/data/test.cu3s",
            "batch_size": 4,
            "processing_mode": "Reflectance",
            "train_ids": [0, 1, 2],
            "val_ids": [3, 4],
            "test_ids": [5, 6],
        },
        "training": {
            "seed": 42,
            "trainer": {
                "max_epochs": 100,
                "accelerator": "auto",
            },
            "optimizer": {
                "name": "adamw",
                "lr": 0.001,
            },
        },
    }


@pytest.fixture
def saved_pipeline(grpc_stub, session, tmp_path, monkeypatch, data_config_factory):
    """Create and save a pipeline with weights for testing.

    This fixture creates a session, saves its pipeline with metadata,
    and provides both the pipeline and weights paths for testing.
    Useful for testing pipeline loading and restoration.

    Args:
        grpc_stub: gRPC stub fixture
        session: Session factory fixture
        tmp_path: Temporary directory fixture
        monkeypatch: Pytest monkeypatch fixture

    Returns:
        dict: Contains 'pipeline_path', 'weights_path', and 'session_id'
    """
    from cuvis_ai_core.grpc.v1 import cuvis_ai_pb2

    monkeypatch.setenv("CUVIS_CONFIGS_DIR", str(tmp_path))
    session_id = session()

    # Initialize weights with a quick statistical training run
    train_request = cuvis_ai_pb2.TrainRequest(
        session_id=session_id,
        trainer_type=cuvis_ai_pb2.TRAINER_TYPE_STATISTICAL,
        data=data_config_factory(),
    )
    for _ in grpc_stub.Train(train_request):
        pass

    save_response = grpc_stub.SavePipeline(
        cuvis_ai_pb2.SavePipelineRequest(
            session_id=session_id,
            pipeline_path=str(tmp_path / "test_pipeline.yaml"),
            metadata=cuvis_ai_pb2.PipelineMetadata(
                name="Test Pipeline",
                description="Pipeline for testing",
                created="2024-11-27",
            ),
        )
    )

    return {
        "pipeline_path": save_response.pipeline_path,
        "weights_path": save_response.weights_path,
        "session_id": session_id,
    }


@pytest.fixture
def pipeline_yaml_only(tmp_path):
    """Create a pipeline YAML file without corresponding weights file.

    This fixture creates a valid pipeline configuration file without the .pt weights.
    Useful for testing scenarios where weights are missing or not yet generated,
    such as pipeline structure validation or loading without weights.

    Args:
        tmp_path: Temporary directory fixture

    Returns:
        str: Path to the pipeline YAML file
    """
    pipeline_path = tmp_path / "yaml_only.yaml"
    pipeline_path.write_text("""
version: 1.0.0
metadata:
  name: "YAML Only Pipeline"
  description: "Pipeline without weights"
  created: "2024-11-27"
  cuvis_ai_version: "0.1.5"
  tags: ["test"]

nodes:
  - name: LentilsAnomalyDataNode
    class: cuvis_ai.node.data.LentilsAnomalyDataNode
    params:
      normal_class_ids: [0]
      wavelengths: null
  - name: MinMaxNormalizer
    class: cuvis_ai.node.normalization.MinMaxNormalizer
    params:
      eps: 1.0e-06
      use_running_stats: true
  - name: SoftChannelSelector
    class: cuvis_ai.node.selector.SoftChannelSelector
    params:
      eps: 1.0e-06
      hard: false
      init_method: uniform
      input_channels: 61
      n_select: 3
      temperature_decay: 0.9
      temperature_init: 5.0
      temperature_min: 0.1
  - name: TopKIndices
    class: cuvis_ai.node.selector.TopKIndices
    params:
      k: 3

connections:
  - from: LentilsAnomalyDataNode.outputs.cube
    to: MinMaxNormalizer.inputs.data
  - from: MinMaxNormalizer.outputs.normalized
    to: SoftChannelSelector.inputs.data
  - from: SoftChannelSelector.outputs.weights
    to: TopKIndices.inputs.weights
""")

    return str(pipeline_path)


@pytest.fixture
def experiment_file(tmp_path, mock_experiment_dict):
    """Create a valid experiment file for testing restoration.

    This fixture creates a properly formatted experiment YAML file
    with pipeline, data, and training configurations. Useful for testing
    experiment restoration and validation.

    Args:
        tmp_path: Temporary directory fixture
        mock_experiment_dict: Mock experiment config fixture

    Returns:
        str: Path to the experiment YAML file
    """
    exp_path = tmp_path / "test_experiment.yaml"

    with open(exp_path, "w") as f:
        yaml.dump(mock_experiment_dict, f)

    return str(exp_path)
