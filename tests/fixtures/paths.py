"""Path and configuration fixtures for tests."""

import tempfile
from pathlib import Path

import pytest


@pytest.fixture
def config_dir():
    """Path to the configs directory."""
    return Path(__file__).parent.parent.parent / "configs"


@pytest.fixture
def pipeline_dir(config_dir):
    """Path to the pipeline configs directory."""
    return config_dir / "pipeline"


@pytest.fixture
def test_data_path():
    """Path to test data directory."""
    return Path("data")


@pytest.fixture
def temp_workspace():
    """Create a temporary workspace with standard subdirectories.

    Creates a workspace directory structure with:
    - pipeline/ - for pipeline YAML and weight files
    - experiments/ - for experiment configuration files
    - models/ - for trained model artifacts

    Useful for end-to-end workflow tests that need organized file structure.

    Yields:
        Path: Path to workspace root directory
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = Path(tmpdir)
        (workspace / "pipeline").mkdir()
        (workspace / "experiments").mkdir()
        (workspace / "models").mkdir()
        yield workspace


@pytest.fixture
def mock_pipeline_dir(tmp_path, monkeypatch):
    """Create and monkeypatch a temporary pipeline directory."""
    pipeline_dir = tmp_path / "pipeline"
    pipeline_dir.mkdir()

    monkeypatch.setenv("CUVIS_CONFIGS_DIR", str(pipeline_dir.parent))
    monkeypatch.setattr("cuvis_ai.grpc.helpers.get_server_base_dir", lambda: pipeline_dir.parent)

    yield pipeline_dir
