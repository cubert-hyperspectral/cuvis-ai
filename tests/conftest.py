from concurrent import futures
from pathlib import Path
from unittest.mock import Mock, patch

import grpc
import numpy as np
import pytest

from cuvis_ai.grpc import CuvisAIService, cuvis_ai_pb2_grpc


def pytest_addoption(parser: pytest.Parser) -> None:
    """Register a CLI flag for including slow tests."""
    parser.addoption(
        "--runslow",
        action="store_true",
        default=False,
        help="Run tests marked as slow.",
    )


def pytest_configure(config: pytest.Config) -> None:
    """Declare the custom marker so pytest does not warn."""
    config.addinivalue_line("markers", "slow: mark test as slow to skip by default")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Skip slow tests unless --runslow was requested explicitly."""
    if config.getoption("--runslow"):
        return

    skip_slow = pytest.mark.skip(reason="need --runslow option to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)


@pytest.fixture
def grpc_stub():
    """Create in-process gRPC client stub."""
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=2))
    service = CuvisAIService()
    cuvis_ai_pb2_grpc.add_CuvisAIServiceServicer_to_server(service, server)
    port = server.add_insecure_port("localhost:0")
    server.start()

    channel = grpc.insecure_channel(f"localhost:{port}")
    stub = cuvis_ai_pb2_grpc.CuvisAIServiceStub(channel)
    try:
        yield stub
    finally:
        channel.close()
        server.stop(None)


@pytest.fixture
def test_data_path():
    """Path to test data directory."""
    return Path("data")


@pytest.fixture
def mock_cuvis_sdk():
    """Mock CUVIS SDK to avoid thread-safety issues in tests."""
    # Create mock measurement object
    mock_measurement = Mock()
    mock_measurement.cube = Mock()
    mock_measurement.cube.array = np.random.rand(64, 64, 61).astype(np.float32)
    mock_measurement.cube.channels = 61
    mock_measurement.cube.wavelength = np.linspace(400, 1000, 61)
    mock_measurement.data = {"cube": True}  # Pretend cube is already loaded

    # Create mock session file
    mock_session = Mock()
    mock_session.get_measurement = Mock(return_value=mock_measurement)
    mock_session.__len__ = Mock(return_value=7)  # 7 measurements total

    # Create mock processing context
    mock_pc = Mock()
    mock_pc.apply = Mock(return_value=mock_measurement)
    mock_pc.processing_mode = Mock()

    # Mock COCO annotations
    mock_coco = Mock()
    mock_coco.category_id_to_name = {0: "background", 1: "anomaly"}
    mock_coco.image_ids = [0, 1, 2, 3, 4, 5, 6]  # Match measurement count
    mock_coco.annotations = Mock()
    mock_coco.annotations.where = Mock(return_value=[])  # No annotations for simplicity

    # Patch cuvis module imports
    with (
        patch("cuvis_ai.data.datasets.cuvis.SessionFile", return_value=mock_session),
        patch("cuvis_ai.data.datasets.cuvis.ProcessingContext", return_value=mock_pc),
        patch("cuvis_ai.data.datasets.cuvis.ProcessingMode") as mock_pm,
        patch("cuvis_ai.data.datasets.cuvis.ReferenceType") as mock_ref_type,
        patch("cuvis_ai.data.datasets.COCOData.from_path", return_value=mock_coco),
    ):
        # Setup ProcessingMode enum
        mock_pm.Raw = "Raw"
        mock_pm.Reflectance = "Reflectance"

        # Provide white/dark references so Reflectance mode passes
        mock_ref_type.White = "White"
        mock_ref_type.Dark = "Dark"
        mock_reference = Mock()
        mock_session.get_reference = Mock(
            side_effect=lambda _idx, ref_type: mock_reference
            if ref_type in (mock_ref_type.White, mock_ref_type.Dark)
            else None
        )

        yield {
            "session": mock_session,
            "processing_context": mock_pc,
            "measurement": mock_measurement,
            "coco": mock_coco,
        }
