"""Fixtures for testing.

All fixtures have been organized into separate modules in tests/fixtures/:
- paths.py: Path and configuration fixtures (temp dirs, config paths)
- data_factory.py: Test data creation factories
- config_factory.py: Pipeline and experiment config factories
- mock_sdk.py: Mock CUVIS SDK
- grpc.py: gRPC testing utilities
- sessions.py: Session management fixtures

Import them via pytest's automatic discovery or explicitly from tests.fixtures.
"""

from .config_factory import create_pipeline_config_proto
from .mock_nodes import MockStatisticalTrainableNode
from .sample_metrics import SampleCustomMetrics

__all__ = [
    "MockStatisticalTrainableNode",
    "create_pipeline_config_proto",
    "SampleCustomMetrics",
]
