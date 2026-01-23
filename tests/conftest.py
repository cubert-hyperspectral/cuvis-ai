"""Pytest configuration and shared fixtures for cuvis.ai tests.

All fixtures have been organized into separate modules in tests/fixtures/:

Core Fixtures:
- paths.py: Path and configuration fixtures (temp dirs, mock_pipeline_dir, etc.)
- data_factory.py: Test data creation (test_data_files, data_config_factory, create_test_cube, training_config_factory)
- mock_sdk.py: Mock CUVIS SDK
- mock_nodes.py: Mock node implementations

For detailed documentation and usage examples, see tests/README.md

Import them via pytest's automatic discovery or explicitly from tests.fixtures.
"""

from __future__ import annotations

import pytest

# Import all fixtures so pytest can discover them
pytest_plugins = [
    "tests.fixtures.paths",
    "tests.fixtures.data_factory",
    "tests.fixtures.mock_sdk",
    "tests.fixtures.mock_nodes",
]


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
