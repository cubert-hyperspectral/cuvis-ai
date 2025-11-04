import pytest


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
