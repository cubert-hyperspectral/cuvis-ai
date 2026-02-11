"""Test CLI commands from documentation.

This module tests that all CLI commands mentioned in the documentation
are valid and execute without errors.
"""

import subprocess

import pytest

# Basic CLI commands that should always work (--help flags)
CLI_COMMANDS_BASIC = [
    "uv run pytest --version",
    "uv run ruff --version",
    "uv run restore-pipeline --help",
    "uv run restore-trainrun --help",
]


@pytest.mark.parametrize("command", CLI_COMMANDS_BASIC)
def test_basic_cli_commands(command):
    """Test that basic CLI commands execute successfully."""
    result = subprocess.run(command.split(), capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, f"Command failed: {command}\n{result.stderr}"


def test_restore_pipeline_help():
    """Test restore-pipeline help output contains expected flags."""
    result = subprocess.run(
        ["uv", "run", "restore-pipeline", "--help"], capture_output=True, text=True
    )
    assert result.returncode == 0, f"Command failed:\n{result.stderr}"

    # Check for expected flags in help output
    expected_flags = ["--pipeline-path", "--device"]
    for flag in expected_flags:
        assert flag in result.stdout, f"Expected flag '{flag}' not found in help output"


def test_restore_trainrun_help():
    """Test restore-trainrun help output contains expected flags."""
    result = subprocess.run(
        ["uv", "run", "restore-trainrun", "--help"], capture_output=True, text=True
    )
    assert result.returncode == 0, f"Command failed:\n{result.stderr}"

    # Check for expected flags in help output
    expected_flags = ["--trainrun-path", "--mode"]
    for flag in expected_flags:
        assert flag in result.stdout, f"Expected flag '{flag}' not found in help output"


def test_mkdocs_build():
    """Test that documentation builds successfully."""
    result = subprocess.run(
        ["uv", "run", "mkdocs", "build"], capture_output=True, text=True, timeout=120
    )
    # Allow warnings but fail on errors
    assert result.returncode == 0, f"mkdocs build failed:\n{result.stderr}"
    assert "ERROR" not in result.stderr, f"Build contains errors:\n{result.stderr}"


def test_ruff_format_check():
    """Test that ruff format check runs successfully."""
    result = subprocess.run(
        ["uv", "run", "ruff", "format", "--check", "."], capture_output=True, text=True, timeout=60
    )
    # This may fail if files need formatting, but should execute without errors
    assert result.returncode in [0, 1], f"Ruff format check error:\n{result.stderr}"


def test_ruff_lint():
    """Test that ruff lint runs successfully."""
    result = subprocess.run(
        ["uv", "run", "ruff", "check", "cuvis_ai"], capture_output=True, text=True, timeout=60
    )
    # This may fail if there are lint errors, but should execute without crashes
    assert result.returncode in [0, 1], f"Ruff lint error:\n{result.stderr}"
