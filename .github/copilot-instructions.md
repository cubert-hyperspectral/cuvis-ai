# Copilot Coding Agent Instructions for cuvis.ai

## Project Overview
- **cuvis.ai** is a modular AI toolkit for hyperspectral imaging, supporting graph-based workflows, remote inference/training via gRPC, and flexible configuration.
- Major components: `cuvis_ai/` (core logic), `configs/` (YAML configs), `examples/`, `tests/`, `proto/` (gRPC/protobuf), and `docs/`.
- gRPC API enables remote control and integration with C++/other clients. See [examples/grpc/](../examples/grpc/) for usage.

## Development Workflow
- **Dependency & environment management:** Use [`uv`](https://docs.astral.sh/uv/) exclusively. Never use bare `python` or `pip`.
  - Sync environment: `uv sync` (add `--locked --extra dev` for full toolchain)
  - Run scripts/tests: `uv run python ...` or `uv run pytest`
- **Testing:**
  - All tests use `pytest` (run with `uv run pytest`).
  - Shared fixtures in `tests/fixtures/` and `tests/conftest.py` (see [tests/README.md](../tests/README.md)).
  - Use built-in `tmp_path` for temp dirs; prefer shared factories over ad-hoc fixtures.
- **Linting/Formatting:**
  - Use Ruff: `uv run ruff check .` and `uv run ruff format .`
  - Configured via `pyproject.toml`.
- **Builds & Docs:**
  - Build package: `uv build`
  - Build docs: `uv sync --locked --extra docs` then `mkdocs build`

## Project Conventions
- **Logging:** Use Loguru (`from loguru import logger`). Never use `print()` in production code.
- **Configuration:** Use Hydra/OmegaConf for CLI/config management. Store configs in `configs/`.
- **Directory structure:**
  - `cuvis_ai/` — core modules (anomaly, data, deciders, grpc, node, pipeline, training, utils)
  - `proto/` — protobuf/gRPC definitions
  - `examples/` — runnable scripts, including gRPC demos
  - `tests/` — all tests, fixtures, and test helpers
- **gRPC:**
  - Service definitions in `proto/cuvis_ai/`, implementation in `cuvis_ai/grpc/`
  - Use in-process gRPC fixtures for tests (`grpc_stub`)

## Patterns & Guardrails
- Prefer Hydra for new CLI tools (not argparse/Fire).
- Use only open-source, permissive dependencies (MIT/BSD/Apache-2.0).
- Keep docstrings concise; use Google/NumPy style for public APIs.
- Update `docs/` and `docs_dev/` for new features and internal design docs.
- Use `uv run` for all commands to ensure correct environment.

## Examples
- Run a test: `uv run pytest tests/`
- Run a script: `uv run python examples/channel_selector.py`
- Coverage: `uv run pytest --cov=cuvis_ai --cov-report=term-missing`
- Build docs: `uv sync --locked --extra docs && mkdocs build`

## References
- [README.md](../README.md) — project intro, setup, and gRPC API
- [tests/README.md](../tests/README.md) — test fixtures and patterns
- [.clinerules](../.clinerules) — project-specific conventions and guardrails

---
For unclear or missing conventions, prefer patterns from the above files and existing code. Ask for review if unsure about project-specific logic or workflows.
