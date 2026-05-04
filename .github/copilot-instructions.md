# Copilot Coding Agent Instructions for cuvis.ai

## Project Overview

Cuvis.AI is an open-source framework for building AI-powered processing pipelines for hyperspectral video data. It is split across **three repositories** — this repo is the **node catalog**, not the framework:

| Repository | Role |
|---|---|
| [cuvis-ai-core](https://github.com/cubert-hyperspectral/cuvis-ai-core) | Framework — base `Node` class, pipeline orchestration, two-phase training, gRPC services, plugin loader |
| [cuvis-ai-schemas](https://github.com/cubert-hyperspectral/cuvis-ai-schemas) | Protobuf / gRPC schema definitions and generated types |
| **cuvis-ai** (this repo) | ~40 domain-specific nodes (anomaly, preprocessing, band selection, visualization, video) and plugin configs |
| [cuvis-ai-cookbook](https://github.com/cubert-hyperspectral/cuvis-ai-cookbook) | Runnable example scripts and notebooks demonstrating cuvis-ai pipelines |

Both `cuvis-ai-core` and `cuvis-ai-schemas` are pinned dependencies in [pyproject.toml](../pyproject.toml). **Do not look for base framework code, pipeline orchestration, gRPC implementation, or Protobuf definitions inside this repo** — they live in the other two packages.

Typical imports to expect in code:

```python
from cuvis_ai_core.node import Node
from cuvis_ai_core.training import ...
from cuvis_ai_schemas.pipeline import PortSpec
from cuvis_ai_schemas.execution import Context, InputStream, Artifact
```

## Directory Structure (actual)

- [cuvis_ai/](../cuvis_ai/) — only `anomaly/`, `deciders/`, `node/`, `utils/`. No local `grpc/`, `pipeline/`, `training/`, `data/`, or `proto/` — those have been extracted.
- [configs/](../configs/) — Hydra/YAML configs. [configs/plugins/cuvis_ai_builtin.yaml](../configs/plugins/cuvis_ai_builtin.yaml) registers every node in this repo with the core plugin loader.
- Runnable example scripts now live in the [cuvis-ai-cookbook](https://github.com/cubert-hyperspectral/cuvis-ai-cookbook) repo (clone alongside this one).
- [tests/](../tests/) — organized by domain (`anomaly`, `deciders`, `node`, `preprocessors`, `training`, `utils`, `docs`, `plugins`). Shared fixtures in [tests/fixtures/](../tests/fixtures/) auto-load via [tests/conftest.py](../tests/conftest.py).
- [tools/](../tools/) — helper scripts: `generate_node_port_stubs.py`, `validate_trainrun_configs.py`.
- [docs/](../docs/) — MkDocs source.

## Development Workflow

- **Python 3.11 only** (`requires-python = ">=3.11,<3.12"`).
- **Dependency & environment management:** Use [`uv`](https://docs.astral.sh/uv/) exclusively. Never use bare `python` or `pip`.
  - Sync environment: `uv sync` (add `--locked --extra dev` for full toolchain)
  - Run scripts/tests: `uv run python ...` or `uv run pytest`
- **Build package:** `uv build`
- **Build docs:** `uv sync --locked --extra docs && mkdocs build`

## CLI Entry Points

Defined in [pyproject.toml](../pyproject.toml) `[project.scripts]`:

- `create-stubs` — local ([tools/generate_node_port_stubs.py](../tools/generate_node_port_stubs.py)), auto-generates node port stubs.
- `dataset`, `restore-pipeline`, `restore-trainrun` — provided by `cuvis_ai_core`, available after install.

## Testing

- Run with `uv run pytest`. See [tests/README.md](../tests/README.md) for the full fixture catalog.
- Pytest markers: `unit`, `integration`, `slow` (opt-in with `--runslow`), `gpu`, `check_links`. CI default excludes `slow` and `check_links`.
- Use built-in `tmp_path` for temp dirs; prefer shared factories in [tests/fixtures/](../tests/fixtures/) over ad-hoc fixtures.
- Coverage: `uv run pytest --cov=cuvis_ai --cov-report=term-missing`.

## Conventions

- **Logging:** Loguru (`from loguru import logger`). Never `print()` in production code.
- **Configuration:** Hydra/OmegaConf. Store configs under [configs/](../configs/).
- **Lint/format:** Ruff (`uv run ruff check .`, `uv run ruff format .`). Line length 100, configured in [pyproject.toml](../pyproject.toml).
- **Docstrings:** Google/NumPy style. Coverage enforced by `interrogate` (`fail-under = 95.0`, excludes tests/docs).
- **Node registration:** every new node in this repo must be added to [configs/plugins/cuvis_ai_builtin.yaml](../configs/plugins/cuvis_ai_builtin.yaml) so the core plugin loader can discover it.
- **Dependencies:** only open-source, permissive licenses (MIT/BSD/Apache-2.0).

## Common Commands

- Run all fast tests: `uv run pytest -m "not slow and not check_links" -v`
- Run a single test file: `uv run pytest tests/node/test_bandpass.py -v`
- Run an example: clone the [cuvis-ai-cookbook](https://github.com/cubert-hyperspectral/cuvis-ai-cookbook) and run `uv run python examples/channel_selector.py` from there.
- Build docs: `uv sync --locked --extra docs && mkdocs build`

## References

- [README.md](../README.md) — three-repo overview, quick start, documentation links.
- [tests/README.md](../tests/README.md) — markers, fixtures, common test patterns.
- [CONTRIBUTING.md](../CONTRIBUTING.md) — contribution guidelines.
- [CHANGELOG.md](../CHANGELOG.md) — release history.

---
For unclear conventions, prefer patterns from the files above and existing code. When a behavior seems to be "missing" in this repo, check `cuvis-ai-core` and `cuvis-ai-schemas` before adding new code here.
