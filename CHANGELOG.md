# Changelog

## Unreleased

### Stage 1 – UV Integration & Project Structure
- Adopted `uv` as the canonical package manager across local workflows and CI, replacing legacy `pip` flows, regenerating `uv.lock`, and aligning shell/CI scripts with the lock file.
- Migrated dependencies into `pyproject.toml` extras, introduced `_version.py` with `importlib.metadata`-based versioning, and pruned obsolete packaging artifacts like `MANIFEST.in`.
- Expanded README/CONTRIBUTING with `uv` setup and validation guidance, then validated the tooling by provisioning a fresh environment and running the unit suite.

### Stage 2 – Linting & Formatting Modernization
- Added Ruff configuration to `pyproject.toml`, covering formatter settings, targeted lint checks, and the `dev` extra dependency.
- Documented `uv run ruff format .` and `uv run ruff check .` workflows alongside refreshed contributor guidance, while intentionally deferring automated lint runs per the refactoring plan.

### Stage 3 – Pytest Migration
- Migrated the test suite from `unittest` to `pytest`, relocating tests under `tests/`, adopting fixtures/asserts, and aligning CI naming.
- Configured `pytest` in `pyproject.toml` (addopts, discovery paths, warning filters) and added `pytest`/`pytest-cov` to the `dev` extra, verified with `uv run --extra dev pytest`.
- Updated README and contributor docs with `uv run pytest` instructions, coverage tips, and refreshed Docker/CI entrypoints.

### Stage 4 – Filename & Casing Standardization
- Renamed CamelCase modules to snake_case, updated package exports and downstream import sites, and refreshed documentation.
- Added `scripts/check_module_case.py` as a guard against future casing regressions and documented how to run it.

### Stage 5 – Absolute Import Adoption
- Replaced relative imports with absolute `cuvis_ai` imports to remove path ambiguity and circular import risk.
- Added `tests/test_imports.py` as a smoke test that skips optional third-party dependencies when absent, validated via `uv run pytest tests/test_imports.py`.

### Stage 6 – Final Polish & Verification
- Closed lingering casing gaps (including `cuvis_ai/tv_transforms/bandpass.py`) and refreshed README packaging validation guidance.
- Re-ran end-to-end checks with `uv build` and `uv run pytest` to confirm packaging metadata and the refactored test suite remain healthy.

### Upgrade Notes
- Run `uv sync --locked --extra dev` (adding extras such as `docs` when needed) after pulling to ensure your environment matches the committed lock file.
- Use `uv build` before release tagging to confirm packaging metadata and artifacts remain healthy.
