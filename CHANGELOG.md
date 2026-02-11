# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Comprehensive documentation site (70+ pages): 6 tutorials, API reference, node catalog (50+ nodes across 11 categories), gRPC guides, config reference, how-to guides, plugin system docs, development guides, 20+ Mermaid/Graphviz diagrams
- MkDocs Material theme with dark mode, versioned deployment via mike, custom branding (deep orange, Lato/Source Code Pro fonts, logo/favicon), numpy-style mkdocstrings
- `AnomalyPixelStatisticsMetric` node in `cuvis_ai.node.metrics` (replaces duplicate `SampleCustomMetrics` in examples)
- `deep_svdd_factory.py` utility module with `ChannelConfig` dataclass in `cuvis_ai/utils/`
- Central plugin registry at `configs/plugins/registry.yaml`
- `configs/trainrun/default_statistical.yaml` for statistical-only training workflows
- CI/CD pipeline (`ci.yml`): test + coverage (Codecov), lint (ruff, interrogate), security (pip-audit, bandit, detect-secrets), typecheck (mypy) — replaces `run_tests.yml`
- PyPI release workflow (`pypi-release.yml`): build validation, TestPyPI publish with smoke tests, production PyPI publish, versioned docs deploy to gh-pages
- Dependabot configuration for GitHub Actions and pip dependencies
- Automated test data downloader (`scripts/download_data.py` with `download-data` CLI entry point)
- Documentation test suite (`tests/docs/`): link checker, CLI command tests, runnable code example validation
- Git hooks for automated code quality checks (ruff format, module case checking)
- `LICENSE` file (Apache-2.0 full text)
- `pytest.ini`, `codecov.yml`, `.secrets.baseline`, `baseline_coverage.txt`

### Changed
- **Breaking**: `TrainablePCA.__init__()` now requires `num_channels` parameter; buffers initialized with correct shapes
- Migrated type imports from `cuvis_ai_core` to new `cuvis-ai-schemas` package across all source files (`PortSpec`, `Context`, `InputStream`, `Metric`, `ExecutionStage`)
- Renamed `RXLogitHead` → `ScoreToLogit` and moved from `cuvis_ai.anomaly.rx_logit_head` to `cuvis_ai.node.conversion`; updated all pipeline configs and examples
- Renamed `BaseDecider` import to `BinaryDecider` in deciders module
- Split `configs/trainrun/default.yaml` into `default_statistical.yaml` and `default_gradient.yaml`
- Enhanced docstrings to 95%+ coverage across all public APIs (NumPy-style)
- `pyproject.toml` updates for PyPI compliance:
  - Package name: `cuvis_ai` → `cuvis-ai`; license: SPDX `Apache-2.0`; author email updated
  - Python classifiers aligned to 3.11 only; ruff target `py310` → `py311`
  - Added tool configs: `[tool.interrogate]` (95% threshold), `[tool.mypy]`, `[tool.bandit]`
- Dependencies: added `cuvis-ai-schemas[full]>=0.1.0`; loosened `cuvis>=3.5.0` (was `==3.5.0`); pinned `cuvis-ai-core>=0.1.2`; removed `graphviz>=0.21`
- Dev deps: added twine, pip-audit, bandit, detect-secrets, pip-licenses, cyclonedx-bom, interrogate
- Docs deps: added mike, pytest-check-links, pytest-md-report
- `restore-pipeline`/`restore-trainrun` CLI entry points now point to `cuvis_ai_core`
- cuvis-ai-core dependency handling: local editable path for dev, PyPI for release
- README refactored; CONTRIBUTING.md enhanced with 7-step plugin contribution workflow
- Examples updated: removed inline `SampleCustomMetrics`, updated all imports for schema migration and ScoreToLogit rename

### Fixed
- LAD detector `reset()`: buffers now initialized with proper shapes instead of `torch.empty(0)`
- LAD detector `unfreeze()`: preserves device when converting buffers to parameters
- TrainablePCA: 17 failing tests fixed by adding required `num_channels` parameter and proper buffer shapes; centralized fixture in `tests/fixtures/mock_nodes.py`
- Node import paths updated for cuvis-ai-schemas migration
- Config references: `trainrun/default` → `default_statistical`/`default_gradient`; `RXLogitHead` → `ScoreToLogit` in pipeline YAMLs
- Documentation: broken internal links, outdated module references, empty placeholder content, incorrect script/path references
- MkDocs build warnings and docstring formatting issues
- Package metadata alignment for PyPI submission

### Removed
- `restore_pipeline.md` from repo root (replaced by docs site)
- `changelog.md` (replaced by `CHANGELOG.md` with Keep a Changelog format)
- `.github/workflows/run_tests.yml` (replaced by `ci.yml`)
- `docs/api/grpc_api.md` and `docs/reference/architecture.md` (replaced by expanded docs sections)

## [0.2.3] - 2026-01-29

### Added
- Plugin system with Git repository and local filesystem support via extended NodeRegistry
- Pydantic plugin configuration models: GitPluginConfig, LocalPluginConfig, PluginManifest with strict validation
- Plugin caching in ~/.cuvis_plugins/ with intelligent cache reuse and version verification
- Session-scoped plugin isolation for gRPC services (each session has independent plugin namespaces)
- New gRPC RPCs: LoadPlugins, ListLoadedPlugins, GetPluginInfo, ListAvailableNodes, ClearPluginCache
- JSON transport pattern for plugin manifests via config_bytes field
- Test migration infrastructure: 426 tests moved to cuvis-ai-core with reusable fixtures

### Changed
- Repository split into cuvis-ai-core (framework) and cuvis-ai (catalog) with clear API boundaries
- Import pattern change: `from cuvis_ai_core.* import ...` for framework components
- Framework extraction: base Node class, port system, Pipeline, training infrastructure, gRPC services, NodeRegistry, data infrastructure moved to cuvis-ai-core

### Fixed
- DataLoader access violation resolved with num_workers=0
- Single-threaded gRPC servers for cuvis SDK compatibility
- 421 tests passing in cuvis-ai-core with independent CI/CD capability

## [0.2.2] - 2026-01-15

### Added
- Restoration utilities in cuvis_ai.utils.restore module
- CLI entry points: `uv run restore-pipeline` and `uv run restore-trainrun`
- restore_pipeline.md guide with CLI and Python API examples

### Changed
- Restoration utilities consolidated for pipeline and trainrun recovery
- restore_trainrun() auto-detects statistical vs gradient workflows
- Python API surface standardized: `from cuvis_ai.utils import restore_pipeline, restore_trainrun`

### Removed
- Duplicate example scripts replaced with library utilities

## [0.2.1] - 2026-01-08

### Added
- Pydantic v2 config models as single source of truth with validation and JSON Schema introspection
- Server-side Hydra composition with session-scoped search paths
- New RPCs: ResolveConfig, ValidateConfig, GetParameterSchema, SetSessionSearchPaths
- Explicit 4-step workflow: CreateSession → Build/Load Pipeline → SetTrainRunConfig → Train

### Changed
- Terminology update: Experiment → TrainRun across configs, RPCs, paths, and examples
- gRPC service refactored into modular components (Session/Config/Pipeline/Training/TrainRun/Inference/Introspection/Discovery)
- Standardized config transport via config_bytes and central config registry

### Fixed
- RPC surface updated for new config resolution flow
- Tests and examples updated (596 tests passing, 65% coverage)

## [0.2.0] - 2025-12-20

### Added
- YAML-driven pipeline configuration with OmegaConf interpolation
- Hybrid NodeRegistry for built-ins and custom nodes
- End-to-end pipeline serialization: YAML structure + single .pt weights file
- Version/schema compatibility guards on load
- gRPC Canvas management and discovery RPCs
- Pipeline path resolution helpers (CUVIS_CANVAS_DIR environment variable)

### Changed
- gRPC API: CreateSession uses PipelineConfig (config_bytes)
- Train RPC requires DataConfig + TrainingConfig
- SaveCanvas/LoadCanvas replace SaveCheckpoint/LoadCheckpoint
- Node state management simplified to state_dict() + buffers/parameters

### Removed
- Custom serialize/load patterns replaced with standard state_dict()

## [0.1.5] - 2025-12-01

### Added
- gRPC service stack with proto definitions
- Buf Schema Registry integration for cross-language codegen
- Session management and PipelineBuilder
- File-based data access via DataConfig
- Two-phase training (statistical init → gradient fine-tuning)
- Pipeline introspection RPCs (inputs/outputs/visualization)
- Streaming training progress support

### Changed
- Output selection via output_specs
- Deterministic counter-based node naming

## [0.1.3] - 2025-11-06

### Added
- Port-based typed I/O system with PortSpec, InputPort, OutputPort
- Graph connection API with auto-validation
- Multi-input/output support for nodes
- Training integration with PyTorch Lightning

### Changed
- Nodes declare INPUT_SPECS/OUTPUT_SPECS with auto-created ports
- Executor refactored for port-based routing and stage-aware execution
- Core and training nodes migrated to typed I/O

---

[unreleased]: https://github.com/cubert-hyperspectral/cuvis-ai/compare/v0.2.3...HEAD
[0.2.3]: https://github.com/cubert-hyperspectral/cuvis-ai/compare/v0.2.2...v0.2.3
[0.2.2]: https://github.com/cubert-hyperspectral/cuvis-ai/compare/v0.2.1...v0.2.2
[0.2.1]: https://github.com/cubert-hyperspectral/cuvis-ai/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/cubert-hyperspectral/cuvis-ai/compare/v0.1.5...v0.2.0
[0.1.5]: https://github.com/cubert-hyperspectral/cuvis-ai/compare/v0.1.3...v0.1.5
[0.1.3]: https://github.com/cubert-hyperspectral/cuvis-ai/releases/tag/v0.1.3
