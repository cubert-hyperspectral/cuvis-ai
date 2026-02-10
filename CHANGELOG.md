# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Comprehensive documentation structure with 70+ markdown files covering all framework aspects
- 6 end-to-end tutorials (RX Statistical, Channel Selector, Deep SVDD, AdaCLIP, gRPC workflow, Remote Deployment)
- Complete API reference documentation with mkdocstrings integration and numpy-style docstrings
- Plugin system documentation (overview, development guide, usage guide) with 3 comprehensive guides
- gRPC API documentation with sequence diagrams and client patterns (4 guides)
- Configuration reference guides (config groups, TrainRun schema, pipeline schema, Hydra composition patterns)
- 7 how-to guides (build pipelines in Python/YAML, restore pipelines, add built-in nodes, monitoring, remote gRPC)
- Complete node catalog documenting 50+ built-in nodes across 11 categories
- Development guides (contributing with plugin workflow, docstring standards, git hooks)
- 20+ diagrams (11 Mermaid diagrams, Graphviz pipeline visualizations)
- Central plugin registry at `configs/plugins/registry.yaml`
- Git hooks for automated code quality checks (ruff format, module case checking)
- AnomalyPixelStatisticsMetric node in cuvis_ai.node.metrics for computing anomaly pixel statistics (replaces duplicate SampleCustomMetrics in examples)
- Automated test data downloader (scripts/download_data.py) for Hugging Face datasets
- Documentation testing infrastructure: link checker, CLI command tests, and runnable code example validation
- Buffer initialization best practices section in node system deep dive documentation
- Review status tracking system with warning banners for documentation quality assurance
- Automated PyPI release workflow with GitHub Actions
- Versioned documentation deployment to gh-pages using mike
- Comprehensive CI/CD pipeline with parallel job execution
  - Test suite with coverage reporting (Codecov integration)
  - Linting and style checks (ruff, interrogate)
  - Security scanning (pip-audit, bandit, detect-secrets)
  - Documentation build validation with link checking
- Automated dependency updates via Dependabot
- Security baseline for secret detection (`.secrets.baseline`)
- Smoke tests for TestPyPI releases

### Changed
- Migrated to cuvis-ai-schemas package for standardized schema definitions across repositories
- Enhanced docstrings to 95%+ coverage across all public APIs
- Updated all code examples to use `uv run` command pattern
- README refactored to eliminate duplicate content - built-in nodes, community plugins, and plugin development sections now link to comprehensive documentation
- Contributing guide enhanced with complete 7-step plugin contribution workflow (develop → test → publish → submit → PR → review → maintain)
- Documentation navigation reorganized with 14 major sections and clear hierarchy
- MkDocs configuration updated to Material theme with advanced search, dark mode support, deep orange color scheme, Lato and Source Code Pro fonts, custom logo and favicon
- Split configs/trainrun/default.yaml into default_statistical.yaml and default_gradient.yaml for clearer workflow separation
- Updated ScoreToLogit module paths in pipeline configurations (rx_statistical.yaml, channel_selector.yaml)
- Updated `pyproject.toml` for PyPI compliance
  - License format changed to SPDX identifier: `Apache-2.0`
  - Python version classifiers aligned to 3.11 only
  - Ruff target version updated to `py311`
  - Added security tooling to dev dependencies: twine, pip-audit, bandit, detect-secrets
  - Added mike for documentation versioning
  - Added mypy and bandit tool configurations
- cuvis-ai-core dependency handling updated for development/release workflow
  - Development: Use local editable path via `[tool.uv.sources]`
  - Release: Resolve from PyPI
- CI/CD uses `cubertgmbh/cuvis_pyil:3.5.0-ubuntu24.04` base image

### Fixed
- All broken internal links in documentation
- Outdated module references (cuvis_ai.pipeline → cuvis_ai_core)
- Empty documentation files and placeholder content
- Plugin installation and dependency management issues
- DataLoader access violation with proper num_workers configuration
- Node import paths for cuvis-ai-core migration
- TrainablePCA test suite (17 failing tests) by adding required num_channels parameter and centralizing fixture in tests/fixtures/mock_nodes.py
- Config name references throughout documentation (trainrun/default → default_statistical/default_gradient)
- Non-existent train.py script references replaced with actual runnable examples (rx_statistical.py, channel_selector.py)
- Output path references corrected from timestamp placeholders to actual directory names (base_trainrun)
- monitoring.md references redirected to visualization.md
- Training API references (pipeline.train() → trainer.fit()) in documentation
- MkDocs build warnings and docstring formatting issues
- Package metadata alignment for PyPI submission requirements
- Documentation build strictness (allows 24 external file reference warnings)
- Test markers properly exclude slow and GPU tests in CI

### Documentation
- Documentation now deployed to https://cubert-hyperspectral.github.io/cuvis-ai/
- Version selector available for accessing docs matching installed package version
- Latest version always available at `/latest/` path

### Removed
- SETUP_TEST_DATA.md (replaced by automated scripts/download_data.py)

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
