# Changelog

## [Unreleased]

- Added reusable `WelfordAccumulator` utility (`cuvis_ai.utils.welford`) for streaming mean/variance/covariance
- Changed RXGlobal, ScoreToLogit, LADGlobal to use `WelfordAccumulator` instead of inline Welford implementations
- Changed `_compute_band_correlation_matrix` to single-pass streaming with `WelfordAccumulator`
- Changed TrainablePCA and LearnableChannelMixer to use streaming covariance + `eigh` instead of concat + SVD
- Changed SoftChannelSelector variance init to use streaming `WelfordAccumulator`
- Changed ZScoreNormalizerGlobal to use streaming `WelfordAccumulator` instead of concat + subsample
- Fixed `pyproject.toml` uv source field (`develop` to `editable`)
- Added `resolve_reduce_dims()` as shared module-level utility in `binary_decider`
- Changed supervised band selectors to use template method pattern, pulling shared `forward()` and `statistical_initialization()` into `SupervisedBandSelectorBase`
- Changed YAML configs and docs to use new schema field names (`hparams`, `class_name`)
- Changed `EXECUTION_STAGE_VALIDATE` references to `VAL` across gRPC docs
- Removed dead `_quantile_threshold()` and duplicate `_resolve_reduce_dims()` from `TwoStageBinaryDecider`
- Removed `frozen_nodes` from pipeline configs and docs
- Fixed wavelength batching in supervised band selector `_collect_training_data` (flatten `[B, C]` to `[C]`)

## 0.3.0 - 2026-02-11

- Fixed README documentation links to use docs.cuvis.ai with correct version prefix
- Fixed Pillow CVE-2026-25990 by bumping 12.1.0 to 12.1.1
- Added comprehensive documentation site with tutorials, API reference, and node catalog
- Added MkDocs Material theme with dark mode and versioned deployment via mike
- Added AnomalyPixelStatisticsMetric node replacing duplicate SampleCustomMetrics
- Added deep_svdd_factory utility module with ChannelConfig dataclass
- Added central plugin registry at configs/plugins/registry.yaml
- Added statistical-only training config (default_statistical.yaml)
- Added CI/CD pipeline with test, lint, security, and typecheck jobs
- Added PyPI release workflow with TestPyPI verification and docs deployment
- Added Dependabot configuration for GitHub Actions and pip dependencies
- Added automated test data downloader script with CLI entry point
- Added documentation test suite for link checking and code example validation
- Added Git hooks for ruff format and module case checking
- Added Apache-2.0 LICENSE file
- Changed TrainablePCA to require num_channels parameter (breaking)
- Changed type imports to use cuvis-ai-schemas package
- Changed RXLogitHead to ScoreToLogit and moved to cuvis_ai.node.conversion
- Changed BaseDecider import to BinaryDecider in deciders module
- Changed trainrun config into separate statistical and gradient variants
- Changed pyproject.toml for PyPI compliance and updated tooling configs
- Changed dependencies to add cuvis-ai-schemas and loosen cuvis version pin
- Changed restore-pipeline/restore-trainrun entry points to use cuvis_ai_core
- Improved README and CONTRIBUTING.md with plugin contribution workflow
- Improved docstring coverage to 95%+ across all public APIs
- Fixed LAD detector reset() initializing buffers with wrong shapes
- Fixed LAD detector unfreeze() losing device when converting buffers
- Fixed TrainablePCA with proper num_channels parameter and buffer shapes
- Fixed node import paths for cuvis-ai-schemas migration
- Fixed config references for trainrun and ScoreToLogit rename
- Fixed documentation links, module references, and placeholder content
- Fixed MkDocs build warnings and docstring formatting
- Fixed package metadata for PyPI submission
- Removed restore_pipeline.md from repo root
- Removed old changelog.md replaced by CHANGELOG.md
- Removed run_tests.yml replaced by ci.yml
- Removed outdated docs pages replaced by expanded docs sections

## 0.2.3 - 2026-01-29

- Added plugin system with Git repository and local filesystem support
- Added Pydantic plugin configuration with strict validation
- Added plugin caching in ~/.cuvis_plugins/ with version verification
- Added session-scoped plugin isolation for gRPC services
- Added plugin management gRPC RPCs (LoadPlugins, ListLoadedPlugins, GetPluginInfo, ClearPluginCache)
- Added JSON transport pattern for plugin manifests
- Added test migration infrastructure with 426 tests moved to cuvis-ai-core
- Changed repository architecture to split into cuvis-ai-core and cuvis-ai
- Changed import pattern to use cuvis_ai_core for framework components
- Fixed DataLoader access violation with num_workers=0
- Fixed gRPC servers to use single-threaded mode for cuvis SDK compatibility

## 0.2.2 - 2026-01-15

- Added restoration utilities in cuvis_ai.utils.restore module
- Added CLI entry points for restore-pipeline and restore-trainrun
- Added restore_pipeline.md guide with CLI and Python API examples
- Changed restoration utilities to auto-detect statistical vs gradient workflows
- Changed Python API surface to standardized imports
- Removed duplicate example scripts replaced with library utilities

## 0.2.1 - 2026-01-08

- Added Pydantic v2 config models as single source of truth with validation
- Added server-side Hydra composition with session-scoped search paths
- Added config RPCs: ResolveConfig, ValidateConfig, GetParameterSchema, SetSessionSearchPaths
- Added explicit 4-step workflow for gRPC API
- Changed terminology from Experiment to TrainRun across configs and RPCs
- Changed gRPC service into modular components
- Changed config transport to use config_bytes with central registry
- Fixed RPC surface for new config resolution flow
- Fixed tests and examples (596 tests passing, 65% coverage)

## 0.2.0 - 2025-12-20

- Added YAML-driven pipeline configuration with OmegaConf interpolation
- Added hybrid NodeRegistry for built-ins and custom nodes
- Added end-to-end pipeline serialization with YAML structure and .pt weights
- Added version/schema compatibility guards on load
- Added gRPC canvas management and discovery RPCs
- Added pipeline path resolution helpers with CUVIS_CANVAS_DIR environment variable
- Changed gRPC API to use PipelineConfig via config_bytes
- Changed Train RPC to require DataConfig and TrainingConfig
- Changed SaveCanvas/LoadCanvas to replace SaveCheckpoint/LoadCheckpoint
- Changed node state management to standard state_dict()
- Removed custom serialize/load patterns

## 0.1.5 - 2025-12-01

- Added gRPC service stack with proto definitions
- Added Buf Schema Registry integration for cross-language codegen
- Added session management and PipelineBuilder
- Added file-based data access via DataConfig
- Added two-phase training (statistical init then gradient fine-tuning)
- Added pipeline introspection RPCs and streaming training progress
- Changed output selection to use output_specs
- Changed node naming to deterministic counter-based scheme

## 0.1.3 - 2025-11-06

- Added port-based typed I/O system with PortSpec, InputPort, OutputPort
- Added graph connection API with auto-validation
- Added multi-input/output support for nodes
- Added training integration with PyTorch Lightning
- Changed nodes to declare INPUT_SPECS/OUTPUT_SPECS with auto-created ports
- Changed executor for port-based routing and stage-aware execution
