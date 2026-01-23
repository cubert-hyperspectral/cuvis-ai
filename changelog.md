# Changelog
## V0.2.3
- Repository split into `cuvis-ai-core` (framework) and `cuvis-ai` (catalog) with clear API boundaries and independent versioning
- Framework extraction: base `Node` class, port system, `Pipeline`, training infrastructure, gRPC services, `NodeRegistry`, data infrastructure moved to cuvis-ai-core
- Plugin system with Git repository and local filesystem support via extended `NodeRegistry`
- Pydantic plugin configuration models: `GitPluginConfig`, `LocalPluginConfig`, `PluginManifest` with strict validation
- Plugin caching in `~/.cuvis_plugins/` with intelligent cache reuse and version verification
- Session-scoped plugin isolation: each gRPC session has independent plugin namespaces
- New gRPC RPCs: `LoadPlugins`, `ListLoadedPlugins`, `GetPluginInfo`, `ListAvailableNodes`, `ClearPluginCache`
- JSON transport pattern for plugin manifests via `config_bytes` field matching existing conventions
- Test migration: 426 tests moved to cuvis-ai-core with reusable fixtures in `tests/fixtures/`
- Bug fixes: DataLoader access violation resolved with `num_workers=0`, single-threaded gRPC servers for cuvis SDK compatibility
- 421 tests passing in cuvis-ai-core, independent CI/CD capability established
- Import pattern change: `from cuvis_ai_core.* import ...` for framework components

## V0.2.2
- **Restoration Utilities Refactoring**: Consolidated `restore_pipeline()` and `restore_trainrun()` functionality into `cuvis_ai.utils.restore` module for better discoverability and reusability
- **Smart TrainRun Restoration**: Single `restore_trainrun()` function auto-detects and handles both gradient and statistical training workflows
- **CLI Scripts**: Added `uv run restore-pipeline` and `uv run restore-trainrun` commands to `pyproject.toml` for direct CLI usage
- **Removed Duplication**: Eliminated separate example scripts (`restore_pipeline.py`, `restore_trainrun.py`, `restore_trainrun_statistical.py`) in favor of library utilities
- **Updated Documentation**: Created consolidated `restore_pipeline.md` guide in root directory with examples using new CLI commands and Python API
- **Python API**: Functions available via `from cuvis_ai.utils import restore_pipeline, restore_trainrun`

## V0.2.1
- Refactored monolithic `service.py` (1,775 lines) into 8 modular service components with delegation pattern
- SessionService, ConfigService, PipelineService, TrainingService, TrainRunService, InferenceService, IntrospectionService, DiscoveryService
- Chnages in the RPCs: `ResolveConfig`, `GetParameterSchema`, `ValidateConfig`, `BuildPipeline`, `LoadPipelineWeights`, `SetTrainRunConfig`, `GetTrainingCapabilities`, `ValidateTrainingConfig`, `SetSessionSearchPaths`
- Introduced explicit 4-step workflow (CreateSession → BuildPipeline → SetTrainRunConfig → Train)
- Server-side Hydra composition with `@package _global_` structure, dynamic schema validation
- Change of Terminology: `Experiment` → `TrainRun` (SaveTrainRun/RestoreTrainRun replace SaveExperiment/RestoreExperiment)
- 596 tests passing, 65% coverage, comprehensive performance benchmarks
- All 13 gRPC examples updated to new ResolveConfig pattern with proper Hydra composition

### V0.2.0
- Added YAML-driven Pipeline/Pipeline configuration with OmegaConf interpolation and versioned schema compatibility so pipelines are rebuilt directly from config files instead of hardcoded graphs.
- Introduced a hybrid NodeRegistry: built-in nodes stay O(1) lookup while custom nodes are auto-loaded via `importlib`, removing manual registration steps.
- Implemented end-to-end pipeline serialization: atomic save/load to a single `.pt` bundle, deterministic counter-based artifact naming, and persisted optimizer/scheduler states alongside model weights/metadata.
- gRPC API breaking changes: `CreateSession` now takes `CanvasConfig` (replacing `pipeline_type`), `Train` requires explicit `DataConfig` + `TrainingConfig`, pipeline discovery/management/experiment RPCs were added, and SaveCheckpoint/LoadCheckpoint were removed.
- Reorganized structure: new `configs/` directory for pipeline/experiment/data configs, core modules renamed from "pipeline" to "pipeline", added serialization examples, and enabled git hooks for pre-commit/pre-push.
- Expanded tests and docs to cover the new configuration/serialization paths with updated fixtures, helpers, API docs, and examples across workflows.

## V0.1.5
- Introduced the gRPC service stack (`cuvis_ai/grpc/*`) with proto definitions, session management, callbacks, helpers, and the production server wiring.
- Added generated protobuf stubs plus Buf config, containerization assets (`Dockerfile`, `docker-compose.yml`, `.env.example`), and onboarding updates in `README.md`/`CONTRIBUTING.md`.
- Documented the gRPC surface and deployment (`docs/api/grpc_api.md`, `docs/deployment/grpc_deployment.md`) alongside the detailed blueprint/implementation notes under `docs_dev/cubert/ALL_4917/*`.
- Expanded automated coverage with comprehensive gRPC and pipeline tests (`tests/grpc_api/*`, updated pipeline/training tests) and supporting fixtures.
- Refined pipeline/node logic (selector, losses, metrics, pipeline, dataset handling, training config) and removed outdated torch example scripts in favor of the new API flows.
- Consolidated pytest fixtures: standardized on `channel_selector`, introduced `session`/`trained_session` factories, added reusable helpers (`pipeline_factory`, `data_config_factory`, `mock_pipeline_dir`, `test_data_files`), and reduced duplicate fixture definitions across gRPC tests.
