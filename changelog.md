# Changelog

### Pipeline serialization system (latest)
- Added YAML-driven Pipeline/Pipeline configuration with OmegaConf interpolation and versioned schema compatibility so pipelines are rebuilt directly from config files instead of hardcoded graphs.
- Introduced a hybrid NodeRegistry: built-in nodes stay O(1) lookup while custom nodes are auto-loaded via `importlib`, removing manual registration steps.
- Implemented end-to-end pipeline serialization: atomic save/load to a single `.pt` bundle, deterministic counter-based artifact naming, and persisted optimizer/scheduler states alongside model weights/metadata.
- gRPC API breaking changes: `CreateSession` now takes `CanvasConfig` (replacing `pipeline_type`), `Train` requires explicit `DataConfig` + `TrainingConfig`, pipeline discovery/management/experiment RPCs were added, and SaveCheckpoint/LoadCheckpoint were removed.
- Reorganized structure: new `configs/` directory for pipeline/experiment/data configs, core modules renamed from "pipeline" to "pipeline", added serialization examples, and enabled git hooks for pre-commit/pre-push.
- Expanded tests and docs to cover the new configuration/serialization paths with updated fixtures, helpers, API docs, and examples across workflows.

## Unreleased (vs `cuvis-ai-v2`)
- Introduced the gRPC service stack (`cuvis_ai/grpc/*`) with proto definitions, session management, callbacks, helpers, and the production server wiring.
- Added generated protobuf stubs plus Buf config, containerization assets (`Dockerfile`, `docker-compose.yml`, `.env.example`), and onboarding updates in `README.md`/`CONTRIBUTING.md`.
- Documented the gRPC surface and deployment (`docs/api/grpc_api.md`, `docs/deployment/grpc_deployment.md`) alongside the detailed blueprint/implementation notes under `docs_dev/cubert/ALL_4917/*`.
- Expanded automated coverage with comprehensive gRPC and pipeline tests (`tests/grpc_api/*`, updated pipeline/training tests) and supporting fixtures.
- Refined pipeline/node logic (selector, losses, metrics, pipeline, dataset handling, training config) and removed outdated torch example scripts in favor of the new API flows.
- Consolidated pytest fixtures: standardized on `channel_selector`, introduced `session`/`trained_session` factories, added reusable helpers (`pipeline_factory`, `data_config_factory`, `mock_pipeline_dir`, `test_data_files`), and reduced duplicate fixture definitions across gRPC tests.