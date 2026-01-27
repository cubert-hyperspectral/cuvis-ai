# Changelog
## V0.2.3
- Repository split into `cuvis-ai-core` (framework) and `cuvis-ai` (catalog at https://github.com/cubert-hyperspectral/cuvis-ai) with clear API boundaries and independent versioning
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
- Restoration utilities consolidated into `cuvis_ai.utils.restore` for pipeline and trainrun recovery
- `restore_trainrun()` auto-detects statistical vs gradient workflows and restores the appropriate trainer state
- CLI entry points added: `uv run restore-pipeline` and `uv run restore-trainrun`
- Removed duplicate example scripts in favor of library utilities
- New `restore_pipeline.md` guide with CLI and Python API examples
- Python API surface standardized: `from cuvis_ai.utils import restore_pipeline, restore_trainrun`

## V0.2.1
- Pydantic v2 config models as the single source of truth with validation, JSON Schema introspection, and JSON/proto serialization
- Standardized config transport across pipeline/data/training/trainrun via `config_bytes` and a central config registry
- Server-side Hydra composition with session-scoped search paths and overrides (`ResolveConfig`, `ValidateConfig`, `GetParameterSchema`, `SetSessionSearchPaths`)
- Explicit 4-step workflow: CreateSession → (ResolveConfig/Build or Load) Pipeline → SetTrainRunConfig → Train
- Terminology update: Experiment → TrainRun across configs, RPCs, paths, and examples
- gRPC service refactor into modular components (Session/Config/Pipeline/Training/TrainRun/Inference/Introspection/Discovery)
- RPC surface updated for the new workflow: `ResolveConfig`, `GetParameterSchema`, `ValidateConfig`, `BuildPipeline`, `LoadPipelineWeights`, `SetTrainRunConfig`, `GetTrainingCapabilities`
- Tests and examples updated for the new config resolution flow (596 tests passing, 65% coverage, performance benchmarks)

## V0.2.0
- YAML-driven canvas configuration with OmegaConf interpolation; CanvasBuilder builds pipelines from config files
- Hybrid NodeRegistry (built-ins + importlib) for custom nodes and pluggable pipelines
- End-to-end canvas serialization: YAML structure + single `.pt` weights file, atomic save/load, counter-based naming
- Version/schema compatibility guards on load, with optional strict checks and clear mismatch errors
- gRPC API changes: CreateSession uses CanvasConfig (`config_bytes`), Train requires DataConfig + TrainingConfig
- Canvas management and discovery RPCs added; SaveCanvas/LoadCanvas replace SaveCheckpoint/LoadCheckpoint
- Canvas path resolution helpers (short names, auto `.yaml`, `CUVIS_CANVAS_DIR` base directory)
- Node state management simplified to `state_dict()` + buffers/parameters; avoid custom serialize/load patterns
- Usage guides added for node serialization and canvas/experiment lifecycle with optimizer/scheduler state and metadata

## V0.1.5
- Introduced the gRPC service stack with proto definitions and Buf Schema Registry for cross-language codegen
- Generated Python stubs, Buf configs, and proto conversion helpers (numpy/tensor <-> proto)
- Session management and CanvasBuilder with hardcoded pipelines for channel selector, statistical, and gradient workflows
- File-based data access via DataConfig and two-phase training (statistical init → gradient fine-tuning)
- Output selection via `output_specs` and streaming training progress; inference and training RPCs wired end-to-end
- Pipeline introspection RPCs (inputs/outputs/visualization) plus deterministic counter-based node naming
- Examples and test suites for gRPC inference/training/introspection; deployment assets and docs updated

## V0.1.3
- Release date: 2025-11-06
- Port-based typed I/O system with `PortSpec`, `InputPort`, `OutputPort`, and dimension resolution
- Nodes declare `INPUT_SPECS`/`OUTPUT_SPECS` with auto-created ports and multi-input/output support
- Graph connection API uses port objects, a MultiDiGraph source of truth, auto-add nodes, and connection-time validation
- Executor refactor for port-based routing, stage-aware execution, batch distribution, and variadic ports
- Training integration with Lightning, context-aware execution, and preserved gradients across the graph
- Core and training nodes migrated to typed I/O with updated examples and tests
