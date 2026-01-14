# Test Fixtures Guide

## Overview
Shared pytest fixtures live in `tests/fixtures/` and are loaded via `tests/conftest.py`. Prefer these factories over defining ad-hoc fixtures inside tests.

## Session Fixtures
- `session` - Factory creating sessions with optional pipeline type; cleans up all created sessions.
- `trained_session` - Factory that creates a session, runs statistical training with test data, and returns `(session_id, data_config)`.

## Path Fixtures
- `tmp_path` (pytest built-in) - **Preferred** temporary directory for all file-based tests. Automatically cleaned up after each test.
- `temp_workspace` - Creates a temporary workspace with `pipeline/` and `experiments/` subdirectories. Use when tests need this specific directory structure.
- `mock_pipeline_dir` - Temporary pipeline directory with `CUVIS_CONFIGS_DIR` and `get_server_base_dir` patched. Use with `pipeline_factory` for pipeline discovery tests.

## Configuration Fixtures
- `pipeline_factory` - Build temporary pipeline directories with YAML and optional weights.
- `data_config_factory` - Build `DataConfig` protos with sensible defaults; supports overrides.
- `test_data_files` - Validated `(cu3s, json)` pair from the Lentils dataset (skips if missing).
- `minimal_pipeline_dict`, `mock_pipeline_dict`, `pipeline_yaml_only`, `saved_pipeline` - Ready-made pipeline helpers.

## gRPC Fixtures
- `grpc_stub` - In-process gRPC client connected to `CuvisAIService`.

## Common Patterns
```python
def test_inference(session):
    session_id = session()  # defaults to channel_selector
    # ... test logic

def test_after_training(trained_session):
    session_id, data_config = trained_session()
    # ... inference or gradient training checks

def test_discovery(pipeline_factory, mock_pipeline_dir, grpc_stub):
    pipeline_dir = pipeline_factory([("rx_statistical", yaml_str, True)])
    # mock_pipeline_dir already patches base dir; use pipeline_dir if you need extra files
```

## Migration Notes
- Old fixtures `base_session`, `session_with_data`, `simple_trained_session`, and `session_factory` have been replaced by `session` and `trained_session`.
- Replace local `_data_files` helpers with the shared `test_data_files` fixture.
- Use `mock_pipeline_dir` instead of hand-written monkeypatches for pipeline directories.
- **Fixture consolidation**: `temp_dir`, `temp_pipeline_dir`, and `temp_experiment_dir` have been removed. Use pytest's built-in `tmp_path` fixture instead for temporary directories.

