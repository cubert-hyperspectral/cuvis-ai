# Test Suite Guide

## Overview

Tests live in `tests/` organized by domain (anomaly, deciders, node, preprocessors, training, utils, docs). Shared fixtures are in `tests/fixtures/` and loaded automatically via `tests/conftest.py`.

## Running Tests

```bash
# All fast tests (default CI command)
uv run pytest -m "not slow and not check_links" -v

# Unit tests only
uv run pytest -m unit

# Integration tests only
uv run pytest -m integration

# Include slow tests
uv run pytest --runslow

# Specific test file
uv run pytest tests/node/test_bandpass.py -v

# With coverage
uv run pytest --cov=cuvis_ai tests/
```

## Markers

All tests are classified with at least one marker:

| Marker | Purpose | CI default |
|--------|---------|------------|
| `unit` | Fast, isolated tests (shapes, dtypes, edge cases, serialization) | Included |
| `integration` | Multi-node pipelines, training loops, doc validation | Included |
| `slow` | Tests taking >10s (full training, mkdocs build) | Excluded unless `--runslow` |
| `gpu` | GPU-dependent tests | Excluded |
| `check_links` | Documentation link validation | Excluded |

## Fixtures

### Path Fixtures (`tests/fixtures/paths.py`)

| Fixture | Scope | Description |
|---------|-------|-------------|
| `config_dir` | function | Path to `configs/` directory |
| `pipeline_dir` | function | Path to `configs/pipeline/` directory |
| `test_data_path` | session | Path to `data/` test data directory |
| `temp_workspace` | function | Temp dir with `pipeline/`, `experiments/`, `models/` subdirs |
| `mock_pipeline_dir` | function | Temp pipeline dir with `CUVIS_CONFIGS_DIR` monkeypatched |

### Data Factory Fixtures (`tests/fixtures/data_factory.py`)

| Fixture | Scope | Description |
|---------|-------|-------------|
| `test_data_files_cached` | session | Validated `(cu3s, json)` file pair from Lentils dataset (skips if missing) |
| `data_config_factory` | function | Factory for `DataConfig` proto objects with sensible defaults |
| `create_test_cube` | session | Factory returning `(cube, wavelengths)` tuples with configurable mode/shape/dtype |
| `synthetic_anomaly_datamodule` | session | Factory creating `SyntheticAnomalyDataModule` instances for training tests |
| `create_batch_with_wavelengths` | session | Helper adding properly formatted 2D wavelengths to batch dicts |
| `training_config_factory` | session | Factory for `TrainingConfig` with CPU defaults for fast tests |

### Mock Node Fixtures (`tests/fixtures/mock_nodes.py`)

| Fixture | Scope | Description |
|---------|-------|-------------|
| `trainable_pca` | session | Pre-initialized, unfrozen `TrainablePCA(num_channels=5, n_components=3)`. Use `copy.deepcopy()` in tests that mutate it. |

## Common Patterns

```python
# Factory fixtures return callables — invoke to create instances
def test_example(create_test_cube, synthetic_anomaly_datamodule):
    cube, waves = create_test_cube(batch_size=2, num_channels=20, mode="random")
    datamodule = synthetic_anomaly_datamodule(batch_size=4, channels=20, seed=42)

# Session-scoped trainable_pca — deepcopy for mutation
def test_training(trainable_pca):
    import copy
    pca = copy.deepcopy(trainable_pca)
    pca._components.data += noise  # safe, won't affect other tests
```
