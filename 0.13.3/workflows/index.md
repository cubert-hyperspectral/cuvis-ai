# Workflows

Task-recipe guides organised around "I want to…" intentions. Workflows are terse and action-oriented — if you know what you're trying to do and need the exact incantation, this is the section.

If you're new and need step-by-step learning instead, start with [Tutorials](https://cubert-hyperspectral.github.io/cuvis-ai/0.13.3/tutorials/index.md).

## Build a pipeline

- **[Build Pipeline (Python)](https://cubert-hyperspectral.github.io/cuvis-ai/0.13.3/workflows/build-pipeline-python/index.md)**

  ______________________________________________________________________

  Author a pipeline programmatically with the Python API.

- **[Build Pipeline (YAML)](https://cubert-hyperspectral.github.io/cuvis-ai/0.13.3/workflows/build-pipeline-yaml/index.md)**

  ______________________________________________________________________

  Define a pipeline declaratively in YAML.

## Train a pipeline

- **[Statistical Training](https://cubert-hyperspectral.github.io/cuvis-ai/0.13.3/workflows/statistical-training/index.md)**

  ______________________________________________________________________

  Fit a pipeline using `StatisticalTrainer` — accumulate background moments, no gradient steps.

- **[Gradient Training](https://cubert-hyperspectral.github.io/cuvis-ai/0.13.3/workflows/gradient-training/index.md)**

  ______________________________________________________________________

  Fit a pipeline using `GradientTrainer` — optimizer, scheduler, callbacks.

## Run, monitor, and profile

- **[Restore Pipeline](https://cubert-hyperspectral.github.io/cuvis-ai/0.13.3/workflows/restore-pipeline/index.md)**

  ______________________________________________________________________

  Replay a saved pipeline with `restore-pipeline` or `restore-trainrun`.

- **[Monitoring & Visualization](https://cubert-hyperspectral.github.io/cuvis-ai/0.13.3/workflows/monitoring/index.md)**

  ______________________________________________________________________

  TensorBoard, metric callbacks, and runtime visualisation of pipeline state.

- **[Profiling](https://cubert-hyperspectral.github.io/cuvis-ai/0.13.3/workflows/profiling/index.md)**

  ______________________________________________________________________

  Measure where a pipeline spends its time and memory.
