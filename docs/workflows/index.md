# Workflows

Task-recipe guides organised around "I want to…" intentions. Workflows
are terse and action-oriented — if you know what you're trying to do
and need the exact incantation, this is the section.

If you're new and need step-by-step learning instead, start with
[Tutorials](../tutorials/index.md).

## Build a pipeline

<div class="grid cards" markdown>

-   :material-code-braces: **[Build Pipeline (Python)](build-pipeline-python.md)**

    ---

    Author a pipeline programmatically with the Python API.

-   :material-file-code: **[Build Pipeline (YAML)](build-pipeline-yaml.md)**

    ---

    Define a pipeline declaratively in YAML.

</div>

## Train a pipeline

<div class="grid cards" markdown>

-   :material-chart-bell-curve: **[Statistical Training](statistical-training.md)**

    ---

    Fit a pipeline using `StatisticalTrainer` — accumulate background moments, no gradient steps.

-   :material-trending-up: **[Gradient Training](gradient-training.md)**

    ---

    Fit a pipeline using `GradientTrainer` — optimizer, scheduler, callbacks.

</div>

## Run, monitor, and profile

<div class="grid cards" markdown>

-   :material-restore: **[Restore Pipeline](restore-pipeline.md)**

    ---

    Replay a saved pipeline with `restore-pipeline` or `restore-trainrun`.

-   :material-monitor-eye: **[Monitoring & Visualization](monitoring.md)**

    ---

    TensorBoard, metric callbacks, and runtime visualisation of pipeline state.

-   :material-speedometer: **[Profiling](profiling.md)**

    ---

    Measure where a pipeline spends its time and memory.

</div>
