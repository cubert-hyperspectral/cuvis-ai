# Gradient Training

Fit a cuvis-ai pipeline using `GradientTrainer` — backpropagation through trainable parameters, driven by PyTorch Lightning.

## Goal

Produce a saved, fully-trained pipeline (and a matching `trainrun.yaml`) that can be replayed with [`restore-trainrun`](https://cubert-hyperspectral.github.io/cuvis-ai/0.9.0/workflows/restore-pipeline/index.md) for reproducible re-runs.

## Prerequisites

- A pipeline with at least one node carrying trainable parameters (Deep SVDD, AdaCLIP, learned Channel Selector, …).
- A pipeline that has already been statistically initialised — see [Statistical Training](https://cubert-hyperspectral.github.io/cuvis-ai/0.9.0/workflows/statistical-training/index.md). Gradient training is Phase 2 of the two-phase model.
- A datamodule producing the data shape your pipeline expects (typically labelled or self-supervised, depending on the loss node).
- Loss and metric nodes wired into the pipeline.

## Recipe

```python
from cuvis_ai_core.training import (
    GradientTrainer,
    OptimizerConfig,
    SchedulerConfig,
    TrainerConfig,
)

trainer = GradientTrainer(
    pipeline=pipeline,
    datamodule=datamodule,
    loss_nodes=loss_nodes,
    metric_nodes=metric_nodes,
    trainer_config=TrainerConfig(max_epochs=50),
    optimizer_config=OptimizerConfig(name="adam", lr=1e-3),
    scheduler_config=SchedulerConfig(name="cosine", t_max=50),
    callbacks=["early_stopping", "model_checkpoint"],
)

trainer.fit()

pipeline.save_to_file("artifacts/trained_pipeline.yaml")
```

## What happens under the hood

1. Trainer wraps the pipeline in a `LightningModule`.
1. For each batch:
1. nodes whose stages include `FORWARD` run a forward pass,
1. nodes whose stages include `LOSS` compute the loss,
1. the optimizer steps,
1. nodes whose stages include `METRIC` log validation metrics.
1. Callbacks (early stopping, model checkpoint) fire at epoch boundaries.
1. The trained pipeline is written with `pipeline.save_to_file(...)`. A `trainrun.yaml` capturing the entire training config (so the run can be reproduced via `restore-trainrun`) is produced by the `SaveTrainRun` gRPC RPC when training under a `cuvis-ai-core` server session.

## Common variations

- **Resume from a checkpoint**: replay the saved `trainrun.yaml` with a Lightning checkpoint via `restore_trainrun(trainrun_path, mode="train", checkpoint_path=...)` (or the `restore-trainrun` CLI).
- **Multi-stage freezing**: drive unfreezing via callbacks (e.g. unfreeze the channel selector after epoch 10).
- **Sweep configurations**: pair with [Hydra sweeps](https://cubert-hyperspectral.github.io/cuvis-ai/0.9.0/reference/configuration/hydra-sweeps/index.md) to run a grid of trainings.

## Related

- [Concepts → Training](https://cubert-hyperspectral.github.io/cuvis-ai/0.9.0/concepts/training/index.md) — two-phase model behind the trainer.
- [Concepts → Execution stages](https://cubert-hyperspectral.github.io/cuvis-ai/0.9.0/concepts/execution-stages/index.md) — which nodes run when.
- [Monitoring & Visualization](https://cubert-hyperspectral.github.io/cuvis-ai/0.9.0/workflows/monitoring/index.md) — TensorBoard, callbacks, runtime visualisation.
- [Profiling](https://cubert-hyperspectral.github.io/cuvis-ai/0.9.0/workflows/profiling/index.md) — find bottlenecks in long training runs.
