# Gradient Training

Fit a cuvis-ai pipeline using `GradientTrainer` — backpropagation through trainable parameters, driven by PyTorch Lightning.

## Goal

Produce a saved, fully-trained pipeline (and a matching `trainrun.yaml`) that can be replayed with [`restore-trainrun`](https://cubert-hyperspectral.github.io/cuvis-ai/0.7.3/workflows/restore-pipeline/index.md) for reproducible re-runs.

## Prerequisites

- A pipeline with at least one node carrying trainable parameters (Deep SVDD, AdaCLIP, learned Channel Selector, …).
- A pipeline that has already been statistically initialised — see [Statistical Training](https://cubert-hyperspectral.github.io/cuvis-ai/0.7.3/workflows/statistical-training/index.md). Gradient training is Phase 2 of the two-phase model.
- A datamodule producing the data shape your pipeline expects (typically labelled or self-supervised, depending on the loss node).
- Loss and metric nodes wired into the pipeline.

## Recipe

```python
from cuvis_ai_core.trainer import GradientTrainer
from cuvis_ai_core.config import OptimizerConfig, SchedulerConfig

trainer = GradientTrainer(
    max_epochs=50,
    optimizer=OptimizerConfig(name="adam", lr=1e-3),
    scheduler=SchedulerConfig(name="cosine", t_max=50),
    callbacks=["early_stopping", "model_checkpoint"],
)

trainer.fit(pipeline=pipeline, datamodule=datamodule)

pipeline.save("artifacts/trained_pipeline.yaml")
trainer.save_trainrun("artifacts/trainrun.yaml")
```

## What happens under the hood

1. Trainer wraps the pipeline in a `LightningModule`.
1. For each batch:
1. nodes whose stages include `FORWARD` run a forward pass,
1. nodes whose stages include `LOSS` compute the loss,
1. the optimizer steps,
1. nodes whose stages include `METRIC` log validation metrics.
1. Callbacks (early stopping, model checkpoint) fire at epoch boundaries.
1. At the end, `save_trainrun()` writes a YAML capturing the entire training config so the run can be reproduced.

## Common variations

- **Resume from a checkpoint**: load both the pipeline YAML and the Lightning checkpoint, then call `trainer.fit(pipeline, datamodule, ckpt_path=...)`.
- **Multi-stage freezing**: drive unfreezing via callbacks (e.g. unfreeze the channel selector after epoch 10).
- **Sweep configurations**: pair with [Hydra sweeps](https://cubert-hyperspectral.github.io/cuvis-ai/0.7.3/reference/configuration/hydra-sweeps/index.md) to run a grid of trainings.

## Related

- [Concepts → Training](https://cubert-hyperspectral.github.io/cuvis-ai/0.7.3/concepts/training/index.md) — two-phase model behind the trainer.
- [Concepts → Execution stages](https://cubert-hyperspectral.github.io/cuvis-ai/0.7.3/concepts/execution-stages/index.md) — which nodes run when.
- [Monitoring & Visualization](https://cubert-hyperspectral.github.io/cuvis-ai/0.7.3/workflows/monitoring/index.md) — TensorBoard, callbacks, runtime visualisation.
- [Profiling](https://cubert-hyperspectral.github.io/cuvis-ai/0.7.3/workflows/profiling/index.md) — find bottlenecks in long training runs.
