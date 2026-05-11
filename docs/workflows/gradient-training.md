# Gradient Training

Fit a cuvis-ai pipeline using `GradientTrainer` — backpropagation
through trainable parameters, driven by PyTorch Lightning.

## Goal

Produce a saved, fully-trained pipeline (and a matching
`trainrun.yaml`) that can be replayed with
[`restore-trainrun`](restore-pipeline.md) for reproducible re-runs.

## Prerequisites

- A pipeline with at least one node carrying trainable parameters (Deep SVDD, AdaCLIP, learned Channel Selector, …).
- A pipeline that has already been statistically initialised — see [Statistical Training](statistical-training.md). Gradient training is Phase 2 of the two-phase model.
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
2. For each batch:
   - nodes whose stages include `FORWARD` run a forward pass,
   - nodes whose stages include `LOSS` compute the loss,
   - the optimizer steps,
   - nodes whose stages include `METRIC` log validation metrics.
3. Callbacks (early stopping, model checkpoint) fire at epoch boundaries.
4. At the end, `save_trainrun()` writes a YAML capturing the entire
   training config so the run can be reproduced.

## Common variations

- **Resume from a checkpoint**: load both the pipeline YAML and the
  Lightning checkpoint, then call `trainer.fit(pipeline, datamodule, ckpt_path=...)`.
- **Multi-stage freezing**: drive unfreezing via callbacks (e.g.
  unfreeze the channel selector after epoch 10).
- **Sweep configurations**: pair with [Hydra sweeps](../reference/configuration/hydra-sweeps.md)
  to run a grid of trainings.

## Related

- [Concepts → Training](../concepts/training.md) — two-phase model behind the trainer.
- [Concepts → Execution stages](../concepts/execution-stages.md) — which nodes run when.
- [Monitoring & Visualization](monitoring.md) — TensorBoard, callbacks, runtime visualisation.
- [Profiling](profiling.md) — find bottlenecks in long training runs.
