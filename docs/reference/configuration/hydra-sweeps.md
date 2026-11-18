!!! warning "Status: Needs Review"
    This page has not been reviewed for accuracy and completeness. Content may be outdated or contain errors.

---

# Hydra Sweeps & Advanced Patterns

Multi-run sweeps for hyperparameter search and advanced composition patterns.

## Multi-Run Sweeps

### Basic Sweep Syntax

Use `-m` flag to enable multi-run mode:

```bash
python train.py -m training.optimizer.lr=0.001,0.0001,0.00001
```

Hydra creates separate runs:
```
outputs/multirun/2026-02-04/10-30-00/
├── 0/  # lr=0.001
├── 1/  # lr=0.0001
└── 2/  # lr=0.00001
```

### Sweep Multiple Parameters

**Cartesian product:**
```bash
python train.py -m \
    training.optimizer.lr=0.001,0.0001 \
    training.optimizer.weight_decay=0.01,0.001
```

Creates 4 runs (2 x 2).

### Comparing Pipelines

The pipeline is not swept. A trainrun names its pipeline yaml by path, and because `pipeline`
also names the config directory, Hydra reads `pipeline=...` on the command line as a config group
override and refuses it. Give each pipeline its own trainrun (Pattern 1 below), as the packaged
ones in `cuvis_ai/configs/trainrun/` do, and sweep the parameters within each:

```bash
python train.py --config-name=trainrun/channel_selector_experiment -m training.optimizer.lr=0.001,0.0001
python train.py --config-name=trainrun/deep_svdd_experiment -m training.optimizer.lr=0.001,0.0001
```

### Custom Sweep Configurations

```yaml
# @package _global_
# cuvis_ai/configs/trainrun/sweep_base.yaml

defaults:
  - /data@data: lentils
  - /training@training: default
  - _self_

name: sweep_lr_${training.optimizer.lr}
# Pipeline reference, resolved relative to this trainrun's directory.
pipeline: ../pipeline/anomaly/rx/channel_selector.yaml
output_dir: ./outputs/sweeps/${name}
```

```bash
python train.py \
    --config-name=trainrun/sweep_base \
    -m \
    training.optimizer.lr=0.001,0.0001,0.00001
```

---

## Advanced Composition Patterns

### Pattern 1: Base + One Trainrun per Pipeline

**Base config:** `cuvis_ai/configs/trainrun/base_experiment.yaml`
```yaml
# @package _global_

defaults:
  - /data@data: lentils
  - /training@training: default
  - _self_

output_dir: ./outputs/${name}
```

**One variant per pipeline**, each inheriting the base and naming its pipeline, for example
`cuvis_ai/configs/trainrun/channel_selector_experiment.yaml`:
```yaml
# @package _global_

defaults:
  - base_experiment
  - _self_

name: channel_selector_experiment
# Pipeline reference, resolved relative to this trainrun's directory.
pipeline: ../pipeline/anomaly/rx/channel_selector.yaml
```

### Pattern 2: Conditional Composition

```yaml
defaults:
  - /training/optimizer@training.optimizer: ${optimizer_type}
  - /training/scheduler@training.scheduler: ${scheduler_type}
  - optional /training/callbacks@training.callbacks: ${callbacks_preset}
  - _self_

optimizer_type: adamw
scheduler_type: reduce_on_plateau
callbacks_preset: null  # Optional
```

### Pattern 3: Config Recipes

```yaml
# cuvis_ai/configs/recipes/fast_prototype.yaml
# @package _global_

defaults:
  - /trainrun@_here_: default
  - _self_

training:
  max_epochs: 3
  log_every_n_steps: 1
data:
  batch_size: 1
  num_workers: 0
output_dir: ./outputs/quick_test
```

```bash
python train.py --config-name=recipes/fast_prototype
```

### Pattern 4: Mixin Configs

```yaml
# cuvis_ai/configs/mixins/debug.yaml
# @package training

max_epochs: 1
log_every_n_steps: 1
enable_progress_bar: true
optimizer:
  lr: 0.01  # Higher LR for fast debugging
```

```yaml
# Usage in trainrun
defaults:
  - /data@data: lentils
  - /training@training: default
  - /mixins/debug@training:_here_  # Merge debug settings
  - _self_

pipeline: ../pipeline/anomaly/rx/channel_selector.yaml
```

---

## Best Practices

1. **Always put `_self_` last** in defaults list
2. **Use `@package _global_`** for trainrun configs
3. **Use clear variable names** — `${training.optimizer.lr}` not `${a}`
4. **Only override what you need** — don't duplicate entire configs
5. **Document complex compositions** with header comments
6. **Use `resolve=True`** when converting to dict: `OmegaConf.to_container(cfg, resolve=True)`

---

## Troubleshooting

| Problem | Cause | Fix |
|---------|-------|-----|
| `Inline pipelines are no longer supported in a trainrun` | The pipeline was composed into the trainrun (a `@pipeline` default or a `pipeline:` mapping) | Set `pipeline:` to the path of the pipeline yaml |
| `Could not override 'pipeline'. No match in the defaults list.` | `pipeline=...` on the command line is read as a config group override | Write one trainrun per pipeline |
| `InterpolationResolutionError` | Referenced key doesn't exist | Define the key before referencing |
| Override not applied | `_self_` not last | Move `_self_` to end of defaults |
| Config at wrong level | Wrong package directive | Use `@package _global_` for trainruns |
| `CircularReferenceError` | Circular interpolation | Break the cycle: `a: ${b}` + `b: ${a}` |

---

## See Also

- [Hydra Basics](hydra-basics.md) — package directives, defaults list
- [Hydra Inheritance & Overrides](hydra-inheritance.md) — config inheritance, interpolation
- [Hydra Documentation](https://hydra.cc/) — official Hydra docs
- [OmegaConf Documentation](https://omegaconf.readthedocs.io/) — OmegaConf reference
