# TrainRun Configuration Schema

Trainruns are the top-level experiment config. They compose pipeline, data, training, and optional
plugin concerns into one runnable unit.

## Core Shape

```yaml
# @package _global_
defaults:
  - /data@data: lentils
  - /training@training: default
  - _self_

name: channel_selector_demo
# Pipeline reference, resolved relative to this trainrun's directory.
pipeline: ../pipeline/anomaly/rx/channel_selector.yaml
output_dir: ./outputs/${name}
loss_nodes:
  - bce
metric_nodes:
  - metrics_anomaly
freeze_nodes: []
unfreeze_nodes:
  - SoftChannelSelector
  - RXGlobal
  - ScoreToLogit
```

## Required Fields

| Field | Meaning |
|---|---|
| `name` | Experiment identifier |
| `pipeline` | Path of the pipeline YAML, resolved relative to the trainrun file; an inline pipeline (also one composed through a Hydra `@pipeline` default) is refused |
| `data` | Data config: module, splits, params (see [Data](#data)) |
| `training` | Training config; a statistical-only run leaves it out |
| `output_dir` | Output root |

## Common Optional Fields

| Field | Meaning |
|---|---|
| `loss_nodes` | Loss node names for gradient training |
| `metric_nodes` | Metric node names to log/evaluate |
| `freeze_nodes` | Node names frozen at startup |
| `unfreeze_nodes` | Node names unfrozen for later phases |
| `tags` | Metadata for run tracking |

## Data

`data` is a `DataConfig`: which DataModule to load, how it is split, and module params.

| Field | Meaning |
|---|---|
| `data_module` | Registered module name (e.g. `cu3s`, `cu3s_multi`, `tiff_paired`, `npz_multi`) |
| `splits` | A selector split (`DataSplitConfig`) or a `splits_path` to a committed `splits.json`. Omit for a module that owns its split. |
| `batch_size` / `num_workers` | DataLoader options |
| `params` | Module-specific arguments (e.g. `cu3s_file_path`, `annotation_json_path`; `universe_csv` for `cu3s_multi` and `npz_multi`) |

A selector split assigns samples to stages by identity. A `universe_csv` (a `universe.csv`) supplies an
explicit sample universe; `cu3s_multi` and `npz_multi` both read one (only `tiff_paired` enumerates
from disk). See [Data Splits](../../concepts/data-splits.md) for the full model (universe, selectors,
baking).

```yaml
data:
  data_module: cu3s_multi
  batch_size: 4
  num_workers: 0                # the cuvis SDK reads in the training process
  splits:                       # inline selectors, or: splits_path: splits/dinomaly.json
    train:
      - { kind: file_indices, source: X.cu3s, ids: [0, 2, 3] }
    val:
      - { kind: file_indices, source: X.cu3s, ids: [1, 5] }
  params:
    universe_csv: universe.csv  # source, index, annotation: one row per frame
    max_open_sessions: 4        # recordings kept open per loader, about 0.3 to 0.6 GB of GPU memory each
    read_ahead: 1               # frames read ahead of the model step
```

More open recordings mean fewer reopens during a shuffled epoch, `read_ahead` overlaps the read
with the model step at batch 1, and `cuda_cubes: true` keeps the cubes on the GPU; the
cuvis-ai-dataloader README covers each option. Every loader (train, val, test) keeps its own open
recordings, so their GPU memory adds up: 3 open with `read_ahead: 1` fit an 8 GB GPU. With all
recordings open, `read_ahead` and `cuda_cubes` together crashed the Cuvis SDK in longer runs
(cuvis-ai-dataloader 0.8.2, SDK 3.6.1).

## Current Patterns

### Statistical Workflow

No `training` block: the run fits the statistical nodes on the train split and evaluates.

```yaml
# @package _global_
defaults:
  - /data@data: lentils
  - _self_

name: rx_statistical_demo
# Pipeline reference, resolved relative to this trainrun's directory.
pipeline: ../pipeline/anomaly/rx/rx_statistical.yaml
output_dir: ./outputs/${name}
metric_nodes:
  - metrics_anomaly
```

### Inference-only pipelines

The SAM3 tracking presets and other pipelines without statistical or trainable nodes have no
trainrun; run them with [`restore-pipeline`](../../workflows/restore-pipeline.md).

## Related Pages

- [Configuration Basics](index.md)
- [Config Groups](config-groups.md)
- [Data Splits](../../concepts/data-splits.md)
- [Restore Pipeline](../../workflows/restore-pipeline.md)
