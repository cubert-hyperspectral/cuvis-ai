# TrainRun Configuration Schema

Trainruns are the top-level experiment config. They compose pipeline, data, training, and optional plugin concerns into one runnable unit.

## Core Shape

```yaml
# @package _global_
defaults:
  - /pipeline/anomaly/rx@pipeline: rx_statistical
  - /data@data: lentils
  - /training@training: default
  - _self_

name: rx_demo
output_dir: ./outputs/${name}
loss_nodes: []
metric_nodes:
  - metrics
freeze_nodes: []
unfreeze_nodes: []
```

## Required Fields

| Field        | Meaning                                                 |
| ------------ | ------------------------------------------------------- |
| `name`       | Experiment identifier                                   |
| `pipeline`   | Composed pipeline config                                |
| `data`       | Data config: module, splits, params (see [Data](#data)) |
| `training`   | Training config                                         |
| `output_dir` | Output root                                             |

## Common Optional Fields

| Field            | Meaning                               |
| ---------------- | ------------------------------------- |
| `loss_nodes`     | Loss node names for gradient training |
| `metric_nodes`   | Metric node names to log/evaluate     |
| `freeze_nodes`   | Node names frozen at startup          |
| `unfreeze_nodes` | Node names unfrozen for later phases  |
| `tags`           | Metadata for run tracking             |

## Data

`data` is a `DataConfig`: which DataModule to load, how it is split, and module params.

| Field                        | Meaning                                                                                                                      |
| ---------------------------- | ---------------------------------------------------------------------------------------------------------------------------- |
| `data_module`                | Registered module name (e.g. `cu3s`, `cu3s_multi`, `tiff_paired`, `npz_multi`)                                               |
| `splits`                     | A selector split (`DataSplitConfig`) or a `splits_path` to a committed `splits.json`. Omit for a module that owns its split. |
| `batch_size` / `num_workers` | DataLoader options                                                                                                           |
| `params`                     | Module-specific arguments (e.g. `cu3s_file_path`, `annotation_json_path`; `universe_csv` for `cu3s_multi` and `npz_multi`)   |

A selector split assigns samples to stages by identity. A `universe_csv` (a `universe.csv`) supplies an explicit sample universe; `cu3s_multi` and `npz_multi` both read one (only `tiff_paired` enumerates from disk). See [Data Splits](https://cubert-hyperspectral.github.io/cuvis-ai/0.11.5/concepts/data-splits/index.md) for the full model (universe, selectors, baking).

```yaml
data:
  data_module: npz_multi
  batch_size: 4
  splits:                       # inline selectors, or: splits_path: splits/dinomaly.json
    train:
      - { kind: file_indices, source: X.cu3s, ids: [0, 2, 3] }
    val:
      - { kind: file_indices, source: X.cu3s, ids: [1, 5] }
  params:
    universe_csv: outputs/npz_local/universe.csv
```

## Current Patterns

### Statistical Workflow

```yaml
defaults:
  - /pipeline/anomaly/rx@pipeline: rx_statistical
  - /data@data: lentils
  - /training@training: default_statistical
  - _self_

name: rx_statistical_demo
metric_nodes:
  - metrics
```

### SAM3 Workflow

```yaml
defaults:
  - /pipeline/sam3@pipeline: sam3_text_propagation
  - /data@data: tracking_cap_and_car
  - /training@training: default
  - _self_

name: sam3_text_demo
output_dir: ./outputs/${name}
```

## Related Pages

- [Configuration Basics](https://cubert-hyperspectral.github.io/cuvis-ai/0.11.5/reference/configuration/index.md)
- [Config Groups](https://cubert-hyperspectral.github.io/cuvis-ai/0.11.5/reference/configuration/config-groups/index.md)
- [Data Splits](https://cubert-hyperspectral.github.io/cuvis-ai/0.11.5/concepts/data-splits/index.md)
- [Restore Pipeline](https://cubert-hyperspectral.github.io/cuvis-ai/0.11.5/workflows/restore-pipeline/index.md)
