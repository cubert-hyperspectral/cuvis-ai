!!! warning "Status: Needs Review"
    This page has not been reviewed for accuracy and completeness. Content may be outdated or contain errors.

---

# Configuration Groups

Hydra config groups keep current experiments modular and composable.

## Current Group Layout

```text
cuvis_ai/configs/
├── data/
├── pipeline/
│   ├── anomaly/
│   └── sam3/
├── plugins/
├── training/
└── trainrun/
```

## Standard Trainrun Composition

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
unfreeze_nodes:
  - SoftChannelSelector
  - RXGlobal
  - ScoreToLogit
loss_nodes:
  - bce
metric_nodes:
  - metrics_anomaly
```

## Pipelines

Current pipeline families:

- `cuvis_ai/configs/pipeline/anomaly/rx/`
- `cuvis_ai/configs/pipeline/anomaly/deep_svdd/`
- `cuvis_ai/configs/pipeline/anomaly/adaclip/`
- `cuvis_ai/configs/pipeline/sam3/`

A pipeline is not composed into a trainrun as a config group: the trainrun names the pipeline
yaml by path (`pipeline:`, resolved relative to the trainrun file), and `TrainRunConfig` refuses a
pipeline mapping in the trainrun. To change a node's hparams, save an edited copy of the pipeline
yaml and point the trainrun at it.

## Data Group

Use `cuvis_ai/configs/data/` for the data module, split selectors, and loader settings.

Example:

```yaml
data:
  data_module: cu3s
  splits:
    train:
      - {kind: file_indices, source: data/Lentils/Lentils_000.cu3s, ids: [0, 2, 3]}
    val:
      - {kind: file_indices, source: data/Lentils/Lentils_000.cu3s, ids: [1]}
    test:
      - {kind: file_indices, source: data/Lentils/Lentils_000.cu3s, ids: [5]}
  batch_size: 2
  params:
    cu3s_file_path: data/Lentils/Lentils_000.cu3s
    annotation_json_path: data/Lentils/Lentils_000.json
```

## Training Group

Use `cuvis_ai/configs/training/` for trainer and optimizer settings.

Example:

```yaml
training:
  max_epochs: 50
  accelerator: auto
  devices: 1
  optimizer:
    name: adamw
    lr: 0.001
```

## Plugin Group

Use the narrow manifest required by the workflow:

- `cuvis_ai/configs/plugins/adaclip.yaml`
- `cuvis_ai/configs/plugins/bytetrack.yaml`
- `cuvis_ai/configs/plugins/deepeiou.yaml`
- `cuvis_ai/configs/plugins/detr.yaml`
- `cuvis_ai/configs/plugins/sam3.yaml`
- `cuvis_ai/configs/plugins/trackeval.yaml`
- `cuvis_ai/configs/plugins/ultralytics.yaml`

Each plugin ships its own `plugins.yaml` manifest; reference the ones you need from your config. The full list of plugin-supplied nodes lives in the [Nodes catalog](../../catalogs/nodes/index.md).

## Related Pages

- [Configuration Basics](index.md)
- [TrainRun Schema](trainrun-schema.md)
