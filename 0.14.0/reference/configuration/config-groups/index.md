Status: Needs Review

This page has not been reviewed for accuracy and completeness. Content may be outdated or contain errors.

______________________________________________________________________

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
  - /pipeline/anomaly/rx@pipeline: rx_statistical
  - /data@data: lentils
  - /training@training: default
  - _self_

name: rx_demo
output_dir: ./outputs/${name}
```

## Pipeline Group

Current pipeline families:

- `cuvis_ai/configs/pipeline/anomaly/rx/`
- `cuvis_ai/configs/pipeline/anomaly/deep_svdd/`
- `cuvis_ai/configs/pipeline/anomaly/adaclip/`
- `cuvis_ai/configs/pipeline/sam3/`

Override example:

```yaml
pipeline:
  nodes:
    - name: RXGlobal
      hparams:
        eps: 1.0e-08
```

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

Each plugin ships its own `plugins.yaml` manifest; reference the ones you need from your config. The full list of plugin-supplied nodes lives in the [Nodes catalog](https://cubert-hyperspectral.github.io/cuvis-ai/0.14.0/catalogs/nodes/index.md).

## Related Pages

- [Configuration Basics](https://cubert-hyperspectral.github.io/cuvis-ai/0.14.0/reference/configuration/index.md)
- [TrainRun Schema](https://cubert-hyperspectral.github.io/cuvis-ai/0.14.0/reference/configuration/trainrun-schema/index.md)
