!!! warning "Status: Needs Review"
    This page has not been reviewed for accuracy and completeness. Content may be outdated or contain errors.

---

# Configuration Groups

Hydra config groups keep current experiments modular and composable.

## Current Group Layout

```text
configs/
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

- `configs/pipeline/anomaly/rx/`
- `configs/pipeline/anomaly/deep_svdd/`
- `configs/pipeline/anomaly/adaclip/`
- `configs/pipeline/sam3/`

Override example:

```yaml
pipeline:
  nodes:
    - name: RXGlobal
      hparams:
        eps: 1.0e-08
```

## Data Group

Use `configs/data/` for dataset paths, IDs, and loader settings.

Example:

```yaml
data:
  cu3s_file_path: data/Lentils/Lentils_000.cu3s
  annotation_json_path: data/Lentils/Lentils_000.json
  train_ids: [0, 2, 3]
  val_ids: [1, 5]
  test_ids: [1, 5]
  batch_size: 2
```

## Training Group

Use `configs/training/` for trainer and optimizer settings.

Example:

```yaml
training:
  trainer:
    max_epochs: 50
    accelerator: auto
    devices: 1
  optimizer:
    name: adamw
    lr: 0.001
```

## Plugin Group

Use the narrow manifest required by the workflow:

- `configs/plugins/adaclip.yaml`
- `configs/plugins/bytetrack.yaml`
- `configs/plugins/deepeiou.yaml`
- `configs/plugins/detr.yaml`
- `configs/plugins/sam3.yaml`
- `configs/plugins/trackeval.yaml`
- `configs/plugins/ultralytics.yaml`

Use `configs/plugins/registry.yaml` only when you want the full registry.

## Related Pages

- [Configuration Basics](../user-guide/configuration.md)
- [Pipeline Configuration Schema](pipeline-schema.md)
- [TrainRun Schema](trainrun-schema.md)
