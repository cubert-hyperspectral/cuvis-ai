# Configuration

CUVIS.AI uses Hydra composition and checked-in YAML files under `configs/` as the current source of
truth for pipeline, data, training, trainrun, and plugin configuration.

## Directory Layout

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

## Configuration Types

| Config type | Location | Purpose |
|---|---|---|
| Pipeline | `configs/pipeline/` | Node graph structure and connections |
| Data | `configs/data/` | Dataset paths, splits, and loader settings |
| Training | `configs/training/` | Trainer, optimizer, scheduler, callbacks |
| Trainrun | `configs/trainrun/` | Composed experiment definition |
| Plugins | `configs/plugins/` | Plugin manifests and registry |

## Current Pipeline Shape

Use the current schema keys only:

```yaml
metadata:
  name: RX_Statistical
  description: RX anomaly detector
  author: cuvis.ai

nodes:
  - name: data_node
    class_name: cuvis_ai.node.data.LentilsAnomalyDataNode
    hparams:
      normal_class_ids: [0, 1]

  - name: normalizer
    class_name: cuvis_ai.node.normalization.MinMaxNormalizer
    hparams:
      eps: 1.0e-06
      use_running_stats: true

connections:
  - source: data_node.outputs.cube
    target: normalizer.inputs.data
```

Current rules:

- Use `class_name`, not `class`.
- Use `hparams`, not `params`.
- Use `source` and `target`, not `from` and `to`.

## Trainrun Composition

Trainruns compose pipeline, data, and training groups:

```yaml
# @package _global_
defaults:
  - /pipeline/anomaly/rx@pipeline: rx_statistical
  - /data@data: lentils
  - /training@training: default
  - _self_

name: rx_lentils_demo
output_dir: ./outputs/${name}
metric_nodes:
  - metrics
```

Typical command-line overrides:

```bash
uv run python train.py training.trainer.max_epochs=50 data.batch_size=4
```

## Shipped Pipeline Families

| Family | Directory | Notes |
|---|---|---|
| RX | `configs/pipeline/anomaly/rx/` | Baseline anomaly workflows |
| Deep SVDD | `configs/pipeline/anomaly/deep_svdd/` | Deep one-class workflows |
| AdaCLIP | `configs/pipeline/anomaly/adaclip/` | CLIP-based anomaly workflows |
| SAM3 | `configs/pipeline/sam3/` | Text, bbox, mask, and segment-everything tracking |

## Plugin Manifests

Current checked-in manifests:

- `configs/plugins/registry.yaml`
- `configs/plugins/adaclip.yaml`
- `configs/plugins/bytetrack.yaml`
- `configs/plugins/deepeiou.yaml`
- `configs/plugins/detr.yaml`
- `configs/plugins/sam3.yaml`
- `configs/plugins/trackeval.yaml`
- `configs/plugins/ultralytics.yaml`

Use the narrow manifest for a workflow when possible; use `registry.yaml` only when you
intentionally want the full registry.

## Related Pages

- [Pipeline Configuration Schema](../config/pipeline-schema.md)
- [TrainRun Schema](../config/trainrun-schema.md)
- [Build Pipeline (YAML)](../how-to/build-pipeline-yaml.md)
- [SAM3 Workflows](../how-to/sam3-workflows.md)
