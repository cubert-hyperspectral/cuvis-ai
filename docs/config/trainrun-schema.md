!!! warning "Status: Needs Review"
    This page has not been reviewed for accuracy and completeness. Content may be outdated or contain errors.

---

# TrainRun Configuration Schema

Trainruns are the top-level experiment config. They compose pipeline, data, training, and optional
plugin concerns into one runnable unit.

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

| Field | Meaning |
|---|---|
| `name` | Experiment identifier |
| `pipeline` | Composed pipeline config |
| `data` | Data config |
| `training` | Training config |
| `output_dir` | Output root |

## Common Optional Fields

| Field | Meaning |
|---|---|
| `loss_nodes` | Loss node names for gradient training |
| `metric_nodes` | Metric node names to log/evaluate |
| `freeze_nodes` | Node names frozen at startup |
| `unfreeze_nodes` | Node names unfrozen for later phases |
| `tags` | Metadata for run tracking |

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

- [Configuration Basics](../user-guide/configuration.md)
- [Config Groups](config-groups.md)
- [Restore Pipeline](../how-to/restore-pipeline-trainrun.md)
