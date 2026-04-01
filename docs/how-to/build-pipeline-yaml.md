!!! warning "Status: Needs Review"
    This page has not been reviewed for accuracy and completeness. Content may be outdated or contain errors.

---

# How-To: Build Pipelines In YAML

Use YAML when you want a reproducible, shareable pipeline definition that can be composed into a
trainrun or resolved remotely over gRPC.

## Minimal Current Pipeline

```yaml
metadata:
  name: MinimalPipeline
  description: Current YAML example
  author: cuvis.ai

nodes:
  - name: data_node
    class_name: cuvis_ai.node.data.LentilsAnomalyDataNode
    hparams:
      normal_class_ids: [0, 1]

  - name: rx
    class_name: cuvis_ai.anomaly.rx_detector.RXGlobal
    hparams:
      num_channels: 61
      eps: 1.0e-06

connections:
  - source: data_node.outputs.cube
    target: rx.inputs.data
```

## Schema Rules

- `nodes` is a list.
- Each node uses `name`, `class_name`, and optional `hparams`.
- Each edge uses `source` and `target`.
- Port names use `node.outputs.port` and `node.inputs.port`.

## Example: Tracking-Oriented Pipeline

```yaml
metadata:
  name: TrackingOverlay
  description: Render overlays from tracking JSON
  author: cuvis.ai

nodes:
  - name: tracks
    class_name: cuvis_ai.node.json_reader.TrackingResultsReader
    hparams:
      json_path: tracking_results.json

  - name: overlay
    class_name: cuvis_ai.node.anomaly_visualization.TrackingOverlayNode

  - name: writer
    class_name: cuvis_ai.node.video.ToVideoNode
    hparams:
      output_video_path: tracking_overlay.mp4

connections:
  - source: tracks.outputs.image
    target: overlay.inputs.image
  - source: tracks.outputs.bboxes
    target: overlay.inputs.bboxes
  - source: overlay.outputs.image
    target: writer.inputs.image
```

## Compose It Into A Trainrun

```yaml
# @package _global_
defaults:
  - /pipeline/sam3@pipeline: sam3_text_propagation
  - /data@data: tracking_cap_and_car
  - /training@training: default
  - _self_

name: sam3_text_demo
output_dir: ./outputs/${name}
```

## Validation Checklist

- Confirm the class path matches a real importable node class.
- Confirm every required input port has a producer.
- Prefer current checked-in configs as templates instead of historical examples.
- Save narrow plugin manifests next to the trainrun when a pipeline depends on plugins.

## Related Pages

- [Pipeline Configuration Schema](../config/pipeline-schema.md)
- [Configuration Basics](../user-guide/configuration.md)
- [Build Pipeline (Python)](build-pipeline-python.md)
