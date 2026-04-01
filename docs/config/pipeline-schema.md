!!! warning "Status: Needs Review"
    This page has not been reviewed for accuracy and completeness. Content may be outdated or contain errors.

---

# Pipeline Configuration Schema

This page documents the current YAML shape used by the checked-in pipeline configs and catalogs the
pipeline families shipped with the repository.

## Core YAML Shape

```yaml
metadata:
  name: MyPipeline
  description: Current pipeline example
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

- `metadata` identifies the pipeline.
- `nodes` is a list of node definitions.
- `class_name` is the importable class path.
- `hparams` contains constructor arguments.
- `connections` is a list of `source` / `target` edges.

## Pipeline Catalog

### RX

| Config | Use case | Input mode | Plugins | Example |
|---|---|---|---|---|
| `configs/pipeline/anomaly/rx/rx_statistical.yaml` | Baseline RX anomaly detection | Lentils / anomaly datasets | None | [RX tutorial](../tutorials/rx-statistical.md) |
| `configs/pipeline/anomaly/rx/channel_selector.yaml` | Channel selector plus RX | Lentils / anomaly datasets | None | [examples/channel_selector.py](../../examples/channel_selector.py) |

### Deep SVDD

| Config | Use case | Input mode | Plugins | Example |
|---|---|---|---|---|
| `configs/pipeline/anomaly/deep_svdd/deep_svdd.yaml` | Deep one-class anomaly detection | Hyperspectral training data | None | [examples/advanced/deep_svdd_gradient_training.py](../../examples/advanced/deep_svdd_gradient_training.py) |

### AdaCLIP

| Config | Use case | Input mode | Plugins | Example |
|---|---|---|---|---|
| `configs/pipeline/anomaly/adaclip/adaclip_baseline.yaml` | Baseline AdaCLIP inference/training | Hyperspectral training data | `configs/plugins/adaclip.yaml` | [examples/grpc/adaclip/adaclip_client.py](../../examples/grpc/adaclip/adaclip_client.py) |
| `configs/pipeline/anomaly/adaclip/adaclip_cir_false_color.yaml` | CIR false-color AdaCLIP | Hyperspectral training data | `configs/plugins/adaclip.yaml` | [examples/grpc/adaclip/adaclip_cir_false_color_client.py](../../examples/grpc/adaclip/adaclip_cir_false_color_client.py) |
| `configs/pipeline/anomaly/adaclip/adaclip_cir_false_color_optimal_threshold.yaml` | CIR false-color with tuned threshold | Hyperspectral training data | `configs/plugins/adaclip.yaml` | [examples/adaclip/statistical_cir_false_color_optimal_threshold.py](../../examples/adaclip/statistical_cir_false_color_optimal_threshold.py) |
| `configs/pipeline/anomaly/adaclip/adaclip_cir_false_rg_color.yaml` | CIR false-RG variant | Hyperspectral training data | `configs/plugins/adaclip.yaml` | [examples/grpc/adaclip/adaclip_cir_false_rg_color_client.py](../../examples/grpc/adaclip/adaclip_cir_false_rg_color_client.py) |
| `configs/pipeline/anomaly/adaclip/adaclip_high_contrast.yaml` | High-contrast false-RGB AdaCLIP | Hyperspectral training data | `configs/plugins/adaclip.yaml` | [examples/grpc/adaclip/adaclip_high_contrast_client.py](../../examples/grpc/adaclip/adaclip_high_contrast_client.py) |
| `configs/pipeline/anomaly/adaclip/adaclip_supervised_cir.yaml` | Supervised CIR selector | Hyperspectral training data | `configs/plugins/adaclip.yaml` | [examples/grpc/adaclip/adaclip_supervised_cir_client.py](../../examples/grpc/adaclip/adaclip_supervised_cir_client.py) |
| `configs/pipeline/anomaly/adaclip/adaclip_supervised_full_spectrum.yaml` | Supervised full-spectrum selector | Hyperspectral training data | `configs/plugins/adaclip.yaml` | [examples/grpc/adaclip/adaclip_supervised_full_spectrum_client.py](../../examples/grpc/adaclip/adaclip_supervised_full_spectrum_client.py) |
| `configs/pipeline/anomaly/adaclip/adaclip_supervised_windowed_false_rgb.yaml` | Supervised windowed false-RGB selector | Hyperspectral training data | `configs/plugins/adaclip.yaml` | [examples/grpc/adaclip/adaclip_supervised_windowed_false_rgb_client.py](../../examples/grpc/adaclip/adaclip_supervised_windowed_false_rgb_client.py) |
| `configs/pipeline/anomaly/adaclip/concrete_adaclip_gradient_two_stage.yaml` | Concrete selector two-stage training | Hyperspectral training data | `configs/plugins/adaclip.yaml` | [examples/adaclip/concrete_adaclip_gradient_training.py](../../examples/adaclip/concrete_adaclip_gradient_training.py) |
| `configs/pipeline/anomaly/adaclip/drcnn_adaclip_gradient.yaml` | DRCNN-style channel mixer training | Hyperspectral training data | `configs/plugins/adaclip.yaml` | [examples/adaclip/drcnn_adaclip_gradient_training.py](../../examples/adaclip/drcnn_adaclip_gradient_training.py) |

### SAM3

| Config | Use case | Input mode | Plugins | Example |
|---|---|---|---|---|
| `configs/pipeline/sam3/sam3_text_propagation.yaml` | Text-prompt tracking on CU3S | CU3S | `configs/plugins/sam3.yaml` | [examples/object_tracking/sam3/sam3_text_propagation.py](../../examples/object_tracking/sam3/sam3_text_propagation.py) |
| `configs/pipeline/sam3/sam3_text_propagation_video.yaml` | Text-prompt tracking on video | Video | `configs/plugins/sam3.yaml` | [examples/grpc/sam3/sam3_text_propagation_client.py](../../examples/grpc/sam3/sam3_text_propagation_client.py) |
| `configs/pipeline/sam3/sam3_bbox_propagation.yaml` | Scheduled bbox prompting on CU3S | CU3S | `configs/plugins/sam3.yaml` | [examples/object_tracking/sam3/sam3_bbox_propagation.py](../../examples/object_tracking/sam3/sam3_bbox_propagation.py) |
| `configs/pipeline/sam3/sam3_bbox_propagation_video.yaml` | Scheduled bbox prompting on video | Video | `configs/plugins/sam3.yaml` | [examples/grpc/sam3/sam3_bbox_propagation_client.py](../../examples/grpc/sam3/sam3_bbox_propagation_client.py) |
| `configs/pipeline/sam3/sam3_mask_propagation.yaml` | Scheduled mask prompting on CU3S | CU3S | `configs/plugins/sam3.yaml` | [examples/object_tracking/sam3/sam3_mask_propagation.py](../../examples/object_tracking/sam3/sam3_mask_propagation.py) |
| `configs/pipeline/sam3/sam3_mask_propagation_video.yaml` | Scheduled mask prompting on video | Video | `configs/plugins/sam3.yaml` | [examples/grpc/sam3/sam3_mask_propagation_client.py](../../examples/grpc/sam3/sam3_mask_propagation_client.py) |
| `configs/pipeline/sam3/sam3_segment_everything.yaml` | Prompt-free segment-everything on CU3S | CU3S | `configs/plugins/sam3.yaml` | [examples/object_tracking/sam3/sam3_segment_everything.py](../../examples/object_tracking/sam3/sam3_segment_everything.py) |
| `configs/pipeline/sam3/sam3_segment_everything_video.yaml` | Prompt-free segment-everything on video | Video | `configs/plugins/sam3.yaml` | [examples/grpc/sam3/sam3_segment_everything_client.py](../../examples/grpc/sam3/sam3_segment_everything_client.py) |

## Related Pages

- [Configuration Basics](../user-guide/configuration.md)
- [Build Pipeline (YAML)](../how-to/build-pipeline-yaml.md)
- [SAM3 Workflows](../how-to/sam3-workflows.md)
