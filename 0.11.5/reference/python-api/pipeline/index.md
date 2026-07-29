Status: Needs Review

This page has not been reviewed for accuracy and completeness. Content may be outdated or contain errors.

______________________________________________________________________

# Pipeline & Graph API

Pipelines in this branch are defined by checked-in YAML configs, built through the core pipeline builder, and restored through the shared restore utilities.

## Current Pipeline Config Shape

Use the current pipeline schema keys only:

```yaml
metadata:
  name: MyPipeline
  description: Current pipeline example
  author: Cuvis.AI

nodes:
  - name: source
    class_name: cuvis_ai.node.data.CU3SDataNode
    hparams:
      processing_mode: Raw

connections:
  - source: source.outputs.cube
    target: some_node.inputs.data
```

Key points:

- `class_name` identifies the importable node class.
- `hparams` carries node constructor arguments.
- `source` / `target` define port-to-port edges.

## Shipped Pipeline Families

Current checked-in pipeline configs live under `cuvis_ai/configs/pipeline/` and are grouped as:

- RX: `cuvis_ai/configs/pipeline/anomaly/rx/`
- Deep SVDD: `cuvis_ai/configs/pipeline/anomaly/deep_svdd/`
- AdaCLIP: `cuvis_ai/configs/pipeline/anomaly/adaclip/`
- SAM3: `cuvis_ai/configs/pipeline/sam3/`

## Restoration And Remote Execution

These interfaces are part of the current pipeline surface:

- `restore-pipeline`
- `restore-trainrun`
- `cuvis_ai.utils.grpc_workflow`

See:

- [Build Pipeline (YAML)](https://cubert-hyperspectral.github.io/cuvis-ai/0.11.5/workflows/build-pipeline-yaml/index.md)
- [Restore Pipeline](https://cubert-hyperspectral.github.io/cuvis-ai/0.11.5/workflows/restore-pipeline/index.md)
- [gRPC client examples](https://github.com/cubert-hyperspectral/cuvis-ai-cookbook/tree/main/examples/grpc)
