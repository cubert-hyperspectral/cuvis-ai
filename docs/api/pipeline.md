!!! warning "Status: Needs Review"
    This page has not been reviewed for accuracy and completeness. Content may be outdated or contain errors.

---

# Pipeline & Graph API

Pipelines in this branch are defined by checked-in YAML configs, built through the core pipeline
builder, and restored through the shared restore utilities.

## Current Pipeline Config Shape

Use the current pipeline schema keys only:

```yaml
metadata:
  name: MyPipeline
  description: Current pipeline example
  author: cuvis.ai

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

Current checked-in pipeline configs live under `configs/pipeline/` and are grouped as:

- RX: `configs/pipeline/anomaly/rx/`
- Deep SVDD: `configs/pipeline/anomaly/deep_svdd/`
- AdaCLIP: `configs/pipeline/anomaly/adaclip/`
- SAM3: `configs/pipeline/sam3/`

## Restoration And Remote Execution

These interfaces are part of the current pipeline surface:

- `restore-pipeline`
- `restore-trainrun`
- `cuvis_ai.utils.grpc_workflow`

See:

- [Pipeline Configuration Schema](../config/pipeline-schema.md)
- [Build Pipeline (YAML)](../how-to/build-pipeline-yaml.md)
- [Restore Pipeline](../how-to/restore-pipeline-trainrun.md)
- [gRPC Example Clients](../grpc/example-clients.md)
