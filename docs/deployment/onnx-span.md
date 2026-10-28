# Exported spans (ONNX)

A pipeline's tensor part can run as one ONNX graph instead of its nodes. cuvis-ai-core's
`export-pipeline` cuts the span between chosen input and output ports, traces it with
`torch.export`, checks it against the live nodes and ONNX Runtime, and writes a deploy
folder. The `ExportedSpan` node in this package runs that folder; everything outside the
span (the data source, deciders, mask operations, writers) keeps running as before.

## The deploy folder

```
deploy/
  dinomaly.onnx          the graph (plus dinomaly.onnx.data above 1 GiB of weights)
  dinomaly.onnx.json     the sidecar: ports, shapes, baked constants, hashes, parity
  dinomaly_deploy.yaml   the pipeline with one ExportedSpan node where the span was
  dinomaly_deploy.pt     the weights of the nodes that stay outside the span
```

The deploy yaml is an ordinary pipeline. `restore-pipeline`, the gRPC `LoadPipeline` and
CuvisNEXT load it like any other; its `plugins:` list carries `cuvis_ai_onnx`, the
manifest that brings ONNX Runtime into a composed environment (the `onnx` extra of this
package). The canonical yaml and `.pt` are untouched: training and weight reloads stay on
them, and a change there means exporting again.

```yaml
- name: exported_span
  class_name: cuvis_ai.node.exported_span.ExportedSpan
  hparams:
    onnx_path: dinomaly.onnx
    backend: onnxruntime
```

How to produce the folder, the cut rules and the sidecar format are on the cuvis-ai-core
page `onnx_export.md`.

## What the node does

- **Ports from the sidecar.** One input per span input, one input per constant baked into the graph, one output per span output. The port names are the span's own port names when they are unique, else `<node>_<port>`. Shapes are the export's example shapes with a free batch dimension.
- **One run per batch element.** The graph was traced at batch size 1; the node loops over the batch and concatenates the outputs.
- **Device.** ONNX Runtime runs where `pipeline.to(device)` moved the node; the `device` hparam overrides it. On CUDA the torch tensors are bound to the session directly, so frames do not pass through host memory.
- **Stages.** Inference, validation and test run. A training stage is refused with a pointer to the canonical yaml.

## Where the ONNX file is looked up

`onnx_path` is written relative to the deploy yaml (the bare file name). The node resolves
a relative path in this order and names all three places in its error:

1. `CUVIS_AI_DEPLOY_DIR`, when set. The way to point a server that receives yaml bytes (gRPC `LoadPipeline`) at a deploy folder.
2. The folder of the pipeline file being loaded, when the loader says where it is (`restore-pipeline`, `CuvisPipeline.load_pipeline`, a trainrun's pipeline reference; cuvis-ai-core 0.19).
3. The working directory.

An absolute `onnx_path` is used as given, which suits a folder that never moves.

## What the node refuses

| Situation | Message names |
|---|---|
| The ONNX file was edited or replaced | the sidecar's `onnx_md5` against the file's hash |
| A frame of another height, width or band count | the exported shape; export again at this resolution |
| A constant with another value (another wavelength grid) | the port, the baked value and the run-time value |
| A CUDA device whose session lands on the CPU provider | the providers ONNX Runtime opened; set `device: cpu` to run on the CPU on purpose |
| A training stage | the canonical yaml |
| ONNX Runtime not installed | `uv sync --extra onnx`, or the `cuvis_ai_onnx` plugin in the pipeline |
| A sidecar of another format | the format this release reads |

## Installing ONNX Runtime

`uv sync --extra onnx` in a checkout, or `cuvis_ai_onnx` under `plugins:` for a composed
environment. The extra installs `onnxruntime-gpu`, which runs the CPU provider as well.
PyPI builds that package with CUDA 13 from 1.27 on, so on x86_64 the extra takes the CUDA
12.8 build from ONNX Runtime's own feed, the one that matches torch cu128 and finds its
CUDA libraries; Linux aarch64 (torch cu130) keeps the PyPI build. One environment holds
one onnxruntime flavour: do not add `cuvis-ai-core[onnx]`, the CPU wheel, on top of this
extra.
