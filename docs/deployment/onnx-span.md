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

- **Ports from the sidecar.** One input per span input, one input per constant baked into the graph, one output per span output. The port names are the span's own port names when they are unique, else `<node>_<port>`. Shapes are the export's example shapes with a free batch dimension. Each port declares the dtype the sidecar's `port_dtype` names (what the original port declared, a numpy type for a wavelength grid), so the deploy yaml connects where the canonical yaml did; a numpy array on an input or constant port is converted on the way in.
- **Numerics on CUDA.** The session runs with TF32 off (`use_tf32: 0` on the CUDA provider), so the outputs match the CPU provider and the export's parity check within 1e-5; the provider's TF32 default moved a PatchCore distance by 0.4.
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
| `backend: tensorrt` on a CPU device | use `backend: onnxruntime` to run on the CPU |
| The TensorRT package not installed | the install command matching torch's CUDA build |
| A sidecar of another format | the format this release reads |

## Which nodes export

A node carries `NodeTag.ONNX` when it exports with the dynamo exporter in its registry
contract default configuration: `torch.export` can trace `forward` as is, or the node
supplies an equivalent through `export_module()`. The tag is a statement about that one
default configuration; a node can still refuse another configuration of its own hparams
(a mode that takes a Python branch on tensor values), in which case its class docstring
says so.

`tests/node/test_onnx_export_contract.py` builds every `NodeTag.ONNX` node from its
registry contract defaults, exports it alone with the dynamo exporter and checks ONNX
Runtime against the eager output. A node that is exportable but not yet tagged, or a node
whose default configuration regresses, fails that test.

To make a node exportable:

- Keep the computation inside `forward` as tensor ops. A numpy lookup, `.item()` or a Python branch on a tensor's value stops `torch.export` outright.

- A check that only ever raises (an unfitted node, an invalid configuration) can stay: guard it with `if not torch.compiler.is_compiling():` so it runs eagerly but never enters the traced graph. The exporter's own eager-versus-traced parity check in `export_deployment` is what catches a node exported in the state the guard would have rejected; it does not need to be caught a second time inside the traced path.

- A wavelength-dependent lookup table (nearest-band indices, a CMF or sensitivity weight matrix) stays numpy when the table only depends on the span's own constant inputs (a sensor's wavelength grid does not change per frame): cache it on first eager call and, under `is_compiling()`, reuse the cached tensor instead of recomputing it.

- When `forward` truly cannot be traced (a numpy or OpenCV round-trip whose shape of computation has no tensor equivalent, such as connected-component labelling), override `export_module()` to return a module that computes the same thing from the node's own weights and buffers; see the docstring of `Node.export_module` for the contract.

## Installing ONNX Runtime

`uv sync --extra onnx` in a checkout, or `cuvis_ai_onnx` under `plugins:` for a composed
environment. The extra installs `onnxruntime-gpu`, which runs the CPU provider as well.
PyPI builds that package with CUDA 13 from 1.27 on, so on x86_64 the extra takes the CUDA
12.8 build from ONNX Runtime's own feed, the one that matches torch cu128 and finds its
CUDA libraries; Linux aarch64 (torch cu130) keeps the PyPI build. One environment holds
one onnxruntime flavour: do not add `cuvis-ai-core[onnx]`, the CPU wheel, on top of this
extra.

## TensorRT backend

```yaml
- name: exported_span
  class_name: cuvis_ai.node.exported_span.ExportedSpan
  hparams:
    onnx_path: dinomaly.onnx
    backend: tensorrt
```

`backend: tensorrt` compiles the span's ONNX graph into a TensorRT engine instead of
running it through ONNX Runtime. Reach for it on a device where ONNX Runtime has no CUDA
kernel for the graph (the Jetson Thor's PyPI `onnxruntime-gpu` wheel is one such device:
TensorRT is its GPU path) or where TensorRT measurably outruns the ONNX Runtime CUDA
provider for the exported graph; otherwise `backend: onnxruntime` needs nothing beyond
the `onnx` extra and is the simpler default.

The engine is found or built the first time `prepare_inference()` runs, keyed by the
sidecar's `fingerprint` and `onnx_md5` plus the GPU and the installed TensorRT version, so
retraining or exporting again always builds a fresh engine. It lands under
`CUVIS_AI_TRT_ENGINE_DIR/exported_span` when that variable is set, else under the
cuvis-ai model cache (`trt_engines/exported_span`); a build takes a few minutes the first
time on a given machine and is instant on every later load. Several span nodes, or the
same node across restarts, reuse the same engine file once it exists.

Requirements: a CUDA device (`backend: tensorrt` on `device: cpu` is refused) and the
TensorRT Python package for the platform, matching torch's CUDA build
(`pip install tensorrt-cu12` or `tensorrt-cu13`), plus `onnx` to build an engine from
scratch. Neither ships as a pip extra of this package: on the Thor, TensorRT comes with
JetPack; on x86_64, install it by hand or through the environment that composes the
pipeline. The same file checks (`onnx_md5`), shape checks, constant checks and training
refusal as the ONNX Runtime backend apply.

## Standing checks

Three checks keep the export path and a deployed folder honest between releases.

**The weekly sweep.** `.github/workflows/onnx_export_sweep.yml` runs
`tests/node/test_onnx_export_contract.py -m slow --runslow` on a schedule (Monday, early UTC) and
on demand (`workflow_dispatch`), so a node that silently regresses between releases is
caught without running the full sweep on every PR. The test writes one JSON line per
node (name, outcome, max abs diff, seconds) when `CUVIS_ONNX_CONTRACT_REPORT` names a
file; the workflow turns that file into the job summary, uploads it as an artifact and fails
the job when no report was written or a node failed. The contract registers every node
subpackage, and `tests/node/test_onnx_export_contract_coverage.py` refuses on every PR an
ONNX-tagged node that the contract would not export. Run it locally the same way:

```bash
CUDA_VISIBLE_DEVICES=-1 PYTHONUTF8=1 CUVIS_ONNX_CONTRACT_REPORT=report.jsonl \
  uv run pytest tests/node/test_onnx_export_contract.py -m slow --runslow -q
```

**`compare-deploy`.** Checks a deploy folder against its canonical pipeline on real
frames: the max abs diff per exported-span output port, pixel decisions differing when a
decider sits downstream, and the per-frame median time for both pipelines. Frames come
from `.npz` files or a data module's predict split, the way `export-pipeline` takes them:

```bash
uv run compare-deploy \
  --canonical pipeline.yaml --deploy deploy/pipeline_deploy.yaml \
  --data-module cu3s --data-arg cu3s_file_path=sample.cu3s \
  --num-frames 5 --device cpu --out compare_report.json
```

The canonical node defaults to the one the deploy yaml's sidecar names as the span's
output (`--canonical-node` overrides it); `--span-node` overrides the exported-span
node's name when it is not `exported_span`.

**The gRPC deploy-folder test.** `tests/grpc/test_deploy_folder_grpc.py` drives
cuvis-ai-core's production servicers in process (the in-memory orchestrator: no
subprocess, no composed venv) over an exported deploy folder: `LoadPipeline`,
`LoadPipelineWeights` and `Inference` reproduce the canonical pipeline's own forward
pass, and a load with no `CUVIS_AI_DEPLOY_DIR` configured and no pipeline-file folder
(gRPC passes the pipeline as bytes, not a path) fails with a message naming the three
places `ExportedSpan` can resolve the ONNX file from. It is the reference for a client
that loads a deploy folder the same way CuvisNEXT does: register every manifest the
deploy yaml names with `LoadPlugin`, then `LoadPipeline` with the yaml as JSON bytes,
`LoadPipelineWeights` with the sibling `.pt`, and `Inference` with `CUVIS_AI_DEPLOY_DIR`
set to the deploy folder.
