# Plugin System

The cuvis-ai plugin system enables extending the framework with custom nodes and functionality without modifying the core codebase. Distribute your algorithms via Git, share with the community, and maintain independent versioning.

A plugin can come from a tagged Git release or a local checkout. Plugins extend `NodeRegistry` with external node classes; no core changes required.

## Quick Start

Pipelines reference plugins by **bare name**. Declare the plugins a pipeline needs in its top-level `plugins:` list, then point the loader at the directory that holds the matching manifests:

```yaml
# my_pipeline.yaml
plugins:
  - trackeval          # bare name → resolves to cuvis_ai/configs/plugins/trackeval.yaml
nodes:
  - name: hota
    class_name: cuvis_ai_trackeval.node.HOTAMetricNode
    hparams: {}
```

```bash
uv run restore-pipeline \
  --pipeline-path my_pipeline.yaml \
  --plugins-dir cuvis_ai/configs/plugins
```

The loader resolves each bare name to a manifest in the plugins directory and materialises only the plugins the pipeline declares — see [Loading Flow](#loading-flow).

## Manifest Shapes

Each plugin manifest is a **single file for a single plugin**: an explicit `name:`, one source (`repo:` + `tag:` for a released plugin, or `path:` for a local checkout), and a `capabilities:` list. The `name:` is explicit and never derived from the filename.

```yaml
# cuvis_ai/configs/plugins/ultralytics.yaml
name: ultralytics
repo: "https://github.com/cubert-hyperspectral/cuvis-ai-ultralytics.git"
tag: "v0.1.4"
package_name: "cuvis-ai-ultralytics"   # optional: real [project].name if it differs from `name`
capabilities:
  - class_name: cuvis_ai_ultralytics.node.YOLOPreprocess
  - class_name: cuvis_ai_ultralytics.node.YOLO26Detection
  - class_name: cuvis_ai_ultralytics.node.YOLOPostprocess
```

A local checkout uses `path:` in place of `repo:` + `tag:` (one plugin per file, as always):

```yaml
# a local development manifest
name: my_plugin
path: "../../../cuvis-ai-my-plugin"
capabilities:
  - class_name: my_plugin.node.custom_node.CustomNode
```

- `name`: the explicit plugin name. Pipelines reference it as a bare name in their `plugins:` list.
- `repo` + `tag`: clone a released plugin. Git **tags** only (branches and commit hashes are not supported, for reproducibility).
- `path`: load a local checkout directly. Relative paths resolve from the manifest directory.
- `package_name`: optional. The PyPI-style `[project].name` from the plugin's `pyproject.toml`; set it when it differs from `name`.
- `capabilities`: the plugin's **node catalog**. Each entry is one node: a fully-qualified `class_name` plus optional palette metadata (`category`, `tags`, `icon_svg`, `input_specs`, `output_specs`, `doc_summary`). The server reads this catalog to populate the node palette *without importing plugin code*. See [`cuvis_ai/configs/plugins/adaclip.yaml`](https://github.com/cubert-hyperspectral/cuvis-ai/blob/main/cuvis_ai/configs/plugins/adaclip.yaml) for a fully populated entry.

## Loading Flow

1. The pipeline yaml's `plugins:` list names the plugins it needs (bare names).
1. The loader resolves each name to a manifest entry in the `--plugins-dir` directory.
1. The declared plugins are **registered import-only**: their node classes are imported from packages already installed in the active environment. Registration never clones, installs dependencies, or mutates `sys.path`, so provision the plugins first (see the `provision` CLI).
1. In the orchestrated gRPC server, the composer builds an **isolated per-pipeline environment** (git plugins pinned to a commit + `uv sync`) and the child registers the now-installed plugins the same import-only way, so one pipeline's dependencies never affect the server or another pipeline.

`NodeRegistry.register_plugin(manifest_path)` is the **in-process** path for registering a manifest directly into a registry instance — handy for quick local checks and notebooks (see the [Plugin Development Guide](https://cubert-hyperspectral.github.io/cuvis-ai/0.13.6/reference/plugin-development/guide/index.md)) — but pipelines normally declare plugins by bare name as shown above.

## Cache and Isolation

- In-process registration imports plugins from the active environment; install them with the `provision` CLI, `uv pip install`, or an editable `[tool.uv.sources]` checkout.
- The orchestrated server composes an isolated venv per plugin set, cached by a content hash of its generated `pyproject.toml`, so identical plugin sets reuse the same child environment.
- Plugin nodes are stored per `NodeRegistry` instance, so one session can register plugins without affecting another.
- How the composed environment resolves plugin dependencies, including how it mirrors the host's torch build, is covered in [Dependency resolution in composed child environments](https://cubert-hyperspectral.github.io/cuvis-ai/0.13.6/reference/plugin-development/guide/#dependency-resolution-in-composed-child-environments).

## Loading multiple plugins

List every plugin a pipeline needs in its `plugins:` block, and keep all the manifests in one directory passed via `--plugins-dir`:

```yaml
plugins:
  - ultralytics
  - trackeval
```

## Official Plugin Manifests

All official plugins ship as git-tagged releases (bare name resolves to the matching manifest file):

- [`adaclip.yaml`](https://github.com/cubert-hyperspectral/cuvis-ai/blob/main/cuvis_ai/configs/plugins/adaclip.yaml): AdaCLIP anomaly detection, pinned to `v0.2.0`
- [`augment.yaml`](https://github.com/cubert-hyperspectral/cuvis-ai/blob/main/cuvis_ai/configs/plugins/augment.yaml): data-augmentation nodes, pinned to `v0.3.3`
- [`cuvis_ai_dataloader.yaml`](https://github.com/cubert-hyperspectral/cuvis-ai/blob/main/cuvis_ai/configs/plugins/cuvis_ai_dataloader.yaml): cu3s / cu3 / paired-TIFF data-module plugin, pinned to `v0.4.0`
- [`cuvis_ai_inspecscrap.yaml`](https://github.com/cubert-hyperspectral/cuvis-ai/blob/main/cuvis_ai/configs/plugins/cuvis_ai_inspecscrap.yaml): metal-scrap inspection nodes, pinned to `v0.2.2`
- [`deepeiou.yaml`](https://github.com/cubert-hyperspectral/cuvis-ai/blob/main/cuvis_ai/configs/plugins/deepeiou.yaml): DeepEIoU tracking plugin, pinned to `v0.2.1`
- [`dinomaly.yaml`](https://github.com/cubert-hyperspectral/cuvis-ai/blob/main/cuvis_ai/configs/plugins/dinomaly.yaml): Dinomaly anomaly detection, pinned to `v0.4.1`
- [`rtsam2.yaml`](https://github.com/cubert-hyperspectral/cuvis-ai/blob/main/cuvis_ai/configs/plugins/rtsam2.yaml): real-time SAM 2 / EfficientTAM plugin, pinned to `v0.3.0`
- [`sam3.yaml`](https://github.com/cubert-hyperspectral/cuvis-ai/blob/main/cuvis_ai/configs/plugins/sam3.yaml): SAM 3.1 tracking plugin, pinned to `v0.2.1`
- [`trackeval.yaml`](https://github.com/cubert-hyperspectral/cuvis-ai/blob/main/cuvis_ai/configs/plugins/trackeval.yaml): tracking-metric plugin, pinned to `v0.1.4`
- [`ultralytics.yaml`](https://github.com/cubert-hyperspectral/cuvis-ai/blob/main/cuvis_ai/configs/plugins/ultralytics.yaml): Ultralytics YOLO26 plugin, pinned to `v0.1.4`

## Official Plugins

- **[cuvis-ai-adaclip](https://github.com/cubert-hyperspectral/cuvis-ai-adaclip)**: AdaCLIP zero-shot vision-language anomaly detection
- **[cuvis-ai-augment](https://github.com/cubert-hyperspectral/cuvis-ai-augment)**: training-time data-augmentation nodes for hyperspectral cubes
- **[cuvis-ai-dataloader](https://github.com/cubert-hyperspectral/cuvis-ai-dataloader)**: cu3s / cu3 (COCO-masked) and paired-TIFF DataModules (data-module plugin)
- **[cuvis-ai-inspecscrap](https://github.com/cubert-hyperspectral/cuvis-ai-inspecscrap)**: metal-scrap material classification nodes
- **[cuvis-ai-deepeiou](https://github.com/cubert-hyperspectral/cuvis-ai-deepeiou)**: DeepEIoU tracking and optional ReID extractors
- **[cuvis-ai-dinomaly](https://github.com/cubert-hyperspectral/cuvis-ai-dinomaly)**: DINOv2-based anomaly detection (Anomalib DinomalyModel)
- **[cuvis-ai-rtsam2](https://github.com/cubert-hyperspectral/cuvis-ai-rtsam2)**: real-time SAM 2 / EfficientTAM streaming segmentation and propagation
- **[cuvis-ai-sam3](https://github.com/cubert-hyperspectral/cuvis-ai-sam3)**: SAM 3.1 tracking, segmentation, and prompt propagation nodes
- **[cuvis-ai-trackeval](https://github.com/cubert-hyperspectral/cuvis-ai-trackeval)**: HOTA, CLEAR-MOT, and Identity tracking metrics
- **[cuvis-ai-ultralytics](https://github.com/cubert-hyperspectral/cuvis-ai-ultralytics)**: YOLO26 detection with composable preprocess / postprocess nodes

## Next steps

- See the [Nodes catalog](https://cubert-hyperspectral.github.io/cuvis-ai/0.13.6/catalogs/nodes/index.md) for CLI and Python examples of loading plugin nodes.
- See the [Plugin Development Guide](https://cubert-hyperspectral.github.io/cuvis-ai/0.13.6/reference/plugin-development/guide/index.md) for packaging rules, testing, and release workflow.
