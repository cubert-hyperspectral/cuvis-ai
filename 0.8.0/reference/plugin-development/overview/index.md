# Plugin System

The cuvis-ai plugin system enables extending the framework with custom nodes and functionality without modifying the core codebase. Distribute your algorithms via Git, share with the community, and maintain independent versioning.

A plugin can come from a tagged Git release or a local checkout. Plugins extend `NodeRegistry` with external node classes; no core changes required.

## Quick Start

Pipelines reference plugins by **bare name**. Declare the plugins a pipeline needs in its top-level `plugins:` list, then point the loader at the directory that holds the matching manifests:

```yaml
# my_pipeline.yaml
plugins:
  - trackeval          # bare name → resolves to configs/plugins/trackeval.yaml
nodes:
  - name: hota
    class_name: cuvis_ai_trackeval.node.HOTAMetricNode
    hparams: {}
```

```bash
uv run restore-pipeline \
  --pipeline-path my_pipeline.yaml \
  --plugins-dir configs/plugins
```

The loader resolves each bare name to a manifest in the plugins directory and materialises only the plugins the pipeline declares — see [Loading Flow](#loading-flow).

## Manifest Shapes

Each plugin manifest uses a `plugins:` mapping and one of two source styles:

```yaml
plugins:
  ultralytics:
    repo: "https://github.com/cubert-hyperspectral/cuvis-ai-ultralytics.git"
    tag: "v0.1.0"
    package_name: "cuvis-ai-ultralytics"   # optional: real [project].name if it differs from the key
    provides:
      - class_name: cuvis_ai_ultralytics.node.YOLOPreprocess
      - class_name: cuvis_ai_ultralytics.node.YOLO26Detection
      - class_name: cuvis_ai_ultralytics.node.YOLOPostprocess

  sam3:
    path: "../../../../cuvis-ai-sam3/sam3-init"
    provides:
      - class_name: cuvis_ai_sam3.node.SAM3TextPropagation
```

- `repo` + `tag`: clone a released plugin. Git **tags** only — branches and commit hashes are not supported, for reproducibility.
- `path`: load a local checkout directly. Relative paths resolve from the manifest directory.
- `package_name`: optional. The PyPI-style `[project].name` from the plugin's `pyproject.toml`; set it when the manifest key (a logical label) differs from the real package name.
- `provides`: the plugin's **node catalog** — each entry is one node: a fully-qualified `class_name` plus optional palette metadata (`category`, `tags`, `icon_svg`, `input_specs`, `output_specs`, `doc_summary`). The server reads this catalog to populate the node palette *without importing plugin code*. See [`configs/plugins/adaclip.yaml`](https://github.com/cubert-hyperspectral/cuvis-ai/blob/main/configs/plugins/adaclip.yaml) for a fully populated entry.

## Loading Flow

1. The pipeline yaml's `plugins:` list names the plugins it needs (bare names).
1. The loader resolves each name to a manifest entry in the `--plugins-dir` directory.
1. Only the declared plugins are **materialised**: clone (Git) or read (local), install dependencies from `pyproject.toml`, and import the provided node classes.
1. In the orchestrated gRPC server this happens in an **isolated per-pipeline environment**, so one pipeline's plugin dependencies never affect the server or another pipeline.

`NodeRegistry.load_plugins(manifest_path)` is the **CLI / dev-mode** path for loading a manifest directly into a registry instance — handy for quick local checks (see the [Plugin Development Guide](https://cubert-hyperspectral.github.io/cuvis-ai/0.8.0/reference/plugin-development/guide/index.md)) — but pipelines normally declare plugins by bare name as shown above.

## Cache and Isolation

- Git plugins are cached under `~/.cuvis_plugins/`.
- Local-path plugins are loaded from the referenced checkout directly.
- Plugin nodes are stored per `NodeRegistry` instance, so one session can load plugins without affecting another.

## Loading multiple plugins

List every plugin a pipeline needs in its `plugins:` block, and keep all the manifests in one directory passed via `--plugins-dir`:

```yaml
plugins:
  - ultralytics
  - trackeval
```

## Official Plugin Manifests

- [`configs/plugins/adaclip.yaml`](https://github.com/cubert-hyperspectral/cuvis-ai/blob/main/configs/plugins/adaclip.yaml): released AdaCLIP plugin manifest
- [`configs/plugins/ultralytics.yaml`](https://github.com/cubert-hyperspectral/cuvis-ai/blob/main/configs/plugins/ultralytics.yaml): released Ultralytics YOLO26 plugin manifest pinned to `v0.1.0`
- [`configs/plugins/deepeiou.yaml`](https://github.com/cubert-hyperspectral/cuvis-ai/blob/main/configs/plugins/deepeiou.yaml): released DeepEIoU plugin manifest pinned to `v0.1.0`
- [`configs/plugins/trackeval.yaml`](https://github.com/cubert-hyperspectral/cuvis-ai/blob/main/configs/plugins/trackeval.yaml): released TrackEval plugin manifest pinned to `v0.1.0`
- [`configs/plugins/sam3.yaml`](https://github.com/cubert-hyperspectral/cuvis-ai/blob/main/configs/plugins/sam3.yaml): local SAM3 plugin manifest

## Official Plugins

- **[cuvis-ai-adaclip](https://github.com/cubert-hyperspectral/cuvis-ai-adaclip)** — AdaCLIP vision-language anomaly detection
- **[cuvis-ai-ultralytics](https://github.com/cubert-hyperspectral/cuvis-ai-ultralytics)** — Ultralytics YOLO26 nodes for detection and tracking pipelines
- **[cuvis-ai-deepeiou](https://github.com/cubert-hyperspectral/cuvis-ai-deepeiou)** — DeepEIoU tracking and optional ReID extractors
- **[cuvis-ai-trackeval](https://github.com/cubert-hyperspectral/cuvis-ai-trackeval)** — HOTA, CLEAR, and Identity tracking metrics
- **[cuvis-ai-sam3](https://github.com/cubert-hyperspectral/cuvis-ai-sam3)** — SAM3 tracking workflows and prompt propagation nodes

## Next steps

- See the [Nodes catalog](https://cubert-hyperspectral.github.io/cuvis-ai/0.8.0/catalogs/nodes/index.md) for CLI and Python examples of loading plugin nodes.
- See the [Plugin Development Guide](https://cubert-hyperspectral.github.io/cuvis-ai/0.8.0/reference/plugin-development/guide/index.md) for packaging rules, testing, and release workflow.
