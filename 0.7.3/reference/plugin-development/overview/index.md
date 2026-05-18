# Plugin System

The cuvis-ai plugin system enables extending the framework with custom nodes and functionality without modifying the core codebase. Distribute your algorithms via Git, share with the community, and maintain independent versioning.

A plugin can come from a tagged Git release or a local checkout. Plugins extend `NodeRegistry` with external node classes; no core changes required.

## Quick Start

```python
from cuvis_ai_core.utils.node_registry import NodeRegistry

registry = NodeRegistry()
registry.load_plugins("configs/plugins/trackeval.yaml")

HOTAMetricNode = registry.get("HOTAMetricNode", instance=registry)
```

## Manifest Shapes

Each plugin manifest uses a `plugins:` mapping and one of two source styles:

```yaml
plugins:
  ultralytics:
    repo: "https://github.com/cubert-hyperspectral/cuvis-ai-ultralytics.git"
    tag: "v0.1.0"
    provides:
      - cuvis_ai_ultralytics.node.YOLOPreprocess
      - cuvis_ai_ultralytics.node.YOLO26Detection
      - cuvis_ai_ultralytics.node.YOLOPostprocess

  trackeval:
    repo: "https://github.com/cubert-hyperspectral/cuvis-ai-trackeval.git"
    tag: "v0.1.0"
    provides:
      - cuvis_ai_trackeval.node.HOTAMetricNode

  sam3:
    path: "../../../../cuvis-ai-sam3/sam3-init"
    provides:
      - cuvis_ai_sam3.node.SAM3TextPropagation
```

- `repo` + `tag`: clone a released plugin into the local cache.
- `path`: load a local checkout directly. Relative paths resolve from the manifest directory.
- `provides`: list fully-qualified class paths that the plugin exports.

## Loading Flow

`NodeRegistry.load_plugins()` validates the manifest, resolves local paths relative to the manifest file, installs plugin dependencies from `pyproject.toml`, and registers each provided node class in the instance registry.

## Cache and Isolation

- Git plugins are cached under `~/.cuvis_plugins/`.
- Local-path plugins are loaded from the referenced checkout directly.
- Plugin nodes are stored per `NodeRegistry` instance, so one session can load plugins without affecting another.

## Loading multiple plugins

Each official plugin ships its own `plugins.yaml` manifest (see below). When a pipeline needs nodes from more than one plugin, pass each manifest path to `NodeRegistry.load_plugins()` in turn — there's no need to author a single combined file.

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

- See the [Nodes catalog](https://cubert-hyperspectral.github.io/cuvis-ai/0.7.3/catalogs/nodes/index.md) for CLI and Python examples of loading plugin nodes.
- See the [Plugin Development Guide](https://cubert-hyperspectral.github.io/cuvis-ai/0.7.3/reference/plugin-development/guide/index.md) for packaging rules, testing, and release workflow.
