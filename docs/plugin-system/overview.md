!!! warning "Status: Needs Review"
    This page has not been reviewed for accuracy and completeness. Content may be outdated or contain errors.

---

# Plugin System Overview

cuvis-ai plugins extend `NodeRegistry` with external node classes without requiring changes to the core repository. A plugin can come from a tagged Git release or a local checkout.

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

```python
from cuvis_ai_core.utils.node_registry import NodeRegistry

registry = NodeRegistry()
registry.load_plugins("configs/plugins/trackeval.yaml")

HOTAMetricNode = registry.get("HOTAMetricNode", instance=registry)
```

`NodeRegistry.load_plugins()` validates the manifest, resolves local paths relative to the manifest file, installs plugin dependencies from `pyproject.toml`, and registers each provided node class in the instance registry.

## Cache and Isolation

- Git plugins are cached under `~/.cuvis_plugins/`.
- Local-path plugins are loaded from the referenced checkout directly.
- Plugin nodes are stored per `NodeRegistry` instance, so one session can load plugins without affecting another.

## Selective Manifests in This Repo

- `configs/plugins/adaclip.yaml`: released AdaCLIP plugin.
- `configs/plugins/ultralytics.yaml`: released Ultralytics YOLO26 plugin at `v0.1.0`.
- `configs/plugins/deepeiou.yaml`: released DeepEIoU plugin at `v0.1.0`.
- `configs/plugins/trackeval.yaml`: released TrackEval plugin at `v0.1.0`.
- `configs/plugins/sam3.yaml`: local SAM3 checkout for active development.

Use a selective manifest when you want one plugin family without building a larger combined manifest first.

## Custom Combined Manifests

Create a local manifest when you want multiple plugins together:

```yaml
plugins:
  ultralytics:
    repo: "https://github.com/cubert-hyperspectral/cuvis-ai-ultralytics.git"
    tag: "v0.1.0"
    provides:
      - cuvis_ai_ultralytics.node.YOLOPreprocess
      - cuvis_ai_ultralytics.node.YOLO26Detection
      - cuvis_ai_ultralytics.node.YOLOPostprocess

  deepeiou:
    repo: "https://github.com/cubert-hyperspectral/cuvis-ai-deepeiou.git"
    tag: "v0.1.0"
    provides:
      - cuvis_ai_deepeiou.node.DeepEIoUTrack
      - cuvis_ai_deepeiou.node.OSNetExtractor
      - cuvis_ai_deepeiou.node.ResNetExtractor

  trackeval:
    repo: "https://github.com/cubert-hyperspectral/cuvis-ai-trackeval.git"
    tag: "v0.1.0"
    provides:
      - cuvis_ai_trackeval.node.HOTAMetricNode
      - cuvis_ai_trackeval.node.CLEARMetricNode
      - cuvis_ai_trackeval.node.IdentityMetricNode

  sam3:
    path: "../../../../cuvis-ai-sam3/sam3-init"
    provides:
      - cuvis_ai_sam3.node.SAM3TrackerInference
      - cuvis_ai_sam3.node.SAM3TextPropagation
      - cuvis_ai_sam3.node.SAM3BboxPropagation
      - cuvis_ai_sam3.node.SAM3PointPropagation
      - cuvis_ai_sam3.node.SAM3MaskPropagation
      - cuvis_ai_sam3.node.SAM3SegmentEverything
```

See [Plugin Nodes](../node-catalog/node-catalog-plugins.md) for CLI and Python examples, and [Plugin Development Guide](development.md) for packaging rules.
