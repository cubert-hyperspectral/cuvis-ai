!!! warning "Status: Needs Review"
    This page has not been reviewed for accuracy and completeness. Content may be outdated or contain errors.

---

# Plugin System

The cuvis-ai plugin system enables extending the framework with custom nodes and functionality without modifying the core codebase. Distribute your algorithms via Git, share with the community, and maintain independent versioning.

## Quick Start

```python
from cuvis_ai_core.utils.node_registry import NodeRegistry

registry = NodeRegistry()
registry.load_plugins("configs/plugins/trackeval.yaml")

HOTAMetricNode = registry.get("HOTAMetricNode", instance=registry)
```

## Guides

<div class="grid cards" markdown>

-   :material-puzzle: **[Plugin System Overview](overview.md)**

    ---

    Architecture, manifest rules, path resolution, and loading behavior.

-   :material-hammer-wrench: **[Plugin Development Guide](development.md)**

    ---

    Create, test, package, and tag a cuvis-ai plugin for local or Git-based use.

</div>

## Official Plugin Manifests

- [`configs/plugins/adaclip.yaml`](../../configs/plugins/adaclip.yaml): released AdaCLIP plugin manifest
- [`configs/plugins/ultralytics.yaml`](../../configs/plugins/ultralytics.yaml): released Ultralytics YOLO26 plugin manifest pinned to `v0.1.0`
- [`configs/plugins/deepeiou.yaml`](../../configs/plugins/deepeiou.yaml): released DeepEIoU plugin manifest pinned to `v0.1.0`
- [`configs/plugins/trackeval.yaml`](../../configs/plugins/trackeval.yaml): released TrackEval plugin manifest pinned to `v0.1.0`
- [`configs/plugins/sam3.yaml`](../../configs/plugins/sam3.yaml): local SAM3 plugin manifest for a checkout at `D:\code-repos\cuvis-ai-sam3\sam3-init`

## Official Plugins

- **[cuvis-ai-adaclip](https://github.com/cubert-hyperspectral/cuvis-ai-adaclip)** - AdaCLIP vision-language anomaly detection
- **[cuvis-ai-ultralytics](https://github.com/cubert-hyperspectral/cuvis-ai-ultralytics)** - Ultralytics YOLO26 nodes for cuvis.ai detection and tracking pipelines
- **[cuvis-ai-deepeiou](https://github.com/cubert-hyperspectral/cuvis-ai-deepeiou)** - DeepEIoU tracking and optional ReID extractors for cuvis.ai tracking pipelines
- **[cuvis-ai-trackeval](https://github.com/cubert-hyperspectral/cuvis-ai-trackeval)** - HOTA, CLEAR, and Identity tracking metrics
- **[cuvis-ai-sam3](https://github.com/cubert-hyperspectral/cuvis-ai-sam3)** - local SAM3 tracking workflows and prompt propagation nodes

See [Plugin Nodes](../node-catalog/node-catalog-plugins.md) for available plugins and loading examples.
