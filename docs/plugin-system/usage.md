!!! warning "Status: Needs Review"
    This page has not been reviewed for accuracy and completeness. Content may be outdated or contain errors.

---

# Plugin Usage Guide

This repo ships selective manifests for the main plugin workflows used during development and validation.

## Selective Manifests

### Ultralytics

Use the released Ultralytics plugin manifest:

```bash
uv run restore-pipeline \
    --pipeline-path configs/pipeline/my_pipeline.yaml \
    --plugins-path configs/plugins/ultralytics.yaml
```

`configs/plugins/ultralytics.yaml` loads:

- `cuvis_ai_ultralytics.node.YOLOPreprocess`
- `cuvis_ai_ultralytics.node.YOLO26Detection`
- `cuvis_ai_ultralytics.node.YOLOPostprocess`

from `https://github.com/cubert-hyperspectral/cuvis-ai-ultralytics.git` at `v0.1.0`.

### DeepEIoU

Use the released DeepEIoU plugin manifest:

```bash
uv run python examples/object_tracking/deepeiou/yolo_deepeiou_reid_hsi.py \
    --video-path path/to/video.mp4 \
    --no-reid \
    --plugins-dir configs/plugins
```

`configs/plugins/deepeiou.yaml` loads:

- `cuvis_ai_deepeiou.node.DeepEIoUTrack`
- `cuvis_ai_deepeiou.node.OSNetExtractor`
- `cuvis_ai_deepeiou.node.ResNetExtractor`

from `https://github.com/cubert-hyperspectral/cuvis-ai-deepeiou.git` at `v0.1.0`.

### TrackEval

Use the released TrackEval plugin manifest:

```bash
uv run restore-pipeline \
    --pipeline-path configs/pipeline/my_pipeline.yaml \
    --plugins-path configs/plugins/trackeval.yaml
```

`configs/plugins/trackeval.yaml` loads:

- `cuvis_ai_trackeval.node.HOTAMetricNode`
- `cuvis_ai_trackeval.node.CLEARMetricNode`
- `cuvis_ai_trackeval.node.IdentityMetricNode`

from `https://github.com/cubert-hyperspectral/cuvis-ai-trackeval.git` at `v0.1.0`.

### SAM3

Use the local SAM3 plugin manifest when working against the local checkout:

```bash
uv run restore-pipeline \
    --pipeline-path configs/pipeline/my_pipeline.yaml \
    --plugins-path configs/plugins/sam3.yaml
```

`configs/plugins/sam3.yaml` points at the local checkout `D:\code-repos\cuvis-ai-sam3\sam3-init`.

### AdaCLIP

Use the released AdaCLIP plugin manifest:

```bash
uv run restore-pipeline \
    --pipeline-path configs/pipeline/my_pipeline.yaml \
    --plugins-path configs/plugins/adaclip.yaml
```

## Python Usage

Load a selective manifest directly:

```python
from cuvis_ai_core.utils.node_registry import NodeRegistry

registry = NodeRegistry()
registry.load_plugins("configs/plugins/trackeval.yaml")

HOTAMetricNode = registry.get("HOTAMetricNode", instance=registry)
```

## Combined Manifest Example

Create a custom manifest when you want released tracking plugins plus a local SAM3 checkout:

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

Then load it with:

```python
registry = NodeRegistry()
registry.load_plugins("plugins.yaml")
```

## Troubleshooting

- If a local plugin path fails, verify the path is correct relative to the manifest file.
- If a Git plugin fails, verify the tag exists and the repo is accessible from the current environment.
- If a node cannot be found after loading, check that the class path appears in `provides:`.

See [Plugin System Overview](overview.md) for loader behavior and [Plugin Development Guide](development.md) for authoring plugins.
