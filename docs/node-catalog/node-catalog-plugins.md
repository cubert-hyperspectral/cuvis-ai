!!! warning "Status: Needs Review"
    This page has not been reviewed for accuracy and completeness. Content may be outdated or contain errors.

---

# Plugin Nodes

Externally contributed nodes loaded through the [plugin system](../plugin-system/overview.md).
Each plugin ships its own Git repository and is loaded at runtime via a YAML manifest
(see [Plugin System Overview](../plugin-system/overview.md)).

| Plugin | Use Case | Nodes |
|---|---|---|
| **[AdaCLIP](https://github.com/cubert-hyperspectral/cuvis-ai-adaclip)** | Vision-language anomaly detection | `AdaCLIPDetector` |
| **[Ultralytics](https://github.com/cubert-hyperspectral/cuvis-ai-ultralytics)** | YOLO26 object detection | `YOLOPreprocess`, `YOLO26Detection`, `YOLOPostprocess` |
| **[DeepEIoU](https://github.com/cubert-hyperspectral/cuvis-ai-deepeiou)** | Multi-object tracking with optional ReID | `DeepEIoUTrack`, `OSNetExtractor`, `ResNetExtractor` |
| **[SAM3](https://github.com/cubert-hyperspectral/cuvis-ai-sam3)** | Video object segmentation and tracking | `SAM3TrackerInference`, `SAM3TextPropagation`, `SAM3BboxPropagation`, `SAM3PointPropagation`, `SAM3MaskPropagation`, `SAM3SegmentEverything` |
| **[TrackEval](https://github.com/cubert-hyperspectral/cuvis-ai-trackeval)** | Tracking evaluation metrics | `HOTAMetricNode`, `CLEARMetricNode`, `IdentityMetricNode` |

---

## AdaCLIP

**Repository:** [cuvis-ai-adaclip](https://github.com/cubert-hyperspectral/cuvis-ai-adaclip)
· **Manifest:** `configs/plugins/adaclip.yaml`
· **Tag:** `v0.1.2`

Adapts [CLIP](https://openai.com/index/clip/) (Contrastive Language-Image Pre-training) for hyperspectral anomaly detection.
[AdaCLIP](https://arxiv.org/abs/2407.15795) performs zero-shot visual anomaly scoring using vision-language alignment, making it
powerful for detecting anomalies without extensive task-specific training.

| Node | Description |
|---|---|
| `AdaCLIPDetector` | Zero-shot anomaly detection using CLIP-adapted vision-language features |

**Tutorial:** [AdaCLIP Workflow](../tutorials/adaclip-workflow.md)

---

## Ultralytics

**Repository:** [cuvis-ai-ultralytics](https://github.com/cubert-hyperspectral/cuvis-ai-ultralytics)
· **Manifest:** `configs/plugins/ultralytics.yaml`
· **Tag:** `v0.1.0`

Wraps [Ultralytics](https://docs.ultralytics.com/) [YOLO](https://arxiv.org/abs/2506.13632) for frame-level object detection. Provides preprocessing,
inference, and postprocessing stages that slot into any tracking pipeline as the
detection front-end.

| Node | Description |
|---|---|
| `YOLOPreprocess` | Resizes and normalizes RGB images for YOLO26 inference |
| `YOLO26Detection` | Runs YOLO26 object detection on prepared frames |
| `YOLOPostprocess` | Applies NMS filtering, confidence thresholding, and class filtering |

**Example:** [examples/object_tracking/bytetrack/](../../examples/object_tracking/bytetrack/),
[examples/object_tracking/deepeiou/](../../examples/object_tracking/deepeiou/)

---

## DeepEIoU

**Repository:** [cuvis-ai-deepeiou](https://github.com/cubert-hyperspectral/cuvis-ai-deepeiou)
· **Manifest:** `configs/plugins/deepeiou.yaml`
· **Tag:** `v0.1.0`

Multi-object tracker using [Deep EIoU](https://arxiv.org/abs/2405.11605) (Efficient IoU) distance with optional appearance-based
association via ReID embeddings. Tracks single or multiple COCO classes and outputs
COCO-format detection and tracking JSON.

| Node | Description |
|---|---|
| `DeepEIoUTrack` | Multi-object tracker using [EIoU](https://arxiv.org/abs/2405.11605) distance metric with optional appearance matching |
| `OSNetExtractor` | Lightweight ReID feature extractor using [OSNet](https://arxiv.org/abs/1905.00953) backbone |
| `ResNetExtractor` | ReID feature extractor using [ResNet](https://arxiv.org/abs/1512.03385) backbone |

**Example:** [examples/object_tracking/deepeiou/](../../examples/object_tracking/deepeiou/)

---

## SAM3

**Repository:** [cuvis-ai-sam3](https://github.com/cubert-hyperspectral/cuvis-ai-sam3)
· **Manifest:** `configs/plugins/sam3.yaml`
· **Tag:** `v0.1.3`

Video object segmentation and tracking using [SAM 2.1](https://ai.meta.com/sam2/) (Segment Anything Model 2.1). Supports
multiple prompt strategies — text, bounding box, point, mask, or fully automatic — and
works on both CU3S hyperspectral and RGB video inputs.

| Node | Description |
|---|---|
| `SAM3TrackerInference` | Streaming SAM3 tracker for video sequences |
| `SAM3TextPropagation` | Text-prompted object tracking (e.g. "person", "dog") |
| `SAM3BboxPropagation` | Bounding-box-prompted object tracking from COCO detection JSON |
| `SAM3PointPropagation` | Single-point-prompted object tracking from detection JSON |
| `SAM3MaskPropagation` | Mask-prompted object tracking from segmentation JSON |
| `SAM3SegmentEverything` | Automatic segmentation of all objects per frame (prompt-free) |

### Prompt contract

Bbox and mask propagation use a scheduled prompt syntax:

```text
--prompt <object_id:detection_id@frame_id>
```

- `object_id` — tracker-side object ID maintained by SAM3
- `detection_id` — annotation ID copied from the supplied detection or tracking JSON
- `frame_id` — frame where the prompt is injected

Frames before the first scheduled prompt emit empty tracking outputs.

**Example:** [examples/object_tracking/sam3/](../../examples/object_tracking/sam3/)

---

## TrackEval

**Repository:** [cuvis-ai-trackeval](https://github.com/cubert-hyperspectral/cuvis-ai-trackeval)
· **Manifest:** `configs/plugins/trackeval.yaml`
· **Tag:** `v0.1.0`

Tracking evaluation metrics from [TrackEval](https://github.com/JonathonLuiten/TrackEval) that compare predicted tracking JSON
against ground-truth in COCO bbox format. Requires aligned frame counts between GT and prediction files.

| Node | Description |
|---|---|
| `HOTAMetricNode` | [HOTA](https://arxiv.org/abs/2009.07736) — Higher Order Tracking Accuracy, reports DetA, AssA, LocA components |
| `CLEARMetricNode` | [CLEAR MOT](https://link.springer.com/article/10.1155/2008/246309) metrics — reports MOTA, MOTP, FP, FN, ID switches |
| `IdentityMetricNode` | [Identity](https://arxiv.org/abs/1609.01775) metrics — reports IDF1, IDP, IDR |

**Example:** [examples/object_tracking/trackeval/](../../examples/object_tracking/trackeval/)

---

## Loading Plugins

Load a single plugin manifest:

```python
from cuvis_ai_core.utils.node_registry import NodeRegistry

registry = NodeRegistry()
registry.load_plugins("configs/plugins/sam3.yaml")
```

Or load all official plugins at once from the central registry:

```bash
uv run restore-pipeline \
    --pipeline-path configs/pipeline/my_pipeline.yaml \
    --plugins-path configs/plugins/registry.yaml
```

### Combined Manifest

Create a custom manifest when you need multiple plugins in one file:

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
    repo: "https://github.com/cubert-hyperspectral/cuvis-ai-sam3.git"
    tag: "v0.1.3"
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

### Troubleshooting

- If a local plugin path fails, verify the path is correct relative to the manifest file.
- If a Git plugin fails, verify the tag exists and the repo is accessible from the current environment.
- If a node cannot be found after loading, check that the class path appears in `provides:`.

See [Plugin System Overview](../plugin-system/overview.md) for architecture details and
[Plugin Development Guide](../plugin-system/development.md) for creating new plugins.
