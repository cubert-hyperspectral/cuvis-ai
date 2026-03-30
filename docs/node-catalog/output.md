!!! warning "Status: Needs Review"
    This page has not been reviewed for accuracy and completeness. Content may be outdated or contain errors.

---

# Output Nodes

Output nodes persist detections, tracks, features, and rendered video artifacts.

## Tracking And Detection Writers

::: cuvis_ai.node.json_writer
    options:
      show_root_heading: true
      heading_level: 3

## NumPy Feature Writers

::: cuvis_ai.node.numpy_writer
    options:
      show_root_heading: true
      heading_level: 3

## Video Outputs

::: cuvis_ai.node.video
    options:
      show_root_heading: true
      heading_level: 3

## Common Tracking Sink Patterns

```text
YOLO / tracker -> DetectionCocoJsonNode or CocoTrackBBoxWriter -> JSON output
SAM3 / mask tracker -> CocoTrackMaskWriter -> tracking JSON output
RGB or overlays -> ToVideoNode -> MP4 output
DeepEIOU embeddings -> NumpyFeatureWriterNode -> .npy output
```
