# Object Tracking Workflows

Current tracking workflows live in `examples/object_tracking/` and are mirrored here as the
published entrypoint for the branch's tracking surface.

## Overview

| Workflow | Example | Purpose | Inputs | Plugin manifest |
|---|---|---|---|---|
| False-RGB export | [examples/object_tracking/export_cu3s_false_rgb_video.py](../../examples/object_tracking/export_cu3s_false_rgb_video.py) | Export a CU3S recording to MP4 without tracking | `.cu3s` | None |
| Tracking overlay render | [examples/object_tracking/render_tracking_overlay.py](../../examples/object_tracking/render_tracking_overlay.py) | Render bbox or mask overlays from COCO tracking JSON | `.mp4` or `.cu3s` + tracking JSON | None |
| Synthetic occlusion | [examples/object_tracking/occlusion/occlude_data.py](../../examples/object_tracking/occlusion/occlude_data.py) | Generate Poisson-filled occlusion videos from tracking masks or bboxes | `.cu3s` + tracking JSON | None |
| TrackEval metric nodes | [examples/object_tracking/trackeval/evaluate_tracking.py](../../examples/object_tracking/trackeval/evaluate_tracking.py) | Score predicted and ground-truth tracking JSONs through TrackEval nodes | GT JSON + prediction JSON | `configs/plugins/trackeval.yaml` |
| ByteTrack HSI | [examples/object_tracking/bytetrack/yolo_bytetrack_hsi.py](../../examples/object_tracking/bytetrack/yolo_bytetrack_hsi.py) | YOLO detection plus ByteTrack on CU3S false-RGB frames | `.cu3s` | `configs/plugins/ultralytics.yaml`, `configs/plugins/bytetrack.yaml` |
| ByteTrack RGB | [examples/object_tracking/bytetrack/yolo_bytetrack_rgb.py](../../examples/object_tracking/bytetrack/yolo_bytetrack_rgb.py) | YOLO detection plus ByteTrack directly on RGB video | `.mp4` | `configs/plugins/ultralytics.yaml`, `configs/plugins/bytetrack.yaml` |
| DeepEIOU HSI / RGB | [examples/object_tracking/deepeiou/yolo_deepeiou_reid_hsi.py](../../examples/object_tracking/deepeiou/yolo_deepeiou_reid_hsi.py) | YOLO + DeepEIOU tracking with optional ReID and feature export | `.cu3s` or `.mp4` | `configs/plugins/ultralytics.yaml`, `configs/plugins/deepeiou.yaml` |
| False-RGB training workflow | [examples/object_tracking/channel_selector_false_rgb.py](../../examples/object_tracking/channel_selector_false_rgb.py) | Train or inspect channel-selector-driven false-RGB pipelines | trainrun config + dataset | None |

## False-RGB Export

Pipeline shape:

```text
CU3SDataNode -> <FalseRGB selector> -> ToVideoNode
```

Supported selector modes:

| `--method` | Node |
|---|---|
| `cie_tristimulus` | `CIETristimulusFalseRGBSelector` |
| `cir` | `CIRSelector` |
| `fast_rgb` / `fastrgb` | `FastRGBSelector` |
| `cuvis-plugin` | `FastRGBSelector` with XML-derived ranges |

Main outputs:

- MP4 export
- Saved pipeline YAML
- Graphviz pipeline image

## Overlay And Occlusion Tools

These helpers are post-processing workflows built on the tracking JSON surface documented by the
tracking writers and readers in the node catalog.

| Example | Main nodes | Outputs |
|---|---|---|
| [render_tracking_overlay.py](../../examples/object_tracking/render_tracking_overlay.py) | `TrackingResultsReader`, `TrackingOverlayNode`, `ToVideoNode` | Overlay MP4 |
| [occlude_data.py](../../examples/object_tracking/occlusion/occlude_data.py) | `TrackingResultsReader`, `PoissonOcclusionNode` or `PoissonCubeOcclusionNode`, `ToVideoNode` | Occluded MP4, profiling summary |
| [extract_track_mask_signature.py](../../examples/object_tracking/extract_track_mask_signature.py) | Tracking JSON reader helpers | Signature exports for analysis |

## Multi-Object Tracking Pipelines

### ByteTrack

Pipeline shape for CU3S mode:

```text
CU3SDataNode -> CIETristimulusFalseRGBSelector -> YOLO detection -> ByteTrack
-> DetectionCocoJsonNode / CocoTrackBBoxWriter -> BBoxesOverlayNode -> ToVideoNode
```

Related helpers:

- [examples/object_tracking/bytetrack/run_bytetrack_sweep.py](../../examples/object_tracking/bytetrack/run_bytetrack_sweep.py)
- [examples/object_tracking/bytetrack/regenerate_bytetrack_sweep_html.py](../../examples/object_tracking/bytetrack/regenerate_bytetrack_sweep_html.py)
- [examples/object_tracking/bytetrack/convert_overlays_in_place.py](../../examples/object_tracking/bytetrack/convert_overlays_in_place.py)
- [examples/object_tracking/trackeval/prepare_trackeval_json.py](../../examples/object_tracking/trackeval/prepare_trackeval_json.py)

Main outputs:

- Detection COCO JSON
- Tracking COCO JSON
- Overlay MP4

### DeepEIOU

Pipeline shape:

```text
<Video or CU3S source> -> YOLO detection -> DeepEIOU tracking
-> DetectionCocoJsonNode / CocoTrackBBoxWriter
-> optional NumpyFeatureWriterNode
-> BBoxesOverlayNode -> ToVideoNode
```

DeepEIOU supports:

- EIoU-only mode
- ReID-enabled tracking
- Optional feature `.npy` export
- Shared logic across CU3S and direct video inputs

## TrackEval Workflow

Use [examples/object_tracking/trackeval/evaluate_tracking.py](../../examples/object_tracking/trackeval/evaluate_tracking.py)
to evaluate predicted and ground-truth COCO tracking JSON files through TrackEval metric nodes.

Inputs:

- Ground-truth tracking JSON
- Prediction tracking JSON
- `configs/plugins/trackeval.yaml`

Outputs:

- Printed metrics summary
- Pipeline image / saved pipeline artifacts when enabled by the example

## Related Pages

- [SAM3 Workflows](sam3-workflows.md)
- [gRPC Example Clients](../grpc/example-clients.md)
- [Pipeline Configuration Schema](../config/pipeline-schema.md)
- [Output Nodes](../node-catalog/output.md)
