# SAM3 Workflows

This page documents the current local and gRPC-facing SAM3 workflows shipped in the repository.
All examples use the checked-in SAM3 plugin manifest at `configs/plugins/sam3.yaml`.

## Workflow Matrix

| Workflow | Prompt style | Script | Input modes |
|---|---|---|---|
| Text propagation | Text prompt | [examples/object_tracking/sam3/sam3_text_propagation.py](../../examples/object_tracking/sam3/sam3_text_propagation.py) | CU3S, video |
| Bbox propagation | Scheduled bbox prompt | [examples/object_tracking/sam3/sam3_bbox_propagation.py](../../examples/object_tracking/sam3/sam3_bbox_propagation.py) | CU3S, video |
| Mask propagation | Scheduled mask prompt | [examples/object_tracking/sam3/sam3_mask_propagation.py](../../examples/object_tracking/sam3/sam3_mask_propagation.py) | CU3S, video |
| Point propagation | Point prompt | [examples/object_tracking/sam3/sam3_point_propagation.py](../../examples/object_tracking/sam3/sam3_point_propagation.py) | CU3S, video |
| Segment everything | Prompt-free | [examples/object_tracking/sam3/sam3_segment_everything.py](../../examples/object_tracking/sam3/sam3_segment_everything.py) | CU3S, video |

## Scheduled Prompt Contract

Bbox and mask propagation use the same repeatable prompt syntax:

```text
--prompt <object_id:detection_id@frame_id>
```

Meaning:

- `object_id`: tracker-side object ID to maintain inside SAM3
- `detection_id`: annotation ID to copy from the supplied detection or tracking JSON
- `frame_id`: frame where the prompt is injected

Behavior:

- Frames before the first scheduled bbox or mask prompt emit empty tracking outputs.
- Local bbox propagation injects `BBoxPrompt` nodes inside the pipeline.
- Local mask propagation injects `MaskPrompt` nodes inside the pipeline.
- gRPC bbox propagation sends prompt boxes directly through `InputBatch.bboxes`.
- gRPC mask propagation sends prompt masks directly through `InputBatch.mask`.

## Local Workflows

### Text Propagation

Pipeline shape:

```text
<Source RGB> -> SAM3TextPropagation -> CocoTrackMaskWriter
<Source RGB> + masks -> TrackingOverlayNode -> ToVideoNode
```

Outputs:

- Tracking COCO JSON
- Overlay MP4
- Saved pipeline YAML
- Optional weights and Graphviz image

### Bbox Propagation

Pipeline shape:

```text
<Source RGB> + DetectionJsonReader -> BBoxPrompt -> SAM3 propagation
-> CocoTrackMaskWriter -> TrackingOverlayNode -> ToVideoNode
```

Use when the seed signal is a bbox annotation stream in a COCO detection or tracking JSON.

### Mask Propagation

Pipeline shape:

```text
<Source RGB> + DetectionJsonReader -> MaskPrompt -> SAM3 propagation
-> CocoTrackMaskWriter -> TrackingOverlayNode -> ToVideoNode
```

Use when the seed signal is an existing segmentation in a COCO detection or tracking JSON.

### Point Propagation

Point propagation is local-only in this branch and is seeded from a COCO detection or tracking JSON.

### Segment Everything

Segment-everything is prompt-free and streams frames through `SAM3SegmentEverything`.

Typical outputs:

- Per-frame or streamed mask JSON
- Overlay MP4 when using the local workflow
- Saved pipeline YAML and Graphviz image

## gRPC Notes

The gRPC SAM3 clients share the standard session/config workflow:

1. `CreateSession`
2. `SetSessionSearchPaths`
3. `ResolveConfig`
4. `SetTrainRunConfig`
5. `Inference`
6. `CloseSession`

Current gRPC-only differences from local runs:

- gRPC SAM3 runs primarily write JSON outputs.
- Local scripts are the source for overlay-video and profiling artifacts.
- Prompt masks and prompt bboxes are injected directly through `InputBatch` rather than being materialized as static prompt nodes.

## Related Pages

- [Object Tracking Workflows](object-tracking.md)
- [gRPC Example Clients](../grpc/example-clients.md)
- [gRPC API Reference](../grpc/api-session.md)
