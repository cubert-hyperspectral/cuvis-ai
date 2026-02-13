# SAM3 Tracking Example (Scaffold)

This folder contains the Phase 1 scaffold for SAM3 integration.

## Files

- `plugins.yaml`: local plugin manifest for loading `cuvis-ai-sam3`
- `sam3_tracking_example.py`: end-to-end usage scaffold (placeholder)

## Status

- Phase 1: scaffolding complete
- Phase 2/3: implementation pending (`CU3SDataNode` + `SAM3VideoTracker`)

## Run

```bash
uv run python examples/sam3/sam3_tracking_example.py --video path/to/video.cu3s
```

The script intentionally raises `NotImplementedError` until T1.2/T1.3 are complete.

