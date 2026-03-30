# gRPC Examples (Model-First Layout)

Run the local server:

```bash
uv run python -m cuvis_ai.grpc.production_server
```

All clients use the same helper module:

- `cuvis_ai/utils/grpc_workflow.py`

Core workflow phases:
1. `CreateSession`
2. `SetSessionSearchPaths`
3. `ResolveConfig`
4. `SetTrainRunConfig`
5. `Train` (statistical / gradient)
6. `SavePipeline` / `SaveTrainRun` / `Inference`

## Folder Index

### `core/` (model-agnostic)

- `examples/grpc/core/capabilities_client.py`
- `examples/grpc/core/pipeline_discovery_client.py`
- `examples/grpc/core/complete_workflow_client.py`
- `examples/grpc/core/run_inference.py`
- `examples/grpc/core/restore_trainrun_grpc.py`
- `examples/grpc/core/restore_trainrun_statistical_grpc.py`

### `deep_svdd/`

- `examples/grpc/deep_svdd/deepsvdd_client.py`
- `examples/grpc/deep_svdd/gradient_training_client.py`
- `examples/grpc/deep_svdd/resume_training_client.py`

### `rx/`

- `examples/grpc/rx/statistical_training_client.py`
- `examples/grpc/rx/inference_with_pretrained_client.py`
- `examples/grpc/rx/introspection_client.py`

### `channel_selector/`

- `examples/grpc/channel_selector/train_from_scratch_client.py`
- `examples/grpc/channel_selector/checkpoint_client.py`
- `examples/grpc/channel_selector/complex_inputs_client.py`

### `adaclip/`

- `examples/grpc/adaclip/adaclip_client.py`
- `examples/grpc/adaclip/adaclip_high_contrast_client.py`
- `examples/grpc/adaclip/adaclip_cir_false_color_client.py`
- `examples/grpc/adaclip/adaclip_cir_false_rg_color_client.py`
- `examples/grpc/adaclip/adaclip_supervised_cir_client.py`
- `examples/grpc/adaclip/adaclip_supervised_full_spectrum_client.py`
- `examples/grpc/adaclip/adaclip_supervised_windowed_false_rgb_client.py`
- `examples/grpc/adaclip/adaclip_lentils_inference.py`

### `sam3/`

- `examples/grpc/sam3/sam3_text_propagation_client.py`
- `examples/grpc/sam3/sam3_bbox_propagation_client.py`
- `examples/grpc/sam3/sam3_mask_propagation_client.py`
- `examples/grpc/sam3/sam3_segment_everything_client.py`

## Canonical Commands

```bash
# core
uv run python examples/grpc/core/capabilities_client.py
uv run python examples/grpc/core/pipeline_discovery_client.py
uv run python examples/grpc/core/complete_workflow_client.py
uv run python examples/grpc/core/run_inference.py --help
uv run python examples/grpc/core/restore_trainrun_grpc.py --help
uv run python examples/grpc/core/restore_trainrun_statistical_grpc.py --help

# deep_svdd
uv run python examples/grpc/deep_svdd/deepsvdd_client.py
uv run python examples/grpc/deep_svdd/gradient_training_client.py
uv run python examples/grpc/deep_svdd/resume_training_client.py

# rx
uv run python examples/grpc/rx/statistical_training_client.py
uv run python examples/grpc/rx/inference_with_pretrained_client.py
uv run python examples/grpc/rx/introspection_client.py

# channel_selector
uv run python examples/grpc/channel_selector/train_from_scratch_client.py
uv run python examples/grpc/channel_selector/checkpoint_client.py
uv run python examples/grpc/channel_selector/complex_inputs_client.py

# adaclip
uv run python examples/grpc/adaclip/adaclip_client.py
uv run python examples/grpc/adaclip/adaclip_high_contrast_client.py
uv run python examples/grpc/adaclip/adaclip_cir_false_color_client.py
uv run python examples/grpc/adaclip/adaclip_cir_false_rg_color_client.py
uv run python examples/grpc/adaclip/adaclip_supervised_cir_client.py
uv run python examples/grpc/adaclip/adaclip_supervised_full_spectrum_client.py
uv run python examples/grpc/adaclip/adaclip_supervised_windowed_false_rgb_client.py
uv run python examples/grpc/adaclip/adaclip_lentils_inference.py --help

# sam3
uv run python examples/grpc/sam3/sam3_text_propagation_client.py --help
uv run python examples/grpc/sam3/sam3_bbox_propagation_client.py --help
uv run python examples/grpc/sam3/sam3_mask_propagation_client.py --help
uv run python examples/grpc/sam3/sam3_segment_everything_client.py --help
```

## SAM3 Mask/Bbox Propagation

Runtime mask propagation uses these pipeline configs:

- `configs/pipeline/sam3/sam3_mask_propagation.yaml`
- `configs/pipeline/sam3/sam3_mask_propagation_video.yaml`

Runtime bbox propagation uses these pipeline configs:

- `configs/pipeline/sam3/sam3_bbox_propagation.yaml`
- `configs/pipeline/sam3/sam3_bbox_propagation_video.yaml`

Runtime segment-everything uses these pipeline configs:

- `configs/pipeline/sam3/sam3_segment_everything.yaml`
- `configs/pipeline/sam3/sam3_segment_everything_video.yaml`

The gRPC mask workflow does not use `MaskPrompt` inside the pipeline. Instead, the
client decodes scheduled segmentations from `--detection-json` and sends prompt
masks directly through `InputBatch.mask` on the requested frames:

```powershell
uv run python examples/grpc/sam3/sam3_mask_propagation_client.py `
  --video-path "D:\experiments\20260319\video_creation\tristimulus\XMR_25mm_CubertParkingLotTracking\2026_03_19_11-27-39\Auto_004+01.mp4" `
  --detection-json "D:\experiments\20260319\video_creation\tristimulus\XMR_25mm_CubertParkingLotTracking\2026_03_19_11-27-39\Auto_004+01.json" `
  --prompt 2:2@65 `
  --prompt 1:1@70 `
  --output-json-path "D:\experiments\20260326\mask_propagation_grpc\video\Auto_004+01.json"
```

The gRPC bbox workflow follows the same scheduled prompt contract, but sends
prompt boxes through `InputBatch.bboxes` instead of prompt masks:

```powershell
uv run python examples/grpc/sam3/sam3_bbox_propagation_client.py `
  --video-path "D:\experiments\20260319\video_creation\tristimulus\XMR_25mm_CubertParkingLotTracking\2026_03_19_11-27-39\Auto_004+01.mp4" `
  --detection-json "D:\experiments\20260319\video_creation\tristimulus\XMR_25mm_CubertParkingLotTracking\2026_03_19_11-27-39\Auto_004+01.json" `
  --prompt 2:2@65 `
  --prompt 1:1@70 `
  --output-json-path "D:\experiments\20260326\bbox_propagation_grpc\video\Auto_004+01.json"
```

The gRPC segment-everything workflow is prompt-free and simply streams frames
through `SAM3SegmentEverything`:

```powershell
uv run python examples/grpc/sam3/sam3_segment_everything_client.py `
  --video-path "D:\experiments\20260319\video_creation\tristimulus\XMR_25mm_CubertParkingLotTracking\2026_03_19_11-27-39\Auto_004+01.mp4" `
  --start-frame 70 `
  --max-frames 1 `
  --output-json-path "D:\experiments\20260330\segment_everything_grpc\video\Auto_004+01_frame0070.json"
```
