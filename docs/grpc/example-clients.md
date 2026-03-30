# gRPC Example Clients

The published gRPC reference pages describe the service. This page maps the checked-in example
clients to their purpose, required inputs, and canonical entry command.

Run the local server with:

```bash
uv run python -m cuvis_ai.grpc.production_server
```

All example clients use `cuvis_ai.utils.grpc_workflow` for the shared session and config flow.

## Core Workflow

| Client | Purpose | Required inputs | Command |
|---|---|---|---|
| [examples/grpc/core/capabilities_client.py](../../examples/grpc/core/capabilities_client.py) | Inspect service capabilities and server-side support | Server only | `uv run python examples/grpc/core/capabilities_client.py` |
| [examples/grpc/core/pipeline_discovery_client.py](../../examples/grpc/core/pipeline_discovery_client.py) | Discover available pipelines and inspect inputs/outputs | Server only | `uv run python examples/grpc/core/pipeline_discovery_client.py` |
| [examples/grpc/core/complete_workflow_client.py](../../examples/grpc/core/complete_workflow_client.py) | Full session, config, training, save flow | Trainrun config search paths | `uv run python examples/grpc/core/complete_workflow_client.py` |
| [examples/grpc/core/run_inference.py](../../examples/grpc/core/run_inference.py) | Run inference against a configured or restored pipeline | Input data + pipeline/trainrun | `uv run python examples/grpc/core/run_inference.py --help` |
| [examples/grpc/core/restore_trainrun_grpc.py](../../examples/grpc/core/restore_trainrun_grpc.py) | Restore a saved trainrun and run it remotely | Saved trainrun, optional plugin manifest | `uv run python examples/grpc/core/restore_trainrun_grpc.py --help` |
| [examples/grpc/core/restore_trainrun_statistical_grpc.py](../../examples/grpc/core/restore_trainrun_statistical_grpc.py) | Restore and run statistical-only workflows remotely | Saved trainrun | `uv run python examples/grpc/core/restore_trainrun_statistical_grpc.py --help` |

## Deep SVDD

| Client | Purpose | Required inputs | Command |
|---|---|---|---|
| [examples/grpc/deep_svdd/deepsvdd_client.py](../../examples/grpc/deep_svdd/deepsvdd_client.py) | End-to-end Deep SVDD workflow | Deep SVDD trainrun config | `uv run python examples/grpc/deep_svdd/deepsvdd_client.py` |
| [examples/grpc/deep_svdd/gradient_training_client.py](../../examples/grpc/deep_svdd/gradient_training_client.py) | Gradient-phase Deep SVDD training | Deep SVDD trainrun config | `uv run python examples/grpc/deep_svdd/gradient_training_client.py` |
| [examples/grpc/deep_svdd/resume_training_client.py](../../examples/grpc/deep_svdd/resume_training_client.py) | Resume a partially trained Deep SVDD run | Saved pipeline / trainrun | `uv run python examples/grpc/deep_svdd/resume_training_client.py` |

## RX

| Client | Purpose | Required inputs | Command |
|---|---|---|---|
| [examples/grpc/rx/statistical_training_client.py](../../examples/grpc/rx/statistical_training_client.py) | Train the RX statistical baseline remotely | RX trainrun config | `uv run python examples/grpc/rx/statistical_training_client.py` |
| [examples/grpc/rx/inference_with_pretrained_client.py](../../examples/grpc/rx/inference_with_pretrained_client.py) | Run inference from a pretrained RX pipeline | Saved pipeline / trainrun | `uv run python examples/grpc/rx/inference_with_pretrained_client.py` |
| [examples/grpc/rx/introspection_client.py](../../examples/grpc/rx/introspection_client.py) | Inspect RX pipeline structure and outputs | Server only | `uv run python examples/grpc/rx/introspection_client.py` |

## Channel Selector

| Client | Purpose | Required inputs | Command |
|---|---|---|---|
| [examples/grpc/channel_selector/train_from_scratch_client.py](../../examples/grpc/channel_selector/train_from_scratch_client.py) | Train the channel-selector workflow remotely | Channel-selector trainrun config | `uv run python examples/grpc/channel_selector/train_from_scratch_client.py` |
| [examples/grpc/channel_selector/checkpoint_client.py](../../examples/grpc/channel_selector/checkpoint_client.py) | Resume or inspect a checkpoint-backed workflow | Saved checkpoint / trainrun | `uv run python examples/grpc/channel_selector/checkpoint_client.py` |
| [examples/grpc/channel_selector/complex_inputs_client.py](../../examples/grpc/channel_selector/complex_inputs_client.py) | Demonstrate richer input payloads | Pipeline plus input tensors | `uv run python examples/grpc/channel_selector/complex_inputs_client.py` |

## AdaCLIP

| Client | Purpose | Required inputs | Command |
|---|---|---|---|
| [examples/grpc/adaclip/adaclip_client.py](../../examples/grpc/adaclip/adaclip_client.py) | Baseline AdaCLIP workflow | AdaCLIP trainrun config + plugin manifest | `uv run python examples/grpc/adaclip/adaclip_client.py` |
| [examples/grpc/adaclip/adaclip_high_contrast_client.py](../../examples/grpc/adaclip/adaclip_high_contrast_client.py) | High-contrast selector variant | AdaCLIP config + plugin manifest | `uv run python examples/grpc/adaclip/adaclip_high_contrast_client.py` |
| [examples/grpc/adaclip/adaclip_cir_false_color_client.py](../../examples/grpc/adaclip/adaclip_cir_false_color_client.py) | CIR false-color AdaCLIP variant | AdaCLIP config + plugin manifest | `uv run python examples/grpc/adaclip/adaclip_cir_false_color_client.py` |
| [examples/grpc/adaclip/adaclip_cir_false_rg_color_client.py](../../examples/grpc/adaclip/adaclip_cir_false_rg_color_client.py) | CIR false-RG variant | AdaCLIP config + plugin manifest | `uv run python examples/grpc/adaclip/adaclip_cir_false_rg_color_client.py` |
| [examples/grpc/adaclip/adaclip_supervised_cir_client.py](../../examples/grpc/adaclip/adaclip_supervised_cir_client.py) | Supervised CIR selection variant | AdaCLIP config + plugin manifest | `uv run python examples/grpc/adaclip/adaclip_supervised_cir_client.py` |
| [examples/grpc/adaclip/adaclip_supervised_full_spectrum_client.py](../../examples/grpc/adaclip/adaclip_supervised_full_spectrum_client.py) | Full-spectrum supervised selector variant | AdaCLIP config + plugin manifest | `uv run python examples/grpc/adaclip/adaclip_supervised_full_spectrum_client.py` |
| [examples/grpc/adaclip/adaclip_supervised_windowed_false_rgb_client.py](../../examples/grpc/adaclip/adaclip_supervised_windowed_false_rgb_client.py) | Windowed supervised false-RGB variant | AdaCLIP config + plugin manifest | `uv run python examples/grpc/adaclip/adaclip_supervised_windowed_false_rgb_client.py` |
| [examples/grpc/adaclip/adaclip_lentils_inference.py](../../examples/grpc/adaclip/adaclip_lentils_inference.py) | Inference-only AdaCLIP example | Saved pipeline + plugin manifest | `uv run python examples/grpc/adaclip/adaclip_lentils_inference.py --help` |

## SAM3

All SAM3 examples require `configs/plugins/sam3.yaml`.

| Client | Purpose | Required inputs | Command |
|---|---|---|---|
| [examples/grpc/sam3/sam3_text_propagation_client.py](../../examples/grpc/sam3/sam3_text_propagation_client.py) | Text-prompt tracking through SAM3 | CU3S or video input + prompt text | `uv run python examples/grpc/sam3/sam3_text_propagation_client.py --help` |
| [examples/grpc/sam3/sam3_bbox_propagation_client.py](../../examples/grpc/sam3/sam3_bbox_propagation_client.py) | Scheduled bbox prompting via `InputBatch.bboxes` | CU3S or video input + detection JSON + `--prompt` specs | `uv run python examples/grpc/sam3/sam3_bbox_propagation_client.py --help` |
| [examples/grpc/sam3/sam3_mask_propagation_client.py](../../examples/grpc/sam3/sam3_mask_propagation_client.py) | Scheduled mask prompting via `InputBatch.mask` | CU3S or video input + detection JSON + `--prompt` specs | `uv run python examples/grpc/sam3/sam3_mask_propagation_client.py --help` |
| [examples/grpc/sam3/sam3_segment_everything_client.py](../../examples/grpc/sam3/sam3_segment_everything_client.py) | Prompt-free per-frame segmentation | CU3S or video input | `uv run python examples/grpc/sam3/sam3_segment_everything_client.py --help` |

## Related Pages

- [SAM3 Workflows](../how-to/sam3-workflows.md)
- [gRPC API Reference](api-reference.md)
- [Client Patterns](client-patterns.md)
