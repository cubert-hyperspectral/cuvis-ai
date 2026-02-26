Run the gRPC server locally:

```bash
uv run python -m cuvis_ai.grpc.production_server
```

All clients now follow the explicit Phase 5 workflow:
1. CreateSession (empty)
2. SetSessionSearchPaths (absolute paths to `configs/*`)
3. ResolveConfig via ConfigService (Hydra composition + overrides)
4. SetTrainRunConfig with the resolved bytes
5. Train (statistical or gradient)
6. SavePipeline/SaveTrainRun or run Inference

`examples/grpc/workflow_utils.py` centralizes stub creation, search-path setup, and TrainRun helpers. Quick starts:

```bash
python examples/grpc/statistical_training_client.py
python examples/grpc/gradient_training_client.py
python examples/grpc/complete_workflow_client.py
```

## Restoring Pipelines and TrainRuns

### Restore Pipeline for Inference

To restore a trained pipeline and run inference on CU3S data:

```bash
# Run inference on CU3S data (all parameters required)
uv run python examples/grpc/run_inference.py \
  --pipeline-path configs/pipeline/anomaly/rx/channel_selector.yaml \
  --weights-path outputs/my_weights.pt \
  --cu3s-file-path data/DemoData/Demo_000.cu3s

# Run inference with custom processing mode
uv run python examples/grpc/run_inference.py \
  --pipeline-path configs/pipeline/anomaly/rx/channel_selector.yaml \
  --weights-path outputs/my_weights.pt \
  --cu3s-file-path data/DemoData/Demo_000.cu3s \
  --processing-mode Raw

# Run inference with config overrides
uv run python examples/grpc/run_inference.py \
  --pipeline-path configs/pipeline/anomaly/rx/channel_selector.yaml \
  --weights-path outputs/my_weights.pt \
  --cu3s-file-path data/DemoData/Demo_000.cu3s \
  --override nodes.10.params.output_dir=outputs/my_tb
```

### Restore and Reproduce Training Runs

To restore complete training runs and reproduce training, validation, or testing:

```bash
# Display trainrun info
uv run python examples/grpc/restore_trainrun_grpc.py --trainrun-path outputs/channel_selector/trained_models/channel_selector_trainrun.yaml

# Re-run training
uv run python examples/grpc/restore_trainrun_grpc.py --trainrun-path outputs/.../trainrun.yaml --mode train

# Run validation only
uv run python examples/grpc/restore_trainrun_grpc.py --trainrun-path outputs/.../trainrun.yaml --mode validate

# Override data and training configs
uv run python examples/grpc/restore_trainrun_grpc.py --trainrun-path outputs/.../trainrun.yaml --mode train --override data.batch_size=16 --override training.optimizer.lr=0.001
```

Adjust paths in the scripts if your configs or outputs live elsewhere.
