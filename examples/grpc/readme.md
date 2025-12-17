Run the gRPC server locally:

```
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

```
python examples/grpc/statistical_training_client.py
python examples/grpc/gradient_training_client.py
python examples/grpc/complete_workflow_client.py
```

Adjust paths in the scripts if your configs or outputs live elsewhere.
