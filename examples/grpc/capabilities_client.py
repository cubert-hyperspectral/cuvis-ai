"""Example client for training capability discovery and validation."""

import grpc

from cuvis_ai.grpc import cuvis_ai_pb2, cuvis_ai_pb2_grpc
from cuvis_ai.training.config import OptimizerConfig, TrainerConfig, TrainingConfig


def main() -> None:
    channel = grpc.insecure_channel("localhost:50051")
    stub = cuvis_ai_pb2_grpc.CuvisAIServiceStub(channel)

    print("Fetching training capabilities...")
    caps = stub.GetTrainingCapabilities(cuvis_ai_pb2.GetTrainingCapabilitiesRequest())

    print(f"Supported optimizers: {', '.join(caps.supported_optimizers)}")
    print(f"Supported schedulers: {', '.join(caps.supported_schedulers)}")
    print("\nCallbacks:")
    for cb in caps.supported_callbacks:
        print(f"- {cb.type}: {cb.description}")
        for param in cb.parameters:
            requirement = "required" if param.required else "optional"
            default = f" (default={param.default_value})" if param.default_value else ""
            validation = f" [{param.validation}]" if param.validation else ""
            print(f"    {param.name} ({param.type}, {requirement}){default}{validation}")

    good_cfg = TrainingConfig(
        trainer=TrainerConfig(max_epochs=10, accelerator="cude"),
        optimizer=OptimizerConfig(name="adam", lr=0.001),
    )
    good_resp = stub.ValidateTrainingConfig(
        cuvis_ai_pb2.ValidateTrainingConfigRequest(
            config=cuvis_ai_pb2.TrainingConfig(config_bytes=good_cfg.to_json().encode())
        )
    )
    print(f"\nValidating good config: valid={good_resp.valid}, errors={list(good_resp.errors)}")

    bad_cfg = TrainingConfig(
        trainer=TrainerConfig(max_epochs=10),
        optimizer=OptimizerConfig(name="invalid_opt", lr=-0.001),
    )
    bad_resp = stub.ValidateTrainingConfig(
        cuvis_ai_pb2.ValidateTrainingConfigRequest(
            config=cuvis_ai_pb2.TrainingConfig(config_bytes=bad_cfg.to_json().encode())
        )
    )
    print(f"\nValidating bad config: valid={bad_resp.valid}")
    for err in bad_resp.errors:
        print(f"  {err}")
    if bad_resp.warnings:
        print("Warnings:")
        for warn in bad_resp.warnings:
            print(f"  - {warn}")


if __name__ == "__main__":
    main()
