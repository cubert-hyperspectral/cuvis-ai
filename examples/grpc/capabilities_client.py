"""Discover and validate training capabilities using ConfigService."""

from __future__ import annotations

import json

from cuvis_ai_schemas.grpc.v1 import cuvis_ai_pb2
from workflow_utils import build_stub, config_search_paths, create_session_with_search_paths


def main() -> None:
    stub = build_stub()
    session_id = create_session_with_search_paths(stub, config_search_paths())

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

    # Resolve a baseline training config via Hydra
    base_training = stub.ResolveConfig(
        cuvis_ai_pb2.ResolveConfigRequest(
            session_id=session_id,
            config_type="training",
            path="training/default",
        )
    )
    base_dict = json.loads(base_training.config_bytes.decode("utf-8"))

    good_resp = stub.ValidateConfig(
        cuvis_ai_pb2.ValidateConfigRequest(
            config_type="training", config_bytes=json.dumps(base_dict).encode("utf-8")
        )
    )
    print(f"\nValidating baseline config: valid={good_resp.valid}, errors={list(good_resp.errors)}")

    # Create an invalid variant for demonstration
    bad_dict = dict(base_dict)
    bad_dict["optimizer"] = {"name": "invalid_opt", "lr": -0.001}
    bad_resp = stub.ValidateConfig(
        cuvis_ai_pb2.ValidateConfigRequest(
            config_type="training", config_bytes=json.dumps(bad_dict).encode("utf-8")
        )
    )
    print(f"\nValidating bad config: valid={bad_resp.valid}")
    for err in bad_resp.errors:
        print(f"  {err}")
    if bad_resp.warnings:
        print("Warnings:")
        for warn in bad_resp.warnings:
            print(f"  - {warn}")

    stub.CloseSession(cuvis_ai_pb2.CloseSessionRequest(session_id=session_id))


if __name__ == "__main__":
    main()
