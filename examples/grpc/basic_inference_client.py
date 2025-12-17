"""Minimal inference client using explicit ResolveConfig + LoadPipeline (bytes)."""

from __future__ import annotations

import numpy as np
from workflow_utils import (
    CONFIG_ROOT,
    build_stub,
    config_search_paths,
    create_session_with_search_paths,
)

from cuvis_ai.grpc import cuvis_ai_pb2, helpers


def main(server_address: str = "localhost:50051") -> None:
    stub = build_stub(server_address)
    session_id = create_session_with_search_paths(stub, config_search_paths())
    print(f"Session created: {session_id}")

    # Resolve pipeline config via ConfigService (explicit Hydra composition path)
    pipeline_config = stub.ResolveConfig(
        cuvis_ai_pb2.ResolveConfigRequest(
            session_id=session_id,
            config_type="pipeline",
            path="pipeline/channel_selector",
        )
    )

    # Build the pipeline structure
    stub.LoadPipeline(
        cuvis_ai_pb2.LoadPipelineRequest(
            session_id=session_id,
            pipeline=cuvis_ai_pb2.PipelineConfig(config_bytes=pipeline_config.config_bytes),
        )
    )

    # Load pretrained weights from the configs directory
    weights_path = str((CONFIG_ROOT / "pipeline" / "channel_selector.pt").resolve())
    stub.LoadPipelineWeights(
        cuvis_ai_pb2.LoadPipelineWeightsRequest(
            session_id=session_id,
            weights_path=weights_path,
            strict=True,
        )
    )
    print(f"Loaded weights from {weights_path}")

    # Run inference with a random cube
    cube = np.random.randn(1, 16, 16, 61).astype(np.uint16)
    wavelengths = np.linspace(430, 910, 61).reshape(1, -1).astype(np.int32)
    response = stub.Inference(
        cuvis_ai_pb2.InferenceRequest(
            session_id=session_id,
            inputs=cuvis_ai_pb2.InputBatch(
                cube=helpers.numpy_to_proto(cube),
                wavelengths=helpers.numpy_to_proto(wavelengths),
            ),
        )
    )
    print(f"Outputs: {list(response.outputs.keys())}")

    for name, tensor_proto in response.outputs.items():
        array = helpers.proto_to_numpy(tensor_proto)
        print(f"  {name}: shape={array.shape}, dtype={array.dtype}")

    stub.CloseSession(cuvis_ai_pb2.CloseSessionRequest(session_id=session_id))
    print("Session closed.")


if __name__ == "__main__":
    main()
