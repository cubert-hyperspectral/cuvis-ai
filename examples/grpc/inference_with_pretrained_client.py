"""Inference with a pre-trained pipeline using explicit config resolution."""

from __future__ import annotations

import numpy as np
from workflow_utils import (
    CONFIG_ROOT,
    build_stub,
    config_search_paths,
    create_session_with_search_paths,
)

from cuvis_ai.grpc import cuvis_ai_pb2, helpers


def main() -> None:
    stub = build_stub()
    session_id = create_session_with_search_paths(stub, config_search_paths())
    print(f"Session: {session_id}")

    # Resolve pipeline config and build pipeline
    pipeline_config = stub.ResolveConfig(
        cuvis_ai_pb2.ResolveConfigRequest(
            session_id=session_id,
            config_type="pipeline",
            path="pipeline/rx_statistical",
        )
    )
    stub.LoadPipeline(
        cuvis_ai_pb2.LoadPipelineRequest(
            session_id=session_id,
            pipeline=cuvis_ai_pb2.PipelineConfig(config_bytes=pipeline_config.config_bytes),
        )
    )

    # Load pretrained weights colocated with the pipeline YAML
    weights_path = str((CONFIG_ROOT / "pipeline" / "rx_statistical.pt").resolve())
    stub.LoadPipelineWeights(
        cuvis_ai_pb2.LoadPipelineWeightsRequest(
            session_id=session_id, weights_path=weights_path, strict=True
        )
    )
    print(f"Loaded weights from {weights_path}")

    # Run inference
    batch_size, height, width, channels = 1, 64, 64, 61
    cube = np.random.randn(batch_size, height, width, channels).astype(np.uint16)
    wavelengths = np.linspace(430, 910, 61).reshape(1, -1).astype(np.int32)
    inference_response = stub.Inference(
        cuvis_ai_pb2.InferenceRequest(
            session_id=session_id,
            inputs=cuvis_ai_pb2.InputBatch(
                cube=helpers.numpy_to_proto(cube), wavelengths=helpers.numpy_to_proto(wavelengths)
            ),
        )
    )

    print(f"Outputs: {len(inference_response.outputs)} tensors")
    for output_name, output_tensor in inference_response.outputs.items():
        output_array = helpers.proto_to_numpy(output_tensor)
        print(
            f"  {output_name}: shape={output_array.shape}, dtype={output_array.dtype}, "
            f"min={output_array.min():.4f}, max={output_array.max():.4f}"
        )

    stub.CloseSession(cuvis_ai_pb2.CloseSessionRequest(session_id=session_id))
    print("Session closed.")


if __name__ == "__main__":
    main()
