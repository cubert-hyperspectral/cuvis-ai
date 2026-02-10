"""Example client demonstrating pipeline introspection RPCs."""

from __future__ import annotations

from pathlib import Path

from cuvis_ai_schemas.grpc.v1 import cuvis_ai_pb2
from workflow_utils import (
    CONFIG_ROOT,
    build_stub,
    config_search_paths,
    create_session_with_search_paths,
)


def main() -> None:
    stub = build_stub()
    session_id = create_session_with_search_paths(stub, config_search_paths())

    # Build pipeline from resolved config and load weights for accurate shapes
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
    stub.LoadPipelineWeights(
        cuvis_ai_pb2.LoadPipelineWeightsRequest(
            session_id=session_id,
            weights_path=str((CONFIG_ROOT / "pipeline" / "rx_statistical.pt").resolve()),
        )
    )
    print(f"Session ready for introspection: {session_id}")

    inputs_resp = stub.GetPipelineInputs(
        cuvis_ai_pb2.GetPipelineInputsRequest(session_id=session_id)
    )
    print("\nPipeline inputs:")
    for name in inputs_resp.input_names:
        spec = inputs_resp.input_specs[name]
        dtype_name = cuvis_ai_pb2.DType.Name(spec.dtype)
        required = "required" if spec.required else "optional"
        print(f"  {name}: shape={list(spec.shape)}, dtype={dtype_name}, {required}")

    outputs_resp = stub.GetPipelineOutputs(
        cuvis_ai_pb2.GetPipelineOutputsRequest(session_id=session_id)
    )
    print("\nPipeline outputs:")
    for name in outputs_resp.output_names:
        spec = outputs_resp.output_specs[name]
        dtype_name = cuvis_ai_pb2.DType.Name(spec.dtype)
        print(f"  {name}: shape={list(spec.shape)}, dtype={dtype_name}")

    viz_resp = stub.GetPipelineVisualization(
        cuvis_ai_pb2.GetPipelineVisualizationRequest(session_id=session_id, format="png")
    )
    output_path = Path("pipeline_visualization.png")
    output_path.write_bytes(viz_resp.image_data)
    print(f"\nVisualization saved: {output_path} ({len(viz_resp.image_data)} bytes)")

    stub.CloseSession(cuvis_ai_pb2.CloseSessionRequest(session_id=session_id))


if __name__ == "__main__":
    main()
