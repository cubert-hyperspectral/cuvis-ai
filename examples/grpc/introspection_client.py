"""Example client demonstrating pipeline introspection RPCs."""

from pathlib import Path

import grpc

from cuvis_ai.grpc import cuvis_ai_pb2, cuvis_ai_pb2_grpc


def main() -> None:
    channel = grpc.insecure_channel("localhost:50051")
    stub = cuvis_ai_pb2_grpc.CuvisAIServiceStub(channel)

    print("Creating session...")
    create_resp = stub.CreateSession(
        cuvis_ai_pb2.CreateSessionRequest(
            pipeline_type="statistical",
            data_config=cuvis_ai_pb2.DataConfig(
                cu3s_file_path="data/Lentils/Lentils_000.cu3s",
                annotation_json_path="data/Lentils/Lentils_000.json",
                batch_size=4,
            ),
        )
    )
    session_id = create_resp.session_id
    print(f"Session ID: {session_id}")

    print("\nPipeline inputs:")
    inputs_resp = stub.GetPipelineInputs(
        cuvis_ai_pb2.GetPipelineInputsRequest(session_id=session_id)
    )
    for name in inputs_resp.input_names:
        spec = inputs_resp.input_specs[name]
        dtype_name = cuvis_ai_pb2.DType.Name(spec.dtype)
        required = "required" if spec.required else "optional"
        print(f"  - {name}: shape={list(spec.shape)}, dtype={dtype_name}, {required}")

    print("\nPipeline outputs:")
    outputs_resp = stub.GetPipelineOutputs(
        cuvis_ai_pb2.GetPipelineOutputsRequest(session_id=session_id)
    )
    for name in outputs_resp.output_names:
        spec = outputs_resp.output_specs[name]
        dtype_name = cuvis_ai_pb2.DType.Name(spec.dtype)
        print(f"  - {name}: shape={list(spec.shape)}, dtype={dtype_name}")

    print("\nFetching visualization (PNG)...")
    viz_resp = stub.GetPipelineVisualization(
        cuvis_ai_pb2.GetPipelineVisualizationRequest(session_id=session_id, format="png")
    )
    output_path = Path("pipeline_visualization.png")
    output_path.write_bytes(viz_resp.image_data)
    print(f"Saved visualization to {output_path} ({len(viz_resp.image_data)} bytes)")

    stub.CloseSession(cuvis_ai_pb2.CloseSessionRequest(session_id=session_id))
    channel.close()


if __name__ == "__main__":
    main()
