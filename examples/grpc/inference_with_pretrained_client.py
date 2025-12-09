"""Inference with Pre-trained Model Client Example - Workflow 2."""

import grpc
import numpy as np

from cuvis_ai.grpc import cuvis_ai_pb2, cuvis_ai_pb2_grpc, helpers


def main() -> None:
    channel = grpc.insecure_channel("localhost:50051")
    stub = cuvis_ai_pb2_grpc.CuvisAIServiceStub(channel)

    response = stub.CreateSession(
        cuvis_ai_pb2.CreateSessionRequest(
            pipeline=cuvis_ai_pb2.PipelineConfig(config_bytes=b"rx_statistical")
        )
    )
    session_id = response.session_id
    print(f"Session: {session_id}")

    batch_size = 1
    height, width = 64, 64
    channels = 61

    cube = np.random.randn(batch_size, height, width, channels).astype(np.uint16)
    print(f"Input shape: {cube.shape}")

    try:
        inference_response = stub.Inference(
            cuvis_ai_pb2.InferenceRequest(
                session_id=session_id,
                inputs=cuvis_ai_pb2.InputBatch(cube=helpers.numpy_to_proto(cube)),
            )
        )

        print(f"Outputs: {len(inference_response.outputs)}")

        for output_name, output_tensor in inference_response.outputs.items():
            output_array = helpers.proto_to_numpy(output_tensor)
            print(
                f"  {output_name}: shape={output_array.shape}, "
                f"dtype={output_array.dtype}, "
                f"min={output_array.min():.4f}, max={output_array.max():.4f}"
            )

        if inference_response.metrics:
            print("Metrics:")
            for metric_name, metric_value in inference_response.metrics.items():
                print(f"  {metric_name}: {metric_value:.4f}")

    except grpc.RpcError as e:
        print(f"Inference failed: {e.details()}")
        return

    num_samples = 5
    for i in range(num_samples):
        cube_i = np.random.randn(1, height, width, channels).astype(np.uint16)
        stub.Inference(
            cuvis_ai_pb2.InferenceRequest(
                session_id=session_id,
                inputs=cuvis_ai_pb2.InputBatch(cube=helpers.numpy_to_proto(cube_i)),
            )
        )
        print(f"Batch {i + 1}/{num_samples} processed")

    stub.CloseSession(cuvis_ai_pb2.CloseSessionRequest(session_id=session_id))


if __name__ == "__main__":
    main()
