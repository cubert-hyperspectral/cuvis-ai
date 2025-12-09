"""Minimal client demonstrating CreateSession -> Inference -> CloseSession."""

from __future__ import annotations

import grpc
import numpy as np

from cuvis_ai.grpc import cuvis_ai_pb2, cuvis_ai_pb2_grpc, helpers


def main(server_address: str = "localhost:50051") -> None:
    channel = grpc.insecure_channel(server_address)
    stub = cuvis_ai_pb2_grpc.CuvisAIServiceStub(channel)

    session_id = stub.CreateSession(
        cuvis_ai_pb2.CreateSessionRequest(
            pipeline=cuvis_ai_pb2.PipelineConfig(config_bytes=b"channel_selector")
        )
    ).session_id
    print(f"Session: {session_id}")

    cube = np.random.randn(1, 16, 16, 61).astype(np.uint16)
    response = stub.Inference(
        cuvis_ai_pb2.InferenceRequest(
            session_id=session_id,
            inputs=cuvis_ai_pb2.InputBatch(cube=helpers.numpy_to_proto(cube)),
        )
    )
    print(f"Outputs: {list(response.outputs.keys())}")

    for name, tensor_proto in response.outputs.items():
        array = helpers.proto_to_numpy(tensor_proto)
        print(f"  {name}: shape={array.shape}, dtype={array.dtype}")

    stub.CloseSession(cuvis_ai_pb2.CloseSessionRequest(session_id=session_id))


if __name__ == "__main__":
    main()
