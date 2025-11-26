"""Minimal client demonstrating CreateSession -> Inference -> CloseSession."""

from __future__ import annotations

import json

import grpc
import numpy as np

from cuvis_ai.grpc import cuvis_ai_pb2, cuvis_ai_pb2_grpc, helpers


def main(server_address: str = "localhost:50051") -> None:
    channel = grpc.insecure_channel(server_address)
    stub = cuvis_ai_pb2_grpc.CuvisAIServiceStub(channel)

    create_req = cuvis_ai_pb2.CreateSessionRequest(
        pipeline_type="statistical",
        pipeline_config=json.dumps({"input_channels": 61, "n_select": 3}),
        data_config=cuvis_ai_pb2.DataConfig(
            cu3s_file_path="data/Lentils/Lentils_000.cu3s",
            annotation_json_path="data/Lentils/Lentils_000.json",
            batch_size=1,
            processing_mode=cuvis_ai_pb2.PROCESSING_MODE_REFLECTANCE,
        ),
    )
    session_id = stub.CreateSession(create_req).session_id
    print(f"Session created: {session_id}")

    cube = np.random.randn(1, 16, 16, 61).astype(np.float32)
    inference_req = cuvis_ai_pb2.InferenceRequest(
        session_id=session_id,
        inputs=cuvis_ai_pb2.InputBatch(cube=helpers.numpy_to_proto(cube)),
        output_specs=["selected", "indices", "decisions"],
    )
    response = stub.Inference(inference_req)
    print(f"Returned outputs: {list(response.outputs.keys())}")

    for name, tensor_proto in response.outputs.items():
        array = helpers.proto_to_numpy(tensor_proto)
        print(f" - {name}: shape={array.shape}, dtype={array.dtype}")

    stub.CloseSession(cuvis_ai_pb2.CloseSessionRequest(session_id=session_id))
    print("Session closed.")


if __name__ == "__main__":
    main()
