"""Example client showing bounding boxes, points, and text prompts."""

import grpc
import numpy as np

from cuvis_ai.grpc import cuvis_ai_pb2, cuvis_ai_pb2_grpc


def main() -> None:
    channel = grpc.insecure_channel("localhost:50051")
    stub = cuvis_ai_pb2_grpc.CuvisAIServiceStub(channel)

    session = stub.CreateSession(
        cuvis_ai_pb2.CreateSessionRequest(
            pipeline_type="gradient",
            data_config=cuvis_ai_pb2.DataConfig(
                cu3s_file_path="/path/to/data.cu3s",
                batch_size=2,
            ),
        )
    )
    session_id = session.session_id
    cube = np.random.rand(1, 32, 32, 61).astype(np.float32)

    print("1) Inference with bounding boxes")
    bbox_resp = stub.Inference(
        cuvis_ai_pb2.InferenceRequest(
            session_id=session_id,
            inputs=cuvis_ai_pb2.InputBatch(
                cube=cuvis_ai_pb2.Tensor(
                    shape=list(cube.shape),
                    dtype=cuvis_ai_pb2.D_TYPE_FLOAT32,
                    raw_data=cube.tobytes(),
                ),
                bboxes=cuvis_ai_pb2.BoundingBoxes(
                    boxes=[
                        cuvis_ai_pb2.BoundingBox(
                            element_id=0,
                            x_min=5,
                            y_min=5,
                            x_max=15,
                            y_max=15,
                        )
                    ]
                ),
            ),
        )
    )
    print(f"   outputs: {list(bbox_resp.outputs.keys())}")

    print("2) Inference with points")
    points_resp = stub.Inference(
        cuvis_ai_pb2.InferenceRequest(
            session_id=session_id,
            inputs=cuvis_ai_pb2.InputBatch(
                cube=cuvis_ai_pb2.Tensor(
                    shape=list(cube.shape),
                    dtype=cuvis_ai_pb2.D_TYPE_FLOAT32,
                    raw_data=cube.tobytes(),
                ),
                points=cuvis_ai_pb2.Points(
                    points=[
                        cuvis_ai_pb2.Point(
                            element_id=0,
                            x=10.5,
                            y=15.5,
                            type=cuvis_ai_pb2.POINT_TYPE_POSITIVE,
                        ),
                        cuvis_ai_pb2.Point(
                            element_id=0,
                            x=20.5,
                            y=25.5,
                            type=cuvis_ai_pb2.POINT_TYPE_NEGATIVE,
                        ),
                    ]
                ),
            ),
        )
    )
    print(f"   outputs: {list(points_resp.outputs.keys())}")

    print("3) Inference with text prompt")
    text_resp = stub.Inference(
        cuvis_ai_pb2.InferenceRequest(
            session_id=session_id,
            inputs=cuvis_ai_pb2.InputBatch(
                cube=cuvis_ai_pb2.Tensor(
                    shape=list(cube.shape),
                    dtype=cuvis_ai_pb2.D_TYPE_FLOAT32,
                    raw_data=cube.tobytes(),
                ),
                text_prompt="Find anomalies",
            ),
        )
    )
    print(f"   outputs: {list(text_resp.outputs.keys())}")

    print("4) Inference with combined inputs")
    combined_resp = stub.Inference(
        cuvis_ai_pb2.InferenceRequest(
            session_id=session_id,
            inputs=cuvis_ai_pb2.InputBatch(
                cube=cuvis_ai_pb2.Tensor(
                    shape=list(cube.shape),
                    dtype=cuvis_ai_pb2.D_TYPE_FLOAT32,
                    raw_data=cube.tobytes(),
                ),
                bboxes=cuvis_ai_pb2.BoundingBoxes(
                    boxes=[
                        cuvis_ai_pb2.BoundingBox(
                            element_id=0,
                            x_min=5,
                            y_min=5,
                            x_max=15,
                            y_max=15,
                        )
                    ]
                ),
                points=cuvis_ai_pb2.Points(
                    points=[
                        cuvis_ai_pb2.Point(
                            element_id=0,
                            x=12.5,
                            y=18.5,
                            type=cuvis_ai_pb2.POINT_TYPE_POSITIVE,
                        )
                    ]
                ),
                text_prompt="Segment defective regions",
            ),
        )
    )
    print(f"   outputs: {list(combined_resp.outputs.keys())}")

    stub.CloseSession(cuvis_ai_pb2.CloseSessionRequest(session_id=session_id))


if __name__ == "__main__":
    main()
