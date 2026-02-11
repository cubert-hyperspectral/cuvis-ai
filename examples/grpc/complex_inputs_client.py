"""Example client showing bounding boxes, points, and text prompts."""

from __future__ import annotations

import numpy as np
from cuvis_ai_core.grpc import helpers
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

    # Build pipeline + load weights via resolved config
    pipeline_config = stub.ResolveConfig(
        cuvis_ai_pb2.ResolveConfigRequest(
            session_id=session_id,
            config_type="pipeline",
            path="pipeline/channel_selector",
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
            weights_path=str((CONFIG_ROOT / "pipeline" / "channel_selector.pt").resolve()),
        )
    )

    cube = np.random.rand(1, 32, 32, 61).astype(np.uint16)
    wavelengths = np.linspace(430, 910, 61).reshape(1, -1).astype(np.int32)

    print("1) Inference with bounding boxes")
    bbox_resp = stub.Inference(
        cuvis_ai_pb2.InferenceRequest(
            session_id=session_id,
            inputs=cuvis_ai_pb2.InputBatch(
                cube=helpers.numpy_to_proto(cube),
                wavelengths=helpers.numpy_to_proto(wavelengths),
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
                cube=helpers.numpy_to_proto(cube),
                wavelengths=helpers.numpy_to_proto(wavelengths),
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
                cube=helpers.numpy_to_proto(cube),
                wavelengths=helpers.numpy_to_proto(wavelengths),
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
                cube=helpers.numpy_to_proto(cube),
                wavelengths=helpers.numpy_to_proto(wavelengths),
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
