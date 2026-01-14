"""AdaCLIP statistical training example using gRPC.

This client demonstrates running AdaCLIP anomaly detection via gRPC:
  - Resolves the adaclip_baseline trainrun config
  - Applies the config to build the pipeline (with AdaCLIPDetector)
  - Runs statistical training (no learnable parameters, just evaluation)
  - Performs inference with sample data
"""

from __future__ import annotations

import numpy as np
from workflow_utils import (
    apply_trainrun_config,
    build_stub,
    config_search_paths,
    create_session_with_search_paths,
    format_progress,
    resolve_trainrun_config,
)

from cuvis_ai.grpc import cuvis_ai_pb2, helpers


def main(server_address: str = "localhost:50051") -> None:
    stub = build_stub(server_address)
    session_id = create_session_with_search_paths(stub, config_search_paths())
    print(f"Session created: {session_id}")

    # Resolve the adaclip_baseline trainrun config
    # This loads the pipeline from configs/pipeline/adaclip_baseline.yaml
    resolved, config_dict = resolve_trainrun_config(
        stub,
        session_id,
        "adaclip_baseline",
        overrides=[
            # Override decider quantile if needed
            # "decider.quantile=0.99",
        ],
    )
    print(
        f"Resolved trainrun '{config_dict['name']}' "
        f"with pipeline '{config_dict['pipeline'].get('metadata', {}).get('name', 'unknown')}'"
    )

    # Apply the trainrun config (builds the pipeline on the server)
    apply_trainrun_config(stub, session_id, resolved.config_bytes)
    print("Applied AdaCLIP trainrun config")

    # Statistical training - for AdaCLIP this is essentially just evaluation
    # since AdaCLIP has no learnable statistical parameters
    print("Starting statistical training (evaluation pass)...")
    for progress in stub.Train(
        cuvis_ai_pb2.TrainRequest(
            session_id=session_id,
            trainer_type=cuvis_ai_pb2.TRAINER_TYPE_STATISTICAL,
        )
    ):
        print(format_progress(progress))

    # Check training status
    status = stub.GetTrainStatus(cuvis_ai_pb2.GetTrainStatusRequest(session_id=session_id))
    print(
        f"Training finished with status: "
        f"{cuvis_ai_pb2.TrainStatus.Name(status.latest_progress.status)}"
    )

    # Quick inference to verify pipeline works and check GPU usage
    # AdaCLIP expects RGB images, but the pipeline handles conversion from hyperspectral
    # Input: hyperspectral cube (B, H, W, C) where C is number of wavelength channels
    cube = (np.random.rand(1, 64, 64, 61) * 65535).astype(np.uint16)  # Random hyperspectral cube
    wavelengths = np.linspace(430, 910, 61).reshape(1, -1).astype(np.int32)

    print("Running inference (first call may be slower due to CUDA warmup)...")
    import time

    start_time = time.time()
    inference = stub.Inference(
        cuvis_ai_pb2.InferenceRequest(
            session_id=session_id,
            inputs=cuvis_ai_pb2.InputBatch(
                cube=helpers.numpy_to_proto(cube),
                wavelengths=helpers.numpy_to_proto(wavelengths),
            ),
        )
    )
    elapsed = time.time() - start_time
    print(f"Inference completed in {elapsed:.3f}s")
    print(f"Inference outputs: {list(inference.outputs.keys())}")

    # Run a second inference to check if it's faster (GPU warmup)
    print("Running second inference (should be faster after GPU warmup)...")
    start_time = time.time()
    stub.Inference(
        cuvis_ai_pb2.InferenceRequest(
            session_id=session_id,
            inputs=cuvis_ai_pb2.InputBatch(
                cube=helpers.numpy_to_proto(cube),
                wavelengths=helpers.numpy_to_proto(wavelengths),
            ),
        )
    )
    elapsed2 = time.time() - start_time
    print(f"Second inference completed in {elapsed2:.3f}s")
    if elapsed2 < elapsed:
        print(f"GPU warmup confirmed: {elapsed / elapsed2:.2f}x speedup")

    # Optionally save the pipeline
    # save_response = stub.SavePipeline(
    #     cuvis_ai_pb2.SavePipelineRequest(
    #         session_id=session_id,
    #         pipeline_path="outputs/adaclip_grpc.yaml",
    #         metadata=cuvis_ai_pb2.PipelineMetadata(
    #             name="AdaCLIP_gRPC",
    #             description="AdaCLIP pipeline trained via gRPC",
    #         ),
    #     )
    # )
    # print(f"Pipeline saved: {save_response.pipeline_path}")

    stub.CloseSession(cuvis_ai_pb2.CloseSessionRequest(session_id=session_id))
    print("Session closed.")


if __name__ == "__main__":
    main()
