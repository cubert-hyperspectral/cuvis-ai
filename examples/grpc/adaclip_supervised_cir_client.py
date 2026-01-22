"""AdaCLIP supervised CIR example using gRPC.

This client demonstrates running AdaCLIP with supervised CIR band selection via gRPC:
  - Resolves the adaclip_supervised_cir trainrun config
  - Applies the config to build the pipeline (with supervised CIR band selector)
  - Runs statistical training
  - Performs inference with sample data
"""

from __future__ import annotations

import numpy as np
from cuvis_ai_core.grpc import cuvis_ai_pb2, helpers
from workflow_utils import (
    apply_trainrun_config,
    build_stub,
    config_search_paths,
    create_session_with_search_paths,
    format_progress,
    resolve_trainrun_config,
)


def main(server_address: str = "localhost:50051") -> None:
    stub = build_stub(server_address)
    session_id = create_session_with_search_paths(stub, config_search_paths())
    print(f"Session created: {session_id}")

    # Resolve the adaclip_supervised_cir trainrun config
    resolved, config_dict = resolve_trainrun_config(
        stub,
        session_id,
        "adaclip_supervised_cir",
        overrides=[],
    )
    print(
        f"Resolved trainrun '{config_dict['name']}' "
        f"with pipeline '{config_dict['pipeline'].get('metadata', {}).get('name', 'unknown')}'"
    )

    # Apply the trainrun config (builds the pipeline on the server)
    apply_trainrun_config(stub, session_id, resolved.config_bytes)
    print("Applied AdaCLIP supervised CIR trainrun config")

    # Statistical training
    print("Starting statistical training...")
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

    # Quick inference to verify pipeline works
    cube = (np.random.rand(1, 64, 64, 61) * 65535).astype(np.uint16)
    wavelengths = np.linspace(430, 910, 61).reshape(1, -1).astype(np.int32)

    print("Running inference...")
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

    stub.CloseSession(cuvis_ai_pb2.CloseSessionRequest(session_id=session_id))
    print("Session closed.")


if __name__ == "__main__":
    main()
