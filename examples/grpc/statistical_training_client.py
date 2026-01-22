"""Statistical training example using explicit ResolveConfig workflow."""

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

    # Resolve and apply the trainrun config (Hydra composition + overrides)
    resolved, config_dict = resolve_trainrun_config(
        stub,
        session_id,
        "rx_statistical",
        overrides=[
            "data.batch_size=4",
            "training.trainer.max_epochs=3",
            "training.seed=123",
        ],
    )
    print(
        f"Resolved trainrun '{config_dict['name']}' "
        f"with pipeline '{config_dict['pipeline']['metadata']['name']}'"
    )
    apply_trainrun_config(stub, session_id, resolved.config_bytes)

    # Statistical training (server streams progress)
    print("Starting statistical training...")
    for progress in stub.Train(
        cuvis_ai_pb2.TrainRequest(
            session_id=session_id,
            trainer_type=cuvis_ai_pb2.TRAINER_TYPE_STATISTICAL,
        )
    ):
        print(format_progress(progress))

    status = stub.GetTrainStatus(cuvis_ai_pb2.GetTrainStatusRequest(session_id=session_id))
    print(
        f"Training finished with status: "
        f"{cuvis_ai_pb2.TrainStatus.Name(status.latest_progress.status)}"
    )

    # Quick inference to verify pipeline is usable
    cube = np.random.rand(1, 32, 32, 61).astype(np.uint16)
    wavelengths = np.linspace(430, 910, 61).reshape(1, -1).astype(np.int32)
    inference = stub.Inference(
        cuvis_ai_pb2.InferenceRequest(
            session_id=session_id,
            inputs=cuvis_ai_pb2.InputBatch(
                cube=helpers.numpy_to_proto(cube),
                wavelengths=helpers.numpy_to_proto(wavelengths),
            ),
        )
    )
    print(f"Inference outputs: {list(inference.outputs.keys())}")

    stub.CloseSession(cuvis_ai_pb2.CloseSessionRequest(session_id=session_id))
    print("Session closed.")


if __name__ == "__main__":
    main()
