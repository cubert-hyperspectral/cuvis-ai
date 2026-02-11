"""Gradient training example using ResolveConfig + SetTrainRunConfig."""

from __future__ import annotations

import numpy as np
from cuvis_ai_core.grpc import helpers
from cuvis_ai_schemas.grpc.v1 import cuvis_ai_pb2
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

    # Resolve gradient trainrun config (Hydra) and apply to the session
    resolved, config_dict = resolve_trainrun_config(
        stub,
        session_id,
        "deep_svdd",
        overrides=[
            "training.trainer.max_epochs=2",
            "training.optimizer.lr=5e-4",
            "training.optimizer.weight_decay=0.005",
        ],
    )
    print(
        f"Resolved gradient trainrun '{config_dict['name']}' "
        f"with loss nodes: {config_dict.get('loss_nodes', [])}"
    )
    apply_trainrun_config(stub, session_id, resolved.config_bytes)

    # Optional: run a quick statistical pass to initialize components
    for progress in stub.Train(
        cuvis_ai_pb2.TrainRequest(
            session_id=session_id, trainer_type=cuvis_ai_pb2.TRAINER_TYPE_STATISTICAL
        )
    ):
        print(f"[statistical init] {format_progress(progress)}")

    # Gradient training with streamed progress
    print("Starting gradient training...")
    for progress in stub.Train(
        cuvis_ai_pb2.TrainRequest(
            session_id=session_id,
            trainer_type=cuvis_ai_pb2.TRAINER_TYPE_GRADIENT,
        )
    ):
        print(format_progress(progress))

    # Quick inference to confirm the pipeline runs
    cube = np.random.rand(1, 8, 8, 61).astype(np.uint16)
    wavelengths = np.linspace(430, 910, 61).reshape(1, -1).astype(np.int32)
    inference = stub.Inference(
        cuvis_ai_pb2.InferenceRequest(
            session_id=session_id,
            inputs=cuvis_ai_pb2.InputBatch(
                cube=helpers.numpy_to_proto(cube), wavelengths=helpers.numpy_to_proto(wavelengths)
            ),
        )
    )
    print(f"Inference outputs: {list(inference.outputs.keys())}")

    stub.CloseSession(cuvis_ai_pb2.CloseSessionRequest(session_id=session_id))
    print("Session closed.")


if __name__ == "__main__":
    main()
