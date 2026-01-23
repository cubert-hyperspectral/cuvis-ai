"""Resume training by saving then restoring a composed trainrun.

This example demonstrates two approaches for resuming training:

1. Basic approach: Save/Restore trainrun config only (requires statistical initialization)
2. Enhanced approach: Save/Restore trainrun with pipeline weights (no statistical init needed)
"""

from __future__ import annotations

from pathlib import Path

from cuvis_ai_core.grpc import cuvis_ai_pb2
from workflow_utils import (
    apply_trainrun_config,
    build_stub,
    config_search_paths,
    create_session_with_search_paths,
    format_progress,
    resolve_trainrun_config,
)


def main() -> None:
    stub = build_stub()

    # Stage 1: Resolve + save a trainrun snapshot
    session_id = create_session_with_search_paths(stub, config_search_paths())
    resolved, _ = resolve_trainrun_config(
        stub,
        session_id,
        "deep_svdd",
        overrides=["training.trainer.max_epochs=1"],
    )
    apply_trainrun_config(stub, session_id, resolved.config_bytes)

    snapshot_path = str(Path("outputs/deep_svdd_snapshot.yaml").resolve())

    # Simplified SaveTrainRun: Only saves trainrun config, optionally saves weights
    save_response = stub.SaveTrainRun(
        cuvis_ai_pb2.SaveTrainRunRequest(
            session_id=session_id,
            trainrun_path=snapshot_path,
            save_weights=True,  # Explicitly request to save weights
        )
    )
    print(f"Saved trainrun snapshot to {save_response.trainrun_path}")

    if save_response.weights_path:
        print(f"Saved weights to {save_response.weights_path}")
    else:
        print("No weights saved (save_weights was False or no pipeline)")

    stub.CloseSession(cuvis_ai_pb2.CloseSessionRequest(session_id=session_id))

    # Stage 2: Restore and resume training using enhanced approach
    # Enhanced RestoreTrainRun: Now can load weights automatically
    print("\n=== Enhanced Approach: Restoring with weights ===")

    # Use the weights that were saved with the trainrun
    if not save_response.weights_path:
        expected_weights_path = str(Path(snapshot_path).with_suffix(".pt"))
        raise RuntimeError(
            f"No weights were saved with the trainrun. Expected weights at {expected_weights_path}"
        )

    weights_path = save_response.weights_path
    print(f"Restoring trainrun with weights from {weights_path}")

    restored = stub.RestoreTrainRun(
        cuvis_ai_pb2.RestoreTrainRunRequest(
            trainrun_path=snapshot_path, weights_path=weights_path, strict=True
        )
    )
    session_id = restored.session_id
    print(f"Restored session: {session_id} from {snapshot_path} with weights")

    # With weights loaded, statistical nodes are already initialized
    # No need for statistical initialization step!

    # Optional: adjust search paths (keeps configs root first)
    stub.SetSessionSearchPaths(
        cuvis_ai_pb2.SetSessionSearchPathsRequest(
            session_id=session_id,
            search_paths=config_search_paths(),
            append=False,
        )
    )

    # Directly start gradient training - no statistical init needed!
    print("Starting gradient training (statistical nodes already initialized from weights)...")
    for progress in stub.Train(
        cuvis_ai_pb2.TrainRequest(
            session_id=session_id,
            trainer_type=cuvis_ai_pb2.TRAINER_TYPE_GRADIENT,
        )
    ):
        print(format_progress(progress))

    # Save updated pipeline + trainrun
    pipeline_path = str(Path("outputs/resumed_pipeline.yaml").resolve())
    trainrun_path = str(Path("outputs/resumed_trainrun.yaml").resolve())

    stub.SavePipeline(
        cuvis_ai_pb2.SavePipelineRequest(
            session_id=session_id,
            pipeline_path=pipeline_path,
            metadata=cuvis_ai_pb2.PipelineMetadata(
                name="Resumed Training Model",
                description="Continued training from snapshot",
            ),
        )
    )
    stub.SaveTrainRun(
        cuvis_ai_pb2.SaveTrainRunRequest(session_id=session_id, trainrun_path=trainrun_path)
    )
    print(f"Pipeline saved to {pipeline_path}")
    print(f"Trainrun saved to {trainrun_path}")

    stub.CloseSession(cuvis_ai_pb2.CloseSessionRequest(session_id=session_id))
    print("Session closed.")


if __name__ == "__main__":
    main()
