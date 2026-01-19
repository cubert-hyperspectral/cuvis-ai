"""Restore and reproduce training runs from saved trainrun configurations using gRPC.

gRPC equivalent of the serialization restore_trainrun.py script.
This script demonstrates how to restore complete training runs (pipeline + data + training settings)
and reproduce training, validation, or testing using the gRPC API.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from loguru import logger
from workflow_utils import (
    build_stub,
    config_search_paths,
    create_session_with_search_paths,
)

from cuvis_ai_core.grpc import cuvis_ai_pb2


def restore_trainrun_grpc(
    trainrun_path: str | Path,
    mode: Literal["info", "train", "validate", "test"] = "info",
    weights_path: str | Path | None = None,
    server_address: str = "localhost:50051",
    device: str = "auto",
    overrides: list[str] | None = None,
) -> None:
    """Restore and reproduce training run from configuration file using gRPC.

    Parameters
    ----------
    trainrun_path : str | Path
        Path to trainrun YAML file
    mode : str
        Execution mode:
        - 'info': Display experiment information only
        - 'train': Re-run training from scratch
        - 'validate': Run validation on trained model
        - 'test': Run test evaluation on trained model
    weights_path : str | Path | None
        Optional path to weights file (.pt). If None, tries to find associated weights
    server_address : str
        gRPC server address (default: localhost:50051)
    device : str
        Device to run on ('cpu', 'cuda', 'auto')
    overrides : list[str] | None
        Hydra-style config overrides (e.g., ["output_dir=outputs/custom", "data.batch_size=16"])
    """
    trainrun_path = Path(trainrun_path)
    if not trainrun_path.exists():
        raise FileNotFoundError(f"TrainRun file not found: {trainrun_path}")

    logger.info(f"Connecting to gRPC server at {server_address}")
    stub = build_stub(server_address)

    logger.info("Creating session and setting search paths")
    session_id = create_session_with_search_paths(stub, config_search_paths())

    try:
        logger.info(f"Loading trainrun from: {trainrun_path}")

        # Restore the trainrun configuration
        restore_response = stub.RestoreTrainRun(
            cuvis_ai_pb2.RestoreTrainRunRequest(
                trainrun_path=str(trainrun_path),
                weights_path=str(weights_path) if weights_path else None,
                strict=True,
            )
        )

        session_id = restore_response.session_id
        logger.info(f"Trainrun restored. Session ID: {session_id}")

        if mode == "info":
            logger.info("Info mode - displaying pipeline specifications")

            # Get pipeline input/output specs
            inputs_response = stub.GetPipelineInputs(
                cuvis_ai_pb2.GetPipelineInputsRequest(session_id=session_id)
            )
            outputs_response = stub.GetPipelineOutputs(
                cuvis_ai_pb2.GetPipelineOutputsRequest(session_id=session_id)
            )

            print("\nInput Specs:")
            for name, spec in inputs_response.input_specs.items():
                shape_str = "x".join(str(dim) for dim in spec.shape)
                print(
                    f"  {name}: shape=[{shape_str}], dtype={cuvis_ai_pb2.DType.Name(spec.dtype)}, required={spec.required}"
                )

            print("\nOutput Specs:")
            for name, spec in outputs_response.output_specs.items():
                shape_str = "x".join(str(dim) for dim in spec.shape)
                print(
                    f"  {name}: shape=[{shape_str}], dtype={cuvis_ai_pb2.DType.Name(spec.dtype)}, required={spec.required}"
                )

            return

        # For training modes, we need to execute the training workflow
        if mode in ["train", "validate", "test"]:
            logger.info(f"Executing {mode} mode")

            # Determine trainer type based on the restored configuration
            # This is a simplified approach - in a real scenario, you might need to inspect
            # the trainrun configuration to determine the appropriate trainer type
            trainer_type = cuvis_ai_pb2.TRAINER_TYPE_GRADIENT

            # Execute training
            train_stream = stub.Train(
                cuvis_ai_pb2.TrainRequest(
                    session_id=session_id,
                    trainer_type=trainer_type,
                )
            )

            # Process training progress
            for progress in train_stream:
                stage = cuvis_ai_pb2.ExecutionStage.Name(progress.context.stage)
                status = cuvis_ai_pb2.TrainStatus.Name(progress.status)

                parts = [f"[{stage}] {status}"]
                if progress.losses:
                    parts.append(f"losses={dict(progress.losses)}")
                if progress.metrics:
                    parts.append(f"metrics={dict(progress.metrics)}")
                if progress.message:
                    parts.append(progress.message)

                logger.info(" | ".join(parts))

                # For validation/test modes, we can break after getting results
                if mode == "validate" and stage == "EXECUTION_STAGE_VAL":
                    break
                if mode == "test" and stage == "EXECUTION_STAGE_TEST":
                    break

            logger.info(f"{mode.capitalize()} mode complete")

    finally:
        stub.CloseSession(cuvis_ai_pb2.CloseSessionRequest(session_id=session_id))
        logger.info("Session closed.")


def main() -> None:
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Restore and reproduce training runs from saved configurations using gRPC",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  # Display trainrun info
  python restore_trainrun_grpc.py --trainrun-path outputs/channel_selector/trained_models/channel_selector_trainrun.yaml

  # Re-run training
  python restore_trainrun_grpc.py --trainrun-path outputs/.../trainrun.yaml --mode train

  # Re-run training with custom weights
  python restore_trainrun_grpc.py --trainrun-path outputs/.../trainrun.yaml --mode train --weights-path outputs/my_weights.pt

  # Override data and training configs
  python restore_trainrun_grpc.py --trainrun-path outputs/.../trainrun.yaml --mode train --override data.batch_size=16 --override training.optimizer.lr=0.001

  # Run validation only
  python restore_trainrun_grpc.py --trainrun-path outputs/.../trainrun.yaml --mode validate

  # Use custom gRPC server
  python restore_trainrun_grpc.py --trainrun-path outputs/.../trainrun.yaml --mode info --server localhost:50052
        """,
    )

    parser.add_argument(
        "--trainrun-path",
        type=str,
        required=True,
        help="Path to trainrun YAML file",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["info", "train", "validate", "test"],
        default="info",
        help="Execution mode (default: info)",
    )
    parser.add_argument(
        "--weights-path",
        type=str,
        default=None,
        help="Path to weights (.pt) file",
    )
    parser.add_argument(
        "--server",
        type=str,
        default="localhost:50051",
        help="gRPC server address (default: localhost:50051)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device to run on (default: auto)",
    )
    parser.add_argument(
        "--override",
        action="append",
        help="Override config values in dot notation (e.g., data.batch_size=16). Can be specified multiple times.",
    )

    args = parser.parse_args()

    restore_trainrun_grpc(
        trainrun_path=args.trainrun_path,
        mode=args.mode,
        weights_path=args.weights_path,
        server_address=args.server,
        device=args.device,
        overrides=args.override,
    )


if __name__ == "__main__":
    main()
