"""Restore and reproduce statistical training runs from saved trainrun configurations using gRPC.

This script is specifically for statistical training runs (those with empty loss_nodes).
For gradient training runs, use restore_trainrun_grpc.py instead.

gRPC equivalent of the serialization restore_trainrun.py script for statistical training.
This script demonstrates how to restore complete training runs (pipeline + data + training settings)
and reproduce statistical training, validation, or testing using the gRPC API.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import click
from cuvis_ai_schemas.grpc.v1 import cuvis_ai_pb2
from cuvis_ai_schemas.training import TrainRunConfig
from loguru import logger
from workflow_utils import (
    build_stub,
    config_search_paths,
    create_session_with_search_paths,
)


def restore_trainrun_statistical_grpc(
    trainrun_path: str | Path,
    mode: Literal["info", "train", "validate", "test"] = "info",
    weights_path: str | Path | None = None,
    server_address: str = "localhost:50051",
    device: str = "auto",
    overrides: list[str] | None = None,
) -> None:
    """Restore and reproduce statistical training run from configuration file using gRPC.

    Parameters
    ----------
    trainrun_path : str | Path
        Path to trainrun YAML file (should have empty loss_nodes for statistical training)
    mode : str
        Execution mode:
        - 'info': Display experiment information only
        - 'train': Re-run statistical training from scratch
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

        # Verify this is a statistical training run
        trainrun_config = TrainRunConfig.from_proto(restore_response.trainrun)
        loss_nodes = trainrun_config.loss_nodes
        if loss_nodes:
            logger.warning(
                f"Warning: This trainrun has {len(loss_nodes)} loss node(s). "
                "Consider using restore_trainrun_grpc.py for gradient training instead."
            )

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

        # For training modes, we need to execute the statistical training workflow
        if mode in ["train", "validate", "test"]:
            logger.info(f"Executing {mode} mode with statistical training")

            # Use statistical trainer type
            trainer_type = cuvis_ai_pb2.TRAINER_TYPE_STATISTICAL

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


@click.command()
@click.option(
    "--trainrun-path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="Path to trainrun YAML file (should have empty loss_nodes for statistical training).",
)
@click.option(
    "--mode",
    type=click.Choice(["info", "train", "validate", "test"], case_sensitive=False),
    default="info",
    show_default=True,
    help="Execution mode.",
)
@click.option(
    "--weights-path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Path to weights (.pt) file.",
)
@click.option(
    "--server",
    "server_address",
    type=str,
    default="localhost:50051",
    show_default=True,
    help="gRPC server address.",
)
@click.option(
    "--device",
    type=click.Choice(["auto", "cpu", "cuda"], case_sensitive=False),
    default="auto",
    show_default=True,
    help="Device to run on.",
)
@click.option(
    "--override",
    "overrides",
    multiple=True,
    help="Override config values in dot notation (e.g., data.batch_size=16). Repeatable.",
)
def cli(
    trainrun_path: Path,
    mode: str,
    weights_path: Path | None,
    server_address: str,
    device: str,
    overrides: tuple[str, ...],
) -> None:
    restore_trainrun_statistical_grpc(
        trainrun_path=trainrun_path,
        mode=mode,
        weights_path=weights_path,
        server_address=server_address,
        device=device,
        overrides=list(overrides) if overrides else None,
    )


if __name__ == "__main__":
    cli()
