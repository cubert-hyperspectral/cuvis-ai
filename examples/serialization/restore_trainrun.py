"""Restore and reproduce training runs from saved trainrun configurations.

CLI equivalent of the gRPC RestoreTrainRun functionality, for users who want to work
directly with the Python API without running a gRPC server.

This script loads a complete trainrun configuration (pipeline + data + training settings)
and can reproduce training, run validation, or execute tests.
"""

from pathlib import Path
from typing import Literal

from loguru import logger

from cuvis_ai.data.lentils_anomaly import SingleCu3sDataModule

# Build pipeline from inline config structure
from cuvis_ai.pipeline.factory import PipelineBuilder
from cuvis_ai.training import GradientTrainer, StatisticalTrainer
from cuvis_ai.training.config import TrainRunConfig


def restore_trainrun(
    trainrun_path: str | Path,
    mode: Literal["train", "validate", "test", "info"] = "info",
    checkpoint_path: str | Path | None = None,
    device: str = "auto",
    overrides: list[str] | None = None,
) -> None:
    """Restore and reproduce training run from configuration file.

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
    checkpoint_path : str | Path | None
        Optional checkpoint path to resume training from
    device : str
        Device to run on ('cpu', 'cuda', 'auto')
    overrides : list[str] | None
        Hydra-style config overrides (e.g., ["output_dir=outputs/custom", "data.batch_size=16"])
    """
    trainrun_path = Path(trainrun_path)
    if not trainrun_path.exists():
        raise FileNotFoundError(f"TrainRun file not found: {trainrun_path}")

    logger.info(f"Loading trainrun from: {trainrun_path}")
    trainrun_config: TrainRunConfig = TrainRunConfig.load_from_file(trainrun_path)

    builder = PipelineBuilder()
    if trainrun_config.pipeline is None:
        raise ValueError("Pipeline configuration is missing in trainrun config.")
    pipeline_dict = trainrun_config.pipeline.to_dict()
    pipeline = builder.build_from_config(pipeline_dict)

    # Move pipeline to specified device if needed
    if device != "auto":
        pipeline = pipeline.to(device)

    if mode == "info":
        logger.info("Info mode complete")
        logger.info(pipeline.get_input_specs())
        logger.info(pipeline.get_output_specs())

        return

    # Create datamodule
    datamodule = SingleCu3sDataModule(
        cu3s_file_path=trainrun_config.data.cu3s_file_path,
        annotation_json_path=trainrun_config.data.annotation_json_path,
        train_ids=trainrun_config.data.train_ids,
        val_ids=trainrun_config.data.val_ids,
        test_ids=trainrun_config.data.test_ids,
        batch_size=trainrun_config.data.batch_size,
        processing_mode=trainrun_config.data.processing_mode,
    )
    datamodule.setup(stage="fit")

    # Use output directory from trainrun config (possibly overridden)
    output_dir = Path(trainrun_config.output_dir)

    # Find loss and metric nodes by name
    loss_nodes = []
    metric_nodes = []
    for node in pipeline.nodes():
        if node.name in trainrun_config.loss_nodes:
            loss_nodes.append(node)
        if node.name in trainrun_config.metric_nodes:
            metric_nodes.append(node)

    # Update checkpoint directory to output_dir
    training_config = trainrun_config.training
    if training_config is None:
        raise ValueError("Training configuration is missing in trainrun config.")

    if training_config.trainer.callbacks and training_config.trainer.callbacks.checkpoint:
        training_config.trainer.callbacks.checkpoint.dirpath = str(output_dir / "checkpoints")
    grad_trainer = GradientTrainer(
        pipeline=pipeline,
        datamodule=datamodule,
        loss_nodes=loss_nodes,
        metric_nodes=metric_nodes,
        trainer_config=training_config.trainer,
        optimizer_config=training_config.optimizer,
    )

    if mode == "train":
        logger.info("Training mode")

        # Record nodes that were fitted with statistical training
        requires_static_fit = any(node.requires_initial_fit for node in pipeline.nodes())

        # Run statistical initialization if static_fit_nodes are specified
        if requires_static_fit:
            stat_trainer = StatisticalTrainer(pipeline=pipeline, datamodule=datamodule)
            stat_trainer.fit()

        # Unfreeze nodes for gradient training
        if trainrun_config.unfreeze_nodes:
            pipeline.unfreeze_nodes_by_name(trainrun_config.unfreeze_nodes)

        # Note: PyTorch Lightning handles checkpoint resumption via trainer_config
        # The checkpoint_path parameter is preserved for future implementation
        if checkpoint_path:
            logger.warning(
                f"Checkpoint path provided: {checkpoint_path}. "
                "Automatic checkpoint resumption is not yet implemented. "
                "Please configure checkpoint resumption in trainer_config."
            )

        grad_trainer.fit()

        # Save trained pipeline
        restored_pipeline_path = output_dir / "trained_models" / f"{pipeline.name}_restored.yaml"
        restored_pipeline_path.parent.mkdir(parents=True, exist_ok=True)
        pipeline.save_to_file(str(restored_pipeline_path))

    elif mode == "validate":
        logger.info("Validation mode")

        _ = grad_trainer.validate()

    elif mode == "test":
        logger.info("Test mode")

        _ = grad_trainer.test()

    logger.info("Complete")


def main() -> None:
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Restore and reproduce training runs from saved configurations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Display trainrun info
  python restore_trainrun.py --trainrun-path outputs/channel_selector/trained_models/channel_selector_trainrun.yaml

  # Re-run training
  python restore_trainrun.py --trainrun-path outputs/.../trainrun.yaml --mode train

  # Re-run training with custom output directory
  python restore_trainrun.py --trainrun-path outputs/.../trainrun.yaml --mode train --output-dir outputs/custom

  # Override data and training configs
  python restore_trainrun.py --trainrun-path outputs/.../trainrun.yaml --mode train --override data.batch_size=16 --override training.optimizer.lr=0.001

  # Run validation only
  python restore_trainrun.py --trainrun-path outputs/.../trainrun.yaml --mode validate

  # Resume training from checkpoint
  python restore_trainrun.py --trainrun-path outputs/.../trainrun.yaml --mode train --checkpoint checkpoints/epoch=05.ckpt
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
        "--checkpoint-path",
        type=str,
        default=None,
        help="Checkpoint path to resume training from",
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

    restore_trainrun(
        trainrun_path=args.trainrun_path,
        mode=args.mode,
        checkpoint_path=args.checkpoint_path,
        device=args.device,
        overrides=args.override,
    )


if __name__ == "__main__":
    main()
