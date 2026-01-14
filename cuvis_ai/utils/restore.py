"""Utilities for restoring and running pipelines and trainruns."""

from pathlib import Path
from typing import Literal

import torch
import yaml
from loguru import logger
from torch.utils.data import DataLoader

from cuvis_ai.data.datasets import SingleCu3sDataset
from cuvis_ai.data.lentils_anomaly import SingleCu3sDataModule
from cuvis_ai.pipeline.factory import PipelineBuilder
from cuvis_ai.pipeline.pipeline import CuvisPipeline
from cuvis_ai.training import GradientTrainer, StatisticalTrainer
from cuvis_ai.training.config import TrainRunConfig
from cuvis_ai.utils.config_helpers import resolve_config_with_hydra
from cuvis_ai.utils.types import Context, ExecutionStage


def restore_pipeline(
    pipeline_path: str | Path,
    weights_path: str | Path | None = None,
    device: str = "auto",
    cu3s_file_path: str | Path | None = None,
    processing_mode: str = "Reflectance",
    config_overrides: list[str] | None = None,
) -> CuvisPipeline:
    """Restore pipeline from configuration and weights for inference.

    Parameters
    ----------
    pipeline_path : str | Path
        Path to pipeline YAML configuration file
    weights_path : str | Path | None
        Optional path to weights file (.pt). If None, defaults to pipeline_path with .pt extension
    device : str
        Device to load weights to ('cpu', 'cuda', 'auto')
    cu3s_file_path : str | Path | None
        Optional path to .cu3s file for inference
    processing_mode : str
        Cuvis processing mode string ("Raw", "Reflectance")
    config_overrides : list[str] | None
        Optional list of config overrides in dot notation (e.g., ["nodes.10.params.output_dir=outputs/my_tb"])

    Returns
    -------
    CuvisPipeline
        Loaded pipeline ready for inference
    """
    pipeline_path = Path(pipeline_path)
    if weights_path is None:
        weights_path = pipeline_path.with_suffix(".pt")
    else:
        weights_path = Path(weights_path)

    logger.info(f"Loading pipeline from {pipeline_path}")

    load_device = device if device != "auto" else None
    pipeline = CuvisPipeline.load_pipeline(
        str(pipeline_path),
        weights_path=str(weights_path) if weights_path.exists() else None,
        device=load_device,
        config_overrides=config_overrides,
    )

    # If cu3s_file_path provided, setup data and run inference
    if cu3s_file_path:
        data = SingleCu3sDataset(
            cu3s_file_path=str(cu3s_file_path),
            processing_mode=processing_mode,
        )
        dataloader = DataLoader(data, shuffle=False, batch_size=1)

        for module in pipeline.torch_layers:
            module.eval()

        # Process all batches
        results = []
        global_step = 0  # Track step across batches
        with torch.no_grad():
            for batch in dataloader:
                # Create context with incrementing step
                context = Context(
                    stage=ExecutionStage.INFERENCE, batch_idx=global_step, global_step=global_step
                )
                outputs = pipeline.forward(batch=batch, context=context)
                results.append(outputs)
                global_step += 1  # Increment for next batch

        logger.info(f"Processed {len(results)} measurements")

    else:
        # Just display input/output specs
        input_specs = pipeline.get_input_specs()
        output_specs = pipeline.get_output_specs()

        print("\nInput Specs:")
        for name, spec in input_specs.items():
            print(f"  {name}: {spec}")

        print("\nOutput Specs:")
        for name, spec in output_specs.items():
            print(f"  {name}: {spec}")

    logger.info("Pipeline ready for inference")
    return pipeline


def _build_pipeline_from_config(
    trainrun_config: TrainRunConfig, device: str = "auto"
) -> CuvisPipeline:
    """Build pipeline from trainrun configuration.

    Parameters
    ----------
    trainrun_config : TrainRunConfig
        Trainrun configuration object
    device : str
        Device to load pipeline to ('cpu', 'cuda', 'auto')

    Returns
    -------
    CuvisPipeline
        Built pipeline ready for training
    """
    builder = PipelineBuilder()
    if trainrun_config.pipeline is None:
        raise ValueError("Pipeline configuration is missing in trainrun config.")
    pipeline_dict = trainrun_config.pipeline.to_dict()
    pipeline = builder.build_from_config(pipeline_dict)

    # Move pipeline to specified device if needed
    if device != "auto":
        pipeline = pipeline.to(device)

    return pipeline


def _create_datamodule_from_config(trainrun_config: TrainRunConfig) -> SingleCu3sDataModule:
    """Create datamodule from trainrun configuration.

    Parameters
    ----------
    trainrun_config : TrainRunConfig
        Trainrun configuration object

    Returns
    -------
    SingleCu3sDataModule
        Configured datamodule
    """
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
    return datamodule


def restore_trainrun(
    trainrun_path: str | Path,
    mode: Literal["train", "validate", "test", "info"] = "info",
    checkpoint_path: str | Path | None = None,
    device: str = "auto",
    overrides: list[str] | None = None,
) -> None:
    """Restore and reproduce training run from configuration file.

    Intelligently detects whether to use GradientTrainer (for gradient-based training)
    or StatisticalTrainer (for statistical-only training) based on trainrun configuration.
    Always runs statistical initialization before gradient training.

    Execution order:
    1. Statistical initialization (if needed) - always runs first
    2. Gradient training (if enabled)
    3. Validation (to populate monitoring/TensorBoard)
    4. Test evaluation (to populate monitoring/TensorBoard)

    Parameters
    ----------
    trainrun_path : str | Path
        Path to trainrun YAML file
    mode : str
        Execution mode:
        - 'info': Display experiment information only
        - 'train': Re-run training from scratch or continue training
        - 'validate': Run validation on trained model
        - 'test': Run test evaluation on trained model
    checkpoint_path : str | Path | None
        Optional checkpoint path to resume training from (gradient training only)
    device : str
        Device to run on ('cpu', 'cuda', 'auto')
    overrides : list[str] | None
        Hydra-style config overrides (e.g., ["output_dir=outputs/custom", "data.batch_size=16"])
    """
    trainrun_path = Path(trainrun_path)
    if not trainrun_path.exists():
        raise FileNotFoundError(f"TrainRun file not found: {trainrun_path}")

    logger.info(f"Loading trainrun from: {trainrun_path}")

    # Check if config has Hydra defaults - if so, resolve them first
    with trainrun_path.open("r", encoding="utf-8") as f:
        raw_config = yaml.safe_load(f)

    if isinstance(raw_config, dict) and "defaults" in raw_config:
        # Config has Hydra defaults - resolve them
        configs_dir = trainrun_path.parent.parent
        relative_path = trainrun_path.relative_to(configs_dir)
        config_path = str(relative_path.with_suffix("")).replace("\\", "/")

        config_dict = resolve_config_with_hydra(
            config_type="trainrun",
            config_path=config_path,
            search_paths=[str(configs_dir)],
            overrides=overrides,
        )
        trainrun_config = TrainRunConfig.model_validate(config_dict)
    else:
        # Config is already resolved - load directly
        trainrun_config: TrainRunConfig = TrainRunConfig.load_from_file(trainrun_path)

    # Build pipeline
    pipeline = _build_pipeline_from_config(trainrun_config, device=device)

    if mode == "info":
        logger.info("Info mode - displaying pipeline specifications")
        input_specs = pipeline.get_input_specs()
        output_specs = pipeline.get_output_specs()

        print("\nInput Specs:")
        for name, spec in input_specs.items():
            print(f"  {name}: {spec}")

        print("\nOutput Specs:")
        for name, spec in output_specs.items():
            print(f"  {name}: {spec}")

        logger.info("Info mode complete")
        return

    # Create datamodule
    datamodule = _create_datamodule_from_config(trainrun_config)
    output_dir = Path(trainrun_config.output_dir)

    # Detect training type: check if we have training config with trainer
    has_gradient_training = (
        trainrun_config.training is not None and trainrun_config.training.trainer is not None
    )

    # Check if statistical initialization is needed
    # Skip if weights are already loaded (indicates pre-trained pipeline)
    requires_static_fit = any(node.requires_initial_fit for node in pipeline.nodes())

    # Create trainers (both will be used based on mode)
    stat_trainer = StatisticalTrainer(pipeline=pipeline, datamodule=datamodule)
    grad_trainer = None

    if has_gradient_training:
        logger.info("Detected gradient training configuration")

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
    else:
        logger.info("Detected statistical-only training configuration")

    # === EXECUTION BASED ON MODE ===

    if mode == "train":
        logger.info("Training mode")

        # Step 1: Statistical initialization (always first, if needed)
        if requires_static_fit:
            logger.info("  Step 1: Statistical initialization...")
            stat_trainer.fit()

        # Step 2: Gradient training (if enabled)
        if grad_trainer is not None:
            logger.info("  Step 2: Gradient-based training...")

            # Unfreeze nodes for gradient training
            if trainrun_config.unfreeze_nodes:
                pipeline.unfreeze_nodes_by_name(trainrun_config.unfreeze_nodes)

            if checkpoint_path:
                logger.warning(
                    f"Checkpoint path provided: {checkpoint_path}. "
                    "Automatic checkpoint resumption is not yet implemented. "
                    "Please configure checkpoint resumption in trainer_config."
                )

            grad_trainer.fit()
        else:
            logger.info("  Step 2: Skipped (statistical-only)")

        # Step 3: Save trained pipeline
        logger.info("  Step 3: Saving trained pipeline...")
        restored_pipeline_path = output_dir / "trained_models" / f"{pipeline.name}_restored.yaml"
        restored_pipeline_path.parent.mkdir(parents=True, exist_ok=True)
        pipeline.save_to_file(str(restored_pipeline_path))
        logger.info(f"     Saved to {restored_pipeline_path}")

        # Step 4: Run validation (populate TensorBoard monitoring)
        if trainrun_config.data.val_ids:
            logger.info("  Step 4: Validation...")
            if grad_trainer is not None:
                grad_trainer.validate()
            else:
                stat_trainer.validate()
        else:
            logger.warning(
                "  Step 4: Validation skipped - no validation data provided (val_ids is empty)"
            )

        # Step 5: Run test evaluation (populate TensorBoard monitoring)
        if trainrun_config.data.test_ids:
            logger.info("  Step 5: Test evaluation...")
            if grad_trainer is not None:
                grad_trainer.test()
            else:
                stat_trainer.test()
        else:
            logger.warning(
                "  Step 5: Test evaluation skipped - no test data provided (test_ids is empty)"
            )

    elif mode == "validate":
        logger.info("Validation mode")

        # Run statistical initialization if needed
        if requires_static_fit:
            logger.info("  Running statistical initialization...")
            stat_trainer.fit()

        # Run validation
        if grad_trainer is not None:
            grad_trainer.validate()
        else:
            stat_trainer.validate()

    elif mode == "test":
        logger.info("Test mode")

        # Run statistical initialization if needed
        if requires_static_fit:
            logger.info("  Running statistical initialization...")
            stat_trainer.fit()

        # Run test
        if grad_trainer is not None:
            grad_trainer.test()
        else:
            stat_trainer.test()

    logger.info("Complete")


def restore_pipeline_cli() -> None:
    """CLI entry point for pipeline restoration."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Restore trained pipeline for inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Display pipeline info
  uv run restore-pipeline --pipeline-path configs/pipeline/adaclip_baseline.yaml

  # Run inference on CU3S file
  uv run restore-pipeline \\
    --pipeline-path configs/pipeline/adaclip_baseline.yaml \\
    --cu3s-file-path data/Lentils/Lentils_000.cu3s

  # Use custom device
  uv run restore-pipeline \\
    --pipeline-path configs/pipeline/adaclip_baseline.yaml \\
    --device cuda
        """,
    )

    parser.add_argument(
        "--pipeline-path",
        type=str,
        required=True,
        help="Path to pipeline YAML file",
    )
    parser.add_argument(
        "--weights-path",
        type=str,
        default=None,
        help="Path to weights (.pt) file (defaults to pipeline_path with .pt extension)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device to run on (default: auto)",
    )
    parser.add_argument(
        "--cu3s-file-path",
        type=str,
        default=None,
        help="Path to .cu3s file for inference",
    )
    parser.add_argument(
        "--processing-mode",
        type=str,
        default="Reflectance",
        choices=["Raw", "Reflectance"],
        help="Processing mode (default: Reflectance)",
    )
    parser.add_argument(
        "--override",
        action="append",
        help="Override config values in dot notation. Can be specified multiple times.",
    )

    args = parser.parse_args()

    restore_pipeline(
        pipeline_path=args.pipeline_path,
        weights_path=args.weights_path,
        device=args.device,
        cu3s_file_path=args.cu3s_file_path,
        processing_mode=args.processing_mode,
        config_overrides=args.override,
    )


def restore_trainrun_cli() -> None:
    """CLI entry point for trainrun restoration."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Restore and reproduce training runs from saved configurations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Display trainrun info
  uv run restore-trainrun --trainrun-path outputs/channel_selector/trained_models/channel_selector_trainrun.yaml

  # Re-run training
  uv run restore-trainrun \\
    --trainrun-path outputs/channel_selector/trained_models/channel_selector_trainrun.yaml \\
    --mode train

  # Run validation
  uv run restore-trainrun \\
    --trainrun-path outputs/channel_selector/trained_models/channel_selector_trainrun.yaml \\
    --mode validate

  # Override data and training configs
  uv run restore-trainrun \\
    --trainrun-path outputs/.../trainrun.yaml \\
    --mode train \\
    --override data.batch_size=16 \\
    --override training.optimizer.lr=0.001
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
