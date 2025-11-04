"""Phase 1 Training Example: Statistical Initialization Only

This example demonstrates the statistical initialization phase of training a cuvis.ai graph.
In Phase 1, nodes with requires_initial_fit=True are initialized from data (e.g., computing
mean/covariance for RX detector, min/max for normalizer).

Usage:
    # Use defaults from train_phase1.yaml
    uv run python examples_torch/phase1_statistical_training.py

    # Override specific settings
    uv run python examples_torch/phase1_statistical_training.py training.trainer.devices=2

    # Override data directory
    uv run python examples_torch/phase1_statistical_training.py datamodule.data_dir=/path/to/data

    # Enable gradient training (move to Phase 2)
    uv run python examples_torch/phase1_statistical_training.py training.trainer.max_epochs=10
"""

import hydra
import torch
from hydra.utils import instantiate
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from cuvis_ai.anomaly.rx_detector import RXGlobal
from cuvis_ai.normalization.normalization import MinMaxNormalizer
from cuvis_ai.pipeline.graph import Graph
from cuvis_ai.training.config import OptimizerConfig, TrainerConfig, TrainingConfig


@hydra.main(version_base=None, config_path="../cuvis_ai/conf", config_name="general")
def main(cfg: DictConfig) -> None:
    """Main training function using Hydra configuration.

    Parameters
    ----------
    cfg : DictConfig
        Hydra configuration loaded from train_phase1.yaml
    """
    logger.info("=" * 80)
    logger.info("Phase 1 Training: Statistical Initialization")
    logger.info("=" * 80)

    # Log the full configuration
    logger.info("Configuration:")
    logger.info("\n" + OmegaConf.to_yaml(cfg))

    # Build graph from configuration
    logger.info("Building graph...")
    graph = Graph("rx_statistical_baseline")

    # Instantiate nodes from config
    rx_node = RXGlobal(eps=1.0e-6, trainable_stats=True)
    normalizer_node = MinMaxNormalizer(eps=1.0e-6, use_running_stats=True)

    # Add nodes to graph
    graph.add_node(rx_node)
    graph.add_node(normalizer_node, parent=rx_node)

    logger.info(f"Graph created with {len(graph.nodes)} nodes:")
    for node_id, node in graph.nodes.items():
        logger.info(f"  - {node_id}: {node.__class__.__name__}")

    # Instantiate datamodule from config
    logger.info("Loading datamodule...")
    from cuvis_ai.data.lentils_anomaly import LentilsAnomaly

    datamodule = LentilsAnomaly(data_dir="../data/Lentils", batch_size=4)

    # Parse training configuration
    logger.info("Parsing training configuration...")
    training_cfg = TrainingConfig(
        seed=42,
        trainer=TrainerConfig(
            max_epochs=0,  # Phase 1: statistical initialization only, no gradient training
            accelerator="auto",
            devices=1,
            precision="32-true",
            enable_checkpointing=False,
            enable_progress_bar=True,
            log_every_n_steps=10,
        ),
        optimizer=OptimizerConfig(
            name="adam",
            lr=0.001,
            weight_decay=0.0,
            betas=(0.9, 0.999),
        ),
        monitor_plugins=[],
    )

    logger.info(
        f"Trainer config: max_epochs={training_cfg.trainer.max_epochs}, "
        f"accelerator={training_cfg.trainer.accelerator}, "
        f"devices={training_cfg.trainer.devices}"
    )
    logger.info(f"Optimizer config: {training_cfg.optimizer.name}, lr={training_cfg.optimizer.lr}")

    # Train the graph
    logger.info("Starting training...")
    trainer = graph.train(datamodule=datamodule, training_config=training_cfg)

    # Report results
    logger.info("=" * 80)
    logger.info("Training Complete!")
    logger.info("=" * 80)

    # Inspect statistical parameters
    logger.info("Statistical parameters initialized:")
    for node_id, node in graph.nodes.items():
        node_name = node_id.split("-")[0]
        if isinstance(node, RXGlobal):
            logger.info(f"  {node_name}:")
            logger.info(f"    - mu shape: {node.mu.shape if hasattr(node, 'mu') else 'N/A'}")
            logger.info(f"    - cov shape: {node.cov.shape if hasattr(node, 'cov') else 'N/A'}")
            logger.info(f"    - is_trainable: {node.is_trainable}")
            logger.info(f"    - frozen: {getattr(node, 'freezed', False)}")
        elif isinstance(node, MinMaxNormalizer):
            logger.info(f"  {node_name}:")
            logger.info(
                f"    - running_min shape: {node.running_min.shape if hasattr(node, 'running_min') else 'N/A'}"
            )
            logger.info(
                f"    - running_max shape: {node.running_max.shape if hasattr(node, 'running_max') else 'N/A'}"
            )

    if training_cfg.trainer.max_epochs > 0:
        logger.info(f"Gradient training completed: {training_cfg.trainer.max_epochs} epochs")
        if trainer is not None:
            logger.info(f"Trainer logs available at: {trainer.log_dir}")
    else:
        logger.info("Statistical initialization only (no gradient training)")

    # Test anomaly detection on validation data
    logger.info("=" * 80)
    logger.info("Testing Anomaly Detection")
    logger.info("=" * 80)

    # Get a test batch from validation dataloader
    test_batch = next(iter(datamodule.val_dataloader()))
    x_test = test_batch.get("cube") if "cube" in test_batch else test_batch.get("x")

    # Forward pass through the trained graph
    with torch.no_grad():
        # Use graph.forward() to pass through all nodes
        normalized_scores, _, _ = graph.forward(x_test)
        # Squeeze the channel dimension if present (BHWC -> BHW)
        if normalized_scores.dim() == 4:
            normalized_scores = normalized_scores.squeeze(-1)

        # Apply threshold for binary decisions (similar to phase0 example)
        threshold = 0.5
        decisions = (normalized_scores >= threshold).to(torch.int32)

        total_pixels = decisions.numel()
        anomalous_pixels = int(decisions.sum().item())
        anomaly_percentage = (anomalous_pixels / total_pixels) * 100

    logger.info(f"Binary decisions (threshold={threshold}):")
    logger.info(f"  - Total pixels: {total_pixels}")
    logger.info(f"  - Anomalous pixels: {anomalous_pixels}")
    logger.info(f"  - Anomaly rate: {anomaly_percentage:.2f}%")

    logger.info("Example complete!")


if __name__ == "__main__":
    main()
