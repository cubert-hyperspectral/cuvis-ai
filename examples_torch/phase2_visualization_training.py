"""Phase 2: Training with Visualization and Monitoring

This example demonstrates:
- Attaching visualization leaf nodes to the graph
- Registering monitoring plugins (DummyMonitor)
- Running statistical training with visualization generation
- Artifact saving to disk

Run with default config:
    python examples_torch/phase2_visualization_training.py

Override config from CLI:
    python examples_torch/phase2_visualization_training.py \
        visualization_leaves.pca_viz.n_components=3 \
        monitoring.dummy.output_dir=./my_artifacts
"""

import hydra
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from cuvis_ai.anomaly.rx_detector import RXGlobal
from cuvis_ai.data.lentils_anomaly import LentilsAnomaly
from cuvis_ai.normalization.normalization import MinMaxNormalizer
from cuvis_ai.pipeline.graph import Graph
from cuvis_ai.training.config import OptimizerConfig, TrainerConfig, TrainingConfig
from cuvis_ai.training.monitors import DummyMonitor, TensorBoardMonitor, WandBMonitor
from cuvis_ai.training.visualizations import AnomalyHeatmap, PCAVisualization, ScoreHistogram
from cuvis_ai.data.lentils_anomaly import LentilsAnomaly


@hydra.main(version_base=None, config_path="../cuvis_ai/conf", config_name="general")
def main(cfg: DictConfig) -> None:
    """Main training function with visualization and monitoring."""

    logger.info("=" * 70)
    logger.info("Phase 2: Visualization & Monitoring Training")
    logger.info("=" * 70)

    # Print configuration
    logger.info("Configuration:")
    logger.info(OmegaConf.to_yaml(cfg))

    # ========================================
    # Step 1: Build Graph
    # ========================================
    logger.info("\n[Step 1] Building graph...")

    graph = Graph("rx_visualization_pipeline")

    # Instantiate nodes from config without Hydra factories
    normalizer = MinMaxNormalizer(eps=1.0e-6, use_running_stats=True)
    graph.add_node(normalizer)

    rx = RXGlobal(eps=1.0e-6, trainable_stats=True)

    graph.add_node(rx, parent=normalizer)

    logger.info(f"  Graph '{graph.name}' created with {len(graph.nodes)} nodes")

    # ========================================
    # Step 2: Add Visualization Leaves
    # ========================================
    logger.info("\n[Step 2] Adding visualization leaves...")

    # PCA visualization on normalizer output
    pca_viz = PCAVisualization(
        n_components=2,
        log_every_n_epochs=1,
        max_samples=1000,
    )
    graph.add_leaf_node(pca_viz, parent=normalizer)
    logger.info(f"  Added PCAVisualization to normalizer (n_components={pca_viz.n_components})")

    # Anomaly heatmap on RX output
    anomaly_heatmap = AnomalyHeatmap(
        log_every_n_epochs=1,
        cmap="hot",
    )
    graph.add_leaf_node(anomaly_heatmap, parent=rx)
    logger.info(f"  Added AnomalyHeatmap to RX detector (cmap={anomaly_heatmap.cmap})")

    # Score histogram on RX output
    score_histogram = ScoreHistogram(
        log_every_n_epochs=1,
        bins=50,
    )
    graph.add_leaf_node(score_histogram, parent=rx)
    logger.info(f"  Added ScoreHistogram to RX detector (bins={score_histogram.bins})")

    logger.info(f"  Total leaf nodes: {len(graph.leaf_nodes)}")

    # ========================================
    # Step 3: Register Monitoring Plugins
    # ========================================
    logger.info("\n[Step 3] Registering monitoring plugins...")

    # Always add DummyMonitor
    dummy_monitor = DummyMonitor(
        output_dir="./outputs/artifacts",
        save_thumbnails=True,
    )
    graph.register_monitor(dummy_monitor)
    logger.info(f"  Registered DummyMonitor (output_dir={dummy_monitor.output_dir})")

    # Optionally add WandB (if enabled in config)
    wandb_monitor = WandBMonitor(
        project="cuvis-ai-phase2",
        tags=["phase2", "visualization", "rx_detector"],
        config={},
        mode="online"
    )
    graph.register_monitor(wandb_monitor)
    logger.info(f"  Registered WandBMonitor (project={wandb_monitor.project})")

    # Optionally add TensorBoard (if enabled in config)
    tb_monitor = TensorBoardMonitor(
        log_dir="./outputs/tensorboard/",
        flush_secs=120,
    )
    graph.register_monitor(tb_monitor)
    logger.info(f"  Registered TensorBoardMonitor (log_dir={tb_monitor.log_dir})")

    logger.info(f"  Total monitoring plugins: {len(graph.monitoring_plugins)}")

    # ========================================
    # Step 4: Setup Data Module
    # ========================================
    logger.info("\n[Step 4] Setting up data module...")

    datamodule = LentilsAnomaly(data_dir="../data/Lentils", batch_size=4)
    
    logger.info(f"  DataModule: {datamodule.__class__.__name__}")
    logger.info(f"  Data directory: {datamodule.data_dir}")
    logger.info(f"  Batch size: {datamodule.batch_size}")

    # ========================================
    # Step 5: Parse Training Config
    # ========================================
    logger.info("\n[Step 5] Parsing training configuration...")

    training_cfg = TrainingConfig(
        seed=42,
        trainer=TrainerConfig(
            max_epochs=5,  # Phase 1: statistical initialization only, no gradient training
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
        
    logger.info(f"  Seed: {training_cfg.seed}")
    logger.info(f"  Max epochs: {training_cfg.trainer.max_epochs}")
    logger.info(f"  Accelerator: {training_cfg.trainer.accelerator}")
    logger.info(f"  Optimizer: {training_cfg.optimizer.name} (lr={training_cfg.optimizer.lr})")

    # ========================================
    # Step 6: Train with Visualization
    # ========================================
    logger.info("\n[Step 6] Starting training...")
    logger.info("=" * 70)

    trainer = graph.train(datamodule=datamodule, training_config=training_cfg)

    logger.info("=" * 70)
    logger.info("Training complete!")

    # ========================================
    # Step 7: Verify Visualizations Generated
    # ========================================
    logger.info("\n[Step 7] Verifying artifacts...")

    import os

    artifact_dir = dummy_monitor.output_dir
    logger.info(f"  Checking artifacts in: {artifact_dir}")

    if artifact_dir.exists():
        # Count artifacts
        pkl_files = list(artifact_dir.rglob("*.pkl"))
        png_files = list(artifact_dir.rglob("*.png"))

        logger.info(f"  Artifacts saved to: {artifact_dir}")
        logger.info(f"  Pickle files: {len(pkl_files)}")
        logger.info(f"  PNG thumbnails: {len(png_files)}")

        if pkl_files:
            logger.info("\n  Generated artifacts:")
            for pkl_file in sorted(pkl_files)[:10]:  # Show first 10
                rel_path = pkl_file.relative_to(artifact_dir)
                logger.info(f"    - {rel_path}")
            if len(pkl_files) > 10:
                logger.info(f"    ... and {len(pkl_files) - 10} more")
    else:
        logger.warning(f"  Artifact directory not found: {artifact_dir}")

    # ========================================
    # Step 8: Run Forward Pass Test
    # ========================================
    logger.info("\n[Step 8] Testing forward pass...")

    # Get a test batch
    datamodule.setup(stage="fit")
    test_loader = datamodule.train_dataloader()
    test_batch = next(iter(test_loader))

    x = test_batch.get("cube") if "cube" in test_batch else test_batch.get("x")
    logger.info(f"  Input shape: {x.shape}")

    # Forward pass
    x_out, y_out, m_out = graph.forward(x)
    logger.info(f"  Output shape: {x_out.shape}")

    # Check for anomalies
    anomaly_scores = x_out.detach().cpu().numpy()
    threshold = anomaly_scores.mean() + 2 * anomaly_scores.std()
    anomalies = (anomaly_scores > threshold).sum()
    total_pixels = anomaly_scores.size
    anomaly_rate = (anomalies / total_pixels) * 100

    logger.info(f"  Anomaly detection threshold: {threshold:.6f}")
    logger.info(f"  Detected anomalies: {anomalies}/{total_pixels} ({anomaly_rate:.2f}%)")

    # ========================================
    # Summary
    # ========================================
    logger.info("\n" + "=" * 70)
    logger.info("PHASE 2 COMPLETE - Summary")
    logger.info("=" * 70)
    logger.info(f"✓ Graph nodes: {len(graph.nodes)}")
    logger.info(f"✓ Visualization leaves: {len(graph.leaf_nodes)}")
    logger.info(f"✓ Monitoring plugins: {len(graph.monitoring_plugins)}")
    logger.info(f"✓ Artifacts generated: {len(pkl_files)} pkl + {len(png_files)} png")
    logger.info(f"✓ Anomaly detection working: {anomaly_rate:.2f}% anomalies detected")
    logger.info("\nVisualization artifacts saved to:")
    logger.info(f"  {artifact_dir.absolute()}")
    logger.info("\nTo view artifacts:")
    logger.info(f"  1. Navigate to: {artifact_dir.absolute()}")
    logger.info(f"  2. Open PNG files to see visualizations")
    logger.info(f"  3. Load PKL files with pickle.load() for full data")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
