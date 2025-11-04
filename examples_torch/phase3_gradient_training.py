"""Phase 3: Gradient-Based Training with Trainable Nodes

This example uses the shared `general.yaml` Hydra configuration and instantiates
all nodes directly with the resolved values from `train_phase3.yaml`. The goal
is to make the script explicit about every argument passed to constructors.

Features demonstrated:
- MinMax normalization followed by a learnable SoftChannelSelector
- Trainable PCA node with orthogonality regularization
- Selector entropy & diversity regularization losses
- PCA variance and orthogonality metrics
- Optional visualization leaf and local artifact logging

Run with default settings:
    python examples_torch/phase3_gradient_training.py

Override options from the CLI (Hydra will still pass through to this script):
    python examples_torch/phase3_gradient_training.py training.trainer.max_epochs=8
"""

from __future__ import annotations

import os

import hydra
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from cuvis_ai.data.lentils_anomaly import LentilsAnomaly
from cuvis_ai.node.pca import TrainablePCA
from cuvis_ai.node.selector import SoftChannelSelector
from cuvis_ai.normalization.normalization import MinMaxNormalizer
from cuvis_ai.pipeline.graph import Graph
from cuvis_ai.training.config import OptimizerConfig, TrainerConfig, TrainingConfig
from cuvis_ai.training.losses import (
    OrthogonalityLoss,
    SelectorDiversityRegularizer,
    SelectorEntropyRegularizer,
)
from cuvis_ai.training.metrics import (
    ComponentOrthogonalityMetric,
    ExplainedVarianceMetric,
)
from cuvis_ai.training.monitors import DummyMonitor
from cuvis_ai.training.visualizations import PCAVisualization


@hydra.main(version_base=None, config_path="../cuvis_ai/conf", config_name="general")
def main(cfg: DictConfig) -> None:
    """Main training function with gradient-based optimization."""

    logger.info("=" * 70)
    logger.info("Phase 3: Gradient-Based Training with Trainable Nodes")
    logger.info("=" * 70)

    logger.info("Configuration:")
    logger.info(OmegaConf.to_yaml(cfg))

    # ========================================
    # Step 1: Build graph with explicit values
    # ========================================
    logger.info("\n[Step 1] Building graph with trainable nodes...")

    graph_name = "phase3_trainable_pca"
    graph = Graph(graph_name)

    normalizer = MinMaxNormalizer(eps=1.0e-6, use_running_stats=True)
    graph.add_node(normalizer)
    logger.info("  Added MinMaxNormalizer (eps=1.0e-6, use_running_stats=True)")

    selector = SoftChannelSelector(
        n_select=15,
        init_method="variance",
        temperature_init=5.0,
        temperature_min=0.1,
        temperature_decay=0.9,
        hard=False,
        trainable=True,
        eps=1.0e-6,
    )
    graph.add_node(selector, parent=normalizer)
    logger.info("  Added SoftChannelSelector:")
    logger.info("    - n_select=15")
    logger.info("    - init_method='variance'")
    logger.info("    - temperature_init=5.0 -> temperature_min=0.1 (decay=0.9)")
    logger.info("    - hard=False, trainable=True, eps=1.0e-6")

    pca = TrainablePCA(
        n_components=3,
        trainable=True,
        whiten=False,
        init_method="svd",
        eps=1.0e-6,
    )
    graph.add_node(pca, parent=selector)
    logger.info("  Added TrainablePCA:")
    logger.info("    - n_components=3, trainable=True, whiten=False")
    logger.info("    - init_method='svd', eps=1.0e-6")

    logger.info(f"  Graph '{graph.name}' created with {len(graph.nodes)} nodes")

    # ========================================
    # Step 2: Add loss leaf nodes
    # ========================================
    logger.info("\n[Step 2] Adding loss leaf nodes...")

    entropy_regularizer = SelectorEntropyRegularizer(weight=0.01, target_entropy=None)
    graph.add_leaf_node(entropy_regularizer, parent=selector)
    logger.info("  Added SelectorEntropyRegularizer (weight=0.01, target_entropy=None)")

    diversity_regularizer = SelectorDiversityRegularizer(weight=0.01)
    graph.add_leaf_node(diversity_regularizer, parent=selector)
    logger.info("  Added SelectorDiversityRegularizer (weight=0.01)")

    orthogonality_loss = OrthogonalityLoss(weight=1.0)
    graph.add_leaf_node(orthogonality_loss, parent=pca)
    logger.info("  Added OrthogonalityLoss (weight=1.0)")

    total_loss_nodes = sum(
        1 for leaf in graph.leaf_nodes.values() if "Loss" in leaf["type"].__name__
    )
    logger.info(f"  Total loss nodes: {total_loss_nodes}")

    # ========================================
    # Step 3: Add metric leaves
    # ========================================
    logger.info("\n[Step 3] Adding metric leaf nodes...")

    explained_variance_metric = ExplainedVarianceMetric()
    graph.add_leaf_node(explained_variance_metric, parent=pca)
    logger.info("  Added ExplainedVarianceMetric")

    component_orthogonality_metric = ComponentOrthogonalityMetric()
    graph.add_leaf_node(component_orthogonality_metric, parent=pca)
    logger.info("  Added ComponentOrthogonalityMetric")

    total_metric_nodes = sum(
        1 for leaf in graph.leaf_nodes.values() if "Metric" in leaf["type"].__name__
    )
    logger.info(f"  Total metric nodes: {total_metric_nodes}")

    # ========================================
    # Step 4: Add visualization leaves (optional)
    # ========================================
    logger.info("\n[Step 4] Adding visualization leaves...")

    pca_visualization = PCAVisualization(
        n_components=2,
        log_every_n_epochs=1,
        max_samples=1000,
    )
    graph.add_leaf_node(pca_visualization, parent=pca)
    logger.info("  Added PCAVisualization (n_components=2, log_every_n_epochs=1, max_samples=1000)")

    logger.info(f"  Total leaf nodes: {len(graph.leaf_nodes)}")

    # ========================================
    # Step 5: Register monitoring plugins
    # ========================================
    logger.info("\n[Step 5] Registering monitoring plugins...")

    dummy_monitor = DummyMonitor(output_dir="./outputs/phase3_artifacts", save_thumbnails=True)
    graph.register_monitor(dummy_monitor)
    logger.info("  Registered DummyMonitor (output_dir='./outputs/phase3_artifacts', save_thumbnails=True)")

    wandb_enabled = False
    tensorboard_enabled = False
    logger.info(f"  WandB monitor enabled: {wandb_enabled}")
    logger.info(f"  TensorBoard monitor enabled: {tensorboard_enabled}")

    # ========================================
    # Step 6: Instantiate datamodule
    # ========================================
    logger.info("\n[Step 6] Loading datamodule...")

    data_dir = os.getenv("DATA_ROOT", "./data/Lentils")
    datamodule = LentilsAnomaly(data_dir=data_dir, batch_size=4)

    logger.info(f"  DataModule: LentilsAnomaly(data_dir='{datamodule.data_dir}', batch_size={datamodule.batch_size})")

    # ========================================
    # Step 7: Create training configuration
    # ========================================
    logger.info("\n[Step 7] Preparing training configuration...")

    training_cfg = TrainingConfig(
        seed=42,
        trainer=TrainerConfig(
            max_epochs=5,
            accelerator="auto",
            devices=1,
            precision="32-true",
            enable_checkpointing=False,
            enable_progress_bar=True,
            log_every_n_steps=10,
            val_check_interval=1.0,
            check_val_every_n_epoch=1,
            accumulate_grad_batches=1,
            gradient_clip_val=None,
            deterministic=False,
            benchmark=False,
        ),
        optimizer=OptimizerConfig(
            name="adam",
            lr=0.001,
            weight_decay=0.0,
            betas=None,
        ),
        monitor_plugins=["loguru"],
    )

    logger.info(
        f"  Trainer: max_epochs={training_cfg.trainer.max_epochs}, "
        f"accelerator='{training_cfg.trainer.accelerator}', devices={training_cfg.trainer.devices}"
    )
    logger.info(
        f"  Optimizer: {training_cfg.optimizer.name} "
        f"(lr={training_cfg.optimizer.lr}, weight_decay={training_cfg.optimizer.weight_decay}, betas={training_cfg.optimizer.betas})"
    )

    # ========================================
    # Step 8: Train the graph
    # ========================================
    logger.info("\n[Step 8] Starting training...")
    logger.info("  1. Statistical initialization (selector + PCA)")
    logger.info("  2. Gradient-based fine-tuning with losses and metrics")
    logger.info("=" * 70)

    trainer = graph.train(datamodule=datamodule, training_config=training_cfg)

    logger.info("=" * 70)
    logger.info("Training complete!")

    # ========================================
    # Step 9: Analyze training results
    # ========================================
    logger.info("\n[Step 9] Analyzing training results...")

    import torch

    logger.info(f"  PCA components shape: {pca.components.shape if pca.components is not None else 'Not initialized'}")
    logger.info(f"  PCA mean shape: {pca.mean.shape if pca.mean is not None else 'Not initialized'}")
    logger.info(f"  Explained variance shape: {pca.explained_variance.shape if pca.explained_variance is not None else 'Not initialized'}")

    var_ratios = pca.get_explained_variance_ratio()
    if var_ratios is not None:
        logger.info("\n  Explained variance ratios:")
        for i, ratio in enumerate(var_ratios):
            logger.info(f"    PC{i + 1}: {ratio.item():.4f} ({ratio.item() * 100:.2f}%)")
        logger.info(f"    Total: {var_ratios.sum().item():.4f} ({var_ratios.sum().item() * 100:.2f}%)")

    final_orth_loss = pca.compute_orthogonality_loss()
    logger.info(f"\n  Final orthogonality loss: {final_orth_loss.item():.6f}")

    if pca.components is not None:
        gram = pca.components @ pca.components.T
        eye = torch.eye(pca.n_components, device=pca.components.device)
        orth_error = torch.norm(gram - eye, p="fro").item()
        logger.info(f"  Frobenius norm error from identity: {orth_error:.6f}")
    else:
        orth_error = float("nan")
        logger.warning("  PCA components not initialized; skipping orthogonality check")

    orth_threshold = 0.1
    if pca.components is not None and orth_error < orth_threshold:
        logger.info(f"  Orthogonality constraint maintained (error < {orth_threshold})")
    else:
        logger.warning(f"  Orthogonality constraint not maintained (error >= {orth_threshold})")

    # ========================================
    # Step 10: Test forward pass
    # ========================================
    logger.info("\n[Step 10] Testing forward pass...")

    datamodule.setup(stage="fit")
    test_loader = datamodule.train_dataloader()
    test_batch = next(iter(test_loader))

    x = test_batch.get("cube") if "cube" in test_batch else test_batch.get("x")
    logger.info(f"  Input shape: {x.shape}")

    x_out, y_out, m_out = graph.forward(x)
    logger.info(f"  Output shape: {x_out.shape}")
    logger.info(f"  Output range: [{x_out.min().item():.4f}, {x_out.max().item():.4f}]")

    # ========================================
    # Step 11: Verify artifacts
    # ========================================
    logger.info("\n[Step 11] Verifying artifacts...")

    artifact_dir = dummy_monitor.output_dir

    if artifact_dir.exists():
        pkl_files = list(artifact_dir.rglob("*.pkl"))
        png_files = list(artifact_dir.rglob("*.png"))
        json_files = list(artifact_dir.rglob("*.jsonl"))

        logger.info(f"  Artifacts saved to: {artifact_dir}")
        logger.info(f"  Pickle files: {len(pkl_files)}")
        logger.info(f"  PNG thumbnails: {len(png_files)}")
        logger.info(f"  Metric logs: {len(json_files)}")
    else:
        logger.warning(f"  Artifact directory not found: {artifact_dir}")

    # ========================================
    # Summary
    # ========================================
    logger.info("\n" + "=" * 70)
    logger.info("PHASE 3 COMPLETE - Summary")
    logger.info("=" * 70)
    logger.info(f"- Trainable nodes: {sum(1 for node in graph.nodes.values() if node.is_trainable)}")
    logger.info(f"- Loss nodes: {total_loss_nodes}")
    logger.info(f"- Metric nodes: {total_metric_nodes}")
    logger.info(f"- Monitoring plugins: {len(graph.monitoring_plugins)}")
    logger.info(f"- Training epochs: {training_cfg.trainer.max_epochs}")
    logger.info(f"- Final orthogonality loss: {final_orth_loss.item():.6f}")
    if var_ratios is not None:
        logger.info(f"- Total variance explained: {var_ratios.sum().item() * 100:.2f}%")
    logger.info("\nArtifacts saved to:")
    logger.info(f"  {artifact_dir.absolute()}")
    logger.info("\nNext steps:")
    logger.info("  1. Phase 4 introduces the soft channel selector with trainable heads")
    logger.info("  2. Experiment with different learning rates and epochs")
    logger.info("  3. Enable external monitors (WandB/TensorBoard) if desired")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()

