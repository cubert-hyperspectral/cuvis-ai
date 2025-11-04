"""Phase 4: Soft Channel Selector with RX Anomaly Detection

This example now relies solely on the shared `general.yaml` Hydra configuration
and instantiates every node with the resolved values from `train_phase4.yaml`.
All constructor arguments are listed explicitly so the sample documents the
final runtime values without relying on target-based instantiation.

Highlights:
- MinMax normalization feeding a soft channel selector (temperature annealing)
- RX anomaly detector followed by a trainable RXLogitHead
- Selector entropy/diversity regularization plus BCE loss on logits
- Anomaly detection metrics and local artifact logging
- Discussion of dynamic graph re-parenting via `graph.set_parent`
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import hydra
import torch
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from cuvis_ai.anomaly.rx_detector import RXGlobal, RXPerBatch
from cuvis_ai.anomaly.rx_logit_head import RXLogitHead
from cuvis_ai.data.lentils_anomaly import LentilsAnomaly
from cuvis_ai.node import BinaryAnomalyLabelMapper, SoftChannelSelector
from cuvis_ai.normalization.normalization import MinMaxNormalizer, SigmoidNormalizer
from cuvis_ai.pipeline.graph import Graph
from cuvis_ai.training.config import OptimizerConfig, TrainerConfig, TrainingConfig
from cuvis_ai.training.losses import (
    AnomalyBCEWithLogits,
    SelectorDiversityRegularizer,
    SelectorEntropyRegularizer,
)
from cuvis_ai.training.metrics import AnomalyDetectionMetrics
from cuvis_ai.training.monitors import DummyMonitor
from cuvis_ai.training.visualizations import AnomalyHeatmap


@hydra.main(version_base=None, config_path="../cuvis_ai/conf", config_name="general")
def main(cfg: DictConfig) -> None:
    """Main training function demonstrating Phase 4 Steps 3 and 4."""

    logger.info("Phase 4 - Soft Channel Selector (RX anomaly)")
    logger.info(
        "Focus: trainable RXLogitHead + dynamic re-parenting overview"
    )
    logger.info("Config:\n{}", OmegaConf.to_yaml(cfg))

    # ========================================
    # Step 1: Build initial graph structure
    # ========================================
    logger.info("[1] Build graph")

    graph = Graph("phase4_soft_channel_selector")

    label_mapper = BinaryAnomalyLabelMapper(
        normal_class_ids=(0, 2),
        add_channel_axis=True,
    )
    graph.add_node(label_mapper)
    logger.info("Added BinaryAnomalyLabelMapper as entry node (0/2 -> 0, others -> 1)")

    normalizer = MinMaxNormalizer(eps=1.0e-6, use_running_stats=True)
    graph.add_node(normalizer, parent=label_mapper)
    logger.info("Added MinMaxNormalizer (eps=1e-6, running stats=True)")

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
    logger.info(
        "Added SoftChannelSelector after MinMaxNormalizer "
        f"(n_select={selector.n_select}, temp={selector.temperature_init}->{selector.temperature_min})"
    )

    # rx = RXGlobal(eps=1.0e-6, trainable_stats=False, cache_inverse=True)
    rx = RXPerBatch(eps=1.0e-6)
    graph.add_node(rx, parent=selector)
    logger.info("Added RXPerBatch (eps=1e-6)")

    logit_head = RXLogitHead(init_scale=1.0, init_bias=5.0, trainable=True)
    graph.add_node(logit_head, parent=rx)
    logger.info("Added RXLogitHead (init_scale=1.0, init_bias=5.0, trainable=True)")
    logger.info(
        "Graph path: Selector -> RX -> LogitHead "
        f"(nodes={len(graph.nodes)})"
    )

    # ========================================
    # Step 2: Dynamic re-parenting note (Step 4)
    # ========================================
    logger.info(
        "[2] Dynamic re-parenting via graph.set_parent(...) "
        "in cuvis_ai/pipeline/graph.py; this example stays linear."
    )

    # ========================================
    # Step 3: Add loss leaf nodes
    # ========================================
    entropy_regularizer = SelectorEntropyRegularizer(weight=0.01, target_entropy=None)
    graph.add_leaf_node(entropy_regularizer, parent=selector)
    diversity_regularizer = SelectorDiversityRegularizer(weight=0.01)
    graph.add_leaf_node(diversity_regularizer, parent=selector)
    anomaly_bce = AnomalyBCEWithLogits(weight=1.0, pos_weight=None, reduction="mean")
    graph.add_leaf_node(anomaly_bce, parent=logit_head)
    logger.info(
        "[3] Loss nodes: SelectorEntropy(0.01), SelectorDiversity(0.01), AnomalyBCEWithLogits"
    )

    anomaly_heatmap = AnomalyHeatmap(log_every_n_epochs=1, cmap="hot")
    graph.add_leaf_node(anomaly_heatmap, parent=rx)
    logger.info(
        "Added AnomalyHeatmap on RX outputs (log_every_n_epochs=1, cmap='hot')"
    )

    total_loss_nodes = sum(
        1 for leaf in graph.leaf_nodes.values() if "Loss" in leaf["type"].__name__
    )
    logger.info(f"Total loss nodes: {total_loss_nodes}")

    # ========================================
    # Step 4: Add metric leaf nodes
    # ========================================
    anomaly_metrics = AnomalyDetectionMetrics(threshold=0.0)
    graph.add_leaf_node(anomaly_metrics, parent=logit_head)
    logger.info("[4] Metric node: AnomalyDetectionMetrics(threshold=0.0)")

    total_metric_nodes = sum(
        1 for leaf in graph.leaf_nodes.values() if "Metric" in leaf["type"].__name__
    )
    logger.info(f"Total metric nodes: {total_metric_nodes}")

    # ========================================
    # Step 5: Register monitoring plugins
    # ========================================
    dummy_monitor = DummyMonitor(output_dir="./outputs/phase4_artifacts", save_thumbnails=True)
    graph.register_monitor(dummy_monitor)
    logger.info(
        "[5] Monitor: DummyMonitor(output_dir='./outputs/phase4_artifacts', thumbnails=True)"
    )

    wandb_enabled = False
    tensorboard_enabled = False
    logger.info(
        f"WandB enabled: {wandb_enabled}; TensorBoard enabled: {tensorboard_enabled}"
    )

    # ========================================
    # Step 6: Setup data module
    # ========================================
    data_dir = os.getenv("DATA_ROOT", "./data/Lentils")
    datamodule = LentilsAnomaly(data_dir=data_dir, batch_size=4, train_ids=[1,], val_ids=[2,3,4,5], test_ids=[10,11,12,13,14])

    logger.info(
        f"[6] Data module: LentilsAnomaly(data_dir='{datamodule.data_dir}', batch_size={datamodule.batch_size})"
    )

    # ========================================
    # Step 7: Build training configuration
    # ========================================
    training_cfg = TrainingConfig(
        seed=42,
        trainer=TrainerConfig(
            max_epochs=10,
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
            lr=0.01,
            weight_decay=0.0,
            betas=None,
        ),
        monitor_plugins=["loguru"],
    )

    logger.info("[7] Training config ready")
    logger.info(
        "Trainer: max_epochs={}, accelerator='{}', devices={}",
        training_cfg.trainer.max_epochs,
        training_cfg.trainer.accelerator,
        training_cfg.trainer.devices,
    )
    logger.info(
        "Optimizer: {} (lr={}, weight_decay={}, betas={})",
        training_cfg.optimizer.name,
        training_cfg.optimizer.lr,
        training_cfg.optimizer.weight_decay,
        training_cfg.optimizer.betas,
    )

    # ========================================
    # Step 8: Train the graph
    # ========================================
    logger.info(
        "[8] Training start (selector/RX init -> BCE updates)"
    )

    graph.train(datamodule=datamodule, training_config=training_cfg)

    logger.info("Training complete")

    # ========================================
    # Step 9: Inspect selector behaviour
    # ========================================
    selector.eval()
    with torch.no_grad():
        entropy_value = selector.compute_entropy().item()
        diversity_value = selector.compute_diversity_loss().item()
        soft_weights = selector.get_selection_weights(hard=False)
        hard_weights = selector.get_selection_weights(hard=True)
        top_k_indices = selector.get_top_k_channels().tolist()

    logger.info(
        "[9] Selector stats: temp {}->{:.4f}, entropy {:.4f}, diversity {:.4f}",
        selector.temperature_init,
        selector.temperature,
        entropy_value,
        diversity_value,
    )
    logger.info(
        "Selector weights: soft[:5]={}, hard_sum={:.0f}, top-{}={}",
        soft_weights[:5].tolist(),
        hard_weights.sum().item(),
        selector.n_select,
        top_k_indices,
    )
    selector.train()

    # ========================================
    # Step 10: Inspect RXLogitHead
    # ========================================
    learned_scale = logit_head.scale.item()
    learned_bias = logit_head.bias.item()
    logger.info(
        "[10] RXLogitHead: scale={:.4f}, bias={:.4f}",
        learned_scale,
        learned_bias,
    )

    # ========================================
    # Step 11: Evaluate anomaly detection on a batch
    # ========================================
    datamodule.setup(stage="fit")
    train_loader = datamodule.train_dataloader()
    batch = next(iter(train_loader))

    x = batch.get("cube") if "cube" in batch else batch.get("x")
    y: Optional[torch.Tensor] = batch.get("y")

    logger.info("[11] Eval batch shape {}", tuple(x.shape))

    with torch.no_grad():
        selector.eval()
        rx.eval()
        logit_head.eval()

        x_selected, _, _ = selector.forward(x)
        rx_scores, _, _ = rx.forward(x_selected)
        logits, _, _ = logit_head.forward(rx_scores)

        predictions = logit_head.predict_anomalies(logits)
        anomaly_rate = predictions.mean().item()

    logger.info(
        "Ranges: RX[{:.2f}, {:.2f}], logits[{:.2f}, {:.2f}], anomaly rate {:.2f}%",
        rx_scores.min().item(),
        rx_scores.max().item(),
        logits.min().item(),
        logits.max().item(),
        anomaly_rate * 100,
    )

    accuracy: Optional[float] = None
    if y is not None:
        if y.shape != predictions.shape:
            if len(y.shape) == 3:
                y = y.unsqueeze(-1)
        accuracy = (predictions == y).float().mean().item()
        logger.info("Accuracy vs labels: {:.2f}%", accuracy * 100)
    else:
        logger.info("Accuracy skipped (no labels)")

    selector.train()
    rx.train()
    logit_head.train()

    # ========================================
    # Step 12: Verify artifacts
    # ========================================
    artifact_dir: Path = dummy_monitor.output_dir
    if artifact_dir.exists():
        pkl_files = list(artifact_dir.rglob("*.pkl"))
        png_files = list(artifact_dir.rglob("*.png"))
        json_files = list(artifact_dir.rglob("*.jsonl"))

        logger.info(
            "[12] Artifacts: dir={}, pkl={}, png={}, json={}",
            artifact_dir,
            len(pkl_files),
            len(png_files),
            len(json_files),
        )
    else:
        logger.warning("Artifact directory not found: {}", artifact_dir)

    # ========================================
    # Summary
    # ========================================
    logger.info(
        "Summary: trainable={}, loss={}, metrics={}, monitors={}, epochs={}",
        sum(1 for node in graph.nodes.values() if node.is_trainable),
        total_loss_nodes,
        total_metric_nodes,
        len(graph.monitoring_plugins),
        training_cfg.trainer.max_epochs,
    )
    logger.info(
        "Selector final temp {:.4f}, entropy {:.4f}, RX bias {:.4f}",
        selector.temperature,
        entropy_value,
        learned_bias,
    )
    if accuracy is not None:
        logger.info("Anomaly accuracy: {:.2f}%", accuracy * 100)
    logger.info("Artifacts: {}", artifact_dir.absolute())
    logger.info(
        "Next: adjust channels/temperature, enable WandB or TensorBoard, try dynamic re-parenting"
    )


if __name__ == "__main__":
    main()
