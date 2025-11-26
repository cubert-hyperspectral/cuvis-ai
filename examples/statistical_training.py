"""Statistical Initialization Only - RX Anomaly Detection Baseline

This example demonstrates the statistical initialization phase of training a cuvis.ai graph.
It showcases a complete RX (Reed-Xiaoli) anomaly detection pipeline that operates purely on
statistical initialization without gradient-based training.

Why RX Detector with Statistical Initialization?
------------------------------------------------
The RX detector is a classical anomaly detection algorithm that models the background distribution
using mean and covariance statistics. It's ideal for demonstrating statistical initialization because:
(1) No gradient training needed - operates on precomputed statistics from normal data
(2) Fast inference - closed-form Mahalanobis distance calculation
(3) Interpretable - anomalies are pixels far from the learned background distribution
(4) Strong baseline - effective for hyperspectral anomaly detection
(5) Normalization synergy - MinMax normalization stabilizes covariance estimation

Run with:
    python examples_torch/statistical_training.py

Graph Visualization:
```mermaid
graph LR
    Data[Data Node<br/>Lentils Cubes] --> Norm[MinMax<br/>Normalizer]
    Norm --> RX[RX Global<br/>Detector]
    RX --> Head[RX Logit<br/>Head]
    Head --> Dec[Binary<br/>Decider]

    Dec --> AM[Anomaly<br/>Metrics]
    Data --> |mask| AM
    AM --> TB[TensorBoard<br/>Monitor]

    Dec --> SM[Sample<br/>Metrics]
    SM --> TB

    Dec --> Viz[Anomaly<br/>Mask Viz]
    Data --> |mask| Viz
    Data --> |cube| Viz
    Viz --> TB

    style Data fill:#e1f5ff
    style Norm fill:#fff4e1
    style RX fill:#ffe1f5
    style Head fill:#ffe1f5
    style Dec fill:#f5e1ff
    style AM fill:#e1ffe1
    style SM fill:#e1ffe1
    style Viz fill:#ffe1e1
    style TB fill:#e1e1ff
```

Training Phases:
----------------
1. Statistical Initialization: Compute statistics for MinMax normalizer (min/max per channel)
   and RX detector (mean vector + covariance matrix) from training data
2. Validation: Evaluate anomaly detection performance on validation set
3. Test: Final evaluation on held-out test set

Pipeline Flow:
Data → MinMax → RX Detector → Logit Head → Decider ─┬→ Anomaly Metrics ─→ TensorBoard
                                                     ├→ Sample Metrics ──→ TensorBoard
                                                     └→ Visualization ───→ TensorBoard
"""

from typing import Any

import hydra
import torch
from loguru import logger
from omegaconf import DictConfig
from torch import Tensor

from cuvis_ai.anomaly.rx_detector import RXGlobal
from cuvis_ai.anomaly.rx_logit_head import RXLogitHead
from cuvis_ai.data.lentils_anomaly import SingleCu3sDataModule
from cuvis_ai.deciders.binary_decider import BinaryDecider
from cuvis_ai.node.data import LentilsAnomalyDataNode
from cuvis_ai.node.metrics import AnomalyDetectionMetrics
from cuvis_ai.node.monitor import TensorBoardMonitorNode
from cuvis_ai.node.node import Node
from cuvis_ai.node.normalization import MinMaxNormalizer
from cuvis_ai.node.visualizations import AnomalyMask
from cuvis_ai.pipeline.canvas import CuvisCanvas
from cuvis_ai.pipeline.ports import PortSpec
from cuvis_ai.training import StatisticalTrainer
from cuvis_ai.utils.types import Context, ExecutionStage, Metric


class SampleCustomMetrics(Node):
    """Compute anomaly pixel statistics from binary decisions.

    Calculates total pixels, anomalous pixels count, and anomaly percentage.
    Executes only during validation and test stages.
    """

    INPUT_SPECS = {
        "decisions": PortSpec(
            dtype=torch.bool,
            shape=(-1, -1, -1, 1),
            description="Binary anomaly decisions [B, H, W, 1]",
        ),
    }

    OUTPUT_SPECS = {"metrics": PortSpec(dtype=list, shape=(), description="List of Metric objects")}

    def __init__(self, **kwargs) -> None:
        super().__init__(execution_stages={ExecutionStage.VAL, ExecutionStage.TEST}, **kwargs)

    def forward(self, decisions: Tensor, context: Context) -> dict[str, Any]:
        """Compute anomaly pixel statistics.

        Parameters
        ----------
        decisions : Tensor
            Binary anomaly decisions [B, H, W, 1]
        context : Context
            Execution context with stage, epoch, batch_idx

        Returns
        -------
        dict[str, Any]
            Dictionary with "metrics" key containing list of Metric objects
        """
        # Calculate statistics
        total_pixels = decisions.numel()
        anomalous_pixels = int(decisions.sum().item())
        anomaly_percentage = (anomalous_pixels / total_pixels) * 100

        metrics = [
            Metric(
                name="anomaly/total_pixels",
                value=float(total_pixels),
                stage=context.stage,
                epoch=context.epoch,
                batch_idx=context.batch_idx,
            ),
            Metric(
                name="anomaly/anomalous_pixels",
                value=float(anomalous_pixels),
                stage=context.stage,
                epoch=context.epoch,
                batch_idx=context.batch_idx,
            ),
            Metric(
                name="anomaly/anomaly_percentage",
                value=anomaly_percentage,
                stage=context.stage,
                epoch=context.epoch,
                batch_idx=context.batch_idx,
            ),
        ]

        return {"metrics": metrics}

    def serialize(self, serial_dir: str) -> dict:
        return {**self.hparams}

    def load(self, params: dict, serial_dir: str) -> None:
        pass


@hydra.main(version_base=None, config_path="../cuvis_ai/conf", config_name="general")
def main(cfg: DictConfig) -> None:
    logger.info("=== Statistical Initialization for RX Anomaly Detection ===")

    # Stage 1: Setup datamodule
    datamodule = SingleCu3sDataModule(
        data_dir="../data/Lentils",
        batch_size=4,
        train_ids=[0, 1, 2],
        val_ids=[3, 4, 5],
        test_ids=[9, 10, 11, 12, 13],
    )
    datamodule.setup(stage="fit")

    # Extract wavelengths from dataset
    wavelengths = datamodule.train_ds.wavelengths
    logger.info(f"Dataset wavelengths: {wavelengths.shape}")
    logger.info(f"Wavelength range: {wavelengths.min():.1f} - {wavelengths.max():.1f} nm")

    # Stage 2: Build graph from configuration
    canvas = CuvisCanvas("RX_Statistical_Baseline")

    data_node = LentilsAnomalyDataNode(
        wavelengths=wavelengths,
        normal_class_ids=[0, 1],  # {0: 'Unlabeled', 1: 'Lentils_black'}
    )
    normalizer_node = MinMaxNormalizer(eps=1.0e-6, use_running_stats=True)
    rx_node = RXGlobal(eps=1.0e-6)
    logit_head = RXLogitHead(init_scale=1.0, init_bias=0.0)
    decider_node = BinaryDecider(threshold=0.5)
    metrics_anomaly = AnomalyDetectionMetrics(name="metrics_anomaly")
    sample_metrics = SampleCustomMetrics(name="sample_metrics")
    viz_mask = AnomalyMask(channel=30, up_to=5)
    tensorboard_node = TensorBoardMonitorNode(
        run_name="rx_statistical_baseline",
        output_dir="./outputs/",
        # run_name will auto-increment: run_01, run_02, run_03, etc.
    )

    # Stage 3: Connect graph
    canvas.connect(
        # Processing flow
        (data_node.outputs.cube, normalizer_node.data),
        (normalizer_node.normalized, rx_node.data),
        (rx_node.scores, logit_head.scores),
        (logit_head.logits, decider_node.logits),
        # Metric flow
        (decider_node.decisions, metrics_anomaly.decisions),
        (data_node.outputs.mask, metrics_anomaly.targets),
        (metrics_anomaly.metrics, tensorboard_node.metrics),
        # Sample custom metrics flow
        (decider_node.decisions, sample_metrics.decisions),
        (sample_metrics.metrics, tensorboard_node.metrics),
        # Visualization flow
        (decider_node.decisions, viz_mask.decisions),
        (data_node.outputs.mask, viz_mask.mask),
        (data_node.outputs.cube, viz_mask.cube),
        (viz_mask.artifacts, tensorboard_node.artifacts),
    )

    canvas.visualize(
        format="render_graphviz",
        output_path=f"outputs/canvases/{canvas.name}.png",
        show_execution_stage=True,
    )

    canvas.visualize(
        format="render_mermaid",
        output_path=f"outputs/canvases/{canvas.name}.md",
        direction="LR",
        include_node_class=True,
        wrap_markdown=True,  # default; set False for raw Mermaid text
        show_execution_stage=True,
    )

    # Stage 4: Statistical initialization
    stat_trainer = StatisticalTrainer(canvas=canvas, datamodule=datamodule)
    stat_trainer.fit()

    # Stage 5: Run validation
    stat_trainer.validate()

    # Stage 6: Run test
    stat_trainer.test()

    # Stage 7: Report results
    logger.info("View logs: uv run tensorboard --logdir=./outputs/rx_statistical_baseline_runs")


if __name__ == "__main__":
    main()
