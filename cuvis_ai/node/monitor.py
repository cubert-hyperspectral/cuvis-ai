"""
TensorBoard Monitoring Nodes.

This module provides nodes for logging artifacts and metrics to TensorBoard
during pipeline execution. The monitoring nodes are sink nodes that accept
artifacts (visualizations) and metrics from upstream nodes and write them to
TensorBoard logs for visualization and analysis.

The primary use case is logging training and validation metrics, along with
visualizations like heatmaps, RGB renderings, and PCA plots during model training.

See Also
--------
cuvis_ai.node.anomaly_visualization : Anomaly visualization nodes
cuvis_ai.node.pipeline_visualization : Pipeline visualization nodes
"""

import re
from pathlib import Path

from cuvis_ai_schemas.enums import ArtifactType, ExecutionStage, NodeCategory, NodeTag
from cuvis_ai_schemas.execution import Artifact, Context, Metric
from cuvis_ai_schemas.pipeline import PortSpec
from loguru import logger
from torch.utils.tensorboard import SummaryWriter

from cuvis_ai_core.node import Node


class TensorBoardMonitorNode(Node):
    """TensorBoard monitoring node for logging artifacts and metrics.

    This is a SINK node that logs visualizations (artifacts) and metrics to TensorBoard.
    Accepts optional inputs for artifacts and metrics, allowing predecessors to be filtered
    by execution_stage without causing errors.

    Runs during train, val and test (the class-level ``EXECUTION_STAGES``); inference
    pipelines skip it, so loading a trained pipeline for prediction writes no TensorBoard
    directory. To log a trained pipeline's artifacts over a dataset, run it at the test
    stage (``Predictor.predict(stage=ExecutionStage.TEST)``). The log directory and the
    ``SummaryWriter`` are created on the first ``forward`` or ``log`` call, never at
    construction, so building the node touches no disk.

    Parameters
    ----------
    output_dir : str, optional
        Directory for TensorBoard logs (default: "./runs")
    comment : str, optional
        Comment to append to log directory name (default: "")
    flush_secs : int, optional
        How often to flush pending events to disk (default: 120)

    Examples
    --------
    >>> heatmap_viz = AnomalyHeatmap(cmap='hot', up_to=10)
    >>> tensorboard_node = TensorBoardMonitorNode(output_dir="./runs")
    >>> graph.connect(
    ...     (heatmap_viz.artifacts, tensorboard_node.artifacts),
    ... )
    """

    _category = NodeCategory.SINK
    _tags = frozenset({NodeTag.METADATA, NodeTag.TRAINING, NodeTag.EVALUATION})
    EXECUTION_STAGES = {ExecutionStage.TRAIN, ExecutionStage.VAL, ExecutionStage.TEST}

    INPUT_SPECS = {
        "artifacts": PortSpec(
            dtype=list,
            shape=(),
            optional=True,
            variadic=True,
            description="Optional list of Artifact objects to log",
        ),
        "metrics": PortSpec(
            dtype=list,
            shape=(),
            optional=True,
            variadic=True,
            description="Optional list of Metric objects to log",
        ),
    }

    OUTPUT_SPECS = {}  # Sink node - no outputs!

    def __init__(
        self,
        output_dir: str = "./runs",
        run_name: str | None = None,
        comment: str = "",
        flush_secs: int = 120,
        **kwargs,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.run_name = run_name
        self.comment = comment
        self.flush_secs = flush_secs
        # Both stay unset until the first forward()/log(): resolving the run directory
        # creates it, and a pipeline that merely loads this node (or never runs it because
        # the stage prunes it) must leave no TensorBoard directory behind.
        self._writer: SummaryWriter | None = None
        self.log_dir: Path | None = None
        self._SummaryWriter = SummaryWriter

        super().__init__(
            output_dir=str(output_dir),
            run_name=run_name,
            comment=comment,
            flush_secs=flush_secs,
            **kwargs,
        )

    def _ensure_writer(self) -> SummaryWriter:
        """Create the log directory and ``SummaryWriter`` on first use; return the writer."""
        if self._writer is None:
            self.log_dir = self._resolve_log_dir()
            self.log_dir.mkdir(parents=True, exist_ok=True)
            self._writer = self._SummaryWriter(
                log_dir=str(self.log_dir),
                comment=self.comment,
                flush_secs=self.flush_secs,
            )
            logger.info(f"TensorBoard writer initialized: {self.log_dir}")
            logger.info(
                f"To view visualizations, run: uv run tensorboard --logdir={self.output_dir}"
            )
        return self._writer

    def _resolve_log_dir(self) -> Path:
        """Resolve the log directory with auto-increment support.

        Returns
        -------
        Path
            The resolved log directory path
        """
        if self.run_name is None:
            # Auto-increment: find next available run_XX
            return self._get_next_run_dir()
        else:
            # Use specified run name
            log_dir = self.output_dir / self.run_name

            # If directory exists, add version suffix
            if log_dir.exists():
                version = 2
                while (self.output_dir / f"{self.run_name}_v{version}").exists():
                    version += 1
                log_dir = self.output_dir / f"{self.run_name}_v{version}"
                logger.warning(
                    f"Run directory '{self.run_name}' already exists. "
                    f"Using '{log_dir.name}' instead."
                )

            return log_dir

    def _get_next_run_dir(self) -> Path:
        """Find the next available run_XX directory.

        Returns
        -------
        Path
            Path to the next run directory (e.g., run_01, run_02, etc.)
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Find all existing run_XX directories
        existing_runs = []
        for item in self.output_dir.iterdir():
            if item.is_dir():
                match = re.match(r"run_(\d+)", item.name)
                if match:
                    existing_runs.append(int(match.group(1)))

        # Get next run number
        next_run = 1 if not existing_runs else max(existing_runs) + 1

        return self.output_dir / f"run_{next_run:02d}"

    def forward(
        self,
        artifacts: list[Artifact] | None = None,
        metrics: list[Metric] | None = None,
        context: Context | None = None,
    ) -> dict:
        """Log artifacts and metrics to TensorBoard.

        Parameters
        ----------
        context : Context
            Execution context with stage, epoch, batch_idx, global_step
        artifacts : list[Artifact], optional
            List of artifacts to log (default: None)
        metrics : list[Metric], optional
            List of metrics to log (default: None)

        Returns
        -------
        dict
            Empty dict (sink node has no outputs)
        """
        if context is None:
            context = Context()

        stage = context.stage.value
        step = context.global_step
        self._ensure_writer()

        # Flatten artifacts if it's a list of lists (variadic port)
        if artifacts is not None:
            if (
                isinstance(artifacts, list)
                and len(artifacts) > 0
                and isinstance(artifacts[0], list)
            ):
                artifacts = [item for sublist in artifacts for item in sublist]

        # Log artifacts
        if artifacts is not None:
            for artifact in artifacts:
                self._log_artifact(artifact, stage, step)
            logger.debug(f"Logged {len(artifacts)} artifacts to TensorBoard at step {step}")

        # Flatten metrics if variadic input provided
        if (
            metrics is not None
            and isinstance(metrics, list)
            and metrics
            and isinstance(metrics[0], list)
        ):
            metrics = [item for sublist in metrics for item in sublist]

        # Log metrics
        if metrics is not None:
            for metric in metrics:
                self._log_metric(metric, stage, step)
            logger.debug(f"Logged {len(metrics)} metrics to TensorBoard at step {step}")

        return {}

    def log(self, name: str, value: float, step: int) -> None:
        """Log a scalar value to TensorBoard.

        This method provides a simple interface for external trainers
        to log metrics directly, complementing the port-based logging.
        Used by GradientTrainer to log train/val losses to the same
        TensorBoard directory as graph metrics and artifacts.

        Parameters
        ----------
        name : str
            Name/tag for the scalar (e.g., "train/loss", "val/accuracy")
        value : float
            Scalar value to log
        step : int
            Global step number

        Examples
        --------
        >>> tensorboard_node = TensorBoardMonitorNode(output_dir="./runs")
        >>> # From external trainer
        >>> tensorboard_node.log("train/loss", 0.5, step=100)
        """
        self._ensure_writer().add_scalar(name, value, step)

    def _log_artifact(self, artifact: Artifact, stage: str, step: int) -> None:
        """Log a single artifact to TensorBoard.

        Parameters
        ----------
        artifact : Artifact
            Artifact object to log
        stage : str
            Training stage ('train', 'val', 'test', 'inference')
        step : int
            Global training step
        """
        # Validate artifact based on type
        if artifact.type == ArtifactType.IMAGE:
            self._validate_image_artifact(artifact)
            # Log as image
            tag = f"{stage}/{artifact.name}"
            # tag = f"{stage}/epoch_{artifact.epoch}/batch_{artifact.batch_idx}/{artifact.name}"
            # TensorBoard expects CHW format or HWC with dataformats parameter
            img_array = artifact.value
            writer = self._ensure_writer()
            if img_array.shape[-1] in [1, 3, 4]:
                # HWC format
                writer.add_image(tag, img_array, step, dataformats="HWC")
            elif img_array.shape[0] in [1, 3, 4]:
                # CHW format
                writer.add_image(tag, img_array, step, dataformats="CHW")
            else:
                logger.warning(
                    f"Image artifact {artifact.name} has unexpected shape: {img_array.shape}"
                )

    def _log_metric(self, metric: Metric, stage: str, step: int) -> None:
        """Log a single metric to TensorBoard.

        Parameters
        ----------
        metric : Metric
            Metric object to log
        stage : str
            Training stage ('train', 'val', 'test', 'inference')
        step : int
            Global training step
        """
        # Add stage prefix if not already present
        tag = f"{stage}/{metric.name}" if not metric.name.startswith(stage) else metric.name
        self._ensure_writer().add_scalar(tag, metric.value, step)

    def _validate_image_artifact(self, artifact: Artifact) -> None:
        """Validate that an IMAGE artifact has correct shape.

        Parameters
        ----------
        artifact : Artifact
            Artifact to validate

        Raises
        ------
        ValueError
            If artifact shape is invalid for IMAGE type
        """
        shape = artifact.value.shape
        if len(shape) != 3:
            raise ValueError(
                f"IMAGE artifact {artifact.name} must have 3 dimensions (H, W, C) or (C, H, W), "
                f"got shape {shape}"
            )

        # Check if either HWC or CHW format
        if shape[-1] not in [1, 3] and shape[0] not in [1, 3]:
            raise ValueError(
                f"IMAGE artifact {artifact.name} must have 1 or 3 channels, got shape {shape}"
            )

    def cleanup(self) -> None:
        """Flush and close the TensorBoard writer if one was created; safe to call twice."""
        writer = self._writer
        if writer is None:
            return
        self._writer = None
        try:
            writer.flush()
            writer.close()
            logger.debug("TensorBoard writer closed successfully")
        except Exception as e:
            logger.error(f"Failed to close TensorBoard writer: {e}")

    def __del__(self) -> None:
        """Clean up TensorBoard writer on deletion."""
        if getattr(self, "_writer", None) is not None:
            self.cleanup()
