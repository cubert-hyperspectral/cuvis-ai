"""Monitoring adapter implementations for external observability backends."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from loguru import logger

from cuvis_ai.training.leaf_nodes import MonitoringNode


class DummyMonitor(MonitoringNode):
    """Filesystem-based monitoring adapter for testing.
    
    Saves artifacts to disk as pickle files with PNG thumbnails for visual artifacts.
    This is useful for testing visualization pipelines without external service dependencies.
    
    Parameters
    ----------
    output_dir : str or Path
        Directory to save artifacts to
    save_thumbnails : bool
        Whether to save PNG thumbnails alongside pickles
        
    Examples
    --------
    >>> monitor = DummyMonitor(output_dir="./outputs/artifacts")
    >>> graph.register_monitor(monitor)
    """

    def __init__(self, output_dir: str | Path = "./outputs/artifacts", save_thumbnails: bool = True):
        super().__init__()
        self.output_dir = Path(output_dir)
        self.save_thumbnails = save_thumbnails
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"DummyMonitor initialized: saving to {self.output_dir}")

    def setup(self, trainer: Any) -> None:
        """Called when trainer is created."""
        logger.info("DummyMonitor setup complete")

    def log_metrics(
        self,
        metrics: dict[str, float],
        step: int,
        *,
        stage: str
    ) -> None:
        """Log scalar metrics to a JSON file.
        
        Parameters
        ----------
        metrics : dict[str, float]
            Dictionary of metric names to values
        step : int
            Global training step
        stage : str
            Training stage ('train', 'val', 'test')
        """
        import json

        # Create stage directory
        stage_dir = self.output_dir / stage
        stage_dir.mkdir(parents=True, exist_ok=True)

        # Append to metrics log file
        metrics_file = stage_dir / "metrics.jsonl"
        with open(metrics_file, "a") as f:
            record = {"step": step, "stage": stage, **metrics}
            f.write(json.dumps(record) + "\n")

        logger.debug(f"Logged {len(metrics)} metrics for {stage} step {step}")

    def log_artifacts(
        self,
        artifacts: dict[str, Any],
        *,
        stage: str,
        step: int
    ) -> None:
        """Save artifacts to disk as pickle + PNG pairs.
        
        Parameters
        ----------
        artifacts : dict[str, Any]
            Dictionary of artifact names to objects
        stage : str
            Training stage ('train', 'val', 'test')
        step : int
            Global training step
        """
        # Create step directory nested under stage
        step_dir = self.output_dir / stage / f"step_{step:06d}"
        step_dir.mkdir(parents=True, exist_ok=True)

        for name, artifact in artifacts.items():
            # Sanitize filename
            safe_name = name.replace("/", "_").replace("\\", "_")

            # Save PNG first (before closing figure)
            if self.save_thumbnails and isinstance(artifact, dict):
                figure = artifact.get('figure')
                if figure is not None and hasattr(figure, 'savefig'):
                    png_path = step_dir / f"{safe_name}.png"
                    try:
                        figure.savefig(png_path, dpi=100, bbox_inches='tight')
                        logger.info(f"Saved visualization: {png_path}")
                    except Exception as e:
                        logger.warning(f"Failed to save PNG: {e}")
                    finally:
                        plt.close(figure)  # Clean up to avoid memory leak

            # Save pickle
            pickle_path = step_dir / f"{safe_name}.pkl"
            with open(pickle_path, "wb") as f:
                pickle.dump(artifact, f)

            logger.debug(f"Saved artifact pickle: {pickle_path}")

    def teardown(self) -> None:
        """Cleanup after training completes."""
        logger.info(f"DummyMonitor teardown: artifacts saved to {self.output_dir}")


class WandBMonitor(MonitoringNode):
    """Weights & Biases monitoring adapter.
    
    Full-featured integration with WandB for experiment tracking, metric logging,
    and artifact management.
    
    Parameters
    ----------
    project : str
        WandB project name
    entity : str, optional
        WandB entity (team) name
    tags : list[str], optional
        Tags for the run
    config : dict, optional
        Configuration dictionary to log
    mode : str, optional
        WandB mode: 'online', 'offline', or 'disabled' (default: 'online')
    name : str, optional
        Run name (default: auto-generated)
    notes : str, optional
        Notes about the run
        
    Examples
    --------
    >>> monitor = WandBMonitor(
    ...     project="cuvis-ai-training",
    ...     tags=["phase3", "rx"],
    ...     mode="online"
    ... )
    >>> graph.register_monitor(monitor)
    """

    def __init__(
        self,
        project: str,
        entity: str | None = None,
        tags: list[str] | None = None,
        config: dict | None = None,
        mode: str = "online",
        name: str | None = None,
        notes: str | None = None,
    ):
        super().__init__()
        self.project = project
        self.entity = entity
        self.tags = tags or []
        self.config = config or {}
        self.mode = mode
        self.name = name
        self.notes = notes
        self._wandb_run = None
        self._wandb_available = False

        # Check if wandb is available
        try:
            import wandb
            self._wandb_available = True
            self._wandb = wandb
            logger.info("WandB integration enabled")
        except ImportError:
            logger.warning(
                "WandB not available. Install with: pip install wandb. "
                "Monitor will operate in no-op mode."
            )

    def setup(self, trainer: Any) -> None:
        """Initialize WandB run.
        
        Parameters
        ----------
        trainer : Any
            PyTorch Lightning trainer instance
        """
        if not self._wandb_available:
            logger.warning("WandB setup skipped (not installed)")
            return

        try:
            # Initialize wandb run
            self._wandb_run = self._wandb.init(
                project=self.project,
                entity=self.entity,
                tags=self.tags,
                config=self.config,
                mode=self.mode,
                name=self.name,
                notes=self.notes,
                resume="allow",
            )

            logger.info(
                f"WandB run initialized: {self._wandb_run.name} "
                f"(project={self.project}, id={self._wandb_run.id})"
            )

            # Log trainer config if available
            if hasattr(trainer, 'logger') and hasattr(trainer.logger, 'experiment'):
                # Link to Lightning's WandB logger if present
                logger.debug("WandB integrated with Lightning trainer")

        except Exception as e:
            logger.error(f"Failed to initialize WandB run: {e}")
            self._wandb_run = None

    def log_metrics(
        self,
        metrics: dict[str, float],
        step: int,
        *,
        stage: str
    ) -> None:
        """Log metrics to WandB.
        
        Parameters
        ----------
        metrics : dict[str, float]
            Dictionary of metric names to values
        step : int
            Global training step
        stage : str
            Training stage ('train', 'val', 'test')
        """
        if not self._wandb_available or self._wandb_run is None:
            return

        try:
            # Prefix metrics with stage
            prefixed_metrics = {
                f"{stage}/{k}" if not k.startswith(stage) else k: v
                for k, v in metrics.items()
            }

            # Log to wandb
            self._wandb.log(prefixed_metrics, step=step)

            logger.debug(f"Logged {len(metrics)} metrics to WandB at step {step}")

        except Exception as e:
            logger.error(f"Failed to log metrics to WandB: {e}")

    def log_artifacts(
        self,
        artifacts: dict[str, Any],
        *,
        stage: str,
        step: int
    ) -> None:
        """Log artifacts to WandB.
        
        Converts matplotlib figures to WandB Images and logs them.
        
        Parameters
        ----------
        artifacts : dict[str, Any]
            Dictionary of artifact names to objects
        stage : str
            Training stage ('train', 'val', 'test')
        step : int
            Global training step
        """
        if not self._wandb_available or self._wandb_run is None:
            return

        try:
            wandb_artifacts = {}

            for name, artifact in artifacts.items():
                # Sanitize name
                safe_name = f"{stage}/{name}".replace("\\", "/")

                # Handle matplotlib figures
                if isinstance(artifact, dict) and 'figure' in artifact:
                    figure = artifact['figure']
                    if figure is not None and hasattr(figure, 'savefig'):
                        # Convert to WandB Image
                        wandb_artifacts[safe_name] = self._wandb.Image(figure)
                        # Close figure to free memory
                        plt.close(figure)

                # Handle PIL Images
                elif hasattr(artifact, 'save'):  # Duck typing for PIL Image
                    wandb_artifacts[safe_name] = self._wandb.Image(artifact)

                # Handle numpy arrays (as images if 2D/3D)
                elif hasattr(artifact, 'shape') and len(artifact.shape) in [2, 3]:
                    import numpy as np
                    if isinstance(artifact, np.ndarray):
                        wandb_artifacts[safe_name] = self._wandb.Image(artifact)

            # Log all artifacts
            if wandb_artifacts:
                self._wandb.log(wandb_artifacts, step=step)
                logger.debug(f"Logged {len(wandb_artifacts)} artifacts to WandB at step {step}")

        except Exception as e:
            logger.error(f"Failed to log artifacts to WandB: {e}")

    def teardown(self) -> None:
        """Finish WandB run."""
        if not self._wandb_available:
            return

        try:
            if self._wandb_run is not None:
                self._wandb.finish()
                logger.info("WandB run finished successfully")
        except Exception as e:
            logger.error(f"Failed to finish WandB run: {e}")


class TensorBoardMonitor(MonitoringNode):
    """TensorBoard monitoring adapter.
    
    Full-featured integration with TensorBoard for experiment tracking, metric logging,
    and visualization.
    
    Parameters
    ----------
    log_dir : str or Path
        Directory for TensorBoard logs
    comment : str, optional
        Comment to append to log directory name
    flush_secs : int, optional
        How often to flush pending events to disk (default: 120)
        
    Examples
    --------
    >>> monitor = TensorBoardMonitor(log_dir="./runs", comment="phase3_experiment")
    >>> graph.register_monitor(monitor)
    """

    def __init__(
        self,
        log_dir: str | Path = "./runs",
        comment: str = "",
        flush_secs: int = 120,
    ):
        super().__init__()
        self.log_dir = Path(log_dir)
        self.comment = comment
        self.flush_secs = flush_secs
        self._writer = None
        self._tensorboard_available = False

        # Check if tensorboard is available
        try:
            from torch.utils.tensorboard import SummaryWriter
            self._tensorboard_available = True
            self._SummaryWriter = SummaryWriter
            logger.info("TensorBoard integration enabled")
        except ImportError:
            logger.warning(
                "TensorBoard not available. Install with: pip install tensorboard. "
                "Monitor will operate in no-op mode."
            )

    def setup(self, trainer: Any) -> None:
        """Initialize TensorBoard SummaryWriter.
        
        Parameters
        ----------
        trainer : Any
            PyTorch Lightning trainer instance
        """
        if not self._tensorboard_available:
            logger.warning("TensorBoard setup skipped (not installed)")
            return

        try:
            # Create log directory
            self.log_dir.mkdir(parents=True, exist_ok=True)

            # Initialize SummaryWriter
            self._writer = self._SummaryWriter(
                log_dir=str(self.log_dir),
                comment=self.comment,
                flush_secs=self.flush_secs,
            )

            logger.info(f"TensorBoard writer initialized: {self.log_dir}")

            # Log hparams if available from trainer
            if hasattr(trainer, 'lightning_module') and hasattr(trainer.lightning_module, 'hparams'):
                hparams = trainer.lightning_module.hparams
                if isinstance(hparams, dict):
                    self._writer.add_hparams(hparams, {})
                    logger.debug("Logged hyperparameters to TensorBoard")

        except Exception as e:
            logger.error(f"Failed to initialize TensorBoard writer: {e}")
            self._writer = None

    def log_metrics(
        self,
        metrics: dict[str, float],
        step: int,
        *,
        stage: str
    ) -> None:
        """Log metrics to TensorBoard.
        
        Parameters
        ----------
        metrics : dict[str, float]
            Dictionary of metric names to values
        step : int
            Global training step
        stage : str
            Training stage ('train', 'val', 'test')
        """
        if not self._tensorboard_available or self._writer is None:
            return

        try:
            # Log each metric with stage prefix
            for name, value in metrics.items():
                # Add stage prefix if not already present
                full_name = f"{stage}/{name}" if not name.startswith(stage) else name
                self._writer.add_scalar(full_name, value, step)

            logger.debug(f"Logged {len(metrics)} metrics to TensorBoard at step {step}")

        except Exception as e:
            logger.error(f"Failed to log metrics to TensorBoard: {e}")

    def log_artifacts(
        self,
        artifacts: dict[str, Any],
        *,
        stage: str,
        step: int
    ) -> None:
        """Log artifacts to TensorBoard.
        
        Converts matplotlib figures to images and logs them to TensorBoard.
        
        Parameters
        ----------
        artifacts : dict[str, Any]
            Dictionary of artifact names to objects
        stage : str
            Training stage ('train', 'val', 'test')
        step : int
            Global training step
        """
        if not self._tensorboard_available or self._writer is None:
            return

        try:
            for name, artifact in artifacts.items():
                # Sanitize name
                full_name = f"{stage}/{name}".replace("\\", "/")

                # Handle matplotlib figures
                if isinstance(artifact, dict) and 'figure' in artifact:
                    figure = artifact['figure']
                    if figure is not None and hasattr(figure, 'savefig'):
                        try:
                            # Log figure directly
                            self._writer.add_figure(full_name, figure, step)
                        finally:
                            # Close figure to free memory
                            plt.close(figure)

                # Handle PIL Images
                elif hasattr(artifact, 'save'):  # Duck typing for PIL Image
                    import numpy as np
                    # Convert PIL to numpy array
                    img_array = np.array(artifact)
                    # TensorBoard expects CHW format
                    if img_array.ndim == 3 and img_array.shape[-1] in [1, 3, 4]:
                        img_array = img_array.transpose(2, 0, 1)
                    self._writer.add_image(full_name, img_array, step)

                # Handle numpy arrays (as images if 2D/3D)
                elif hasattr(artifact, 'shape'):
                    import numpy as np
                    if isinstance(artifact, np.ndarray):
                        if len(artifact.shape) == 2:
                            # 2D array - add as grayscale
                            self._writer.add_image(full_name, artifact, step, dataformats='HW')
                        elif len(artifact.shape) == 3:
                            # 3D array - check if HWC or CHW format
                            if artifact.shape[-1] in [1, 3, 4]:
                                # HWC format
                                self._writer.add_image(full_name, artifact, step, dataformats='HWC')
                            elif artifact.shape[0] in [1, 3, 4]:
                                # CHW format
                                self._writer.add_image(full_name, artifact, step, dataformats='CHW')

            logger.debug(f"Logged {len(artifacts)} artifacts to TensorBoard at step {step}")

        except Exception as e:
            logger.error(f"Failed to log artifacts to TensorBoard: {e}")

    def teardown(self) -> None:
        """Close TensorBoard writer."""
        if not self._tensorboard_available:
            return

        try:
            if self._writer is not None:
                self._writer.flush()
                self._writer.close()
                logger.info("TensorBoard writer closed successfully")
        except Exception as e:
            logger.error(f"Failed to close TensorBoard writer: {e}")


__all__ = [
    'DummyMonitor',
    'WandBMonitor',
    'TensorBoardMonitor',
]
