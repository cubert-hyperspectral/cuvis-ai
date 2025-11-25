"""External trainer orchestrators for cuvis.ai graphs."""

from collections.abc import Mapping, Sequence
from typing import Any

import pytorch_lightning as pl
import torch
from loguru import logger
from torch.optim import Optimizer

from cuvis_ai.node.node import Node
from cuvis_ai.pipeline.canvas import CuvisCanvas
from cuvis_ai.training.config import OptimizerConfig, TrainerConfig, create_callbacks_from_config
from cuvis_ai.utils.types import ExecutionStage, InputStream


class GradientTrainer(pl.LightningModule):
    """Gradient-based trainer with explicit loss and metric handling.

    Parameters
    ----------
    canvas : CuvisCanvas
        Computation canvas
    datamodule : LightningDataModule
        Data module
    loss_nodes : list[Node]
        List of loss nodes whose outputs should be summed for backpropagation.
        Each loss node should output {"loss": scalar} or multiple scalar outputs.
    metric_nodes : list[Node], optional
        List of metric nodes whose outputs should be logged during validation/testing.
        Each metric node should output {"metrics": list[Metric]}.
    trainer_config : TrainerConfig, optional
        PyTorch Lightning trainer configuration
    optimizer_config : OptimizerConfig, optional
        Optimizer and scheduler configuration
    monitors : list, optional
        List of monitoring objects (WandB, TensorBoard, etc.)

    Examples
    --------
    >>> # Define loss and metric nodes
    >>> bce_loss = AnomalyBCEWithLogits(name="bce", weight=1.0)
    >>> entropy_loss = SelectorEntropyRegularizer(name="entropy", weight=0.01)
    >>> iou_metric = AnomalyDetectionMetrics(name="detection")
    >>>
    >>> # Connect in canvas
    >>> canvas.connect(
    ...     (predictions, bce_loss.predictions),
    ...     (targets, bce_loss.targets),
    ...     (selector_weights, entropy_loss.weights),
    ...     (predictions, iou_metric.predictions),
    ...     (targets, iou_metric.targets),
    ... )
    >>>
    >>> # Create trainer with explicit node lists
    >>> trainer = GradientTrainer(
    ...     canvas=canvas,
    ...     datamodule=datamodule,
    ...     loss_nodes=[bce_loss, entropy_loss],
    ...     metric_nodes=[iou_metric],
    ...     trainer_config=cfg.trainer,
    ...     optimizer_config=cfg.optimizer,
    ... )
    >>> trainer.fit()

    Logged Metrics
    --------------
    Training:
        - train_loss: Sum of all losses (for backpropagation)
        - train/{node.name}: Individual loss components (e.g., train/bce, train/entropy)
        - train/{node.name}_{port}: If loss node has multiple outputs (e.g., train/bce_loss1)

    Validation:
        - val_loss: Sum of all losses
        - val/{node.name}: Individual loss components
        - {node.name}_{metric.name}: All metrics (e.g., detection_anomaly/iou)
        - {node.name}_{metric.name}_1: If duplicate names occur

    Testing:
        - test_loss: Sum of all losses (if loss nodes execute in test stage)
        - test/{node.name}: Individual loss components
        - {node.name}_{metric.name}: All metrics
    """

    def __init__(
        self,
        canvas: CuvisCanvas,
        datamodule: pl.LightningDataModule,
        loss_nodes: list[Node],
        metric_nodes: list[Node] | None = None,
        trainer_config: TrainerConfig | None = None,
        optimizer_config: OptimizerConfig | None = None,
        monitors: list | None = None,
        callbacks: list | None = None,
    ) -> None:
        super().__init__()

        # Register graph's modules so Lightning can move them to correct device
        # Without this, graph nodes stay on CPU while trainer moves to CUDA
        self.canvas_modules = canvas.torch_layers
        self.canvas = canvas
        self.datamodule = datamodule
        self.loss_nodes = loss_nodes
        self.metric_nodes = metric_nodes or []
        self.trainer_config = trainer_config or TrainerConfig()
        self.optimizer_config = optimizer_config or OptimizerConfig()
        self.monitors = monitors or []
        self.callbacks = callbacks or []
        self.trainer = None  # Store trainer instance for reuse in test()

    def fit(self) -> None:
        """Start training using PyTorch Lightning.

        Unified API - both GradientTrainer and StatisticalTrainer use .fit()
        """

        # Convert TrainerConfig to kwargs for pl.Trainer
        trainer_kwargs = dict(self.trainer_config.__dict__)

        # Filter out None values and callbacks field (handled separately)
        trainer_kwargs = {
            k: v for k, v in trainer_kwargs.items() if v is not None and k != "callbacks"
        }

        # Determine which callbacks to use:
        # 1. If explicit callbacks passed to __init__, use those (backward compatibility)
        # 2. Otherwise, create callbacks from trainer_config.callbacks if present
        if self.callbacks:
            trainer_kwargs["callbacks"] = self.callbacks
        elif self.trainer_config.callbacks is not None:
            trainer_kwargs["callbacks"] = create_callbacks_from_config(
                self.trainer_config.callbacks
            )

            # Automatic checkpointing: enable if ModelCheckpoint callback is configured
            if self.trainer_config.callbacks.model_checkpoint is not None:
                trainer_kwargs["enable_checkpointing"] = True

        self.trainer = pl.Trainer(**trainer_kwargs)
        self.trainer.fit(model=self, datamodule=self.datamodule)

    def validate(self, ckpt_path: str | None = None) -> Sequence[Mapping[str, float]]:
        """Run validation evaluation.

        Parameters
        ----------
        ckpt_path : str | None
            Checkpoint path to load. Use "best" to load the best checkpoint,
            or None to validate with current model state.

        Returns
        -------
        Sequence[Mapping[str, float]]
            Sequence of mappings containing validation metrics.
        """
        if self.trainer is None:
            raise RuntimeError("validate() called before fit(). Please call fit() first.")

        self.datamodule.setup(stage="val")
        return self.trainer.validate(model=self, datamodule=self.datamodule, ckpt_path=ckpt_path)

    def test(self, ckpt_path: str | None = "best") -> Sequence[Mapping[str, float]]:
        """Run test evaluation, loading best checkpoint if available.

        Reuses the trainer instance from fit() to maintain callback state and
        enable checkpoint loading. If a ModelCheckpoint callback was used during
        training, this will automatically load the best checkpoint.

        Parameters
        ----------
        ckpt_path : str | None
            Checkpoint path to load. Use "best" to load the best checkpoint saved
            during training, or None to test with current model state.

        Returns
        -------
        Sequence[Mapping[str, float]]
            Sequence of mappings containing test metrics. Each mapping maps
            metric names (e.g., 'test_loss', 'metrics_anomaly/iou') to their float values.
            Typically contains one mapping with all test metrics.

        Raises
        ------
        RuntimeError
            If test() is called before fit()

        Examples
        --------
        >>> grad_trainer = GradientTrainer(...)
        >>> grad_trainer.fit()
        >>> test_results = grad_trainer.test()  # Loads best checkpoint if available
        >>> # test_results = [{'test_loss': 0.123, 'metrics_anomaly/iou': 0.95, ...}]
        """
        if self.trainer is None:
            raise RuntimeError(
                "test() called before fit(). Please call fit() first to initialize the trainer."
            )

        self.datamodule.setup(stage="test")
        return self.trainer.test(model=self, datamodule=self.datamodule, ckpt_path=ckpt_path)

    def _collect_losses(
        self, node_outputs: dict[str, dict[str, Any]], stage: str, epoch: int, batch_idx: int
    ) -> torch.Tensor:
        """Collect and log losses from loss_nodes.

        Parameters
        ----------
        node_outputs : dict[str, dict[str, Any]]
            Restructured graph outputs with node_id as primary key.
            Created by restructure_output_to_node_dict() for O(1) access.
        stage : str
            Execution stage: "train", "val", "test"
        epoch : int
            Current epoch number
        batch_idx : int
            Current batch index

        Returns
        -------
        torch.Tensor
            Sum of all losses for backpropagation
        """
        all_losses = []
        loss_name_counter = {}  # Track duplicate names

        for loss_node in self.loss_nodes:
            # O(1) lookup instead of O(n_outputs) iteration
            loss_outputs = node_outputs.get(loss_node.name, {})

            # Warn on first batch of first epoch if multiple outputs
            if epoch == 0 and batch_idx == 0 and len(loss_outputs) > 1:
                logger.warning(
                    f"Loss node '{loss_node.name}' produces multiple outputs: "
                    f"{list(loss_outputs.keys())}. All outputs will be summed for backpropagation."
                )

            # Collect and log each loss output
            for port_name, loss_val in loss_outputs.items():
                all_losses.append(loss_val)

                # Generate base log name
                if len(loss_outputs) == 1:
                    base_name = f"{stage}/{loss_node.name}"
                else:
                    base_name = f"{stage}/{loss_node.name}_{port_name}"

                # Handle duplicate names with numeric increment
                if base_name in loss_name_counter:
                    loss_name_counter[base_name] += 1
                    log_name = f"{base_name}_{loss_name_counter[base_name]}"
                else:
                    loss_name_counter[base_name] = 0
                    log_name = base_name

                self.log(log_name, loss_val, prog_bar=True)

        # Sum all losses for backpropagation
        if all_losses:
            total_loss = torch.stack(all_losses).sum()
        elif stage == "train":
            raise ValueError("No losses collected for training stage!")
        else:
            total_loss = torch.tensor(0.0, device=self.device)
        self.log(f"{stage}_loss", total_loss, prog_bar=True)

        return total_loss

    def _collect_metrics(self, node_outputs: dict[str, dict[str, Any]]) -> None:
        """Collect and log metrics from metric_nodes.

        Parameters
        ----------
        node_outputs : dict[str, dict[str, Any]]
            Restructured graph outputs with node_id as primary key.
            Created by restructure_output_to_node_dict() for O(1) access.
        """
        if not self.metric_nodes:
            return

        metric_name_counter = {}  # Track duplicate names

        for metric_node in self.metric_nodes:
            # O(1) lookup instead of O(n_outputs) iteration
            metric_outputs = node_outputs.get(metric_node.name, {}).get("metrics", [])

            # Each metric node outputs a list of Metric dataclass objects
            for metric in metric_outputs:  # list[Metric] objects
                # Generate base log name: NodeName_metric_name
                base_name = f"{metric_node.name}/{metric.name}"

                # Handle duplicate names with numeric increment
                if base_name in metric_name_counter:
                    metric_name_counter[base_name] += 1
                    log_name = f"{base_name}_{metric_name_counter[base_name]}"
                else:
                    metric_name_counter[base_name] = 0
                    log_name = base_name

                self.log(log_name, metric.value, prog_bar=True)

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        """Execute graph and collect losses for training."""
        from cuvis_ai.utils.graph_helper import restructure_output_to_node_dict
        from cuvis_ai.utils.types import Context

        context = Context(
            stage=ExecutionStage.TRAIN,
            epoch=self.current_epoch,
            batch_idx=batch_idx,
            global_step=self.global_step,
        )

        # Execute graph
        outputs = self.canvas.forward(batch=batch, context=context)

        # Transform outputs once for efficient access (O(n) operation)
        node_outputs = restructure_output_to_node_dict(outputs)

        # Collect losses with O(1) lookups per node
        total_loss = self._collect_losses(node_outputs, "train", self.current_epoch, batch_idx)

        # Log to external monitors
        for monitor in self.monitors:
            monitor.log("train/loss", total_loss, step=self.global_step)

        return total_loss

    def validation_step(self, batch, batch_idx) -> torch.Tensor:
        """Execute graph and collect losses + metrics for validation."""
        from cuvis_ai.utils.graph_helper import restructure_output_to_node_dict
        from cuvis_ai.utils.types import Context

        context = Context(
            stage=ExecutionStage.VAL,
            epoch=self.current_epoch,
            batch_idx=batch_idx,
            global_step=self.global_step,
        )

        # Execute graph
        outputs = self.canvas.forward(batch=batch, context=context)

        # Transform outputs once for efficient access (O(n) operation)
        node_outputs = restructure_output_to_node_dict(outputs)

        # Collect losses and metrics with O(1) lookups per node
        total_loss = self._collect_losses(node_outputs, "val", self.current_epoch, batch_idx)
        self._collect_metrics(node_outputs)

        # Log to external monitors
        for monitor in self.monitors:
            monitor.log("val/loss", total_loss, step=self.global_step)

        return total_loss

    def test_step(self, batch, batch_idx) -> torch.Tensor:
        """Execute graph and collect losses + metrics for testing."""
        from cuvis_ai.utils.graph_helper import restructure_output_to_node_dict
        from cuvis_ai.utils.types import Context

        context = Context(
            stage=ExecutionStage.TEST,
            epoch=self.current_epoch,
            batch_idx=batch_idx,
            global_step=self.global_step,
        )

        # Execute graph
        outputs = self.canvas.forward(batch=batch, context=context)

        # Transform outputs once for efficient access (O(n) operation)
        node_outputs = restructure_output_to_node_dict(outputs)

        # Collect losses and metrics with O(1) lookups per node
        total_loss = self._collect_losses(node_outputs, "test", self.current_epoch, batch_idx)
        self._collect_metrics(node_outputs)

        return total_loss

    def configure_optimizers(self) -> Optimizer | dict:
        """Configure optimizer and optional scheduler via the OptimizerConfig.

        Returns
        -------
        Optimizer | dict
            If no scheduler is configured, returns the optimizer.
            If a scheduler is configured, returns a dictionary with optimizer and scheduler config.
        """
        params = [p for p in self.canvas.parameters() if p.requires_grad]

        # Create optimizer based on config
        if self.optimizer_config.name.lower() == "adam":
            optimizer = torch.optim.Adam(
                params,
                lr=self.optimizer_config.lr,
                weight_decay=self.optimizer_config.weight_decay,
                betas=self.optimizer_config.betas or (0.9, 0.999),
            )
        elif self.optimizer_config.name.lower() == "adamw":
            optimizer = torch.optim.AdamW(
                params,
                lr=self.optimizer_config.lr,
                weight_decay=self.optimizer_config.weight_decay,
                betas=self.optimizer_config.betas or (0.9, 0.999),
            )
        elif self.optimizer_config.name.lower() == "sgd":
            optimizer = torch.optim.SGD(
                params,
                lr=self.optimizer_config.lr,
                weight_decay=self.optimizer_config.weight_decay,
            )
        else:
            raise ValueError(f"Unsupported optimizer: {self.optimizer_config.name}")

        # Create scheduler if configured
        if self.optimizer_config.scheduler is not None:
            scheduler_config = self.optimizer_config.scheduler

            if scheduler_config.name.lower() == "reduce_on_plateau":
                if scheduler_config.monitor is None:
                    raise ValueError(
                        "ReduceLROnPlateau scheduler requires 'monitor' to be specified in SchedulerConfig"
                    )

                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer,
                    mode=scheduler_config.mode,
                    factor=scheduler_config.factor,
                    patience=scheduler_config.patience,
                    threshold=scheduler_config.threshold,
                    threshold_mode=scheduler_config.threshold_mode,
                    cooldown=scheduler_config.cooldown,
                    min_lr=scheduler_config.min_lr,
                    eps=scheduler_config.eps,
                )

                return {
                    "optimizer": optimizer,
                    "lr_scheduler": {
                        "scheduler": scheduler,
                        "monitor": scheduler_config.monitor,
                        "interval": "epoch",
                        "frequency": 1,
                    },
                }
            else:
                raise ValueError(f"Unsupported scheduler: {scheduler_config.name}")

        return optimizer


class StatisticalTrainer:
    """External orchestrator for statistical initialization.

    NOT a Node - it's an external orchestrator that initializes
    statistical nodes using clean port-based streams.

    This trainer:
    - Finds all nodes with requires_initial_fit=True
    - Trains them in topological order using node.train()
    - Uses executor with upto_node for clean data transformation
    - No legacy tuple code - pure port-based architecture

    Examples
    --------
    >>> # Statistical initialization with unified API
    >>> stat_trainer = StatisticalTrainer(graph=graph, datamodule=datamodule)
    >>> stat_trainer.fit()  # Unified API - same as GradientTrainer
    >>>
    >>> # Run validation and test
    >>> stat_trainer.validate()
    >>> stat_trainer.test()
    """

    def __init__(self, canvas: CuvisCanvas, datamodule: pl.LightningDataModule) -> None:
        """Initialize statistical trainer.

        Parameters
        ----------
        canvas : CuvisCanvas
            The cuvis-ai canvas containing statistical nodes
        datamodule : pl.LightningDataModule
            Data provider for initialization
        """
        self.canvas = canvas
        self.datamodule = datamodule

    def fit(self) -> None:
        """Train all statistical nodes in topological order.

        Unified API - both GradientTrainer and StatisticalTrainer use .fit()
        """
        # Setup datamodule
        self.datamodule.setup("fit")
        train_loader = self.datamodule.train_dataloader()

        # Find nodes requiring statistical initialization
        stat_nodes = [
            node for node in self.canvas.nodes() if getattr(node, "requires_initial_fit", False)
        ]

        if not stat_nodes:
            logger.info("No statistical nodes to train")
            return

        logger.info(f"Training {len(stat_nodes)} statistical nodes...")

        # Train in topological order
        for node in self.canvas._sorted_nodes:
            if node not in stat_nodes:
                continue

            logger.info(f"  Training {type(node).__name__}...")

            # Create port-based input stream using upto_node
            input_stream = self._create_input_stream(node, train_loader)

            # Initialize the node from data
            node.fit(input_stream)

    def validate(self) -> None:
        """Run validation on the validation dataset.

        Returns
        -------
        list[dict[tuple[str, str], Any]]
            List of output dictionaries (one per batch), where each dict maps
            (node_id, port_name) tuples to output tensors
        """
        self.datamodule.setup(stage="val")
        val_loader = self.datamodule.val_dataloader()

        for batch in val_loader:
            self.canvas.forward(batch=batch, stage=ExecutionStage.VAL)

    def test(self) -> None:
        """Run test on the test dataset.

        Returns
        -------
        list[dict[tuple[str, str], Any]]
            List of output dictionaries (one per batch), where each dict maps
            (node_id, port_name) tuples to output tensors
        """
        self.datamodule.setup(stage="test")
        test_loader = self.datamodule.test_dataloader()

        for batch in test_loader:
            self.canvas.forward(batch=batch, stage=ExecutionStage.TEST)

    def _create_input_stream(self, target_node, dataloader) -> InputStream:
        """Create port-based input stream for statistical node.

        Uses graph.forward() with upto_node to get clean transformed inputs.
        Yields dicts matching target_node.INPUT_SPECS.

        Parameters
        ----------
        target_node : Node
            Node to create input stream for
        dataloader : DataLoader
            Source dataloader providing batches

        Yields
        ------
        dict[str, Any]
            Input dict with keys from target_node.INPUT_SPECS
        """
        for batch in dataloader:
            # Execute ancestors, stop before target (using upto_node)
            outputs = self.canvas.forward(
                batch=batch,
                stage=ExecutionStage.INFERENCE,
                upto_node=target_node,  # Partial execution
            )

            # Gather inputs for target node from predecessor outputs
            node_inputs = {}
            predecessors = list(self.canvas._graph.predecessors(target_node))

            if not predecessors:
                # Entry node - get directly from batch
                for port_name in target_node.INPUT_SPECS:
                    if port_name in batch:
                        node_inputs[port_name] = batch[port_name]
            else:
                # Get from parent outputs via graph edges
                for parent_node in predecessors:
                    for edge_data in self.canvas._graph[parent_node][target_node].values():
                        from_port = edge_data["from_port"]
                        to_port = edge_data["to_port"]
                        node_inputs[to_port] = outputs[(parent_node.name, from_port)]

            yield node_inputs  # Clean dict matching INPUT_SPECS!
