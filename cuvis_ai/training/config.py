"""Training configuration infrastructure with Hydra support for full reproducibility."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml
from omegaconf import DictConfig, OmegaConf

from cuvis_ai import __version__


@dataclass
class EarlyStoppingConfig:
    """Configuration for PyTorch Lightning EarlyStopping callback.

    Parameters
    ----------
    monitor : str
        Metric name to monitor (e.g., 'train/bce', 'metrics_anomaly/iou')
    patience : int
        Number of epochs with no improvement after which training will be stopped
    mode : str
        One of 'min' or 'max'. In 'min' mode, training stops when monitored metric stops decreasing.
        In 'max' mode, training stops when monitored metric stops increasing.
    min_delta : float
        Minimum change in the monitored quantity to qualify as an improvement
    stopping_threshold : float | None
        Stop training immediately once the monitored quantity reaches this threshold
    verbose : bool
        Whether to print messages when stopping
    """

    monitor: str
    patience: int = 10
    mode: str = "min"
    min_delta: float = 1e-7
    stopping_threshold: float | None = None
    verbose: bool = True


@dataclass
class ModelCheckpointConfig:
    """Configuration for PyTorch Lightning ModelCheckpoint callback.

    Parameters
    ----------
    dirpath : str
        Directory to save checkpoints
    filename : str | None
        Checkpoint filename pattern (e.g., 'best-{epoch:02d}-{val_loss:.4f}')
    monitor : str
        Metric name to monitor for best checkpoint
    mode : str
        One of 'min' or 'max'
    save_top_k : int
        Save top k models. -1 saves all, 0 disables saving
    save_last : bool
        Save a checkpoint with name 'last.ckpt' in addition to top k
    verbose : bool
        Whether to print messages when saving checkpoints
    auto_insert_metric_name : bool
        Whether to automatically insert metric name in filename
    """

    dirpath: str
    monitor: str
    filename: str | None = None
    mode: str = "max"
    save_top_k: int = 1
    save_last: bool = False
    verbose: bool = True
    auto_insert_metric_name: bool = True


@dataclass
class LearningRateMonitorConfig:
    """Configuration for PyTorch Lightning LearningRateMonitor callback.

    Parameters
    ----------
    logging_interval : str
        Set to 'epoch' or 'step' to log lr at epoch or batch level
    log_momentum : bool
        Whether to log momentum alongside learning rate
    """

    logging_interval: str = (
        "epoch"  # Note: Could be Literal['step', 'epoch'] but keeping as str for flexibility
    )
    log_momentum: bool = False


@dataclass
class CallbacksConfig:
    """Container for all callback configurations.

    Parameters
    ----------
    early_stopping : list[EarlyStoppingConfig]
        List of early stopping configurations. Multiple early stopping callbacks
        can be used to monitor different metrics simultaneously.
    model_checkpoint : ModelCheckpointConfig | None
        Model checkpoint configuration
    learning_rate_monitor : LearningRateMonitorConfig | None
        Learning rate monitor configuration
    """

    early_stopping: list[EarlyStoppingConfig] = field(default_factory=list)
    model_checkpoint: ModelCheckpointConfig | None = None
    learning_rate_monitor: LearningRateMonitorConfig | None = None


def create_callbacks_from_config(config: CallbacksConfig | None) -> list:
    """Create PyTorch Lightning callback instances from configuration.

    Parameters
    ----------
    config : CallbacksConfig | None
        Callback configuration. If None, returns empty list.

    Returns
    -------
    list
        List of instantiated PyTorch Lightning callback objects

    Examples
    --------
    >>> config = CallbacksConfig(
    ...     early_stopping=[
    ...         EarlyStoppingConfig(monitor="train/bce", patience=10, mode="min"),
    ...         EarlyStoppingConfig(monitor="val/iou", patience=10, mode="max"),
    ...     ],
    ...     model_checkpoint=ModelCheckpointConfig(
    ...         dirpath="./checkpoints",
    ...         monitor="val/iou",
    ...         mode="max",
    ...     ),
    ... )
    >>> callbacks = create_callbacks_from_config(config)
    """
    if config is None:
        return []

    from pytorch_lightning.callbacks import (
        EarlyStopping,
        LearningRateMonitor,
        ModelCheckpoint,
    )

    callbacks = []

    # Create EarlyStopping callbacks
    for es_config in config.early_stopping:
        callbacks.append(
            EarlyStopping(
                monitor=es_config.monitor,
                patience=es_config.patience,
                mode=es_config.mode,
                min_delta=es_config.min_delta,
                stopping_threshold=es_config.stopping_threshold,
                verbose=es_config.verbose,
            )
        )

    # Create ModelCheckpoint callback
    if config.model_checkpoint is not None:
        mc_config = config.model_checkpoint
        callbacks.append(
            ModelCheckpoint(
                dirpath=mc_config.dirpath,
                filename=mc_config.filename,
                monitor=mc_config.monitor,
                mode=mc_config.mode,
                save_top_k=mc_config.save_top_k,
                save_last=mc_config.save_last,
                verbose=mc_config.verbose,
                auto_insert_metric_name=mc_config.auto_insert_metric_name,
            )
        )

    # Create LearningRateMonitor callback
    if config.learning_rate_monitor is not None:
        lr_config = config.learning_rate_monitor
        callbacks.append(
            LearningRateMonitor(
                logging_interval=lr_config.logging_interval,
                log_momentum=lr_config.log_momentum,
            )
        )

    return callbacks


@dataclass
class TrainerConfig:
    """Hydra-serializable configuration for the Lightning Trainer.

    Parameters
    ----------
    max_epochs : int
        Maximum number of training epochs
    accelerator : str
        Device accelerator ('cpu', 'gpu', 'tpu', 'auto')
    devices : int | str | None
        Number of devices or device IDs to use
    default_root_dir : str | None
        Default path for logs and weights. If None, uses current working directory.
    precision : str | int
        Precision mode ('32-true', '16-mixed', 'bf16-mixed', etc.)
    accumulate_grad_batches : int
        Accumulate gradients over N batches
    enable_progress_bar : bool
        Show training progress bar
    enable_checkpointing : bool
        Enable model checkpointing
    log_every_n_steps : int
        Log metrics every N steps
    val_check_interval : float | int | None
        How often to check validation set (fraction of epoch or num batches)
    check_val_every_n_epoch : int | None
        Check validation every N epochs
    gradient_clip_val : float | None
        Gradient clipping value (None to disable)
    deterministic : bool
        Use deterministic algorithms for reproducibility
    benchmark : bool
        Enable cudnn.benchmark for performance
    callbacks : CallbacksConfig | None
        Callback configurations (early stopping, checkpointing, LR monitoring, etc.)
    """

    max_epochs: int = 100
    accelerator: str = "auto"
    devices: int | str | None = None
    default_root_dir: str | None = None
    precision: str | int = "32-true"
    accumulate_grad_batches: int = 1
    enable_progress_bar: bool = True
    enable_checkpointing: bool = False
    log_every_n_steps: int = 50
    val_check_interval: float | int | None = 1.0
    check_val_every_n_epoch: int | None = 1
    gradient_clip_val: float | None = None
    deterministic: bool = False
    benchmark: bool = False
    callbacks: CallbacksConfig | None = None


@dataclass
class SchedulerConfig:
    """Learning rate scheduler configuration.

    Parameters
    ----------
    name : str
        Scheduler name ('reduce_on_plateau', 'step', 'cosine', etc.)
    monitor : str | None
        Metric name to monitor for ReduceLROnPlateau (e.g., 'metrics_anomaly/iou')
    mode : str
        One of 'min' or 'max'. In 'min' mode, lr will be reduced when the quantity
        monitored has stopped decreasing; in 'max' mode it will be reduced when the
        quantity monitored has stopped increasing.
    factor : float
        Factor by which the learning rate will be reduced. new_lr = lr * factor
    patience : int
        Number of epochs with no improvement after which learning rate will be reduced
    threshold : float
        Threshold for measuring the new optimum, to only focus on significant changes
    threshold_mode : str
        One of 'rel', 'abs'. In 'rel' mode, dynamic_threshold = best * (1 + threshold)
        in 'max' mode or best * (1 - threshold) in 'min' mode
    cooldown : int
        Number of epochs to wait before resuming normal operation after lr has been reduced
    min_lr : float
        A lower bound on the learning rate
    eps : float
        Minimal decay applied to lr. If the difference between new and old lr is smaller
        than eps, the update is ignored
    verbose : bool
        If True, prints a message to stdout for each update
    """

    name: str = "reduce_on_plateau"
    monitor: str | None = None
    mode: str = "min"
    factor: float = 0.1
    patience: int = 10
    threshold: float = 1e-4
    threshold_mode: str = "rel"
    cooldown: int = 0
    min_lr: float = 1e-6
    eps: float = 1e-8
    verbose: bool = False


@dataclass
class OptimizerConfig:
    """Optimizer settings consumed by the Trainer.

    Parameters
    ----------
    name : str
        Optimizer name ('adam', 'sgd', 'adamw')
    lr : float
        Learning rate
    weight_decay : float
        Weight decay (L2 regularization)
    betas : tuple[float, float] | None
        Adam beta parameters (beta1, beta2)
    scheduler : SchedulerConfig | None
        Learning rate scheduler configuration
    """

    name: str = "adamw"
    lr: float = 1e-3
    weight_decay: float = 0.0
    betas: tuple[float, float] | None = None
    scheduler: SchedulerConfig | None = None


@dataclass
class TrainingConfig:
    """Top-level training configuration bundle.

    Parameters
    ----------
    seed : int
        Random seed for reproducibility
    trainer : TrainerConfig
        Lightning Trainer configuration
    optimizer : OptimizerConfig
        Optimizer configuration
    """

    seed: int = 42
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)

    # ------------------------------------------------------------------
    # Serialization methods (consistent with other config classes)
    # ------------------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary (consistent with other config classes).

        Returns
        -------
        dict[str, Any]
            Dictionary representation of the training config
        """
        return {
            "seed": self.seed,
            "trainer": asdict(self.trainer),
            "optimizer": asdict(self.optimizer),
        }

    def to_dict_config(self) -> DictConfig:
        """Convert to OmegaConf DictConfig with schema validation.

        Returns
        -------
        DictConfig
            OmegaConf DictConfig representation
        """
        base = OmegaConf.structured(TrainingConfig)
        payload = OmegaConf.create(asdict(self))
        merged = OmegaConf.merge(base, payload)
        assert isinstance(merged, DictConfig)
        return merged

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TrainingConfig:
        """Create TrainingConfig from dictionary.

        Parameters
        ----------
        data : dict[str, Any]
            Dictionary containing training config data

        Returns
        -------
        TrainingConfig
            Parsed training configuration
        """
        return cls.from_dict_config(data)

    @classmethod
    def from_dict_config(cls, config: DictConfig | Mapping[str, Any]) -> TrainingConfig:
        """Create TrainingConfig from DictConfig or plain mapping.

        Uses OmegaConf's structured config instantiation to properly handle
        nested configs and default value overrides without manual construction.

        Parameters
        ----------
        config : DictConfig | Mapping[str, Any]
            Configuration dict to parse

        Returns
        -------
        TrainingConfig
            Parsed configuration object

        Raises
        ------
        TypeError
            If the result is not a TrainingConfig instance
        """
        cfg = config if isinstance(config, DictConfig) else OmegaConf.create(dict(config))
        merged = OmegaConf.merge(OmegaConf.structured(TrainingConfig), cfg)
        assert isinstance(merged, DictConfig)

        # Use OmegaConf's native structured config instantiation
        # This properly handles nested configs and default values automatically
        instantiated = OmegaConf.to_object(merged)
        if not isinstance(instantiated, TrainingConfig):
            raise TypeError(f"Expected TrainingConfig, got {type(instantiated)}")
        return instantiated

    # ------------------------------------------------------------------
    # JSON helpers for RPC serialization
    # ------------------------------------------------------------------
    def to_json(self) -> str:
        """Serialize the training config to JSON (for gRPC transport).

        Returns
        -------
        str
            JSON string representation
        """
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, payload: str) -> TrainingConfig:
        """Deserialize a TrainingConfig from a JSON string (for gRPC transport).

        Parameters
        ----------
        payload : str
            JSON string to deserialize

        Returns
        -------
        TrainingConfig
            Parsed training configuration
        """
        return cls.from_dict(json.loads(payload))


@dataclass
class PipelineMetadata:
    """Pipeline metadata for documentation and discovery.

    Parameters
    ----------
    name : str
        Pipeline name
    description : str
        Pipeline description
    created : str
        Creation timestamp (ISO format)
    tags : list[str]
        Tags for categorization and discovery
    author : str
        Pipeline author
    cuvis_ai_version : str
        Version of cuvis_ai used to create the pipeline
    """

    name: str
    description: str = ""
    created: str = ""
    tags: list[str] = field(default_factory=list)
    author: str = ""
    cuvis_ai_version: str = field(default_factory=lambda: __version__)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PipelineMetadata:
        """Create from dictionary."""
        return cls(
            name=data.get("name", ""),
            description=data.get("description", ""),
            created=data.get("created", ""),
            tags=list(data.get("tags", [])),
            author=data.get("author", ""),
            cuvis_ai_version=data.get("cuvis_ai_version", __version__),
        )

    def to_proto(self) -> Any:
        """Convert to proto PipelineMetadata.

        Returns
        -------
        Any
            Proto PipelineMetadata message
        """
        from cuvis_ai.grpc.v1 import cuvis_ai_pb2

        return cuvis_ai_pb2.PipelineMetadata(
            name=self.name,
            description=self.description,
            created=self.created,
            cuvis_ai_version="",
            tags=self.tags,
            author=self.author,
        )

    @classmethod
    def from_proto(cls, proto_metadata: Any) -> PipelineMetadata:
        """Convert from proto PipelineMetadata.

        Parameters
        ----------
        proto_metadata : Any
            Proto PipelineMetadata message

        Returns
        -------
        PipelineMetadata
            Python pipeline metadata
        """
        return cls(
            name=proto_metadata.name,
            description=proto_metadata.description,
            created=proto_metadata.created,
            tags=list(proto_metadata.tags),
            author=proto_metadata.author,
        )


@dataclass
class PipelineConfig:
    """Typed schema for pipeline structure and metadata.

    This represents the complete pipeline configuration including structure,
    metadata, nodes, and connections. Similar to TrainingConfig and DataConfig,
    this can be serialized/deserialized and passed via gRPC.

    Parameters
    ----------
    metadata : PipelineMetadata
        Pipeline metadata (name, description, tags, etc.)
    nodes : list[dict[str, Any]]
        List of node configurations with class, params, etc.
    connections : list[dict[str, str]]
        List of connection specifications (from/to port references)
    """

    metadata: PipelineMetadata
    nodes: list[dict[str, Any]]
    connections: list[dict[str, str]]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary.

        Returns
        -------
        dict[str, Any]
            Dictionary representation suitable for YAML serialization
        """
        return {
            "metadata": self.metadata.to_dict(),
            "nodes": self.nodes,
            "connections": self.connections,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PipelineConfig:
        """Create from dictionary.

        Parameters
        ----------
        data : dict[str, Any]
            Dictionary containing pipeline config data

        Returns
        -------
        PipelineConfig
            Parsed pipeline configuration
        """
        return cls(
            metadata=PipelineMetadata.from_dict(data.get("metadata", {})),
            nodes=list(data.get("nodes", [])),
            connections=list(data.get("connections", [])),
        )

    def to_proto(self) -> Any:
        """Convert to proto PipelineConfig.

        For gRPC transport, we serialize the full config as JSON bytes.

        Returns
        -------
        Any
            Proto PipelineConfig message with config_bytes field
        """
        from cuvis_ai.grpc.v1 import cuvis_ai_pb2

        config_json = json.dumps(self.to_dict())
        return cuvis_ai_pb2.PipelineConfig(
            config_bytes=config_json.encode("utf-8"),
        )

    @classmethod
    def from_proto(cls, proto_config: Any) -> PipelineConfig:
        """Convert from proto PipelineConfig.

        Parameters
        ----------
        proto_config : Any
            Proto PipelineConfig message

        Returns
        -------
        PipelineConfig
            Python pipeline configuration
        """
        config_json = proto_config.config_bytes.decode("utf-8")
        config_dict = json.loads(config_json)
        return cls.from_dict(config_dict)

    def to_json(self) -> str:
        """Serialize to JSON string.

        Returns
        -------
        str
            JSON representation
        """
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, payload: str) -> PipelineConfig:
        """Deserialize from JSON string.

        Parameters
        ----------
        payload : str
            JSON string to deserialize

        Returns
        -------
        PipelineConfig
            Parsed pipeline configuration
        """
        return cls.from_dict(json.loads(payload))


@dataclass
class DataConfig:
    """Data configuration for training.

    Parameters
    ----------
    cu3s_file_path : str
        Path to .cu3s file
    annotation_json_path : str | None
        Path to annotation JSON file (optional for unsupervised training)
    train_ids : list[int]
        Training sample IDs
    val_ids : list[int]
        Validation sample IDs
    test_ids : list[int]
        Test sample IDs
    batch_size : int
        Batch size for data loading
    processing_mode : str
        Processing mode ('Raw' or 'Reflectance')
    """

    cu3s_file_path: str
    annotation_json_path: str | None = None
    train_ids: list[int] = field(default_factory=list)
    val_ids: list[int] = field(default_factory=list)
    test_ids: list[int] = field(default_factory=list)
    batch_size: int = 1
    processing_mode: str = "Reflectance"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DataConfig:
        """Create from dictionary."""
        return cls(
            cu3s_file_path=data["cu3s_file_path"],
            annotation_json_path=data.get("annotation_json_path"),
            train_ids=list(data.get("train_ids", [])),
            val_ids=list(data.get("val_ids", [])),
            test_ids=list(data.get("test_ids", [])),
            batch_size=data.get("batch_size", 1),
            processing_mode=data.get("processing_mode", "Reflectance"),
        )

    def to_proto(self) -> Any:
        """Convert to proto DataConfig.

        Returns
        -------
        Any
            Proto DataConfig message
        """
        from cuvis_ai.grpc.v1 import cuvis_ai_pb2

        # Map processing mode string to proto enum
        processing_mode_map = {
            "Raw": cuvis_ai_pb2.PROCESSING_MODE_RAW,
            "Reflectance": cuvis_ai_pb2.PROCESSING_MODE_REFLECTANCE,
        }
        processing_mode = processing_mode_map.get(
            self.processing_mode, cuvis_ai_pb2.PROCESSING_MODE_REFLECTANCE
        )

        return cuvis_ai_pb2.DataConfig(
            cu3s_file_path=self.cu3s_file_path,
            annotation_json_path=self.annotation_json_path,
            train_ids=self.train_ids,
            val_ids=self.val_ids,
            test_ids=self.test_ids,
            batch_size=self.batch_size,
            processing_mode=processing_mode,
        )

    @classmethod
    def from_proto(cls, proto_config: Any) -> DataConfig:
        """Convert from proto DataConfig.

        Parameters
        ----------
        proto_config : Any
            Proto DataConfig message

        Returns
        -------
        DataConfig
            Python data configuration
        """
        from cuvis_ai.grpc.v1 import cuvis_ai_pb2

        # Map proto enum to processing mode string
        processing_mode_map = {
            cuvis_ai_pb2.PROCESSING_MODE_RAW: "Raw",
            cuvis_ai_pb2.PROCESSING_MODE_REFLECTANCE: "Reflectance",
        }
        processing_mode = processing_mode_map.get(proto_config.processing_mode, "Reflectance")

        return cls(
            cu3s_file_path=proto_config.cu3s_file_path,
            annotation_json_path=proto_config.annotation_json_path,
            train_ids=list(proto_config.train_ids),
            val_ids=list(proto_config.val_ids),
            test_ids=list(proto_config.test_ids),
            batch_size=proto_config.batch_size,
            processing_mode=processing_mode,
        )


@dataclass
class ExperimentConfig:
    """Complete experiment configuration for reproducibility.

    Parameters
    ----------
    name : str
        Experiment name
    pipeline : PipelineConfig
        Pipeline configuration
    data : DataConfig
        Data configuration
    training : TrainingConfig
        Training configuration
    output_dir : str
        Root directory for outputs (models, logs, etc.)
    unfreeze_nodes : list[str]
        List of node names to unfreeze for gradient training
    freeze_nodes : list[str]
        List of node names to keep frozen during training
    metric_nodes : list[str]
        List of metric node names used in the experiment
    loss_nodes : list[str]
        List of loss node names used in the experiment

    """

    name: str
    pipeline: PipelineConfig
    data: DataConfig
    training: TrainingConfig
    output_dir: str = "./outputs"
    unfreeze_nodes: list[str] = field(default_factory=list)
    freeze_nodes: list[str] = field(default_factory=list)
    metric_nodes: list[str] = field(default_factory=list)
    loss_nodes: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Convert DictConfig/dict to proper config objects if needed."""
        # Convert data to DataConfig if it's a dict/DictConfig
        if not isinstance(self.data, DataConfig):
            if isinstance(self.data, DictConfig):
                data_container = OmegaConf.to_container(self.data, resolve=True)
                assert isinstance(data_container, dict), "Expected dict from OmegaConf.to_container"
                data_dict: dict[str, Any] = data_container  # type: ignore[assignment]
                self.data = DataConfig.from_dict(data_dict)
            elif isinstance(self.data, dict):
                self.data = DataConfig.from_dict(self.data)

        # Convert pipeline to PipelineConfig if it's a dict/DictConfig
        if not isinstance(self.pipeline, PipelineConfig):
            if isinstance(self.pipeline, DictConfig):
                pipeline_container = OmegaConf.to_container(self.pipeline, resolve=True)
                assert isinstance(pipeline_container, dict), (
                    "Expected dict from OmegaConf.to_container"
                )
                pipeline_dict: dict[str, Any] = pipeline_container  # type: ignore[assignment]
                self.pipeline = PipelineConfig.from_dict(pipeline_dict)
            elif isinstance(self.pipeline, dict):
                self.pipeline = PipelineConfig.from_dict(self.pipeline)

        # Convert training to TrainingConfig if it's a dict/DictConfig
        if not isinstance(self.training, TrainingConfig):
            if isinstance(self.training, DictConfig):
                self.training = TrainingConfig.from_dict_config(self.training)
            elif isinstance(self.training, dict):
                self.training = TrainingConfig.from_dict_config(self.training)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "pipeline": self.pipeline.to_dict(),
            "data": self.data.to_dict(),
            "training": self.training.to_dict(),
            "output_dir": self.output_dir,
            "unfreeze_nodes": self.unfreeze_nodes,
            "freeze_nodes": self.freeze_nodes,
            "metric_nodes": self.metric_nodes,
            "loss_nodes": self.loss_nodes,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExperimentConfig:
        """Create from dictionary."""
        return cls(
            name=data["name"],
            pipeline=PipelineConfig.from_dict(data["pipeline"]),
            data=DataConfig.from_dict(data["data"]),
            training=TrainingConfig.from_dict_config(data["training"]),
            output_dir=data.get("output_dir", "./outputs"),
            unfreeze_nodes=list(data.get("unfreeze_nodes", [])),
            freeze_nodes=list(data.get("freeze_nodes", [])),
            metric_nodes=list(data.get("metric_nodes", [])),
            loss_nodes=list(data.get("loss_nodes", [])),
        )

    @classmethod
    def from_config(cls, cfg: DictConfig) -> ExperimentConfig:
        """Create from OmegaConf DictConfig.

        Parameters
        ----------
        cfg : DictConfig
            OmegaConf configuration object

        Returns
        -------
        ExperimentConfig
            Experiment configuration instance
        """
        # Convert to plain dict for existing from_dict infrastructure
        config_dict = OmegaConf.to_container(cfg, resolve=True)
        return cls.from_dict(config_dict)  # type: ignore

    def save_to_file(self, path: str) -> None:
        """Save experiment config to YAML file.

        Parameters
        ----------
        path : str
            Path to save YAML file
        """
        from pathlib import Path

        import yaml

        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with output_path.open("w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)

    @classmethod
    def load_from_file(cls, path: str, overrides: list[str] | None = None) -> ExperimentConfig:
        """Load experiment config from YAML file with Hydra composition support.

        Automatically locates the repo's config root directory and uses
        Hydra's composition system to resolve nested configs (e.g., defaults).
        If no 'configs' directory is found (e.g., in test scenarios), falls back
        to simple YAML loading without Hydra composition.

        Parameters
        ----------
        path : str
            Path to YAML config file
        overrides : list[str] | None
            Hydra-style config overrides in dot notation.
            Examples: ["output_dir=outputs/custom", "data.batch_size=16"]

        Returns
        -------
        ExperimentConfig
            Loaded experiment configuration

        Examples
        --------
        >>> config = ExperimentConfig.load_from_file(
        ...     "configs/experiment/deep_svdd.yaml",
        ...     overrides=["output_dir=outputs/custom", "training.optimizer.lr=0.001"]
        ... )
        """
        from hydra import compose, initialize_config_dir
        from hydra.core.global_hydra import GlobalHydra

        config_path = Path(path).resolve()

        # Check if file exists
        if not config_path.exists():
            raise FileNotFoundError(f"Experiment file not found: {config_path}")

        # Find the repo's config root directory
        # Traverse up from the config file to find the 'configs' directory
        config_root = config_path.parent
        while config_root.name != "configs" and config_root.parent != config_root:
            config_root = config_root.parent

        # If no configs directory found, fall back to simple YAML loading
        if config_root.name != "configs":
            # Fallback: Load YAML directly without Hydra composition
            with config_path.open() as f:
                data: dict[str, Any] = yaml.safe_load(f)

            # Apply overrides manually if provided
            if overrides:
                data = cls._apply_overrides(data, overrides)

            return cls.from_dict(data)

        # Hydra-based loading for configs in the standard directory structure
        # Get relative path from config root (e.g., "experiment/deep_svdd")
        relative_path = config_path.relative_to(config_root)
        config_name = str(relative_path.with_suffix("")).replace("\\", "/")

        # Clean up any existing Hydra instance
        GlobalHydra.instance().clear()

        try:
            with initialize_config_dir(config_dir=str(config_root.absolute()), version_base=None):
                cfg = compose(config_name=config_name, overrides=overrides or [])

            # Convert OmegaConf to dict and create ExperimentConfig
            data = OmegaConf.to_container(cfg, resolve=True)  # type: ignore[assignment]
            assert isinstance(data, dict), "Expected dict from OmegaConf.to_container"

            # Hydra may nest the config under a group name (e.g., {'experiment': {...}})
            # Only unwrap if we have exactly one key and that key is NOT an expected experiment field
            # This ensures we don't unwrap valid experiment configs that happen to have one top-level field
            if len(data) == 1:
                single_key = list(data.keys())[0]
                # Only unwrap if the single key is a group name (not an experiment config field)
                if single_key not in [
                    "name",
                    "pipeline",
                    "data",
                    "training",
                    "output_dir",
                    "unfreeze_nodes",
                    "freeze_nodes",
                    "metric_nodes",
                    "loss_nodes",
                    "defaults",
                ]:
                    nested_data = data[single_key]
                    if isinstance(nested_data, dict) and "name" in nested_data:
                        # This looks like a wrapped experiment config
                        data = nested_data

            return cls.from_dict(data)
        finally:
            # Always clean up Hydra singleton
            GlobalHydra.instance().clear()

    @staticmethod
    def _apply_overrides(data: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
        """Apply Hydra-style overrides to a config dict.

        Parameters
        ----------
        data : dict[str, Any]
            Configuration dictionary
        overrides : list[str]
            List of overrides in dot notation (e.g., "data.batch_size=16")

        Returns
        -------
        dict[str, Any]
            Configuration dictionary with overrides applied
        """
        for override in overrides:
            if "=" not in override:
                continue

            key_path, value_str = override.split("=", 1)
            keys = key_path.split(".")

            # Navigate to the nested dict
            current = data
            for key in keys[:-1]:
                if key not in current:
                    current[key] = {}
                current = current[key]

            # Try to parse the value as the appropriate type
            final_key = keys[-1]
            try:
                # Try parsing as number
                if "." in value_str:
                    current[final_key] = float(value_str)
                else:
                    current[final_key] = int(value_str)
            except ValueError:
                # Try parsing as boolean
                if value_str.lower() in {"true", "false"}:
                    current[final_key] = value_str.lower() == "true"
                else:
                    # Keep as string
                    current[final_key] = value_str

        return data

    def to_proto(self) -> Any:
        """Convert to proto ExperimentConfig.

        Returns
        -------
        Any
            Proto ExperimentConfig message
        """
        from cuvis_ai.grpc.v1 import cuvis_ai_pb2

        training_json = self.training.to_json()

        return cuvis_ai_pb2.ExperimentConfig(
            name=self.name,
            pipeline=self.pipeline.to_proto(),
            data=self.data.to_proto(),
            training=cuvis_ai_pb2.TrainingConfig(config_bytes=training_json.encode("utf-8")),
        )

    @classmethod
    def from_proto(cls, proto_config: Any) -> ExperimentConfig:
        """Convert from proto ExperimentConfig.

        Parameters
        ----------
        proto_config : Any
            Proto ExperimentConfig message

        Returns
        -------
        ExperimentConfig
            Python experiment configuration
        """
        training_json = proto_config.training.config_bytes.decode("utf-8")
        training_config = TrainingConfig.from_json(training_json)

        return cls(
            name=proto_config.name,
            pipeline=PipelineConfig.from_proto(proto_config.pipeline),
            data=DataConfig.from_proto(proto_config.data),
            training=training_config,
        )


__all__ = [
    "EarlyStoppingConfig",
    "ModelCheckpointConfig",
    "LearningRateMonitorConfig",
    "CallbacksConfig",
    "create_callbacks_from_config",
    "TrainerConfig",
    "SchedulerConfig",
    "OptimizerConfig",
    "TrainingConfig",
    "PipelineMetadata",
    "PipelineConfig",
    "DataConfig",
    "ExperimentConfig",
]
