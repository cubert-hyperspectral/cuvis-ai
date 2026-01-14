from collections.abc import Iterator
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np

InputStream = Iterator[dict[str, Any]]


class ExecutionStage(str, Enum):
    """Execution stages for node filtering.

    Nodes can specify which stages they should execute in to enable
    stage-aware graph execution (e.g., loss nodes only in training).
    """

    ALWAYS = "always"
    TRAIN = "train"
    VAL = "val"
    VALIDATE = "val"
    TEST = "test"
    INFERENCE = "inference"


@dataclass
class Context:
    """Execution context passed to executor and nodes.

    Contains runtime information that doesn't flow through data edges.
    This replaces mutable global state with explicit context parameters.

    Attributes
    ----------
    stage : str
        Execution stage: "train", "val", "test", "inference"
    epoch : int
        Current training epoch
    batch_idx : int
        Current batch index within epoch
    global_step : int
        Global training step across all epochs

    Examples
    --------
    >>> context = Context(stage="train", epoch=5, batch_idx=42, global_step=1337)
    >>> executor.forward(context=context, batch=batch)

    Notes
    -----
    Future extensions for distributed training:
    - rank: int (process rank in distributed training)
    - world_size: int (total number of processes)
    """

    stage: ExecutionStage = ExecutionStage.INFERENCE
    epoch: int = 0
    batch_idx: int = 0
    global_step: int = 0


class ArtifactType(str, Enum):
    """Types of artifacts with different validation/logging policies.

    Attributes
    ----------
    IMAGE : str
        Image artifact - expects shape (H, W, 1) monocular or (H, W, 3) RGB
    """

    IMAGE = "image"


@dataclass
class Artifact:
    """Artifact for logging visualizations and data to monitoring systems.

    Attributes
    ----------
    name : str
        Name/identifier for the artifact
    value : np.ndarray
        Numpy array containing the artifact data (shape validated by type)
    el_id : int
        Element ID (e.g., batch item index, image index)
    desc : str
        Human-readable description of the artifact
    type : ArtifactType
        Type of artifact, determines validation and logging policy

    Examples
    --------
    >>> artifact = Artifact(
    ...     name="heatmap_img_0",
    ...     value=np.random.rand(256, 256, 3),
    ...     el_id=0,
    ...     desc="Anomaly heatmap for first image",
    ...     type=ArtifactType.IMAGE
    ... )
    """

    name: str
    value: np.ndarray
    el_id: int
    desc: str
    type: ArtifactType
    stage: ExecutionStage = ExecutionStage.INFERENCE
    epoch: int = 0
    batch_idx: int = 0


@dataclass
class Metric:
    """Metric for logging scalar values to monitoring systems.

    Attributes
    ----------
    name : str
        Name/identifier for the metric
    value : float
        Scalar metric value

    Examples
    --------
    >>> metric = Metric(name="loss/train", value=0.123)
    """

    name: str
    value: float
    stage: ExecutionStage = ExecutionStage.INFERENCE
    epoch: int = 0
    batch_idx: int = 0
