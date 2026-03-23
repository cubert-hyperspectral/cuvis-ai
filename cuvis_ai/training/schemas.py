"""Local schema overrides for multi-file dataset support.

Temporary workaround until ``cuvis-ai-schemas`` DataConfig is extended
upstream to support ``splits_csv``.  See:
    docs/tasks/dataconfig-multi-file-upstream.md
"""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

import yaml
from cuvis_ai_schemas.base import BaseSchemaModel
from cuvis_ai_schemas.pipeline.config import PipelineConfig
from cuvis_ai_schemas.training.config import TrainingConfig
from pydantic import ConfigDict, Field, model_validator


class MultiFileDataConfig(BaseSchemaModel):
    """Data config supporting both single-file and multi-file (splits CSV) sources.

    Exactly one of ``cu3s_file_path`` or ``splits_csv`` must be provided.
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)
    __proto_message__: ClassVar[str] = "DataConfig"

    cu3s_file_path: str | None = Field(default=None, description="Path to single .cu3s file")
    splits_csv: str | None = Field(
        default=None, description="Path to splits CSV for multi-file datasets"
    )
    annotation_json_path: str | None = Field(
        default=None, description="Path to annotation JSON (optional)"
    )
    train_ids: list[int] = Field(default_factory=list, description="Training sample IDs")
    val_ids: list[int] = Field(default_factory=list, description="Validation sample IDs")
    test_ids: list[int] = Field(default_factory=list, description="Test sample IDs")
    train_split: float | None = Field(
        default=None, ge=0.0, le=1.0, description="Training split ratio"
    )
    val_split: float | None = Field(
        default=None, ge=0.0, le=1.0, description="Validation split ratio"
    )
    shuffle: bool = Field(default=True, description="Shuffle dataset")
    batch_size: int = Field(default=1, ge=1, description="Batch size")
    processing_mode: str = Field(default="Reflectance", description="Raw or Reflectance mode")

    @model_validator(mode="after")
    def _validate_data_source(self) -> MultiFileDataConfig:
        """Ensure exactly one data source is specified."""
        if not self.cu3s_file_path and not self.splits_csv:
            raise ValueError("Either 'cu3s_file_path' or 'splits_csv' must be provided")
        if self.cu3s_file_path and self.splits_csv:
            raise ValueError(
                "Provide only one of 'cu3s_file_path' (single-file) or 'splits_csv' (multi-file)"
            )
        return self


class MultiFileTrainRunConfig(BaseSchemaModel):
    """TrainRunConfig that accepts MultiFileDataConfig.

    Drop-in replacement for ``cuvis_ai_schemas.training.TrainRunConfig``
    when using multi-file datasets. Remove once the upstream schema is extended.
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)
    __proto_message__: ClassVar[str] = "TrainRunConfig"

    name: str = Field(description="Train run identifier")
    pipeline: PipelineConfig | None = Field(
        default=None, description="Pipeline configuration (optional if already built)"
    )
    data: MultiFileDataConfig = Field(description="Data configuration")
    training: TrainingConfig | None = Field(
        default=None, description="Training configuration (required if gradient training)"
    )
    loss_nodes: list[str] = Field(
        default_factory=list, description="Loss node names for gradient training"
    )
    metric_nodes: list[str] = Field(
        default_factory=list, description="Metric node names for monitoring"
    )
    freeze_nodes: list[str] = Field(
        default_factory=list, description="Nodes to freeze for this training run (runtime action)"
    )
    unfreeze_nodes: list[str] = Field(
        default_factory=list, description="Nodes to unfreeze for this training run (runtime action)"
    )
    output_dir: str = Field(default="./outputs", description="Output directory for artifacts")
    tags: dict[str, str] = Field(default_factory=dict, description="Metadata tags for tracking")

    def save_to_file(self, path: str | Path) -> None:
        """Save configuration to YAML file."""
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False)

    @classmethod
    def load_from_file(cls, path: str | Path) -> MultiFileTrainRunConfig:
        """Load configuration from YAML file."""
        with Path(path).open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)

    @model_validator(mode="after")
    def _validate_training_config(self) -> MultiFileTrainRunConfig:
        """Ensure training config has optimizer if provided."""
        if self.training is not None and self.training.optimizer is None:
            raise ValueError("Training configuration must include optimizer when provided")
        return self


__all__ = ["MultiFileDataConfig", "MultiFileTrainRunConfig"]
