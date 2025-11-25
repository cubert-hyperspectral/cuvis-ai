"""Custom PyTorch Lightning callbacks for streaming gRPC progress."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from pytorch_lightning import Callback, LightningModule, Trainer

from cuvis_ai.utils.types import Context, ExecutionStage


class ProgressStreamCallback(Callback):
    """Callback that forwards training progress to a handler function."""

    def __init__(
        self, progress_handler: Callable[[Context, dict[str, float], dict[str, float], str], None]
    ) -> None:  # noqa: E501
        super().__init__()
        self.progress_handler = progress_handler
        self._current_epoch = 0
        self._current_batch = 0
        self._global_step = 0

    # ----------------------------- Lifecycle hooks -----------------------------
    def on_train_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self._current_epoch = trainer.current_epoch
        self._emit_progress(ExecutionStage.TRAIN, {}, {}, "running")

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        self._current_batch = batch_idx
        self._global_step = trainer.global_step
        losses = self._extract_losses(outputs)
        metrics = self._extract_metrics(trainer, "train")
        self._emit_progress(ExecutionStage.TRAIN, losses, metrics, "running")

    def on_validation_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self._emit_progress(ExecutionStage.VAL, {}, {}, "running")

    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        losses = self._extract_losses(outputs)
        metrics = self._extract_metrics(trainer, "val")
        self._emit_progress(ExecutionStage.VAL, losses, metrics, "running")

    def on_test_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self._emit_progress(ExecutionStage.TEST, {}, {}, "running")

    def on_test_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        losses = self._extract_losses(outputs)
        metrics = self._extract_metrics(trainer, "test")
        self._emit_progress(ExecutionStage.TEST, losses, metrics, "running")

    def on_train_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self._emit_progress(ExecutionStage.TRAIN, {}, {}, "complete")

    # ----------------------------- Helpers -----------------------------
    def _extract_losses(self, outputs: Any) -> dict[str, float]:
        """Extract loss values from batch outputs."""
        losses: dict[str, float] = {}

        if outputs is None:
            return losses

        if isinstance(outputs, dict):
            if "loss" in outputs:
                losses["total"] = float(outputs["loss"])
            for key, value in outputs.items():
                if "loss" in key.lower() and key != "loss":
                    try:
                        losses[key] = float(value)
                    except Exception:
                        continue
        elif hasattr(outputs, "item"):
            losses["total"] = float(outputs.item())

        return losses

    def _extract_metrics(self, trainer: Trainer, stage: str) -> dict[str, float]:
        """Extract metrics from trainer.callback_metrics."""
        metrics: dict[str, float] = {}
        callback_metrics = getattr(trainer, "callback_metrics", {}) or {}

        for key, value in callback_metrics.items():
            key_lower = key.lower()
            if "loss" in key_lower:
                continue

            if stage in key_lower:
                metric_name = key.replace(f"{stage}_", "").replace(f"{stage}/", "")
            else:
                metric_name = key

            try:
                metrics[metric_name] = float(value.item())
            except Exception:
                try:
                    metrics[metric_name] = float(value)
                except Exception:
                    continue

        return metrics

    def _emit_progress(
        self,
        stage: ExecutionStage,
        losses: dict[str, float],
        metrics: dict[str, float],
        status: str,
    ) -> None:
        context = Context(
            stage=stage,
            epoch=self._current_epoch,
            batch_idx=self._current_batch,
            global_step=self._global_step,
        )
        self.progress_handler(context, losses, metrics, status)


__all__ = ["ProgressStreamCallback"]
