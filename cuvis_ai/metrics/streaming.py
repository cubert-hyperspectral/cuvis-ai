from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from cuvis_ai.utils.general import to_numpy_np


@dataclass
class StreamingBinaryMetrics:
    """Accumulate predictions and labels to compute AUROC and AUPRC."""

    _scores: list[np.ndarray] = field(default_factory=list)
    _labels: list[np.ndarray] = field(default_factory=list)

    def update(self, scores: Any, mask: Any | None) -> dict[str, float | None]:
        """Update metric buffers with the latest batch.

        Args:
            scores: Tensor/array shaped (B, H, W) or (B, H, W, C) containing anomaly scores.
            mask: Optional tensor/array shaped like scores indicating ground-truth labels.

        Returns:
            Dict containing current AUROC and AUPRC (None when not computable).
        """
        if mask is None:
            return self.compute()

        scores_np = to_numpy_np(scores)
        labels_np = to_numpy_np(mask)

        scores_flat = scores_np.reshape(-1)
        labels_flat = labels_np.reshape(-1)

        binary_labels = (labels_flat > 0).astype(np.float32)

        valid = np.isfinite(scores_flat) & np.isfinite(binary_labels)
        if not np.any(valid):
            return self.compute()

        self._scores.append(scores_flat[valid])
        self._labels.append(binary_labels[valid])

        return self.compute()

    def compute(self) -> dict[str, float | None]:
        if not self._labels:
            return {"auroc": None, "auprc": None}

        labels = np.concatenate(self._labels)
        scores = np.concatenate(self._scores)

        unique_labels = np.unique(labels)
        if unique_labels.size < 2:
            return {"auroc": None, "auprc": None}

        auroc = roc_auc_score(labels, scores)
        auprc = average_precision_score(labels, scores)
        return {"auroc": float(auroc), "auprc": float(auprc)}

    def reset(self) -> None:
        self._scores.clear()
        self._labels.clear()
