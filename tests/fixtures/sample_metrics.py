"""Test-friendly custom metrics node used in pipeline fixtures."""

from __future__ import annotations

import sys

import torch

from cuvis_ai_core.node.node import Node
from cuvis_ai_core.pipeline.ports import PortSpec
from cuvis_ai_core.utils.types import Context, ExecutionStage, Metric


class SampleCustomMetrics(Node):
    """Lightweight metrics node to satisfy custom class references in configs."""

    INPUT_SPECS = {
        "decisions": PortSpec(
            dtype=torch.bool,
            shape=(-1, -1, -1, -1),
            description="Binary decisions mask",
        )
    }
    OUTPUT_SPECS = {
        "metrics": PortSpec(
            dtype=list,
            shape=(),
            description="Custom metrics list",
        )
    }

    def forward(self, decisions, context: Context | None = None, **kwargs) -> dict:
        """Compute a simple mean-of-decisions metric."""
        ctx = context or Context()
        decision_tensor = torch.as_tensor(decisions)
        mean_decision = float(decision_tensor.float().mean().item())

        metric = Metric(
            name="sample/mean_decision",
            value=mean_decision,
            stage=ctx.stage if ctx.stage is not None else ExecutionStage.INFERENCE,
            epoch=ctx.epoch,
            batch_idx=ctx.batch_idx,
        )
        return {"metrics": [metric]}


# Make the class discoverable via the legacy path used in some configs
sys.modules["__main__"].SampleCustomMetrics = SampleCustomMetrics
