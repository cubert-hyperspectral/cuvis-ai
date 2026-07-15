"""Tests for DistinctLabelCount: per-frame count of distinct non-zero labels."""

from __future__ import annotations

import torch
from cuvis_ai_schemas.enums import ExecutionStage
from cuvis_ai_schemas.execution import Context

from cuvis_ai.node.metrics import DistinctLabelCount

CTX = Context(stage=ExecutionStage.INFERENCE, epoch=0, batch_idx=0)


def test_counts_distinct_nonzero_labels():
    """Background (0) is ignored; each distinct positive id counts once."""
    mask = torch.zeros(1, 4, 4, dtype=torch.int32)
    mask[0, 0, 0] = 1
    mask[0, 1, 1] = 2
    mask[0, 2, 2] = 2  # repeat of label 2 -> still one distinct label
    mask[0, 3, 3] = 5
    out = DistinctLabelCount().forward(mask=mask, context=CTX)
    assert out["count"].tolist() == [3]  # labels {1, 2, 5}
    assert out["count"].dtype == torch.int64


def test_all_background_is_zero():
    out = DistinctLabelCount().forward(mask=torch.zeros(1, 3, 3, dtype=torch.int32), context=CTX)
    assert out["count"].tolist() == [0]


def test_batch_counts_per_frame():
    mask = torch.zeros(2, 3, 3, dtype=torch.int32)
    mask[0, 0, 0] = 1  # frame 0: one label
    mask[1, 0, 0] = 7
    mask[1, 1, 1] = 9  # frame 1: two labels
    out = DistinctLabelCount().forward(mask=mask, context=CTX)
    assert out["count"].tolist() == [1, 2]


def test_emits_metric_per_frame():
    mask = torch.zeros(1, 3, 3, dtype=torch.int32)
    mask[0, 0, 0] = 1
    mask[0, 1, 1] = 2
    out = DistinctLabelCount().forward(mask=mask, context=CTX)
    assert [m.name for m in out["metrics"]] == ["num_distinct_labels"]
    assert out["metrics"][0].value == 2.0


def test_runs_in_inference_stage_by_default():
    """Defaulting to ALWAYS keeps the node active under Predictor inference."""
    node = DistinctLabelCount()
    assert ExecutionStage.ALWAYS in node.execution_stages
