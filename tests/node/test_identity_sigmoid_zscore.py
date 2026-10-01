"""IdentityNormalizer, SigmoidTransform and ZScoreNormalizer: values and recorded hparams."""

from __future__ import annotations

import pytest
import torch

from cuvis_ai.node.normalization import IdentityNormalizer, SigmoidTransform, ZScoreNormalizer

pytestmark = pytest.mark.unit


def test_identity_normalizer_returns_its_input_and_records_no_hparams() -> None:
    node = IdentityNormalizer()
    data = torch.rand(1, 3, 4, 2)
    assert torch.equal(node.forward(data=data)["normalized"], data)
    assert dict(node.hparams) == {}


def test_sigmoid_transform_matches_torch_sigmoid_and_records_no_hparams() -> None:
    node = SigmoidTransform()
    data = torch.randn(1, 3, 4, 2)
    torch.testing.assert_close(node.forward(data=data)["transformed"], torch.sigmoid(data))
    assert dict(node.hparams) == {}


def test_zscore_normalizer_standardizes_over_its_dims() -> None:
    data = torch.arange(24.0).reshape(1, 3, 4, 2) ** 1.5
    node = ZScoreNormalizer()
    out = node.forward(data=data)["normalized"]
    mean = data.mean(dim=[1, 2], keepdim=True)
    std = data.std(dim=[1, 2], keepdim=True, unbiased=False)
    torch.testing.assert_close(out, (data - mean) / (std + 1e-6))
    assert dict(node.hparams) == {"dims": [1, 2], "eps": 1e-6, "keepdim": True}
