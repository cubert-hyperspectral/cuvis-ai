"""IdentityNormalizer and SigmoidTransform: values and recorded hparams."""

from __future__ import annotations

import pytest
import torch

from cuvis_ai.node.normalization import IdentityNormalizer, SigmoidTransform

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
