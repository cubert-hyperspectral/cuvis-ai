from __future__ import annotations

from pathlib import Path

import torch

from cuvis_ai.deciders.binary_decider import QuantileBinaryDecider


def _make_linear_map() -> torch.Tensor:
    return torch.arange(1, 101, dtype=torch.float32).reshape(1, 10, 10, 1)


def _multi_dim_quantile(tensor: torch.Tensor, q: float, dims: tuple[int, ...]) -> torch.Tensor:
    ndim = tensor.dim()
    dims = tuple(sorted(dims))
    keep = tuple(i for i in range(ndim) if i not in dims)
    perm = (*keep, *dims)
    permuted = tensor.permute(*perm)
    sizes_keep = [permuted.size(i) for i in range(len(keep))]
    flattened = permuted.reshape(*sizes_keep, -1)
    thresh_flat = torch.quantile(flattened, q, dim=len(keep), keepdim=True)
    thresh_perm = thresh_flat.reshape(*sizes_keep, *([1] * len(dims)))
    inverse = [0] * ndim
    for original_idx, permuted_idx in enumerate(perm):
        inverse[permuted_idx] = original_idx
    return thresh_perm.permute(*inverse)


def test_quantile_binary_decider_selects_top_fraction():
    tensor = _make_linear_map()
    decider = QuantileBinaryDecider(quantile=0.995)

    mask = decider.forward(logits=tensor)["decisions"]
    threshold = _multi_dim_quantile(tensor, decider.quantile, dims=(1, 2, 3))

    assert mask.sum().item() == 1
    assert mask[0, -1, -1, 0]
    assert mask.dtype == torch.bool
    expected_mask = tensor >= threshold
    assert torch.equal(mask, expected_mask)


def test_quantile_binary_decider_matches_torch_quantile():
    tensor = _make_linear_map()
    quantiles = [0.1, 0.5, 0.9]

    for q in quantiles:
        decider = QuantileBinaryDecider(quantile=q)
        mask = decider.forward(logits=tensor)["decisions"]
        q_threshold = _multi_dim_quantile(tensor, q, dims=(1, 2, 3))
        expected_mask = tensor >= q_threshold
        assert torch.equal(mask, expected_mask)


def test_quantile_binary_decider_serialization_roundtrip(tmp_path: Path):
    tensor = _make_linear_map()
    decider = QuantileBinaryDecider(quantile=0.8, reduce_dims=(1, 2, 3))

    original = decider.forward(logits=tensor)["decisions"]

    # PyTorch handles the module state via state_dict; hparams capture init kwargs
    state_path = tmp_path / "decider_state.pt"
    torch.save(decider.state_dict(), state_path)

    assert decider.hparams["quantile"] == decider.quantile
    assert decider.hparams["reduce_dims"] == decider.reduce_dims

    restored = QuantileBinaryDecider(**decider.hparams)
    state = torch.load(state_path)
    restored.load_state_dict(state)

    recreated = restored.forward(logits=tensor)["decisions"]
    assert torch.equal(original, recreated)
    assert restored.quantile == decider.quantile
    assert restored.reduce_dims == decider.reduce_dims
