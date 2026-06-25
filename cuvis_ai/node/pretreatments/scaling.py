"""Global mean-centring and unit-variance scaling pretreatment nodes.

Both nodes fit per-channel statistics once during
``statistical_initialization`` by streaming every training pixel through a
:class:`~cuvis_ai.utils.welford.WelfordAccumulator` (exact full-data moments,
no subsampling), then apply the frozen statistics at inference time.
"""

import torch
from cuvis_ai_schemas.enums import NodeCategory, NodeTag
from cuvis_ai_schemas.execution import InputStream
from cuvis_ai_schemas.pipeline import PortSpec

from cuvis_ai.node._statistical_fit import _StatisticalFitNode
from cuvis_ai.utils.welford import WelfordAccumulator


class MeanCenter(_StatisticalFitNode):
    """Subtract a globally-fitted per-channel mean from the cube.

    During ``statistical_initialization`` every training pixel is streamed
    through a Welford accumulator to compute the exact per-channel mean over
    the full dataset; ``forward`` then subtracts that mean from each spectrum.

    Notes
    -----
    The fitted ``mean_c`` is registered as a persistent buffer so a
    checkpointed node reloads ready for inference.
    """

    _category = NodeCategory.TRANSFORM
    _tags = frozenset(
        {NodeTag.HYPERSPECTRAL, NodeTag.PREPROCESSING, NodeTag.NORMALIZATION, NodeTag.TORCH}
    )

    OUTPUT_SPECS = {
        "cube": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, -1),
            description="Mean-centred cube [B, H, W, C]",
        )
    }

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.register_buffer("mean_c", torch.zeros(0, dtype=torch.float32))
        self._welford: WelfordAccumulator | None = None

    def statistical_initialization(self, input_stream: InputStream) -> None:
        """Fit the per-channel mean from the training stream via Welford.

        Parameters
        ----------
        input_stream : InputStream
            Iterable of port-keyed batch dicts matching ``INPUT_SPECS``.
        """
        self._welford = None
        for batch in input_stream:
            x = batch.get("cube")
            if x is None:
                continue
            flat = x.reshape(-1, x.shape[-1]).to(torch.float32)
            if self._welford is None:
                self._welford = WelfordAccumulator(flat.shape[-1], track_covariance=False).to(
                    device=flat.device
                )
            self._welford.update(flat)

        count = 0 if self._welford is None else self._welford.count
        self._reject_if_insufficient(count)
        self.mean_c = self._welford.mean
        self._mark_initialized()

    def forward(self, cube: torch.Tensor, **_) -> dict[str, torch.Tensor]:
        """Subtract the fitted per-channel mean from the cube.

        Parameters
        ----------
        cube : torch.Tensor
            Input cube in BHWC format.

        Returns
        -------
        dict[str, torch.Tensor]
            ``{"cube": centred}`` with the same shape as the input.
        """
        self._require_initialized()
        return {"cube": cube - self.mean_c}


class UnitVarianceScaling(_StatisticalFitNode):
    """Divide the cube by a globally-fitted per-channel standard deviation.

    During ``statistical_initialization`` every training pixel is streamed
    through a Welford accumulator to compute the exact per-channel standard
    deviation (sample, ``ddof=1``) over the full dataset; ``forward`` then
    divides each spectrum by that standard deviation.

    Notes
    -----
    The fitted ``std_c`` is registered as a persistent buffer so a
    checkpointed node reloads ready for inference.
    """

    _category = NodeCategory.TRANSFORM
    _tags = frozenset(
        {NodeTag.HYPERSPECTRAL, NodeTag.PREPROCESSING, NodeTag.NORMALIZATION, NodeTag.TORCH}
    )

    OUTPUT_SPECS = {
        "cube": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, -1),
            description="Unit-variance-scaled cube [B, H, W, C]",
        )
    }

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.register_buffer("std_c", torch.zeros(0, dtype=torch.float32))
        self._welford: WelfordAccumulator | None = None

    def statistical_initialization(self, input_stream: InputStream) -> None:
        """Fit the per-channel standard deviation from the stream via Welford.

        Parameters
        ----------
        input_stream : InputStream
            Iterable of port-keyed batch dicts matching ``INPUT_SPECS``.
        """
        self._welford = None
        for batch in input_stream:
            x = batch.get("cube")
            if x is None:
                continue
            flat = x.reshape(-1, x.shape[-1]).to(torch.float32)
            if self._welford is None:
                self._welford = WelfordAccumulator(flat.shape[-1], track_covariance=False).to(
                    device=flat.device
                )
            self._welford.update(flat)

        count = 0 if self._welford is None else self._welford.count
        self._reject_if_insufficient(count)
        self.std_c = self._welford.std
        self._mark_initialized()

    def forward(self, cube: torch.Tensor, **_) -> dict[str, torch.Tensor]:
        """Divide the cube by the fitted per-channel standard deviation.

        Parameters
        ----------
        cube : torch.Tensor
            Input cube in BHWC format.

        Returns
        -------
        dict[str, torch.Tensor]
            ``{"cube": scaled}`` with the same shape as the input.
        """
        self._require_initialized()
        return {"cube": cube / self.std_c.clamp_min(1e-8)}
