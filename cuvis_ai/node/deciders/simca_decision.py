"""DD-SIMCA decision: two distances in, one calibrated anomaly score and a decision out."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import torch
from cuvis_ai_schemas.enums import NodeCategory, NodeTag
from cuvis_ai_schemas.execution import InputStream
from cuvis_ai_schemas.pipeline import PortSpec
from torch import Tensor

from cuvis_ai.utils.welford import WelfordAccumulator
from cuvis_ai_core.node import Node

# Bounds of the degrees of freedom, as in the reference implementations of DD-SIMCA.
_MIN_DOF = 1.0
_MAX_DOF = 250.0
_DISTANCE_PORTS = ("score_distance", "orthogonal_distance")


def _valid_alpha(value: float) -> float:
    """Return ``alpha`` as a float, or raise when it is not strictly between 0 and 1."""
    alpha = float(value)
    if not (math.isfinite(alpha) and 0.0 < alpha < 1.0):
        raise ValueError(f"SIMCADecision: alpha must lie strictly between 0 and 1, got {value!r}.")
    return alpha


class SIMCADecision(Node):
    """Combine a score distance and an orthogonal distance into one decision (DD-SIMCA).

    Each distance ``u`` of normal data is modelled as a scaled chi-square variable,
    ``N u / u0 ~ chi2(N)``. The statistical fit estimates both parameters per distance by the
    method of moments: ``u0`` is the mean and ``N = 2 (u0 / sd)^2``, rounded and kept within
    1 to 250. The combined distance

        ``f = N_h h / h0 + N_q q / q0``

    then follows ``chi2(N_h + N_q)`` for normal pixels, as far as the two distances are
    independent, and a pixel is flagged when ``f`` exceeds the ``1 - alpha`` quantile of that
    distribution. ``alpha`` is the share of normal pixels the limit gives up: at 0.01 one
    normal pixel in a hundred is flagged.

    ``combined_distance`` is the anomaly score (larger is more anomalous) and works without
    the limit, for a ranking metric or for a fixed threshold from calibration data
    (``ScoreThreshold`` on ``combined_distance`` in place of ``decisions``).

    The node takes any two non-negative distances, so a different model of normal can supply
    either. The fit must see normal data only. The limit is recomputed from ``alpha`` whenever
    weights are loaded, so changing ``alpha`` in a saved pipeline changes the decisions
    without a new fit.

    Parameters
    ----------
    alpha : float
        Significance level, strictly between 0 and 1 (default 0.01).

    References
    ----------
    Pomerantsev, A. L., and Rodionova, O. Ye. (2014). Concept and role of extreme objects in
    PCA/SIMCA. Journal of Chemometrics, 28, 429-438.
    """

    _category = NodeCategory.TRANSFORM
    _tags = frozenset(
        {
            NodeTag.ONNX,
            NodeTag.ANOMALY,
            NodeTag.CLASSIFICATION,
            NodeTag.POSTPROCESSING,
            NodeTag.STATEFUL,
            NodeTag.TORCH,
        }
    )

    INPUT_SPECS = {
        "score_distance": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, 1),
            description="Distance inside the model of normal, e.g. Hotelling T2 [B, H, W, 1]",
        ),
        "orthogonal_distance": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, 1),
            description="Distance off the model of normal, e.g. the Q residual [B, H, W, 1]",
        ),
    }

    OUTPUT_SPECS = {
        "combined_distance": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, 1),
            description="Anomaly score f = N_h h / h0 + N_q q / q0, larger is more anomalous "
            "[B, H, W, 1]",
        ),
        "decisions": PortSpec(
            dtype=torch.bool,
            shape=(-1, -1, -1, 1),
            description="Pixels whose combined distance exceeds the limit at alpha [B, H, W, 1]",
        ),
    }

    def __init__(self, alpha: float = 0.01, **kwargs: Any) -> None:
        self.alpha = _valid_alpha(alpha)

        super().__init__(alpha=self.alpha, **kwargs)

        for name in ("h0", "n_h", "q0", "n_q", "threshold"):
            self.register_buffer(name, torch.ones((), dtype=torch.float32))

    def statistical_initialization(self, input_stream: InputStream) -> None:
        """Fit the scaled chi-square parameters of both distances on normal data.

        Parameters
        ----------
        input_stream : InputStream
            Batches keyed by input port name, each carrying both distances.

        Raises
        ------
        ValueError
            The stream holds no pixels, a distance is not finite, or a distance is not
            positive on average.
        """
        accumulator: WelfordAccumulator | None = None
        for batch in input_stream:
            score = batch.get("score_distance")
            orthogonal = batch.get("orthogonal_distance")
            if score is None or orthogonal is None:
                continue
            pair = torch.stack((score.reshape(-1), orthogonal.reshape(-1)), dim=1)
            if accumulator is None:
                # On the stream's device: the accumulator merges each batch into its own buffers.
                accumulator = WelfordAccumulator(2, dtype=torch.float64).to(pair.device)
            accumulator.update(pair)

        if accumulator is None or accumulator.count < 2:
            raise ValueError("SIMCADecision received no pixels to fit on (it needs at least two).")

        mean = accumulator.mean
        variance = accumulator.var
        if not (torch.isfinite(mean).all() and torch.isfinite(variance).all()):
            raise ValueError(
                "SIMCADecision found non-finite distances in the fit data; check the upstream "
                "nodes for NaN or inf."
            )
        for index, port in enumerate(_DISTANCE_PORTS):
            if mean[index] <= 0:
                raise ValueError(
                    f"SIMCADecision: `{port}` is not positive on the fit data (mean "
                    f"{mean[index].item():.3g}), so it cannot be scaled. A residual that is "
                    "zero everywhere usually means the subspace spans the whole data space; "
                    "lower the number of components."
                )

        # No spread gives an infinite estimate, which the reference convention sets to 1.
        dof = 2.0 * mean.square() / variance
        dof = torch.where(torch.isfinite(dof), dof, torch.ones_like(dof))
        dof = dof.round().clamp(_MIN_DOF, _MAX_DOF)

        self.h0.copy_(mean[0].to(self.h0.dtype))
        self.q0.copy_(mean[1].to(self.q0.dtype))
        self.n_h.copy_(dof[0].to(self.n_h.dtype))
        self.n_q.copy_(dof[1].to(self.n_q.dtype))
        self._refresh_threshold()
        self._statistically_initialized = True

    def _refresh_threshold(self) -> None:
        """Set the limit to the ``1 - alpha`` quantile of ``chi2(N_h + N_q)``.

        The quantile is read from the upper tail (``isf``): ``1 - alpha`` rounds to 1 for a
        very small ``alpha``, which would put the limit at infinity.
        """
        from scipy.stats import chi2

        dof = float(self.n_h) + float(self.n_q)
        self.threshold.fill_(float(chi2.isf(self.alpha, dof)))

    def load_state_dict(
        self, state_dict: Mapping[str, Any], strict: bool = True, assign: bool = False
    ) -> Any:
        """Load the fitted parameters, then derive the limit from this node's ``alpha``.

        The degrees of freedom come from the weights and the significance level from the
        node's hparams, so an ``alpha`` edited in the pipeline yaml takes effect on restore.
        """
        result = super().load_state_dict(state_dict, strict=strict, assign=assign)
        if self._statistically_initialized:
            self._refresh_threshold()
        return result

    def forward(
        self, score_distance: Tensor, orthogonal_distance: Tensor, **_: Any
    ) -> dict[str, Tensor]:
        """Return the combined distance and the decision at ``alpha``.

        Parameters
        ----------
        score_distance : Tensor
            Distance inside the model of normal [B, H, W, 1].
        orthogonal_distance : Tensor
            Distance off the model of normal [B, H, W, 1].

        Returns
        -------
        dict[str, Tensor]
            ``combined_distance`` (float32) and ``decisions`` (bool), each [B, H, W, 1].
        """
        if not self._statistically_initialized:
            raise RuntimeError(
                "SIMCADecision is not fitted. Run statistical_initialization() on normal data "
                "first."
            )

        combined = self.n_h * score_distance / self.h0 + self.n_q * orthogonal_distance / self.q0
        return {"combined_distance": combined, "decisions": combined > self.threshold}


__all__ = ["SIMCADecision"]
