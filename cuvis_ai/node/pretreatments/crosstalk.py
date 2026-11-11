"""Band crosstalk correction: a learned square band-mixing matrix applied per pixel."""

from __future__ import annotations

from typing import Any

import torch
from cuvis_ai_schemas.enums import NodeCategory, NodeTag
from cuvis_ai_schemas.pipeline import PortSpec
from torch import Tensor

from cuvis_ai_core.node import Node


class BandCrosstalkCorrector(Node):
    """Apply a learned square band-mixing matrix to every pixel spectrum.

    Removes crosstalk between spectral bands by replacing each pixel's spectrum with
    a linear combination of all bands: ``out[..., d] = sum_c weights[d, c] * cube[..., c]``.
    ``weights`` starts as the identity (no correction) and is listed in
    ``TRAINABLE_BUFFERS``, so a trainrun can ``unfreeze`` the node and fit the matrix by
    gradient against a target with an existing loss node.

    With ``normalize_trace=True`` the matrix applied in ``forward`` is rescaled to a
    fixed trace on the fly: ``w = weights * (C / trace(weights))``. The saved
    ``weights`` stay exactly as trained; only the applied matrix changes. This fixes
    the mean of the diagonal at one but does not preserve the total intensity or the
    norm of a pixel in general: output values stay near their input level only when
    the matrix is close to diagonal. The rule holds for any nonzero trace, a negative
    one included. The scale of a trained matrix is then free (any nonzero multiple of
    it gives the same output), and turning the option on after training multiplies
    every output by the constant ``C / trace(weights)``.

    A trace whose absolute value falls below ``eps`` raises ``ValueError`` in eager
    ``forward`` instead of dividing by a near-zero number into NaN or infinity. That
    check reads a tensor value and cannot be traced by ``torch.export``, so
    :meth:`export_module` returns an adapter that runs the same matrix mixing without
    the check.

    A cube whose channel count differs from ``input_channels`` raises ``ValueError``;
    both ports carry ``input_channels`` as their channel dimension.

    :class:`cuvis_ai.node.channel_mixer.LearnableChannelMixer` also mixes bands with
    learned weights and can be set up square. It is a model block built on 1x1
    convolutions with optional bias, activation and output normalization, and its
    output port is ``rgb``. This node is the calibration step of the pretreatment
    family: ``cube`` in and ``cube`` out, one ``[C, C]`` matrix that starts as the
    identity, no bias, and no output normalization beyond the optional ReLU clip.

    Parameters
    ----------
    input_channels : int
        Number of spectral bands ``C``; ``weights`` is ``[C, C]``.
    clip_negative : bool, optional
        Apply ReLU to the output (default True).
    normalize_trace : bool, optional
        Rescale the applied matrix to trace ``C`` on the fly (default False).
    eps : float, optional
        Floor on the absolute trace magnitude before :meth:`forward` raises when
        ``normalize_trace`` is set (default 1e-8). Must be positive.
    """

    _category = NodeCategory.TRANSFORM
    _tags = frozenset(
        {
            NodeTag.HYPERSPECTRAL,
            NodeTag.PREPROCESSING,
            NodeTag.CALIBRATION,
            NodeTag.LEARNABLE,
            NodeTag.TORCH,
            NodeTag.ONNX,
        }
    )

    TRAINABLE_BUFFERS = ("weights",)

    INPUT_SPECS = {
        "cube": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, "input_channels"),
            description="Hyperspectral cube [B, H, W, C]",
        )
    }

    OUTPUT_SPECS = {
        "cube": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, "input_channels"),
            description="Crosstalk-corrected cube, same shape as the input",
        )
    }

    def __init__(
        self,
        input_channels: int,
        clip_negative: bool = True,
        normalize_trace: bool = False,
        eps: float = 1e-8,
        **kwargs: Any,
    ) -> None:
        if int(input_channels) < 1:
            raise ValueError(f"input_channels must be a positive integer, got {input_channels!r}")
        if float(eps) <= 0.0:
            raise ValueError(f"eps must be positive, got {eps!r}")
        self.input_channels = int(input_channels)
        self.clip_negative = bool(clip_negative)
        self.normalize_trace = bool(normalize_trace)
        self.eps = float(eps)
        super().__init__(
            input_channels=self.input_channels,
            clip_negative=self.clip_negative,
            normalize_trace=self.normalize_trace,
            eps=self.eps,
            **kwargs,
        )
        self.register_buffer("weights", torch.eye(self.input_channels, dtype=torch.float32))

    def _resolve_matrix(self, weights: Tensor) -> Tensor:
        """The matrix ``forward`` applies: ``weights`` itself, or trace-rescaled.

        Raises ``ValueError`` when ``normalize_trace`` is set and the trace magnitude
        falls below ``eps``; reads a tensor value, so this path is eager-only (see
        :meth:`export_module`).
        """
        if not self.normalize_trace:
            return weights
        trace = torch.trace(weights)
        trace_value = float(trace.detach())
        if abs(trace_value) < self.eps:
            raise ValueError(
                f"{type(self).__name__}: weights trace magnitude {trace_value:.3e} is below "
                f"eps={self.eps:.3e}; normalize_trace cannot rescale a near-zero trace"
            )
        return weights * (self.input_channels / trace)

    @staticmethod
    def _mix(cube: Tensor, matrix: Tensor, clip_negative: bool = True) -> Tensor:
        """Apply ``matrix`` to every pixel spectrum of ``cube`` and optionally clip.

        Raises ``ValueError`` when the cube's channel count differs from the matrix size:
        the product would otherwise broadcast a single channel on either side.
        """
        if cube.shape[-1] != matrix.shape[1]:
            raise ValueError(
                f"BandCrosstalkCorrector was built for input_channels={matrix.shape[1]}, "
                f"got a cube with {cube.shape[-1]} channels"
            )
        out = torch.einsum("dc,bhwc->bhwd", matrix, cube)
        if clip_negative:
            out = torch.relu(out)
        return out

    def forward(self, cube: Tensor, **_: Any) -> dict[str, Tensor]:
        """Apply the band-mixing matrix to ``cube``.

        Parameters
        ----------
        cube : Tensor
            Hyperspectral cube ``[B, H, W, C]``.

        Returns
        -------
        dict[str, Tensor]
            ``{"cube": corrected}`` with the input's shape.
        """
        matrix = self._resolve_matrix(self.weights)
        return {"cube": self._mix(cube, matrix, self.clip_negative)}

    def export_module(self) -> torch.nn.Module:
        """Adapter without the trace-magnitude guard; see ``_ConstantMatrixAdapter``.

        The guard in :meth:`forward` reads the trace of ``weights`` as a Python float
        to decide whether to raise, which ``torch.export`` cannot trace even though the
        matrix mixing itself is tensor ops throughout. This resolves the matrix once,
        outside tracing (as :meth:`export_module`'s contract requires), and returns an
        adapter that applies that fixed matrix with no further trace check.
        """
        matrix = self._resolve_matrix(self.weights)
        return _ConstantMatrixAdapter(matrix, self.clip_negative)


class _ConstantMatrixAdapter(torch.nn.Module):
    """Export stand-in: applies a fixed matrix (and the clip flag) to ``cube``.

    Built from an already trace-resolved matrix so the exported graph needs no
    trace check at all; the matrix is a plain tensor, not a parameter, so it traces
    as a constant in the exported program.
    """

    def __init__(self, matrix: Tensor, clip_negative: bool) -> None:
        super().__init__()
        self.register_buffer("matrix", matrix.detach().clone())
        self.clip_negative = clip_negative

    def forward(self, cube: Tensor) -> dict[str, Tensor]:
        """Apply the fixed matrix to ``cube``."""
        return {"cube": BandCrosstalkCorrector._mix(cube, self.matrix, self.clip_negative)}


__all__ = ["BandCrosstalkCorrector"]
