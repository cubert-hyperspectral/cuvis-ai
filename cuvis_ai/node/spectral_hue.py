"""Split a per-pixel spectrum into its hue, saturation and value."""

from __future__ import annotations

import math
from typing import Any

import torch
from cuvis_ai_schemas.enums import NodeCategory, NodeTag
from cuvis_ai_schemas.pipeline import PortSpec
from torch import Tensor

from cuvis_ai_core.node import Node

_HUE_MODES = ("cartesian", "polar")


class SpectralHueSaturationValue(Node):
    """Split each pixel spectrum into hue, saturation and value.

    Hue is the direction of the spectrum once the all-bands-equal part is
    removed: the coordinates of the spectrum in an orthonormal (Helmert) basis
    of the hyperplane orthogonal to the all-ones direction, normalized to unit
    length. With ``hue="cartesian"`` the output is those ``C - 1`` Cartesian
    coordinates; with ``hue="polar"`` it is their ``C - 2`` hyperspherical
    angles, which do not depend on the coordinate vector's length. Saturation
    is ``max(x) - min(x)`` and value is ``mean(x)`` over the channels, both
    with the channel dimension kept.

    The Cartesian coordinates are computed from consecutive differences of the
    spectrum rather than from products of the raw values: a flat spectrum then
    gives exactly zero in float32 even at a large constant level, where the
    raw-product form can leave a residual above ``eps`` from floating-point
    cancellation and produce a spurious unit hue. For the ``C x (C-1)`` Helmert
    basis matrix ``U`` the coordinates are ``U^T (x - mean(x))``, so their norm
    is ``||x - mean(x)||`` and the Cartesian hue equals ``U^T`` applied to the
    output of :class:`cuvis_ai.node.normalization.PerPixelUnitNorm`.

    Invariance under a positive gain and a constant offset on the spectrum,
    and the reconstruction of the Cartesian unit vector from the polar angles,
    hold only for spectra whose coefficient norm sits above ``eps``; below it
    the Cartesian hue is ``coefficients / eps``, shorter than a unit vector,
    exactly as :class:`cuvis_ai.node.normalization.PerPixelUnitNorm` documents
    for its own floor.

    Norms are taken on the coordinates divided by their largest magnitude, so a
    finite float32 spectrum keeps its hue although the square of its values
    would overflow or underflow. ``hue="polar"`` costs roughly three times
    ``hue="cartesian"`` on a CPU, because of the masks described next.

    Every polar angle is an ``atan2(y, x)`` with a coordinate as ``x``. For all
    angles but the last, ``y`` is the norm of the coordinates after ``x`` and
    the angle lies in ``[0, pi]``; for the last angle ``y`` is the final
    coordinate and the angle lies in ``(-pi, pi]``. Where ``y`` is exactly
    zero the angle is fixed by convention instead of evaluating ``atan2`` at
    an axis point: zero when ``x`` is nonnegative, pi otherwise. A flat
    spectrum then gives every angle zero. A zero ``x`` counts as positive zero
    whatever its sign bit. Both rules also keep the eager and the ONNX-exported
    graph in agreement: the ONNX lowering of ``atan2`` returns ``-pi`` where
    eager PyTorch returns ``pi`` for a zero ``y`` and a negative ``x``, and the
    opposite sign of ``pi/2`` for an ``x`` of ``-0.0``. A square root behind
    the suffix norms is masked like ``y``, so it is never differentiated at
    exactly zero.

    Exactly degenerate inputs give finite (zero) gradients through these
    masks, but this node is a fixed preprocessing step, not a general-purpose
    one: inputs close to a degenerate point (a near-flat spectrum, coordinates
    close to an axis) can give very large gradients, since the hue direction
    itself is only weakly determined there.

    Parameters
    ----------
    hue : str, optional
        ``"cartesian"`` for the ``C - 1`` Helmert coordinates, ``"polar"`` for
        their ``C - 2`` hyperspherical angles (default ``"cartesian"``).
    eps : float, optional
        Floor on the Cartesian coefficient norm before normalizing to unit
        length (default ``1e-8``). Must be a normal float32 number, so that the
        floor is neither zero nor infinite in the dtype of the ports.
    **kwargs : dict
        Additional arguments passed to the ``Node`` base class.
    """

    _category = NodeCategory.TRANSFORM
    _tags = frozenset(
        {
            NodeTag.HYPERSPECTRAL,
            NodeTag.PREPROCESSING,
            NodeTag.NORMALIZATION,
            NodeTag.TORCH,
            NodeTag.ONNX,
        }
    )

    INPUT_SPECS = {
        "cube": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, -1),
            description="Hyperspectral cube [B, H, W, C]",
        )
    }

    OUTPUT_SPECS = {
        "hue": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, -1),
            description="Hue: C-1 unit-length Cartesian coordinates or C-2 polar angles",
        ),
        "saturation": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, 1),
            description="max(x) - min(x) over the channels",
        ),
        "value": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, 1),
            description="mean(x) over the channels",
        ),
    }

    def __init__(self, hue: str = "cartesian", eps: float = 1e-8, **kwargs: Any) -> None:
        if hue not in _HUE_MODES:
            raise ValueError(f"hue must be one of {_HUE_MODES}, got {hue!r}")
        float32 = torch.finfo(torch.float32)
        if not float32.tiny <= eps <= float32.max:
            raise ValueError(
                f"eps must be a positive float32 number between {float32.tiny:.3g} and "
                f"{float32.max:.3g}, got {eps!r}"
            )
        self.hue = hue
        self.eps = float(eps)
        super().__init__(hue=self.hue, eps=self.eps, **kwargs)

    @staticmethod
    def _cartesian_coefficients(cube: Tensor) -> Tensor:
        """The ``C - 1`` Helmert coordinates of every pixel spectrum, unnormalized.

        Weighted consecutive differences of the spectrum, summed as a flipped
        cumulative sum, reproduce the raw-sum formula exactly while keeping a
        flat spectrum at precisely zero in float32 (see the class docstring).
        """
        c = cube.shape[-1]
        diffs = cube[..., 1:] - cube[..., :-1]  # d_k = x_{k+1} - x_k, k = 1..C-1
        index = torch.arange(1, c, device=cube.device, dtype=cube.dtype)
        remaining = c - index  # C - k for the differences, n_i - 1 for the coordinates
        suffix = torch.cumsum((remaining * diffs).flip(-1), dim=-1).flip(-1)
        scale = -1.0 / torch.sqrt(remaining * (remaining + 1))
        return scale * suffix

    @staticmethod
    def _peak_scaled(coefficients: Tensor) -> tuple[Tensor, Tensor]:
        """The coordinates divided by their largest magnitude per pixel, and that magnitude.

        The scaled coordinates lie in ``[-1, 1]``, so their squares stay inside
        the float range whatever the level of the spectrum. A zero vector keeps
        a magnitude of one and stays zero. The magnitude is detached: dividing
        by it as a constant leaves the gradients of the norm and of the angles
        unchanged, since the norm is homogeneous and the angles do not depend
        on the length.
        """
        peak = coefficients.detach().abs().amax(dim=-1, keepdim=True)
        peak = torch.where(peak > 0.0, peak, torch.ones_like(peak))
        return coefficients / peak, peak

    @staticmethod
    def _safe_atan2(y: Tensor, x: Tensor) -> Tensor:
        """``atan2(y, x)`` fixed by convention at ``y == 0`` and blind to the sign of a zero ``x``.

        Where ``y`` is exactly zero the result is 0 for ``x >= 0`` and pi for
        ``x < 0``, and ``atan2`` itself never sees that zero: ``y`` is replaced
        by a nonzero placeholder there via ``torch.where`` before the call, so
        the undefined axis point is never evaluated or differentiated.

        A zero ``x`` loses its sign before the call. The coordinates are a
        negative scale times a sum, so an exactly cancelling sum gives ``-0.0``;
        eager ``atan2(y, -0.0)`` is still ``+-pi/2``, but the ONNX lowering
        divides ``y`` by ``x`` and returns the opposite sign. The replacement is
        added as a detached difference, so the gradient with respect to ``x``
        stays that of ``atan2``.
        """
        is_zero = y == 0.0
        safe_y = torch.where(is_zero, torch.ones_like(y), y)
        unsigned_zero_x = torch.where(x == 0.0, torch.zeros_like(x), x)
        x = x + (unsigned_zero_x - x).detach()
        angle = torch.atan2(safe_y, x)
        axis_value = torch.where(x >= 0.0, torch.zeros_like(x), torch.full_like(x, math.pi))
        return torch.where(is_zero, axis_value, angle)

    @classmethod
    def _polar_angles(cls, coefficients: Tensor) -> Tensor:
        """The ``C - 2`` hyperspherical angles of the Helmert coordinates.

        Each ``phi_k`` for ``k = 1..C-3`` is ``atan2`` of the L2 norm of the
        coordinates after it (nonnegative) against it; the last angle is
        ``atan2`` of the final two coordinates directly. The suffix norms are
        taken as the square root of a sum of squares masked the same way as
        ``_safe_atan2``, so the square root is never differentiated at zero.
        """
        squares = coefficients**2
        suffix_sq = torch.cumsum(squares.flip(-1), dim=-1).flip(-1)  # suffix_sq[k-1] = sum c_k..c_m
        tail_sq = suffix_sq[..., 1:-1]  # sum_{j > k} c_j^2 for k = 1..m-2
        tail_norm = cls._safe_sqrt(tail_sq)
        leading = cls._safe_atan2(tail_norm, coefficients[..., :-2])
        last = cls._safe_atan2(coefficients[..., -1:], coefficients[..., -2:-1])
        return torch.cat([leading, last], dim=-1)

    @staticmethod
    def _safe_sqrt(value: Tensor) -> Tensor:
        """``sqrt(value)`` with the square root never evaluated or differentiated at zero.

        A zero suffix sum of squares is replaced by one before the square root,
        then the result is masked back to zero, so ``_polar_angles`` sees an
        exact zero for the ``y`` argument of ``_safe_atan2`` instead of a
        square root's (finite but large) derivative at zero.
        """
        is_zero = value == 0.0
        safe_value = torch.where(is_zero, torch.ones_like(value), value)
        return torch.where(is_zero, torch.zeros_like(value), torch.sqrt(safe_value))

    def forward(self, cube: Tensor, **_: Any) -> dict[str, Tensor]:
        """Split ``cube`` into hue, saturation and value.

        Parameters
        ----------
        cube : Tensor
            Hyperspectral cube ``[B, H, W, C]``. Raises ``ValueError`` when
            ``C`` is below the minimum the configured ``hue`` mode needs (two
            channels for ``"cartesian"``, three for ``"polar"``), a comparison
            of the input's Python shape.

        Returns
        -------
        dict[str, Tensor]
            ``{"hue": ..., "saturation": ..., "value": ...}``: hue is
            ``[B, H, W, C-1]`` (Cartesian) or ``[B, H, W, C-2]`` (polar);
            saturation and value are ``[B, H, W, 1]``.
        """
        channels = cube.shape[-1]
        minimum = 3 if self.hue == "polar" else 2
        if channels < minimum:
            raise ValueError(
                f"{type(self).__name__} with hue={self.hue!r} needs at least {minimum} "
                f"channels, got {channels}"
            )
        coefficients = self._cartesian_coefficients(cube)
        scaled, peak = self._peak_scaled(coefficients)
        if self.hue == "cartesian":
            norm = (peak * scaled.norm(p=2, dim=-1, keepdim=True)).clamp_min(self.eps)
            hue = coefficients / norm
        else:
            hue = self._polar_angles(scaled)
        saturation = cube.amax(dim=-1, keepdim=True) - cube.amin(dim=-1, keepdim=True)
        value = cube.mean(dim=-1, keepdim=True)
        return {"hue": hue, "saturation": saturation, "value": value}


__all__ = ["SpectralHueSaturationValue"]
