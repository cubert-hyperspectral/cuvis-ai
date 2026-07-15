"""One-class SVM novelty / anomaly detection node.

Fits a one-class support vector machine on background (in-distribution) pixels
during statistical initialization, then scores every pixel of an incoming cube
with the signed decision function at inference time.

The node delegates the (CPU, NumPy) fit to scikit-learn's
:class:`sklearn.svm.OneClassSVM`, but does **not** keep the sklearn estimator
around. Instead it distils the fitted model down to the four tensors the RBF
decision function actually needs (support vectors, dual coefficients, the
resolved scalar ``gamma`` and the offset) and re-implements the forward pass in
pure torch. That keeps inference device-agnostic, free of a NumPy round-trip,
and serialisable through the normal buffer / ``state_dict`` machinery.

For an RBF kernel scikit-learn defines::

    decision_function(x) = sum_i dual_coef_i * exp(-gamma * ||x - sv_i||^2) - offset
    predict(x)           = sign(decision_function(x))

so ``decision_function > 0`` marks an inlier and ``< 0`` an outlier. The node
emits the raw signed score on ``scores`` and the outlier mask
(``decision_function < 0``) on ``decisions``.
"""

from __future__ import annotations

from typing import Any

import torch
from cuvis_ai_schemas.enums import NodeCategory, NodeTag
from cuvis_ai_schemas.pipeline import PortSpec

from cuvis_ai.node._statistical_fit import _StatisticalFitNode


class OneClassSVMDetector(_StatisticalFitNode):
    """One-class SVM novelty detector (sklearn fit, pure-torch RBF forward).

    During ``statistical_initialization`` the node collects background pixels
    and fits a :class:`sklearn.svm.OneClassSVM`. The fitted estimator is reduced
    to torch buffers (support vectors, dual coefficients, resolved ``gamma`` and
    the offset) so that ``forward`` can evaluate the RBF decision function in
    pure torch, chunked over the pixel axis to bound peak memory.

    Parameters
    ----------
    kernel : str, optional
        Kernel passed to scikit-learn at fit time. Only ``"rbf"`` is supported
        by the torch forward pass; any other value raises at fit (default:
        ``"rbf"``).
    nu : float, optional
        Upper bound on the fraction of training outliers and lower bound on the
        fraction of support vectors, in ``(0, 1]`` (default: ``0.5``).
    gamma : str or float, optional
        RBF kernel coefficient. Either a positive float or one of scikit-learn's
        string presets (``"scale"`` / ``"auto"``); the numeric value sklearn
        actually resolves is stored and used at inference (default: ``"scale"``).
    chunk_size : int, optional
        Number of pixels scored per chunk in ``forward``. Caps the transient
        ``[chunk_size, n_support_vectors]`` kernel matrix (default: ``65536``).

    Attributes
    ----------
    support_vectors : torch.Tensor
        Fitted support vectors ``[n_sv, C]``.
    dual_coef : torch.Tensor
        Signed dual coefficients ``[n_sv]``.
    gamma_buf : torch.Tensor
        Resolved scalar RBF ``gamma`` as a ``[1]`` tensor.
    offset_buf : torch.Tensor
        Decision-function offset as a ``[1]`` tensor.

    Examples
    --------
    >>> from cuvis_ai.node.svm import OneClassSVMDetector
    >>> detector = OneClassSVMDetector(nu=0.1, gamma="scale")
    >>> # detector.statistical_initialization(background_stream)
    >>> # out = detector.forward(cube=cube)
    >>> # scores, decisions = out["scores"], out["decisions"]
    """

    _category = NodeCategory.MODEL
    _tags = frozenset({NodeTag.HYPERSPECTRAL, NodeTag.ANOMALY, NodeTag.STATEFUL, NodeTag.TORCH})

    INPUT_SPECS = {
        "cube": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, -1),
            description="Input hyperspectral cube [B, H, W, C]",
        )
    }

    OUTPUT_SPECS = {
        "scores": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, 1),
            description="Signed one-class SVM decision function [B, H, W, 1]; "
            ">0 inlier, <0 outlier",
        ),
        "decisions": PortSpec(
            dtype=torch.bool,
            shape=(-1, -1, -1, 1),
            description="Outlier mask [B, H, W, 1]; True where decision function < 0",
        ),
    }

    def __init__(
        self,
        kernel: str = "rbf",
        nu: float = 0.5,
        gamma: str | float = "scale",
        chunk_size: int = 65536,
        **kwargs: Any,
    ) -> None:
        """Store hyperparameters and register placeholder fit buffers.

        Parameters
        ----------
        kernel : str, optional
            scikit-learn kernel; only ``"rbf"`` is supported at inference
            (default: ``"rbf"``).
        nu : float, optional
            One-class SVM ``nu`` in ``(0, 1]`` (default: ``0.5``).
        gamma : str or float, optional
            RBF kernel coefficient or a string preset (default: ``"scale"``).
        chunk_size : int, optional
            Pixels scored per chunk in ``forward`` (default: ``65536``).
        **kwargs : Any
            Forwarded to the statistical-fit base (``max_fit_pixels``,
            ``fit_seed``) and the node base.
        """
        self.kernel = str(kernel)
        self.nu = float(nu)
        self.gamma = gamma if isinstance(gamma, str) else float(gamma)
        self.chunk_size = int(chunk_size)
        super().__init__(
            kernel=self.kernel,
            nu=self.nu,
            gamma=self.gamma,
            chunk_size=self.chunk_size,
            **kwargs,
        )
        # Placeholders resized by the base's _load_from_state_dict on reload, and
        # by _fit on a fresh statistical_initialization.
        self.register_buffer("support_vectors", torch.zeros(0))
        self.register_buffer("dual_coef", torch.zeros(0))
        self.register_buffer("gamma_buf", torch.zeros(0))
        self.register_buffer("offset_buf", torch.zeros(0))

    @torch.no_grad()
    def _fit(self, pixels: torch.Tensor) -> None:
        """Fit a scikit-learn one-class SVM and distil it into torch buffers.

        Parameters
        ----------
        pixels : torch.Tensor
            Background pixel matrix ``[N, C]`` collected by the base.

        Raises
        ------
        ValueError
            If ``kernel`` is not ``"rbf"`` (the only kernel the torch forward
            pass implements).
        """
        if self.kernel != "rbf":
            raise ValueError(
                f"{type(self).__name__} only supports kernel='rbf' at inference; "
                f"got kernel={self.kernel!r}."
            )

        from sklearn.svm import OneClassSVM

        model = OneClassSVM(kernel=self.kernel, nu=self.nu, gamma=self.gamma)
        model.fit(pixels.cpu().numpy())

        # model._gamma is the numeric gamma sklearn resolved (for "scale" this is
        # 1 / (n_features * X.var()), for "auto" it is 1 / n_features).
        resolved_gamma = float(model._gamma)

        self.support_vectors = torch.as_tensor(model.support_vectors_, dtype=torch.float32)
        self.dual_coef = torch.as_tensor(model.dual_coef_.ravel(), dtype=torch.float32)
        self.gamma_buf = torch.tensor([resolved_gamma], dtype=torch.float32)
        self.offset_buf = torch.tensor([float(model.offset_[0])], dtype=torch.float32)

    @torch.no_grad()
    def forward(self, cube: torch.Tensor, **_: Any) -> dict[str, torch.Tensor]:
        """Score a cube with the signed RBF decision function, chunked over pixels.

        Parameters
        ----------
        cube : torch.Tensor
            Input hyperspectral cube ``[B, H, W, C]``.
        **_ : Any
            Additional unused keyword arguments.

        Returns
        -------
        dict[str, torch.Tensor]
            Dictionary with:

            - ``"scores"`` : signed decision function ``[B, H, W, 1]`` (``>0`` inlier, ``<0`` outlier).
            - ``"decisions"`` : outlier mask ``[B, H, W, 1]`` (``True`` where the decision function is ``< 0``).

        Raises
        ------
        RuntimeError
            If the node has not been initialized via
            ``statistical_initialization`` (or a fitted checkpoint load).
        """
        self._require_initialized()

        B, H, W, C = cube.shape
        support = self.support_vectors.to(device=cube.device, dtype=cube.dtype)
        dual = self.dual_coef.to(device=cube.device, dtype=cube.dtype)
        gamma = self.gamma_buf.to(device=cube.device, dtype=cube.dtype)
        offset = self.offset_buf.to(device=cube.device, dtype=cube.dtype)

        flat = cube.reshape(-1, C)
        P = flat.shape[0]
        chunk_size = max(1, self.chunk_size)

        df_chunks: list[torch.Tensor] = []
        for start in range(0, P, chunk_size):
            chunk = flat[start : start + chunk_size]  # [chunk, C]
            sq_dist = torch.cdist(chunk, support) ** 2  # [chunk, n_sv]
            kernel = torch.exp(-gamma * sq_dist)  # [chunk, n_sv]
            df_chunks.append(kernel @ dual - offset)  # [chunk]
        df = torch.cat(df_chunks, dim=0) if df_chunks else flat.new_zeros(0)

        scores = df.reshape(B, H, W, 1)
        decisions = scores < 0
        return {"scores": scores, "decisions": decisions}


__all__ = ["OneClassSVMDetector"]
