"""Learnable channel mixer nodes for spectral data reduction.

Channel mixers project across channels using a full weight matrix:
``output[k] = Σ_c W[k,c] * input[c]``, which can change the channel count.
Contrast with *channel selectors* (see :mod:`cuvis_ai.node.channel_selector`),
which gate/reweight individual channels independently (diagonal operation).

This module provides two mixer variants:

* :class:`LearnableChannelMixer` — 1×1 convolution-based mixer (DRCNN-style,
  Zeegers et al. 2020).
* :class:`ConcreteChannelMixer` — Gumbel-Softmax differentiable band selection
  that learns soft-to-hard channel weighting via temperature annealing.
"""

from __future__ import annotations

import math
from typing import Any, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from cuvis_ai_core.node import Node
from cuvis_ai_schemas.enums import ExecutionStage
from cuvis_ai_schemas.execution import Context, InputStream
from cuvis_ai_schemas.pipeline import PortSpec
from torch import Tensor

from cuvis_ai.utils.welford import WelfordAccumulator


class LearnableChannelMixer(Node):
    """Learnable channel mixer for hyperspectral data reduction (DRCNN-style).

    This node implements a learnable linear combination layer that reduces the number
    of spectral channels through spectral pixel-wise 1x1 convolutions. Based on the
    DRCNN approach, it uses:
    - 1x1 convolution (linear combination across spectral dimension)
    - Leaky ReLU activation (a=0.01)
    - Bias parameters
    - Optional PCA-based initialization

    The mixer is designed to be trained end-to-end with a downstream model (e.g., AdaClip)
    while keeping the downstream model frozen. This allows the mixer to learn optimal
    spectral combinations for the specific task.

    Parameters
    ----------
    input_channels : int
        Number of input spectral channels (e.g., 61 for hyperspectral cube)
    output_channels : int
        Number of output channels (e.g., 3 for RGB compatibility)
    leaky_relu_negative_slope : float, optional
        Negative slope for Leaky ReLU activation (default: 0.01, as per DRCNN paper)
    use_bias : bool, optional
        Whether to use bias parameters (default: True, as per DRCNN paper)
    use_activation : bool, optional
        Whether to apply Leaky ReLU activation (default: True, as per DRCNN paper)
    normalize_output : bool, optional
        Whether to apply per-channel min-max normalization to [0, 1] range (default: True).
        This matches the behavior of band selectors and ensures compatibility with AdaClip.
        When True, each output channel is normalized independently using per-batch statistics.
    init_method : {"xavier", "kaiming", "pca", "zeros"}, optional
        Weight initialization method (default: "xavier")
        - "xavier": Xavier/Glorot uniform initialization
        - "kaiming": Kaiming/He uniform initialization
        - "pca": Initialize from PCA components (requires statistical_initialization)
        - "zeros": Zero initialization (weights and bias start at zero)
    eps : float, optional
        Small constant for numerical stability (default: 1e-6)
    reduction_scheme : list[int] | None, optional
        Multi-layer reduction scheme for gradual channel reduction (default: None).
        If None, uses single-layer reduction (input_channels → output_channels).
        If provided, must start with input_channels and end with output_channels.
        Example: [61, 16, 8, 3] means:
        - Layer 1: 61 → 16 channels
        - Layer 2: 16 → 8 channels
        - Layer 3: 8 → 3 channels
        This matches the DRCNN paper's multi-layer architecture for better optimization.

    Attributes
    ----------
    conv : nn.Conv2d
        1x1 convolutional layer performing spectral mixing
    activation : nn.LeakyReLU or None
        Leaky ReLU activation function (if use_activation=True)

    Examples
    --------
    >>> # Create mixer: 61 channels → 3 channels (single-layer)
    >>> mixer = LearnableChannelMixer(
    ...     input_channels=61,
    ...     output_channels=3,
    ...     leaky_relu_negative_slope=0.01,
    ...     init_method="xavier"
    ... )
    >>>
    >>> # Create mixer with multi-layer reduction (matches DRCNN paper)
    >>> mixer = LearnableChannelMixer(
    ...     input_channels=61,
    ...     output_channels=3,
    ...     reduction_scheme=[61, 16, 8, 3],  # Gradual reduction
    ...     leaky_relu_negative_slope=0.01,
    ...     init_method="xavier"
    ... )
    >>>
    >>> # Optional: Initialize from PCA
    >>> # mixer.statistical_initialization(input_stream)
    >>>
    >>> # Enable gradient training
    >>> mixer.unfreeze()
    >>>
    >>> # Forward pass: [B, H, W, 61] → [B, H, W, 3]
    >>> output = mixer.forward(data=hsi_cube)
    >>> rgb_like = output["rgb"]  # [B, H, W, 3]
    """

    INPUT_SPECS = {
        "data": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, -1),
            description="Input hyperspectral cube (BHWC format)",
        )
    }

    OUTPUT_SPECS = {
        "rgb": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, "output_channels"),
            description="Reduced channel output (e.g., RGB-like) [B, H, W, output_channels]",
        )
    }

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        leaky_relu_negative_slope: float = 0.01,
        use_bias: bool = True,
        use_activation: bool = True,
        normalize_output: bool = True,
        init_method: Literal["xavier", "kaiming", "pca", "zeros"] = "xavier",
        eps: float = 1e-6,
        reduction_scheme: list[int] | None = None,
        **kwargs,
    ) -> None:
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.leaky_relu_negative_slope = leaky_relu_negative_slope
        self.use_bias = use_bias
        self.use_activation = use_activation
        self.normalize_output = normalize_output
        self.init_method = init_method
        self.eps = eps

        # Determine reduction scheme: if None, use single-layer (backward compatible)
        # If provided, use multi-layer gradual reduction (e.g., [61, 16, 8, 3])
        if reduction_scheme is None:
            reduction_scheme = [input_channels, output_channels]
        else:
            # Validate reduction scheme
            if reduction_scheme[0] != input_channels:
                raise ValueError(
                    f"First element of reduction_scheme must match input_channels: "
                    f"got {reduction_scheme[0]}, expected {input_channels}"
                )
            if reduction_scheme[-1] != output_channels:
                raise ValueError(
                    f"Last element of reduction_scheme must match output_channels: "
                    f"got {reduction_scheme[-1]}, expected {output_channels}"
                )
            if len(reduction_scheme) < 2:
                raise ValueError(
                    f"reduction_scheme must have at least 2 elements, got {len(reduction_scheme)}"
                )

        self.reduction_scheme = reduction_scheme
        self.num_layers = len(reduction_scheme) - 1  # Number of reduction layers

        super().__init__(
            input_channels=input_channels,
            output_channels=output_channels,
            leaky_relu_negative_slope=leaky_relu_negative_slope,
            use_bias=use_bias,
            use_activation=use_activation,
            normalize_output=normalize_output,
            init_method=init_method,
            eps=eps,
            reduction_scheme=reduction_scheme,
            **kwargs,
        )

        # Create multi-layer reduction architecture (as per DRCNN paper)
        # Each layer performs: C_in → C_out reduction via 1x1 convolution
        self.convs = nn.ModuleList()
        for i in range(self.num_layers):
            in_ch = reduction_scheme[i]
            out_ch = reduction_scheme[i + 1]
            conv = nn.Conv2d(
                in_channels=in_ch,
                out_channels=out_ch,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=use_bias,
            )
            self.convs.append(conv)

        # Leaky ReLU activation (as per DRCNN paper)
        # Note: Leaky ReLU with a=0.01 can be very aggressive, killing most negative values
        # Consider using a higher value (e.g., 0.1) or removing activation if issues occur
        if use_activation:
            self.activation = nn.LeakyReLU(negative_slope=leaky_relu_negative_slope)
        else:
            self.activation = None

        # Initialize weights based on method
        self._initialize_weights()

        # Track initialization state
        self._statistically_initialized = False

    def _initialize_weights(self) -> None:
        """Initialize convolution weights based on init_method."""
        for conv in self.convs:
            if self.init_method == "xavier":
                nn.init.xavier_uniform_(conv.weight)
            elif self.init_method == "kaiming":
                nn.init.kaiming_uniform_(conv.weight, a=self.leaky_relu_negative_slope)
            elif self.init_method == "zeros":
                nn.init.zeros_(conv.weight)
            elif self.init_method == "pca":
                # PCA initialization will be done in statistical_initialization
                # For now, use xavier as placeholder
                nn.init.xavier_uniform_(conv.weight)
            else:
                raise ValueError(f"Unknown init_method: {self.init_method}")

            # Initialize bias to zero (as per DRCNN paper)
            if self.use_bias:
                nn.init.zeros_(conv.bias)

    @property
    def requires_initial_fit(self) -> bool:
        """Whether this node requires statistical initialization."""
        return self.init_method == "pca"

    def statistical_initialization(self, input_stream: InputStream) -> None:
        """Initialize mixer weights from PCA components.

        This method computes PCA on the input data and initializes the mixer weights
        to the top principal components. This provides a good starting point for
        gradient-based optimization.

        Parameters
        ----------
        input_stream : InputStream
            Iterator yielding dicts matching INPUT_SPECS (port-based format)
            Expected format: {"data": tensor} where tensor is [B, H, W, C_in]

        Notes
        -----
        This method is only used when init_method="pca". For other initialization
        methods, weights are set in __init__.
        """
        if self.init_method != "pca":
            return  # No statistical initialization needed

        acc = WelfordAccumulator(self.input_channels, track_covariance=True)
        for batch_data in input_stream:
            x = batch_data["data"]
            if x is not None:
                flat = x.reshape(-1, x.shape[-1])  # [B*H*W, C]
                acc.update(flat)

        if acc.count == 0:
            raise ValueError("No data provided for PCA initialization")

        cov = acc.cov.to(torch.float64)  # [C_in, C_in]

        # Eigen decomposition (equivalent to SVD on centered data)
        eigenvalues, eigenvectors = torch.linalg.eigh(cov)
        eigenvalues = eigenvalues.flip(0)
        eigenvectors = eigenvectors.flip(1)

        # For multi-layer, initialize only the first layer with PCA
        # Subsequent layers use xavier initialization (already done in _initialize_weights)
        first_layer_out_channels = self.reduction_scheme[1]

        n_components = min(first_layer_out_channels, eigenvectors.shape[1])
        components = eigenvectors[:, :n_components].T.float()  # [n_components, C_in]

        # If we need more output channels than components, pad with zeros
        if n_components < first_layer_out_channels:
            padding = torch.zeros(
                first_layer_out_channels - n_components,
                self.input_channels,
                device=components.device,
                dtype=components.dtype,
            )
            components = torch.cat([components, padding], dim=0)

        # Set weights for first layer: conv weight shape is [C_out, C_in, 1, 1]
        # We need to transpose components: [C_out, C_in]
        with torch.no_grad():
            self.convs[0].weight.data = components.view(
                first_layer_out_channels, self.input_channels, 1, 1
            )

        self._statistically_initialized = True

    def freeze(self) -> None:
        """Disable gradient-based training of mixer weights."""
        for conv in self.convs:
            for param in conv.parameters():
                param.requires_grad = False
        super().freeze()

    def unfreeze(self) -> None:
        """Enable gradient-based training of mixer weights."""
        for conv in self.convs:
            for param in conv.parameters():
                param.requires_grad = True
        super().unfreeze()

    # NOTE(debug-cleanup): Debug tensor saving is disabled for production.
    # Keeping this stub commented so it can be fully removed later if no longer needed.
    # def _save_debug_tensor(
    #     self, tensor: Tensor, name: str, context: Context | None, frame_idx: int
    # ) -> None:
    #     \"\"\"Save tensor for debugging if debug mode is enabled.\"\"\"
    #     if not (hasattr(self, "_debug_save_dir") and self._debug_save_dir):
    #         return
    #
    #     if context is None:
    #         return
    #
    #     # Create directory structure: {stage}/epoch_{epoch}/batch_{batch_idx}/frame_{frame_idx}/
    #     # Convert ExecutionStage enum to string (e.g., ExecutionStage.TRAIN -> "train")
    #     stage_str = context.stage.value if hasattr(context.stage, "value") else str(context.stage)
    #     save_dir = (
    #         Path(self._debug_save_dir)
    #         / stage_str
    #         / f"epoch_{context.epoch:03d}"
    #         / f"batch_{context.batch_idx:03d}"
    #         / f"frame_{frame_idx:03d}"
    #     )
    #     save_dir.mkdir(parents=True, exist_ok=True)
    #
    #     # Convert tensor to numpy and save
    #     tensor_np = tensor.detach().cpu().numpy()
    #     save_path = save_dir / f"{self.name}_{name}.npy"
    #     np.save(save_path, tensor_np)

    def forward(self, data: Tensor, context: Context | None = None, **_: Any) -> dict[str, Tensor]:
        """Apply learnable channel mixing to input.

        Parameters
        ----------
        data : Tensor
            Input tensor [B, H, W, C_in] in BHWC format
        context : Context, optional
            Execution context with epoch, batch_idx, stage info

        Returns
        -------
        dict[str, Tensor]
            Dictionary with "rgb" key containing reduced channels [B, H, W, C_out]
        """
        B, H, W, C_in = data.shape

        # DEBUG: Print input info
        if hasattr(self, "_debug") and self._debug:
            print(
                f"[LearnableChannelMixer] Input: shape={data.shape}, "
                f"min={data.min().item():.4f}, max={data.max().item():.4f}, "
                f"mean={data.mean().item():.4f}, requires_grad={data.requires_grad}"
            )

        # Validate input channels
        if C_in != self.input_channels:
            raise ValueError(
                f"Expected {self.input_channels} input channels, got {C_in}. "
                f"Input shape: {data.shape}"
            )

        # DEBUG disabled: previously saved input tensor here (_save_debug_tensor).
        # for b in range(B):
        #     self._save_debug_tensor(data[b], "input", context, frame_idx=b)

        # Convert from BHWC to BCHW for Conv2d
        data_bchw = data.permute(0, 3, 1, 2)  # [B, C_in, H, W]

        # Apply multi-layer reduction (as per DRCNN paper)
        # Each layer: 1x1 conv → Leaky ReLU (if enabled)
        mixed = data_bchw
        for i, conv in enumerate(self.convs):
            # Apply 1x1 convolution (spectral mixing)
            mixed = conv(mixed)  # [B, C_out_i, H, W]

            # Apply Leaky ReLU activation if enabled (except after last layer if we normalize)
            # For multi-layer, we apply activation after each layer except the last
            # The last layer's output will be normalized, so we skip activation there if normalize_output=True
            if self.activation is not None:
                if i < len(self.convs) - 1 or not self.normalize_output:
                    mixed = self.activation(mixed)

        # Convert back from BCHW to BHWC
        mixed_bhwc = mixed.permute(0, 2, 3, 1)  # [B, H, W, C_out]

        # DEBUG disabled: previously saved output_before_norm tensor here (_save_debug_tensor).
        # for b in range(B):
        #     self._save_debug_tensor(mixed_bhwc[b], "output_before_norm", context, frame_idx=b)

        # Apply per-channel normalization to [0, 1] range (matching band selector behavior)
        # This ensures compatibility with AdaClip preprocessing
        if self.normalize_output:
            # Per-image, per-channel min/max normalization to [0, 1]
            # This ensures each image is normalized independently for visual consistency
            # Shape: [B, H, W, C_out]
            B_norm, H_norm, W_norm, C_norm = mixed_bhwc.shape
            # Reshape to [B, H*W, C] for easier per-image processing
            mixed_flat = mixed_bhwc.view(B_norm, H_norm * W_norm, C_norm)
            # Compute min/max per image, per channel: [B, 1, C]
            rgb_min = mixed_flat.amin(dim=1, keepdim=True)  # [B, 1, C]
            rgb_max = mixed_flat.amax(dim=1, keepdim=True)  # [B, 1, C]
            denom = (rgb_max - rgb_min).clamp_min(self.eps)
            # Normalize: [B, H*W, C]
            mixed_normalized = (mixed_flat - rgb_min) / denom
            # Reshape back and clamp
            mixed_bhwc = mixed_normalized.view(B_norm, H_norm, W_norm, C_norm).clamp_(0.0, 1.0)

        # DEBUG disabled: previously saved output_after_norm tensor here (_save_debug_tensor).
        # for b in range(B):
        #     self._save_debug_tensor(mixed_bhwc[b], "output_after_norm", context, frame_idx=b)

        # DEBUG: Print output info
        if hasattr(self, "_debug") and self._debug:
            print(
                f"[LearnableChannelMixer] Output: shape={mixed_bhwc.shape}, "
                f"min={mixed_bhwc.min().item():.4f}, max={mixed_bhwc.max().item():.4f}, "
                f"mean={mixed_bhwc.mean().item():.4f}, requires_grad={mixed_bhwc.requires_grad}"
            )

        return {"rgb": mixed_bhwc}


def _sample_gumbel(shape: tuple[int, ...], device: torch.device, eps: float = 1e-10) -> Tensor:
    """Sample Gumbel(0, 1) noise."""
    u = torch.rand(shape, device=device)
    return -torch.log(-torch.log(u + eps) + eps)


class ConcreteChannelMixer(Node):
    """Concrete/Gumbel-Softmax channel mixer for hyperspectral cubes.

    Learns ``K`` categorical distributions over ``T`` input bands,
    and during training uses the Gumbel-Softmax trick to produce differentiable
    approximate one-hot selection weights that become increasingly peaked as the
    temperature :math:`\\tau` is annealed.

    For each output channel :math:`c \\in {1, \\dots, K}`, we learn logits
    ``L_c in R^T`` and sample:

    .. math::

        w_c = \\text{softmax}\\left( \\frac{L_c + g}{\\tau} \\right), \\quad
        g \\sim \\text{Gumbel}(0, 1)

    The resulting weights are used to form K-channel RGB-like images:

    .. math::

        Y[:, :, c] = \\sum_{t=1}^T w_c[t] \\cdot X[:, :, t]

    where ``X`` is the input hyperspectral cube in ``[0, 1]``.

    Parameters
    ----------
    input_channels : int
        Number of input spectral channels (e.g., 61 for hyperspectral cube).
    output_channels : int, optional
        Number of output channels (default: 3 for RGB/AdaClip compatibility).
    tau_start : float, optional
        Initial temperature for Gumbel-Softmax (default: 10.0).
    tau_end : float, optional
        Final temperature for Gumbel-Softmax (default: 0.1).
    max_epochs : int, optional
        Number of epochs over which to exponentially anneal :math:`\\tau`
        from ``tau_start`` to ``tau_end`` (default: 20).
    use_hard_inference : bool, optional
        If True, uses hard argmax selection at inference/validation time
        (one-hot weights). If False, uses softmax over logits (default: True).
    eps : float, optional
        Small constant for numerical stability (default: 1e-6).

    Notes
    -----
    - During training (``context.stage == 'train'``), the node samples
      Gumbel noise and uses the Concrete relaxation with the current
      temperature :math:`\\tau(\\text{epoch})``.
    - During validation/test/inference, it uses deterministic weights
      without Gumbel noise.
    - The node exposes ``selection_weights`` so that repulsion penalties
      (e.g., DistinctnessLoss) can be attached in the pipeline.
    """

    INPUT_SPECS = {
        "data": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, -1),
            description="Input hyperspectral cube [B, H, W, C_in] in BHWC format.",
        )
    }

    OUTPUT_SPECS = {
        "rgb": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, "output_channels"),
            description="Selected-band RGB-like image [B, H, W, C_out].",
        ),
        "selection_weights": PortSpec(
            dtype=torch.float32,
            shape=("output_channels", -1),
            description="Current selection weights [C_out, C_in].",
        ),
    }

    def __init__(
        self,
        input_channels: int,
        output_channels: int = 3,
        tau_start: float = 10.0,
        tau_end: float = 0.1,
        max_epochs: int = 20,
        use_hard_inference: bool = True,
        eps: float = 1e-6,
        **kwargs: Any,
    ) -> None:
        self.input_channels = int(input_channels)
        self.output_channels = int(output_channels)
        self.tau_start = float(tau_start)
        self.tau_end = float(tau_end)
        self.max_epochs = int(max_epochs)
        self.use_hard_inference = bool(use_hard_inference)
        self.eps = float(eps)

        if self.output_channels <= 0:
            raise ValueError(f"output_channels must be positive, got {output_channels}")
        if self.input_channels <= 0:
            raise ValueError(f"input_channels must be positive, got {input_channels}")
        if self.tau_start <= 0.0 or self.tau_end <= 0.0:
            raise ValueError("tau_start and tau_end must be positive.")

        super().__init__(
            input_channels=self.input_channels,
            output_channels=self.output_channels,
            tau_start=self.tau_start,
            tau_end=self.tau_end,
            max_epochs=self.max_epochs,
            use_hard_inference=self.use_hard_inference,
            eps=self.eps,
            **kwargs,
        )

        # Learnable logits for Categorical over input channels: [C_out, C_in]
        self.logits = nn.Parameter(torch.zeros(self.output_channels, self.input_channels))

    def _current_tau(self, context: Context | None) -> float:
        """Compute current temperature based on epoch (exponential schedule)."""
        if context is None or context.stage != ExecutionStage.TRAIN:
            return self.tau_end

        if self.max_epochs <= 1:
            return self.tau_end

        epoch = max(0, min(context.epoch, self.max_epochs - 1))
        return self._get_tau(epoch)

    def _get_tau(self, epoch: int) -> float:
        """Get temperature for a specific epoch (exponential schedule).

        Parameters
        ----------
        epoch : int
            Epoch number (0-indexed).

        Returns
        -------
        float
            Temperature value for the given epoch.
        """
        if self.max_epochs <= 1:
            return self.tau_end

        epoch = max(0, min(epoch, self.max_epochs - 1))
        frac = epoch / float(self.max_epochs - 1)

        # Exponential interpolation in log-space between tau_start and tau_end
        log_tau_start = math.log(self.tau_start)
        log_tau_end = math.log(self.tau_end)
        log_tau = (1.0 - frac) * log_tau_start + frac * log_tau_end
        return float(math.exp(log_tau))

    def get_selection_weights(self, deterministic: bool = True) -> Tensor:
        """Return current selection weights without data dependency.

        Parameters
        ----------
        deterministic : bool, optional
            If True, uses softmax over logits (no Gumbel noise) at a
            "midpoint" temperature (geometric mean of start/end). If False,
            uses current logits with ``tau_end``.
        """
        if deterministic:
            tau = math.sqrt(self.tau_start * self.tau_end)
        else:
            tau = self.tau_end

        return F.softmax(self.logits / tau, dim=-1)

    def get_selected_bands(self) -> Tensor:
        """Return argmax band indices per output channel."""
        with torch.no_grad():
            return torch.argmax(self.logits, dim=-1)

    def forward(
        self,
        data: Tensor,
        context: Context | None = None,
        **_: Any,
    ) -> dict[str, Tensor]:
        """Apply Concrete/Gumbel-Softmax channel mixing.

        Parameters
        ----------
        data : Tensor
            Input tensor [B, H, W, C_in] in BHWC format.
        context : Context, optional
            Execution context with stage and epoch information.

        Returns
        -------
        dict[str, Tensor]
            Dictionary with:
            - ``"rgb"``: [B, H, W, C_out] RGB-like image.
            - ``"selection_weights"``: [C_out, C_in] current weights.
        """
        B, H, W, C_in = data.shape

        tau = self._current_tau(context)
        device = data.device

        if self.training and context is not None and context.stage == ExecutionStage.TRAIN:
            # Gumbel-Softmax sampling during training
            g = _sample_gumbel(self.logits.shape, device=device, eps=self.eps)
            weights = F.softmax((self.logits + g) / tau, dim=-1)  # [C_out, C_in]
        else:
            # Deterministic selection for val/test/inference
            if self.use_hard_inference:
                # Hard argmax → one-hot
                indices = torch.argmax(self.logits, dim=-1)  # [C_out]
                weights = torch.zeros_like(self.logits)
                weights.scatter_(1, indices.unsqueeze(-1), 1.0)
            else:
                # Softmax over logits at low temperature
                weights = F.softmax(self.logits / self.tau_end, dim=-1)

        # Weighted sum over spectral dimension: [B, H, W, C_in] x [C_out, C_in] -> [B, H, W, C_out]
        rgb = torch.einsum("bhwc,kc->bhwk", data, weights)

        return {
            "rgb": rgb,
            "selection_weights": weights,
        }


__all__ = ["ConcreteChannelMixer", "LearnableChannelMixer"]
