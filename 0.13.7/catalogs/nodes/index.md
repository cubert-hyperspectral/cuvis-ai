# Nodes Catalog

Every node available in cuvis-ai pipelines, in one place. Built-in nodes ship with the `cuvis_ai` package; plugin nodes and data modules come from separately-installable plugin manifests — see [Plugin Development](https://cubert-hyperspectral.github.io/cuvis-ai/0.13.7/reference/plugin-development/overview/index.md).

CategoryTagsSource

216 items

Clear

lossmetricmodelregularizersinksourcetransformunspecifiedvisualizer

anomaugbatchedbboxclassdetdiffdim-redembevalhsiimginferkplearnmaskmetanormnumpypostprereconregrgbsegstatefulstochstreamtexttorchtracktrainvid

Built-inPluginData module

`AdaCLIPFocalDiceLoss`cuvis_ai_adaclip.node.losseslossdifftorchtrainadaclip[↗](https://github.com/cubert-hyperspectral/cuvis-ai-adaclip "Open plugin repo")Combined Focal + Dice loss for AdaCLIP training.

Combined Focal + Dice loss for AdaCLIP training.

Inputs

| Port                        | Dtype     | Shape              | Description                                              |
| --------------------------- | --------- | ------------------ | -------------------------------------------------------- |
| `predictions`               | `float32` | `[-1, -1, -1, 1]`  | Aggregated anomaly scores [B, H, W, 1] for fallback path |
| `targets`                   | `bool`    | `[-1, -1, -1, 1]`  | Ground truth binary masks [B, H, W, 1]                   |
| `per_layer_scores` optional | `float32` | `[-1, -1, -1, -1]` | Per-layer softmaxed maps [B, num_layers\*2, H, W]        |
| `image_score_2ch` optional  | `float32` | `[-1, -1]`         | Image-level score [B, 2] in [normal, anomaly] order      |

Outputs

| Port   | Dtype     | Shape | Description                |
| ------ | --------- | ----- | -------------------------- |
| `loss` | `float32` | `any` | Combined focal + dice loss |

[View plugin repo (v0.3.1)](https://github.com/cubert-hyperspectral/cuvis-ai-adaclip/tree/v0.3.1)

`AnomalyBCEWithLogits`cuvis_ai.node.losseslossanomdifftorchtrainBinary cross-entropy loss for anomaly detection with logits.

#### AnomalyBCEWithLogits

```python
AnomalyBCEWithLogits(
    weight=1.0, pos_weight=None, reduction="mean", **kwargs
)
```

Bases: `LossNode`

Binary cross-entropy loss for anomaly detection with logits.

Computes BCE loss between predicted anomaly scores and ground truth masks. Uses BCEWithLogitsLoss for numerical stability.

Parameters:

| Name         | Type    | Description                                                                   | Default  |
| ------------ | ------- | ----------------------------------------------------------------------------- | -------- |
| `weight`     | `float` | Overall weight for this loss component (default: 1.0)                         | `1.0`    |
| `pos_weight` | `float` | Weight for positive class (anomaly) to handle class imbalance (default: None) | `None`   |
| `reduction`  | `str`   | Reduction method: 'mean', 'sum', or 'none' (default: 'mean')                  | `'mean'` |

Source code in `cuvis_ai/node/losses.py`

```python
def __init__(
    self,
    weight: float = 1.0,
    pos_weight: float | None = None,
    reduction: str = "mean",
    **kwargs,
) -> None:
    self.weight = weight
    self.pos_weight = pos_weight
    self.reduction = reduction

    super().__init__(
        weight=weight,
        pos_weight=pos_weight,
        reduction=reduction,
        **kwargs,
    )

    # Create loss function
    if pos_weight is not None:
        pos_weight_tensor = torch.tensor([pos_weight])
        self.register_buffer("_pos_weight", pos_weight_tensor)
        self.loss_fn = nn.BCEWithLogitsLoss(
            pos_weight=self._pos_weight,
            reduction=reduction,
        )
    else:
        self.loss_fn = nn.BCEWithLogitsLoss(reduction=reduction)
```

##### forward

```python
forward(predictions, targets, **_)
```

Compute weighted BCE loss.

Parameters:

| Name          | Type     | Description                     | Default    |
| ------------- | -------- | ------------------------------- | ---------- |
| `predictions` | `Tensor` | Predicted scores [B, H, W, 1]   | *required* |
| `targets`     | `Tensor` | Ground truth masks [B, H, W, 1] | *required* |

Returns:

| Type                | Description                                       |
| ------------------- | ------------------------------------------------- |
| `dict[str, Tensor]` | Dictionary with "loss" key containing scalar loss |

Source code in `cuvis_ai/node/losses.py`

```python
def forward(self, predictions: Tensor, targets: Tensor, **_: Any) -> dict[str, Tensor]:
    """Compute weighted BCE loss.

    Parameters
    ----------
    predictions : Tensor
        Predicted scores [B, H, W, 1]
    targets : Tensor
        Ground truth masks [B, H, W, 1]

    Returns
    -------
    dict[str, Tensor]
        Dictionary with "loss" key containing scalar loss
    """
    # Squeeze channel dimension to [B, H, W] for BCEWithLogitsLoss
    if predictions.dim() == 4 and predictions.shape[-1] == 1:
        predictions = predictions.squeeze(-1)

    if targets.dim() == 4 and targets.shape[-1] == 1:
        targets = targets.squeeze(-1)

    # Convert labels to float
    targets = targets.float()

    # Compute loss
    loss = self.loss_fn(predictions, targets)

    # Apply weight
    weighted_loss = self.weight * loss

    return {"loss": weighted_loss}
```

`CrossEntropyLoss`cuvis_ai.node.losseslossdiffsegtorchtrainPixel-wise cross-entropy over dense segmentation logits (`K >= 2`).

#### CrossEntropyLoss

```python
CrossEntropyLoss(
    weight=1.0,
    class_weights=None,
    ignore_index=-100,
    **kwargs,
)
```

Bases: `LossNode`

Pixel-wise cross-entropy over dense segmentation logits (`K >= 2`).

Consumes BHWC per-pixel class logits plus an integer class-index mask and emits a scalar loss. The class count is the logits' last axis; no `num_classes` hyperparameter is needed.

Parameters:

| Name            | Type                    | Description                                                                                                                       | Default |
| --------------- | ----------------------- | --------------------------------------------------------------------------------------------------------------------------------- | ------- |
| `weight`        | `float`                 | Scalar multiplier applied to the loss (default: 1.0)                                                                              | `1.0`   |
| `class_weights` | `list of float or None` | Per-class rescaling passed to F.cross_entropy; helps with strong class imbalance such as small foreground objects (default: None) | `None`  |
| `ignore_index`  | `int`                   | Target value excluded from the loss (default: -100, PyTorch's default)                                                            | `-100`  |

Raises:

| Type         | Description                                                                                                                                                                                           |
| ------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `ValueError` | If the logits carry a single class channel (K == 1). Softmax over one logit is constant 1, so the loss would be identically zero; use DiceLoss or AnomalyBCEWithLogits for single-logit binary heads. |

Examples:

```pycon
>>> ce = CrossEntropyLoss(class_weights=[0.1, 1.0])
>>> loss = ce.forward(logits=logits_bhwk, targets=mask_bhw)["loss"]
```

Source code in `cuvis_ai/node/losses.py`

```python
def __init__(
    self,
    weight: float = 1.0,
    class_weights: list[float] | None = None,
    ignore_index: int = -100,
    **kwargs,
) -> None:
    self.weight = float(weight)
    self.class_weights = class_weights
    self.ignore_index = int(ignore_index)
    super().__init__(
        weight=self.weight,
        class_weights=class_weights,
        ignore_index=self.ignore_index,
        **kwargs,
    )
    if class_weights is not None:
        self.register_buffer("_class_weights", torch.tensor(class_weights, dtype=torch.float32))
```

##### forward

```python
forward(logits, targets, **_)
```

Compute the weighted cross-entropy from BHWC logits and a mask.

Parameters:

| Name      | Type     | Description                                                                   | Default    |
| --------- | -------- | ----------------------------------------------------------------------------- | ---------- |
| `logits`  | `Tensor` | Per-pixel class logits [B, H, W, K]                                           | *required* |
| `targets` | `Tensor` | Integer class-index mask [B, H, W] (a trailing singleton channel is squeezed) | *required* |

Returns:

| Type                | Description                                                    |
| ------------------- | -------------------------------------------------------------- |
| `dict[str, Tensor]` | Dictionary with "loss" key containing the scalar cross-entropy |

Source code in `cuvis_ai/node/losses.py`

```python
def forward(self, logits: Tensor, targets: Tensor, **_: Any) -> dict[str, Tensor]:
    """Compute the weighted cross-entropy from BHWC logits and a mask.

    Parameters
    ----------
    logits : Tensor
        Per-pixel class logits [B, H, W, K]
    targets : Tensor
        Integer class-index mask [B, H, W] (a trailing singleton channel
        is squeezed)

    Returns
    -------
    dict[str, Tensor]
        Dictionary with "loss" key containing the scalar cross-entropy
    """
    if logits.shape[-1] == 1:
        raise ValueError(
            "CrossEntropyLoss needs multiclass logits [B, H, W, K>=2]; softmax over a "
            "single logit is constant, so the loss would always be zero. Use DiceLoss or "
            "AnomalyBCEWithLogits for single-logit binary heads."
        )
    logits_bchw, targets_bhw = _to_bchw_targets(logits, targets)
    buf = getattr(self, "_class_weights", None)
    weights = buf.to(logits_bchw.dtype) if buf is not None else None
    loss = F.cross_entropy(
        logits_bchw, targets_bhw, weight=weights, ignore_index=self.ignore_index
    )
    return {"loss": self.weight * loss}
```

`CrossEntropyLoss`cuvis_ai_unet.node.losseslossdiffsegtorchtrainunet[↗](https://github.com/cubert-hyperspectral/cuvis-ai-unet "Open plugin repo")Pixel-wise cross-entropy over the DynUNet logits (multiclass, `K >= 2`).

Pixel-wise cross-entropy over the DynUNet logits (multiclass, `K &gt;= 2`).

Inputs

| Port      | Dtype     | Shape              | Description                                                                      |
| --------- | --------- | ------------------ | -------------------------------------------------------------------------------- |
| `logits`  | `float32` | `[-1, -1, -1, -1]` | Per-pixel class logits [B, H, W, num_classes]                                    |
| `targets` | `int32`   | `[-1, -1, -1]`     | Integer class-index mask [B, H, W] (cast to int64 internally; trailing [.,1] ok) |

Outputs

| Port   | Dtype     | Shape | Description       |
| ------ | --------- | ----- | ----------------- |
| `loss` | `float32` | `any` | Scalar loss value |

[View plugin repo (v0.2.2)](https://github.com/cubert-hyperspectral/cuvis-ai-unet/tree/v0.2.2)

`DeepSVDDSoftBoundaryLoss`cuvis_ai.node.losseslossanomdifftorchtrainSoft-boundary Deep SVDD objective operating on BHWD embeddings.

#### DeepSVDDSoftBoundaryLoss

```python
DeepSVDDSoftBoundaryLoss(nu=0.05, weight=1.0, **kwargs)
```

Bases: `LossNode`

Soft-boundary Deep SVDD objective operating on BHWD embeddings.

Source code in `cuvis_ai/node/losses.py`

```python
def __init__(self, nu: float = 0.05, weight: float = 1.0, **kwargs) -> None:
    if not (0.0 < nu < 1.0):
        raise ValueError("nu must be in (0, 1)")
    self.nu = float(nu)
    self.weight = float(weight)

    super().__init__(nu=self.nu, weight=self.weight, **kwargs)

    self.r_unconstrained = nn.Parameter(torch.tensor(0.0))
```

##### forward

```python
forward(embeddings, center, **_)
```

Compute Deep SVDD soft-boundary loss.

The loss consists of the hypersphere radius R² plus a slack penalty for points outside the hypersphere. The radius R is learned via an unconstrained parameter with softplus activation.

Parameters:

| Name         | Type     | Description                                                     | Default    |
| ------------ | -------- | --------------------------------------------------------------- | ---------- |
| `embeddings` | `Tensor` | Embedded feature representations [B, H, W, D] from the network. | *required* |
| `center`     | `Tensor` | Center of the hypersphere [D] computed during initialization.   | *required* |
| `**_`        | `Any`    | Additional unused keyword arguments.                            | `{}`       |

Returns:

| Type                | Description                                                  |
| ------------------- | ------------------------------------------------------------ |
| `dict[str, Tensor]` | Dictionary with "loss" key containing the scalar loss value. |

Notes

The loss formula is: loss = weight * (R² + (1/ν) * mean(ReLU(dist - R²))) where dist is the squared distance from embeddings to the center.

Source code in `cuvis_ai/node/losses.py`

```python
def forward(self, embeddings: Tensor, center: Tensor, **_: Any) -> dict[str, Tensor]:
    """Compute Deep SVDD soft-boundary loss.

    The loss consists of the hypersphere radius R² plus a slack penalty
    for points outside the hypersphere. The radius R is learned via
    an unconstrained parameter with softplus activation.

    Parameters
    ----------
    embeddings : Tensor
        Embedded feature representations [B, H, W, D] from the network.
    center : Tensor
        Center of the hypersphere [D] computed during initialization.
    **_ : Any
        Additional unused keyword arguments.

    Returns
    -------
    dict[str, Tensor]
        Dictionary with "loss" key containing the scalar loss value.

    Notes
    -----
    The loss formula is: loss = weight * (R² + (1/ν) * mean(ReLU(dist - R²)))
    where dist is the squared distance from embeddings to the center.
    """
    B, H, W, D = embeddings.shape
    z = embeddings.reshape(B * H * W, D)
    R = torch.nn.functional.softplus(self.r_unconstrained, beta=10.0)
    dist = torch.sum((z - center.view(1, -1)) ** 2, dim=1)
    slack = torch.relu(dist - R**2)
    base_loss = R**2 + (1.0 / self.nu) * slack.mean()
    loss = self.weight * base_loss

    return {"loss": loss}
```

`DiceLoss`cuvis_ai.node.losseslossdiffsegtorchtrainSoft Dice loss over dense segmentation logits.

#### DiceLoss

```python
DiceLoss(
    weight=1.0,
    ignore_index=None,
    include_background=True,
    eps=1e-06,
    **kwargs,
)
```

Bases: `LossNode`

Soft Dice loss over dense segmentation logits.

Consumes BHWC per-pixel class logits (last axis = classes; the class count is inferred at runtime, `K == 1` is treated as sigmoid binary, `K > 1` as softmax multiclass over one-hot targets) plus an integer class-index mask, and emits a scalar loss `1 - mean(dice)`. Dice is accumulated per class over the whole batch (nnU-Net-style batch Dice) rather than averaged per sample. Multiclass targets outside `[0, K)` that are not `ignore_index` raise.

Parameters:

| Name                 | Type          | Description                                                                                                                                                                                                                                      | Default |
| -------------------- | ------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------- |
| `weight`             | `float`       | Scalar multiplier applied to the loss, for combining several losses (default: 1.0)                                                                                                                                                               | `1.0`   |
| `ignore_index`       | `int or None` | Target value excluded from both prediction and target, or None to use every pixel (default: None)                                                                                                                                                | `None`  |
| `include_background` | `bool`        | Whether class 0 contributes to the per-class Dice mean; multiclass only (default: True). Disabling it is the standard choice for heavily imbalanced segmentation, where the background Dice term is saturated and dilutes the foreground signal. | `True`  |
| `eps`                | `float`       | Smoothing constant for the Dice ratio (default: 1e-6)                                                                                                                                                                                            | `1e-06` |

Examples:

```pycon
>>> dice = DiceLoss(weight=1.0, include_background=False)
>>> loss = dice.forward(logits=logits_bhwk, targets=mask_bhw)["loss"]
```

Source code in `cuvis_ai/node/losses.py`

```python
def __init__(
    self,
    weight: float = 1.0,
    ignore_index: int | None = None,
    include_background: bool = True,
    eps: float = 1e-6,
    **kwargs,
) -> None:
    self.weight = float(weight)
    self.ignore_index = ignore_index
    self.include_background = bool(include_background)
    self.eps = float(eps)
    super().__init__(
        weight=self.weight,
        ignore_index=ignore_index,
        include_background=self.include_background,
        eps=self.eps,
        **kwargs,
    )
```

##### forward

```python
forward(logits, targets, **_)
```

Compute the weighted soft Dice loss from BHWC logits and a mask.

Parameters:

| Name      | Type     | Description                                                                   | Default    |
| --------- | -------- | ----------------------------------------------------------------------------- | ---------- |
| `logits`  | `Tensor` | Per-pixel class logits [B, H, W, K]                                           | *required* |
| `targets` | `Tensor` | Integer class-index mask [B, H, W] (a trailing singleton channel is squeezed) | *required* |

Returns:

| Type                | Description                                                |
| ------------------- | ---------------------------------------------------------- |
| `dict[str, Tensor]` | Dictionary with "loss" key containing the scalar Dice loss |

Source code in `cuvis_ai/node/losses.py`

```python
def forward(self, logits: Tensor, targets: Tensor, **_: Any) -> dict[str, Tensor]:
    """Compute the weighted soft Dice loss from BHWC logits and a mask.

    Parameters
    ----------
    logits : Tensor
        Per-pixel class logits [B, H, W, K]
    targets : Tensor
        Integer class-index mask [B, H, W] (a trailing singleton channel
        is squeezed)

    Returns
    -------
    dict[str, Tensor]
        Dictionary with "loss" key containing the scalar Dice loss
    """
    logits_bchw, targets_bhw = _to_bchw_targets(logits, targets)
    loss = _soft_dice_loss(
        logits_bchw,
        targets_bhw,
        ignore_index=self.ignore_index,
        include_background=self.include_background,
        eps=self.eps,
    )
    return {"loss": self.weight * loss}
```

`DiceLoss`cuvis_ai_unet.node.losseslossdiffsegtorchtrainunet[↗](https://github.com/cubert-hyperspectral/cuvis-ai-unet "Open plugin repo")Soft Dice loss over the DynUNet logits.

Soft Dice loss over the DynUNet logits.

Inputs

| Port      | Dtype     | Shape              | Description                                                                      |
| --------- | --------- | ------------------ | -------------------------------------------------------------------------------- |
| `logits`  | `float32` | `[-1, -1, -1, -1]` | Per-pixel class logits [B, H, W, num_classes]                                    |
| `targets` | `int32`   | `[-1, -1, -1]`     | Integer class-index mask [B, H, W] (cast to int64 internally; trailing [.,1] ok) |

Outputs

| Port   | Dtype     | Shape | Description       |
| ------ | --------- | ----- | ----------------- |
| `loss` | `float32` | `any` | Scalar loss value |

[View plugin repo (v0.2.2)](https://github.com/cubert-hyperspectral/cuvis-ai-unet/tree/v0.2.2)

`DinomalyTrainLossBridge`cuvis_ai_dinomaly.node.dinomaly_train_loss_bridgelossdifftorchtraindinomaly[↗](https://github.com/cubert-hyperspectral/cuvis-ai-dinomaly "Open plugin repo")Passes through the scalar reconstruction loss from :class:`DinomalyDetector`.

Passes through the scalar reconstruction loss from :class:`DinomalyDetector`.

Inputs

| Port                | Dtype     | Shape | Description                                |
| ------------------- | --------- | ----- | ------------------------------------------ |
| `raw_loss` optional | `float32` | `any` | Scalar training loss from DinomalyDetector |

Outputs

| Port   | Dtype     | Shape | Description                |
| ------ | --------- | ----- | -------------------------- |
| `loss` | `float32` | `any` | Weighted loss for backprop |

[View plugin repo (v0.6.3)](https://github.com/cubert-hyperspectral/cuvis-ai-dinomaly/tree/v0.6.3)

`DistinctnessLoss`cuvis_ai.node.losseslossdifftorchtrainRepulsion loss encouraging different selectors to choose different bands.

#### DistinctnessLoss

```python
DistinctnessLoss(weight=0.1, eps=1e-06, **kwargs)
```

Bases: `LossNode`

Repulsion loss encouraging different selectors to choose different bands.

This loss is designed for band/channel selector nodes that output a 2D weight matrix `[output_channels, input_channels]`. It computes the mean pairwise cosine similarity between all pairs of selector weight vectors and penalizes high similarity:

.. math::

```text
L_\text{repel} = \frac{1}{N_\text{pairs}} \sum_{i < j}
    \cos(\mathbf{w}_i, \mathbf{w}_j)
```

Minimizing this loss encourages selectors to focus on different bands, preventing the common failure mode where all channels collapse onto the same band.

Parameters:

| Name     | Type    | Description                                                              | Default |
| -------- | ------- | ------------------------------------------------------------------------ | ------- |
| `weight` | `float` | Overall weight for this loss component (default: 0.1).                   | `0.1`   |
| `eps`    | `float` | Small constant for numerical stability when normalizing (default: 1e-6). | `1e-06` |

Source code in `cuvis_ai/node/losses.py`

```python
def __init__(self, weight: float = 0.1, eps: float = 1e-6, **kwargs) -> None:
    self.weight = float(weight)
    self.eps = float(eps)

    super().__init__(weight=self.weight, eps=self.eps, **kwargs)
```

##### forward

```python
forward(selection_weights, **_)
```

Compute mean pairwise cosine similarity penalty.

Parameters:

| Name                | Type     | Description                                               | Default    |
| ------------------- | -------- | --------------------------------------------------------- | ---------- |
| `selection_weights` | `Tensor` | Weight matrix of shape [output_channels, input_channels]. | *required* |

Returns:

| Type                | Description                                                     |
| ------------------- | --------------------------------------------------------------- |
| `dict[str, Tensor]` | Dictionary with a single key "loss" containing the scalar loss. |

Source code in `cuvis_ai/node/losses.py`

```python
def forward(self, selection_weights: Tensor, **_: Any) -> dict[str, Tensor]:
    """Compute mean pairwise cosine similarity penalty.

    Parameters
    ----------
    selection_weights : Tensor
        Weight matrix of shape [output_channels, input_channels].

    Returns
    -------
    dict[str, Tensor]
        Dictionary with a single key ``"loss"`` containing the scalar loss.
    """
    # Normalize each selector vector to unit length
    w = selection_weights
    w_norm = F.normalize(w, p=2, dim=-1, eps=self.eps)  # [C, T]

    num_channels = w_norm.shape[0]
    if num_channels < 2:
        # Nothing to compare - no repulsion needed
        return {"loss": torch.zeros((), device=w_norm.device, dtype=w_norm.dtype)}

    # Compute all pairwise cosine similarities using matrix multiplication (optimized)
    similarity_matrix = w_norm @ w_norm.T  # [C, C] matrix of cosine similarities

    # Extract upper triangular part (i < j pairs), excluding diagonal
    upper_tri = torch.triu(similarity_matrix, diagonal=1)

    # Compute mean of non-zero elements (i < j pairs)
    mean_cos = upper_tri[upper_tri != 0].mean()

    # Minimize mean cosine similarity (repulsion)
    loss = self.weight * mean_cos
    return {"loss": loss}
```

`ForegroundContrastLoss`cuvis_ai.node.losseslossdifftorchtrainMaximize visual separation between foreground and background mean colors.

#### ForegroundContrastLoss

```python
ForegroundContrastLoss(
    weight=1.0,
    compactness_weight=0.0,
    anchor_weight=0.0,
    eps=1e-06,
    color_space="rgb",
    assume_srgb=True,
    **kwargs,
)
```

Bases: `LossNode`

Maximize visual separation between foreground and background mean colors.

Loss per image::

```text
-||mean_fg - mean_bg||_2

+ compactness_weight * Var_fg
+ anchor_weight * (||mean_fg - mean_img||^2 + ||mean_bg - mean_img||^2)
```

Parameters:

| Name                 | Type                             | Description                                                                                                                           | Default |
| -------------------- | -------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------- | ------- |
| `weight`             | `float`                          | Overall weight for this loss component (default: 1.0).                                                                                | `1.0`   |
| `compactness_weight` | `float`                          | Weight for foreground variance penalty (default: 0.0, disabled).                                                                      | `0.0`   |
| `anchor_weight`      | `float`                          | Anti-gaming penalty that keeps fg/bg means near the image mean, discouraging extreme color pushes (default: 0.0, disabled).           | `0.0`   |
| `eps`                | `float`                          | Small constant for numerical stability in sqrt (default: 1e-6).                                                                       | `1e-06` |
| `color_space`        | ``` ``"rgb"`` or ``"oklab"`` ``` | Color space in which to compute the fg/bg distance (default: "rgb").                                                                  | `'rgb'` |
| `assume_srgb`        | `bool`                           | When color_space="oklab", whether to apply inverse sRGB gamma before OKLab conversion. Ignored when color_space="rgb". Default: True. | `True`  |

Notes

- When `color_space="oklab"`, the OKLab conversion expects linear RGB in [0, 1]. If the upstream RGB has no sRGB gamma curve applied (e.g. output of `LearnableChannelMixer` with `normalize_output=True`), set `assume_srgb=False`.
- Vectorized over batch.
- Fallback loss uses `0.0 * rgb.sum()` so it remains connected to the model graph.

Source code in `cuvis_ai/node/losses.py`

```python
def __init__(
    self,
    weight: float = 1.0,
    compactness_weight: float = 0.0,
    anchor_weight: float = 0.0,
    eps: float = 1e-6,
    color_space: str = "rgb",
    assume_srgb: bool = True,
    **kwargs,
) -> None:
    self.weight = float(weight)
    self.compactness_weight = float(compactness_weight)
    self.anchor_weight = float(anchor_weight)
    self.eps = float(eps)
    self.color_space = str(color_space)
    self.assume_srgb = bool(assume_srgb)

    if self.color_space not in ("rgb", "oklab"):
        raise ValueError(f"color_space must be 'rgb' or 'oklab', got '{self.color_space}'")

    super().__init__(
        weight=self.weight,
        compactness_weight=self.compactness_weight,
        anchor_weight=self.anchor_weight,
        eps=self.eps,
        color_space=self.color_space,
        assume_srgb=self.assume_srgb,
        **kwargs,
    )
```

##### forward

```python
forward(rgb, mask, **_)
```

Compute foreground/background contrast loss.

Parameters:

| Name   | Type     | Description                                                           | Default    |
| ------ | -------- | --------------------------------------------------------------------- | ---------- |
| `rgb`  | `Tensor` | RGB image tensor of shape [B, H, W, 3].                               | *required* |
| `mask` | `Tensor` | Segmentation mask of shape [B, H, W] where values > 0 are foreground. | *required* |

Returns:

| Type                | Description                                                     |
| ------------------- | --------------------------------------------------------------- |
| `dict[str, Tensor]` | Dictionary with a single key "loss" containing the scalar loss. |

Source code in `cuvis_ai/node/losses.py`

```python
def forward(self, rgb: Tensor, mask: Tensor, **_: Any) -> dict[str, Tensor]:
    """Compute foreground/background contrast loss.

    Parameters
    ----------
    rgb : Tensor
        RGB image tensor of shape [B, H, W, 3].
    mask : Tensor
        Segmentation mask of shape [B, H, W] where values > 0 are foreground.

    Returns
    -------
    dict[str, Tensor]
        Dictionary with a single key ``"loss"`` containing the scalar loss.
    """
    # Optionally convert to OKLab perceptual space
    if self.color_space == "oklab":
        from cuvis_ai.utils.color_spaces import rgb_to_oklab

        pixels = rgb_to_oklab(rgb.to(torch.float32), assume_srgb=self.assume_srgb)
    else:
        pixels = rgb.to(torch.float32)  # [B, H, W, 3]

    fg = (mask > 0).unsqueeze(-1).to(dtype=pixels.dtype)  # [B, H, W, 1]
    bg = 1.0 - fg

    fg_count = fg.sum(dim=(1, 2))  # [B, 1]
    bg_count = bg.sum(dim=(1, 2))  # [B, 1]
    valid = (fg_count.squeeze(-1) > 0) & (bg_count.squeeze(-1) > 0)  # [B]

    # Passthrough fallback (connected to model graph)
    if not valid.any():
        logger.warning(
            f"ForegroundContrastLoss: all frames skipped — "
            f"mask shape={list(mask.shape)}, dtype={mask.dtype}, "
            f"fg_pixels={(mask > 0).sum().item()}, "
            f"min={mask.min().item()}, max={mask.max().item()}"
        )
        return {"loss": 0.0 * rgb.sum()}

    # Masked means (vectorized)
    fg_sum = (pixels * fg).sum(dim=(1, 2))  # [B, 3]
    bg_sum = (pixels * bg).sum(dim=(1, 2))  # [B, 3]
    fg_mean = fg_sum / fg_count.clamp_min(1.0)  # [B, 3]
    bg_mean = bg_sum / bg_count.clamp_min(1.0)  # [B, 3]

    diff = fg_mean - bg_mean
    dist = torch.sqrt((diff * diff).sum(dim=-1) + self.eps)  # [B]
    frame_loss = -dist

    # Compactness (foreground variance): Var = E[x^2] - (E[x])^2
    if self.compactness_weight > 0.0:
        fg_sq_sum = ((pixels * pixels) * fg).sum(dim=(1, 2))  # [B, 3]
        ex2 = fg_sq_sum / fg_count.clamp_min(1.0)  # [B, 3]
        var = (ex2 - fg_mean * fg_mean).clamp_min(0.0).mean(dim=-1)  # [B]
        frame_loss = frame_loss + self.compactness_weight * var

    # Anti-gaming anchor: keep fg/bg means near the image mean
    if self.anchor_weight > 0.0:
        img_mean = pixels.mean(dim=(1, 2))  # [B, 3]
        anchor_pen = (fg_mean - img_mean).pow(2).mean(dim=-1) + (bg_mean - img_mean).pow(
            2
        ).mean(dim=-1)  # [B]
        frame_loss = frame_loss + self.anchor_weight * anchor_pen

    loss = self.weight * frame_loss[valid].mean()
    return {"loss": loss}
```

`IoULoss`cuvis_ai.node.losseslossdiffsegtorchtrainDifferentiable IoU (Intersection over Union) loss.

#### IoULoss

```python
IoULoss(
    weight=1.0,
    smooth=1e-06,
    normalize_method="sigmoid",
    **kwargs,
)
```

Bases: `LossNode`

Differentiable IoU (Intersection over Union) loss.

Computes: 1 - (|A ∩ B| + smooth) / (|A U B| + smooth) Works directly on continuous scores (not binary decisions), preserving gradients.

The scores are normalized to [0, 1] range using sigmoid or clamp before computing IoU, ensuring differentiability.

Parameters:

| Name               | Type                             | Description                                                                                                                                                                                                                                                                           | Default     |
| ------------------ | -------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------- |
| `weight`           | `float`                          | Overall weight for this loss component (default: 1.0)                                                                                                                                                                                                                                 | `1.0`       |
| `smooth`           | `float`                          | Small constant for numerical stability (default: 1e-6)                                                                                                                                                                                                                                | `1e-06`     |
| `normalize_method` | `('sigmoid', 'clamp', 'minmax')` | Method to normalize predictions to [0, 1] range (default: "sigmoid") "sigmoid": Apply sigmoid activation (good for unbounded scores) "clamp": Clamp to [0, 1] (good for scores already in reasonable range) "minmax": Min-max normalization per batch (good for varying score ranges) | `"sigmoid"` |

Examples:

```pycon
>>> iou_loss = IoULoss(weight=1.0, smooth=1e-6)
>>> # Use with anomaly scores directly (no thresholding needed)
>>> loss = iou_loss.forward(predictions=anomaly_scores, targets=ground_truth_mask)
```

Source code in `cuvis_ai/node/losses.py`

```python
def __init__(
    self,
    weight: float = 1.0,
    smooth: float = 1e-6,
    normalize_method: str = "sigmoid",
    **kwargs,
) -> None:
    self.weight = weight
    self.smooth = smooth
    self.normalize_method = normalize_method

    if normalize_method not in ["sigmoid", "clamp", "minmax"]:
        raise ValueError(
            f"normalize_method must be one of ['sigmoid', 'clamp', 'minmax'], got {normalize_method}"
        )

    super().__init__(
        weight=weight,
        smooth=smooth,
        normalize_method=normalize_method,
        **kwargs,
    )
```

##### forward

```python
forward(predictions, targets, **_)
```

Compute differentiable IoU loss.

Parameters:

| Name          | Type     | Description                                             | Default    |
| ------------- | -------- | ------------------------------------------------------- | ---------- |
| `predictions` | `Tensor` | Predicted anomaly scores [B, H, W, 1] (any real values) | *required* |
| `targets`     | `Tensor` | Ground truth binary masks [B, H, W, 1]                  | *required* |

Returns:

| Type                | Description                                           |
| ------------------- | ----------------------------------------------------- |
| `dict[str, Tensor]` | Dictionary with "loss" key containing scalar IoU loss |

Source code in `cuvis_ai/node/losses.py`

```python
def forward(self, predictions: Tensor, targets: Tensor, **_: Any) -> dict[str, Tensor]:
    """Compute differentiable IoU loss.

    Parameters
    ----------
    predictions : Tensor
        Predicted anomaly scores [B, H, W, 1] (any real values)
    targets : Tensor
        Ground truth binary masks [B, H, W, 1]

    Returns
    -------
    dict[str, Tensor]
        Dictionary with "loss" key containing scalar IoU loss
    """
    # Normalize predictions to [0, 1] range based on method
    if self.normalize_method == "sigmoid":
        # Sigmoid: good for unbounded scores (e.g., logits)
        pred = torch.sigmoid(predictions)
    elif self.normalize_method == "clamp":
        # Clamp: good for scores already in reasonable range
        pred = torch.clamp(predictions, 0.0, 1.0)
    elif self.normalize_method == "minmax":
        # Min-max normalization per batch
        pred_min = predictions.min()
        pred_max = predictions.max()
        if pred_max > pred_min:
            pred = (predictions - pred_min) / (pred_max - pred_min + self.smooth)
        else:
            pred = torch.ones_like(predictions) * 0.5
    else:
        raise ValueError(f"Unknown normalize_method: {self.normalize_method}")

    # Convert targets to float
    target = targets.float()

    # Flatten for computation
    pred_flat = pred.view(-1)  # [B*H*W]
    target_flat = target.view(-1)  # [B*H*W]

    # Compute IoU: intersection / union
    # intersection = |A ∩ B| = sum(pred * target)
    # union = |A ∪ B| = sum(pred) + sum(target) - intersection
    intersection = (pred_flat * target_flat).sum()
    union = pred_flat.sum() + target_flat.sum() - intersection

    # IoU coefficient
    iou = (intersection + self.smooth) / (union + self.smooth)

    # IoU loss: 1 - IoU (minimize loss = maximize IoU)
    loss = 1.0 - iou

    return {"loss": self.weight * loss}
```

`LossNode`cuvis_ai.node.losseslossdifftorchtrainBase class for loss nodes that restricts execution to training stages.

#### LossNode

```python
LossNode(**kwargs)
```

Bases: `Node`

Base class for loss nodes that restricts execution to training stages.

Loss nodes should not execute during inference - only during train, val, and test.

Source code in `cuvis_ai/node/losses.py`

```python
def __init__(self, **kwargs) -> None:
    # Default to train/val/test stages, but allow override
    assert "execution_stages" not in kwargs, (
        "Loss nodes can only execute in train, val, and test stages."
    )

    super().__init__(
        execution_stages={
            ExecutionStage.TRAIN,
            ExecutionStage.VAL,
            ExecutionStage.TEST,
        },
        **kwargs,
    )
```

`MSEReconstructionLoss`cuvis_ai.node.losseslossdiffrecontorchtrainMean squared error reconstruction loss.

#### MSEReconstructionLoss

```python
MSEReconstructionLoss(
    weight=1.0, reduction="mean", **kwargs
)
```

Bases: `LossNode`

Mean squared error reconstruction loss.

Computes MSE between reconstruction and target. Useful for autoencoder-style architectures.

Parameters:

| Name        | Type    | Description                                                  | Default  |
| ----------- | ------- | ------------------------------------------------------------ | -------- |
| `weight`    | `float` | Weight for this loss component (default: 1.0)                | `1.0`    |
| `reduction` | `str`   | Reduction method: 'mean', 'sum', or 'none' (default: 'mean') | `'mean'` |

Source code in `cuvis_ai/node/losses.py`

```python
def __init__(self, weight: float = 1.0, reduction: str = "mean", **kwargs) -> None:
    self.weight = weight
    self.reduction = reduction
    # Extract Node base parameters from kwargs to avoid duplication
    super().__init__(
        weight=weight,
        reduction=reduction,
        **kwargs,
    )
    self.loss_fn = nn.MSELoss(reduction=reduction)
```

##### forward

```python
forward(reconstruction, target, **_)
```

Compute MSE reconstruction loss.

Parameters:

| Name             | Type     | Description                                                                   | Default    |
| ---------------- | -------- | ----------------------------------------------------------------------------- | ---------- |
| `reconstruction` | `Tensor` | Reconstructed data                                                            | *required* |
| `target`         | `Tensor` | Target for reconstruction                                                     | *required* |
| `**_`            | `Any`    | Additional arguments (e.g., context) - ignored but accepted for compatibility | `{}`       |

Returns:

| Type                | Description                                       |
| ------------------- | ------------------------------------------------- |
| `dict[str, Tensor]` | Dictionary with "loss" key containing scalar loss |

Source code in `cuvis_ai/node/losses.py`

```python
def forward(self, reconstruction: Tensor, target: Tensor, **_: Any) -> dict[str, Tensor]:
    """Compute MSE reconstruction loss.

    Parameters
    ----------
    reconstruction : Tensor
        Reconstructed data
    target : Tensor
        Target for reconstruction
    **_ : Any
        Additional arguments (e.g., context) - ignored but accepted for compatibility

    Returns
    -------
    dict[str, Tensor]
        Dictionary with "loss" key containing scalar loss
    """
    # Ensure consistent shapes
    if target.shape != reconstruction.shape:
        raise ValueError(
            f"Shape mismatch: reconstruction {reconstruction.shape} vs target {target.shape}"
        )

    # Compute loss
    loss = self.loss_fn(reconstruction, target)

    # Apply weight
    return {"loss": self.weight * loss}
```

`OHEMCrossEntropyLoss`cuvis_ai_unet.node.losseslossdiffsegtorchtrainunet[↗](https://github.com/cubert-hyperspectral/cuvis-ai-unet "Open plugin repo")Cross-entropy with online hard-example mining (OHEM) on the background.

Cross-entropy with online hard-example mining (OHEM) on the background.

Inputs

| Port      | Dtype     | Shape              | Description                                                                      |
| --------- | --------- | ------------------ | -------------------------------------------------------------------------------- |
| `logits`  | `float32` | `[-1, -1, -1, -1]` | Per-pixel class logits [B, H, W, num_classes]                                    |
| `targets` | `int32`   | `[-1, -1, -1]`     | Integer class-index mask [B, H, W] (cast to int64 internally; trailing [.,1] ok) |

Outputs

| Port   | Dtype     | Shape | Description       |
| ------ | --------- | ----- | ----------------- |
| `loss` | `float32` | `any` | Scalar loss value |

[View plugin repo (v0.2.2)](https://github.com/cubert-hyperspectral/cuvis-ai-unet/tree/v0.2.2)

`WeightedCrossEntropyLoss`cuvis_ai_inspecscrap.node.losseslossclasstraincuvis_ai_inspecscrap[↗](https://github.com/cubert-hyperspectral/cuvis-ai-inspecscrap "Open plugin repo")

[View plugin repo (v0.2.4)](https://github.com/cubert-hyperspectral/cuvis-ai-inspecscrap/tree/v0.2.4)

`AnomalyAUROCMetrics`cuvis_ai_dinomaly.node.auroc_metricsmetricanomevaldinomaly[↗](https://github.com/cubert-hyperspectral/cuvis-ai-dinomaly "Open plugin repo")Streaming pixel/image AUROC via torchmetrics (val/test only).

Streaming pixel/image AUROC via torchmetrics (val/test only).

Inputs

| Port            | Dtype     | Shape             | Description                           |
| --------------- | --------- | ----------------- | ------------------------------------- |
| `scores`        | `float32` | `[-1, -1, -1, 1]` | Raw anomaly map [B, H, W, 1]          |
| `targets`       | `bool`    | `[-1, -1, -1, 1]` | Ground-truth pixel masks [B, H, W, 1] |
| `anomaly_score` | `float32` | `[-1]`            | Per-image anomaly score [B]           |

Outputs

| Port      | Dtype | Shape | Description                            |
| --------- | ----- | ----- | -------------------------------------- |
| `metrics` | `any` | `any` | List of Metric objects (running AUROC) |

[View plugin repo (v0.6.3)](https://github.com/cubert-hyperspectral/cuvis-ai-dinomaly/tree/v0.6.3)

`AnomalyDetectionMetrics`cuvis_ai.node.metricsmetricanomevalCompute anomaly detection metrics (precision, recall, F1, etc.).

#### AnomalyDetectionMetrics

```python
AnomalyDetectionMetrics(
    execution_stages=None, ap_thresholds=200, **kwargs
)
```

Bases: `Node`

Compute anomaly detection metrics (precision, recall, F1, etc.).

Uses torchmetrics for GPU-optimized, robust metric computation. Expects binary decisions and targets to be binary masks. Executes only during validation and test stages.

Source code in `cuvis_ai/node/metrics.py`

```python
def __init__(
    self,
    execution_stages: set[ExecutionStage] | None = None,
    ap_thresholds: int = 200,
    **kwargs,
) -> None:
    self.ap_thresholds = ap_thresholds
    name, execution_stages = Node.consume_base_kwargs(
        kwargs, execution_stages or {ExecutionStage.VAL, ExecutionStage.TEST}
    )
    super().__init__(
        name=name,
        execution_stages=execution_stages,
        ap_thresholds=ap_thresholds,
        **kwargs,
    )

    # Precision/Recall/F1/IoU keep O(1) running confmat state and are stateless
    # under torchmetrics __call__ (full_state_update=False) — per-batch values.
    # BinaryAveragePrecision uses histogram-based AP (thresholds=N) so state is
    # O(N) instead of O(n_pixels). We accumulate via update() across batches
    # within a (stage, epoch) and reset only at the boundary, so the value
    # emitted each batch is a *running* AP across batches seen so far in the
    # current epoch — the last batch's value is true epoch-level AP.
    self.precision_metric = BinaryPrecision()
    self.recall_metric = BinaryRecall()
    self.f1_metric = BinaryF1Score()
    self.iou_metric = BinaryJaccardIndex()
    self.average_precision_metric = BinaryAveragePrecision(thresholds=ap_thresholds)
    self._ap_last_key: tuple[ExecutionStage, int] | None = None
```

##### forward

```python
forward(decisions, targets, context, logits=None)
```

Compute anomaly detection metrics using torchmetrics.

Parameters:

| Name        | Type      | Description                                    | Default    |
| ----------- | --------- | ---------------------------------------------- | ---------- |
| `decisions` | `Tensor`  | Binary anomaly decisions [B, H, W, 1]          | *required* |
| `targets`   | `Tensor`  | Ground truth binary masks [B, H, W, 1]         | *required* |
| `context`   | `Context` | Execution context with stage, epoch, batch_idx | *required* |

Returns:

| Type             | Description                                                     |
| ---------------- | --------------------------------------------------------------- |
| `dict[str, Any]` | Dictionary with "metrics" key containing list of Metric objects |

Source code in `cuvis_ai/node/metrics.py`

```python
def forward(
    self,
    decisions: Tensor,
    targets: Tensor,
    context: Context,
    logits: Tensor | None = None,
) -> dict[str, Any]:
    """Compute anomaly detection metrics using torchmetrics.

    Parameters
    ----------
    decisions : Tensor
        Binary anomaly decisions [B, H, W, 1]
    targets : Tensor
        Ground truth binary masks [B, H, W, 1]
    context : Context
        Execution context with stage, epoch, batch_idx

    Returns
    -------
    dict[str, Any]
        Dictionary with "metrics" key containing list of Metric objects
    """
    # Ensure consistent shapes and flatten spatial dimensions
    decisions = decisions.squeeze(-1)  # [B, H, W]
    targets = targets.squeeze(-1)  # [B, H, W]

    # Flatten to [N] where N = B*H*W for torchmetrics
    preds_flat = decisions.flatten()  # [B*H*W]
    targets_flat = targets.flatten()  # [B*H*W]

    # Compute metrics using torchmetrics (they handle edge cases robustly)
    precision = self.precision_metric(preds_flat, targets_flat)
    recall = self.recall_metric(preds_flat, targets_flat)
    f1 = self.f1_metric(preds_flat, targets_flat)
    iou = self.iou_metric(preds_flat, targets_flat)

    metrics = [
        Metric(
            name="precision",
            value=precision.item(),
            stage=context.stage,
            epoch=context.epoch,
            batch_idx=context.batch_idx,
        ),
        Metric(
            name="recall",
            value=recall.item(),
            stage=context.stage,
            epoch=context.epoch,
            batch_idx=context.batch_idx,
        ),
        Metric(
            name="f1_score",
            value=f1.item(),
            stage=context.stage,
            epoch=context.epoch,
            batch_idx=context.batch_idx,
        ),
        Metric(
            name="iou",
            value=iou.item(),
            stage=context.stage,
            epoch=context.epoch,
            batch_idx=context.batch_idx,
        ),
    ]

    if logits is not None:
        raw_scores = logits.squeeze(-1).flatten().float()
        probs_for_ap = torch.sigmoid(raw_scores)

        current_key = (context.stage, context.epoch)
        if self._ap_last_key != current_key:
            self.average_precision_metric.reset()
            self._ap_last_key = current_key

        self.average_precision_metric.update(probs_for_ap, targets_flat)
        average_precision = self.average_precision_metric.compute()

        metrics.append(
            Metric(
                name="average_precision",
                value=average_precision.item(),
                stage=context.stage,
                epoch=context.epoch,
                batch_idx=context.batch_idx,
            )
        )

    return {"metrics": metrics}
```

##### pooled_metrics

```python
pooled_metrics()
```

Live torchmetrics objects for the epoch-pooled metrics, keyed by name.

`average_precision` accumulates across the epoch (reset only at the `(stage, epoch)` boundary), so the trainer logs this object with `on_epoch=True` and Lightning computes the single pooled AP and resets it at epoch end, exact and batch-size-invariant. Returns an empty mapping until at least one batch with `logits` has been seen, so nothing is logged for a run that never produced scores.

Source code in `cuvis_ai/node/metrics.py`

```python
def pooled_metrics(self) -> dict[str, TorchMetric]:
    """Live torchmetrics objects for the epoch-pooled metrics, keyed by name.

    ``average_precision`` accumulates across the epoch (reset only at the
    ``(stage, epoch)`` boundary), so the trainer logs this object with
    ``on_epoch=True`` and Lightning computes the single pooled AP and resets
    it at epoch end, exact and batch-size-invariant. Returns an empty mapping
    until at least one batch with ``logits`` has been seen, so nothing is
    logged for a run that never produced scores.
    """
    if self._ap_last_key is None:
        return {}
    return {"average_precision": self.average_precision_metric}
```

`AnomalyPixelStatisticsMetric`cuvis_ai.node.metricsmetricanomevalCompute anomaly pixel statistics from binary decisions.

#### AnomalyPixelStatisticsMetric

```python
AnomalyPixelStatisticsMetric(
    execution_stages=None, **kwargs
)
```

Bases: `Node`

Compute anomaly pixel statistics from binary decisions.

Calculates total pixels, anomalous pixels count, and anomaly percentage. Useful for monitoring the proportion of detected anomalies in batches. Executes only during validation and test stages.

Source code in `cuvis_ai/node/metrics.py`

```python
def __init__(
    self,
    execution_stages: set[ExecutionStage] | None = None,
    **kwargs,
) -> None:
    name, execution_stages = Node.consume_base_kwargs(
        kwargs, execution_stages or {ExecutionStage.VAL, ExecutionStage.TEST}
    )
    super().__init__(
        name=name,
        execution_stages=execution_stages,
        **kwargs,
    )
```

##### forward

```python
forward(decisions, context)
```

Compute anomaly pixel statistics.

Parameters:

| Name        | Type      | Description                                    | Default    |
| ----------- | --------- | ---------------------------------------------- | ---------- |
| `decisions` | `Tensor`  | Binary anomaly decisions [B, H, W, 1]          | *required* |
| `context`   | `Context` | Execution context with stage, epoch, batch_idx | *required* |

Returns:

| Type             | Description                                                     |
| ---------------- | --------------------------------------------------------------- |
| `dict[str, Any]` | Dictionary with "metrics" key containing list of Metric objects |

Source code in `cuvis_ai/node/metrics.py`

```python
def forward(self, decisions: Tensor, context: Context) -> dict[str, Any]:
    """Compute anomaly pixel statistics.

    Parameters
    ----------
    decisions : Tensor
        Binary anomaly decisions [B, H, W, 1]
    context : Context
        Execution context with stage, epoch, batch_idx

    Returns
    -------
    dict[str, Any]
        Dictionary with "metrics" key containing list of Metric objects
    """
    # Calculate statistics
    total_pixels = decisions.numel()
    anomalous_pixels = int(decisions.sum().item())
    anomaly_percentage = (anomalous_pixels / total_pixels) * 100 if total_pixels > 0 else 0.0

    metrics = [
        Metric(
            name="anomaly/total_pixels",
            value=float(total_pixels),
            stage=context.stage,
            epoch=context.epoch,
            batch_idx=context.batch_idx,
        ),
        Metric(
            name="anomaly/anomalous_pixels",
            value=float(anomalous_pixels),
            stage=context.stage,
            epoch=context.epoch,
            batch_idx=context.batch_idx,
        ),
        Metric(
            name="anomaly/anomaly_percentage",
            value=anomaly_percentage,
            stage=context.stage,
            epoch=context.epoch,
            batch_idx=context.batch_idx,
        ),
    ]

    return {"metrics": metrics}
```

`CLEARMetricNode`cuvis_ai_trackeval.nodemetricbboxevalnumpytracktrackeval[↗](https://github.com/cubert-hyperspectral/cuvis-ai-trackeval "Open plugin repo")Accumulate per-frame tracking data and compute CLEAR metrics in finalize().

Accumulate per-frame tracking data and compute CLEAR metrics in finalize().

Inputs

| Port                     | Dtype     | Shape        | Description |
| ------------------------ | --------- | ------------ | ----------- |
| `frame_id`               | `int64`   | `[1]`        |             |
| `pred_frame_id` optional | `int64`   | `[1]`        |             |
| `gt_bboxes`              | `float32` | `[1, -1, 4]` |             |
| `gt_track_ids`           | `int64`   | `[1, -1]`    |             |
| `pred_bboxes`            | `float32` | `[1, -1, 4]` |             |
| `pred_track_ids`         | `int64`   | `[1, -1]`    |             |

Outputs

| Port   | Dtype     | Shape | Description     |
| ------ | --------- | ----- | --------------- |
| `mota` | `float32` | `[1]` | MOTA            |
| `motp` | `float32` | `[1]` | MOTP            |
| `fp`   | `int64`   | `[1]` | False positives |
| `fn`   | `int64`   | `[1]` | False negatives |
| `idsw` | `int64`   | `[1]` | ID switches     |

[View plugin repo (v0.1.4)](https://github.com/cubert-hyperspectral/cuvis-ai-trackeval/tree/v0.1.4)

`ComponentOrthogonalityMetric`cuvis_ai.node.metricsmetricdim-redevalTrack orthogonality of PCA components during training.

#### ComponentOrthogonalityMetric

```python
ComponentOrthogonalityMetric(
    execution_stages=None, **kwargs
)
```

Bases: `Node`

Track orthogonality of PCA components during training.

Measures how close the component matrix is to being orthonormal. Executes only during validation and test stages.

Source code in `cuvis_ai/node/metrics.py`

```python
def __init__(
    self,
    execution_stages: set[ExecutionStage] | None = None,
    **kwargs,
) -> None:
    name, execution_stages = Node.consume_base_kwargs(
        kwargs, execution_stages or {ExecutionStage.VAL, ExecutionStage.TEST}
    )
    super().__init__(
        name=name,
        execution_stages=execution_stages,
        **kwargs,
    )
```

##### forward

```python
forward(components, context)
```

Compute component orthogonality metrics.

Parameters:

| Name         | Type      | Description                                      | Default    |
| ------------ | --------- | ------------------------------------------------ | ---------- |
| `components` | `Tensor`  | PCA components matrix [n_components, n_features] | *required* |
| `context`    | `Context` | Execution context with stage, epoch, batch_idx   | *required* |

Returns:

| Type             | Description                                                     |
| ---------------- | --------------------------------------------------------------- |
| `dict[str, Any]` | Dictionary with "metrics" key containing list of Metric objects |

Source code in `cuvis_ai/node/metrics.py`

```python
def forward(self, components: Tensor, context: Context) -> dict[str, Any]:
    """Compute component orthogonality metrics.

    Parameters
    ----------
    components : Tensor
        PCA components matrix [n_components, n_features]
    context : Context
        Execution context with stage, epoch, batch_idx

    Returns
    -------
    dict[str, Any]
        Dictionary with "metrics" key containing list of Metric objects
    """
    # Compute gram matrix: W @ W.T
    gram = components @ components.T
    n = components.shape[0]

    # Target: identity matrix
    eye = torch.eye(n, device=components.device, dtype=components.dtype)

    # Frobenius norm of difference
    orth_error = torch.norm(gram - eye, p="fro").item()

    # Average absolute deviation from identity
    avg_off_diagonal = (gram - eye).abs().mean().item()

    # Diagonal elements (should be close to 1)
    diagonal = torch.diagonal(gram)
    diagonal_mean = diagonal.mean().item()
    diagonal_std = diagonal.std().item()

    metrics = [
        Metric(
            name="orthogonality_error",
            value=orth_error,
            stage=context.stage,
            epoch=context.epoch,
            batch_idx=context.batch_idx,
        ),
        Metric(
            name="avg_off_diagonal",
            value=avg_off_diagonal,
            stage=context.stage,
            epoch=context.epoch,
            batch_idx=context.batch_idx,
        ),
        Metric(
            name="diagonal_mean",
            value=diagonal_mean,
            stage=context.stage,
            epoch=context.epoch,
            batch_idx=context.batch_idx,
        ),
        Metric(
            name="diagonal_std",
            value=diagonal_std,
            stage=context.stage,
            epoch=context.epoch,
            batch_idx=context.batch_idx,
        ),
    ]

    return {"metrics": metrics}
```

`DistinctLabelCount`cuvis_ai.node.metricsmetricevalmaskCount the distinct non-zero labels per frame in an integer label map.

#### DistinctLabelCount

```python
DistinctLabelCount(execution_stages=None, **kwargs)
```

Bases: `Node`

Count the distinct non-zero labels per frame in an integer label map.

Reports how many separate segments a label map contains, e.g. how many compartments survived a per-blob majority vote or how many clusters a frame holds. Emits the per-frame count both as a `count` tensor (for pipeline reads / notebook printing) and as `Metric` objects for training-time logging. Defaults to `ExecutionStage.ALWAYS` so it also runs under `Predictor` inference, not only validation / test.

Source code in `cuvis_ai/node/metrics.py`

```python
def __init__(
    self,
    execution_stages: set[ExecutionStage] | None = None,
    **kwargs,
) -> None:
    name, execution_stages = Node.consume_base_kwargs(
        kwargs, execution_stages or {ExecutionStage.ALWAYS}
    )
    super().__init__(name=name, execution_stages=execution_stages, **kwargs)
```

##### forward

```python
forward(mask, context)
```

Count distinct non-zero labels in each frame of *mask*.

Parameters:

| Name      | Type      | Description                                     | Default    |
| --------- | --------- | ----------------------------------------------- | ---------- |
| `mask`    | `Tensor`  | Integer label map [B, H, W]; 0 is background.   | *required* |
| `context` | `Context` | Execution context with stage, epoch, batch_idx. | *required* |

Returns:

| Type             | Description                                                                |
| ---------------- | -------------------------------------------------------------------------- |
| `dict[str, Any]` | count [B] int64 and a metrics list with one num_distinct_labels per frame. |

Source code in `cuvis_ai/node/metrics.py`

```python
def forward(self, mask: Tensor, context: Context) -> dict[str, Any]:
    """Count distinct non-zero labels in each frame of *mask*.

    Parameters
    ----------
    mask : Tensor
        Integer label map [B, H, W]; 0 is background.
    context : Context
        Execution context with stage, epoch, batch_idx.

    Returns
    -------
    dict[str, Any]
        ``count`` [B] int64 and a ``metrics`` list with one ``num_distinct_labels`` per frame.
    """
    counts = [int((torch.unique(mask[b]) != 0).sum().item()) for b in range(mask.shape[0])]
    count = torch.tensor(counts, dtype=torch.int64, device=mask.device)
    metrics = [
        Metric(
            name="num_distinct_labels",
            value=float(c),
            stage=context.stage,
            epoch=context.epoch,
            batch_idx=context.batch_idx,
        )
        for c in counts
    ]
    return {"count": count, "metrics": metrics}
```

`ExplainedVarianceMetric`cuvis_ai.node.metricsmetricdim-redevalTrack explained variance ratio for PCA components.

#### ExplainedVarianceMetric

```python
ExplainedVarianceMetric(execution_stages=None, **kwargs)
```

Bases: `Node`

Track explained variance ratio for PCA components.

Executes only during validation and test stages.

Source code in `cuvis_ai/node/metrics.py`

```python
def __init__(
    self,
    execution_stages: set[ExecutionStage] | None = None,
    **kwargs,
) -> None:
    name, execution_stages = Node.consume_base_kwargs(
        kwargs, execution_stages or {ExecutionStage.VAL, ExecutionStage.TEST}
    )
    super().__init__(
        name=name,
        execution_stages=execution_stages,
        **kwargs,
    )
```

##### forward

```python
forward(explained_variance_ratio, context)
```

Compute explained variance metrics.

Parameters:

| Name                       | Type      | Description                                    | Default    |
| -------------------------- | --------- | ---------------------------------------------- | ---------- |
| `explained_variance_ratio` | `Tensor`  | Explained variance ratios from PCA node        | *required* |
| `context`                  | `Context` | Execution context with stage, epoch, batch_idx | *required* |

Returns:

| Type             | Description                                                     |
| ---------------- | --------------------------------------------------------------- |
| `dict[str, Any]` | Dictionary with "metrics" key containing list of Metric objects |

Source code in `cuvis_ai/node/metrics.py`

```python
def forward(self, explained_variance_ratio: Tensor, context: Context) -> dict[str, Any]:
    """Compute explained variance metrics.

    Parameters
    ----------
    explained_variance_ratio : Tensor
        Explained variance ratios from PCA node
    context : Context
        Execution context with stage, epoch, batch_idx

    Returns
    -------
    dict[str, Any]
        Dictionary with "metrics" key containing list of Metric objects
    """
    metrics = []

    # Per-component variance
    for i, ratio in enumerate(explained_variance_ratio):
        metrics.append(
            Metric(
                name=f"explained_variance_pc{i + 1}",
                value=ratio.item(),
                stage=context.stage,
                epoch=context.epoch,
                batch_idx=context.batch_idx,
            )
        )

    # Total variance explained
    metrics.append(
        Metric(
            name="total_explained_variance",
            value=explained_variance_ratio.sum().item(),
            stage=context.stage,
            epoch=context.epoch,
            batch_idx=context.batch_idx,
        )
    )

    # Cumulative variance
    cumulative = torch.cumsum(explained_variance_ratio, dim=0)
    for i, cum_var in enumerate(cumulative):
        metrics.append(
            Metric(
                name=f"cumulative_variance_pc{i + 1}",
                value=cum_var.item(),
                stage=context.stage,
                epoch=context.epoch,
                batch_idx=context.batch_idx,
            )
        )

    return {"metrics": metrics}
```

`HOTAMetricNode`cuvis_ai_trackeval.nodemetricbboxevalnumpytracktrackeval[↗](https://github.com/cubert-hyperspectral/cuvis-ai-trackeval "Open plugin repo")Accumulate per-frame tracking data and compute HOTA in finalize().

Accumulate per-frame tracking data and compute HOTA in finalize().

Inputs

| Port                     | Dtype     | Shape        | Description |
| ------------------------ | --------- | ------------ | ----------- |
| `frame_id`               | `int64`   | `[1]`        |             |
| `pred_frame_id` optional | `int64`   | `[1]`        |             |
| `gt_bboxes`              | `float32` | `[1, -1, 4]` |             |
| `gt_track_ids`           | `int64`   | `[1, -1]`    |             |
| `pred_bboxes`            | `float32` | `[1, -1, 4]` |             |
| `pred_track_ids`         | `int64`   | `[1, -1]`    |             |
| `pred_scores` optional   | `float32` | `[1, -1]`    |             |

Outputs

| Port   | Dtype     | Shape | Description |
| ------ | --------- | ----- | ----------- |
| `hota` | `float32` | `[1]` | Mean HOTA   |
| `deta` | `float32` | `[1]` | Mean DetA   |
| `assa` | `float32` | `[1]` | Mean AssA   |
| `loca` | `float32` | `[1]` | Mean LocA   |

[View plugin repo (v0.1.4)](https://github.com/cubert-hyperspectral/cuvis-ai-trackeval/tree/v0.1.4)

`IdentityMetricNode`cuvis_ai_trackeval.nodemetricbboxevalnumpytracktrackeval[↗](https://github.com/cubert-hyperspectral/cuvis-ai-trackeval "Open plugin repo")Accumulate per-frame tracking data and compute ID metrics in finalize().

Accumulate per-frame tracking data and compute ID metrics in finalize().

Inputs

| Port                     | Dtype     | Shape        | Description |
| ------------------------ | --------- | ------------ | ----------- |
| `frame_id`               | `int64`   | `[1]`        |             |
| `pred_frame_id` optional | `int64`   | `[1]`        |             |
| `gt_bboxes`              | `float32` | `[1, -1, 4]` |             |
| `gt_track_ids`           | `int64`   | `[1, -1]`    |             |
| `pred_bboxes`            | `float32` | `[1, -1, 4]` |             |
| `pred_track_ids`         | `int64`   | `[1, -1]`    |             |

Outputs

| Port   | Dtype     | Shape | Description |
| ------ | --------- | ----- | ----------- |
| `idf1` | `float32` | `[1]` | IDF1        |
| `idp`  | `float32` | `[1]` | IDP         |
| `idr`  | `float32` | `[1]` | IDR         |

[View plugin repo (v0.1.4)](https://github.com/cubert-hyperspectral/cuvis-ai-trackeval/tree/v0.1.4)

`MulticlassSegmentationMetrics`cuvis_ai_inspecscrap.node.metricsmetricclassevalcuvis_ai_inspecscrap[↗](https://github.com/cubert-hyperspectral/cuvis-ai-inspecscrap "Open plugin repo")

[View plugin repo (v0.2.4)](https://github.com/cubert-hyperspectral/cuvis-ai-inspecscrap/tree/v0.2.4)

`PerClassAnomalyAUROC`cuvis_ai_dinomaly.node.per_class_aurocmetricanomevaldinomaly[↗](https://github.com/cubert-hyperspectral/cuvis-ai-dinomaly "Open plugin repo")Streaming one-vs-background pixel AUROC per class (val/test only).

Streaming one-vs-background pixel AUROC per class (val/test only).

Inputs

| Port         | Dtype     | Shape             | Description                                                         |
| ------------ | --------- | ----------------- | ------------------------------------------------------------------- |
| `scores`     | `float32` | `[-1, -1, -1, 1]` | Raw anomaly map [B, H, W, 1]                                        |
| `class_mask` | `int32`   | `[-1, -1, -1, 1]` | Multi-class ground-truth mask [B, H, W, 1] (background_id = normal) |

Outputs

| Port      | Dtype | Shape | Description                                      |
| --------- | ----- | ----- | ------------------------------------------------ |
| `metrics` | `any` | `any` | List of Metric objects (running per-class AUROC) |

[View plugin repo (v0.6.3)](https://github.com/cubert-hyperspectral/cuvis-ai-dinomaly/tree/v0.6.3)

`ScoreStatisticsMetric`cuvis_ai.node.metricsmetricanomevalCompute statistical properties of score distributions.

#### ScoreStatisticsMetric

```python
ScoreStatisticsMetric(execution_stages=None, **kwargs)
```

Bases: `Node`

Compute statistical properties of score distributions.

Tracks mean, std, min, max, median, and quantiles of scores. Executes only during validation and test stages.

Source code in `cuvis_ai/node/metrics.py`

```python
def __init__(
    self,
    execution_stages: set[ExecutionStage] | None = None,
    **kwargs,
) -> None:
    name, execution_stages = Node.consume_base_kwargs(
        kwargs, execution_stages or {ExecutionStage.VAL, ExecutionStage.TEST}
    )
    super().__init__(
        name=name,
        execution_stages=execution_stages,
        **kwargs,
    )
```

##### forward

```python
forward(scores, context)
```

Compute score statistics.

Parameters:

| Name      | Type      | Description                                    | Default    |
| --------- | --------- | ---------------------------------------------- | ---------- |
| `scores`  | `Tensor`  | Score values [B, H, W]                         | *required* |
| `context` | `Context` | Execution context with stage, epoch, batch_idx | *required* |

Returns:

| Type             | Description                                                     |
| ---------------- | --------------------------------------------------------------- |
| `dict[str, Any]` | Dictionary with "metrics" key containing list of Metric objects |

Source code in `cuvis_ai/node/metrics.py`

```python
def forward(self, scores: Tensor, context: Context) -> dict[str, Any]:
    """Compute score statistics.

    Parameters
    ----------
    scores : Tensor
        Score values [B, H, W]
    context : Context
        Execution context with stage, epoch, batch_idx

    Returns
    -------
    dict[str, Any]
        Dictionary with "metrics" key containing list of Metric objects
    """
    # Flatten scores
    scores_flat = scores.reshape(-1)

    metrics = [
        Metric(
            name="scores/mean",
            value=scores_flat.mean().item(),
            stage=context.stage,
            epoch=context.epoch,
            batch_idx=context.batch_idx,
        ),
        Metric(
            name="scores/std",
            value=scores_flat.std().item(),
            stage=context.stage,
            epoch=context.epoch,
            batch_idx=context.batch_idx,
        ),
        Metric(
            name="scores/min",
            value=scores_flat.min().item(),
            stage=context.stage,
            epoch=context.epoch,
            batch_idx=context.batch_idx,
        ),
        Metric(
            name="scores/max",
            value=scores_flat.max().item(),
            stage=context.stage,
            epoch=context.epoch,
            batch_idx=context.batch_idx,
        ),
        Metric(
            name="scores/median",
            value=scores_flat.median().item(),
            stage=context.stage,
            epoch=context.epoch,
            batch_idx=context.batch_idx,
        ),
        Metric(
            name="scores/q25",
            value=torch.quantile(scores_flat, 0.25).item(),
            stage=context.stage,
            epoch=context.epoch,
            batch_idx=context.batch_idx,
        ),
        Metric(
            name="scores/q75",
            value=torch.quantile(scores_flat, 0.75).item(),
            stage=context.stage,
            epoch=context.epoch,
            batch_idx=context.batch_idx,
        ),
        Metric(
            name="scores/q95",
            value=torch.quantile(scores_flat, 0.95).item(),
            stage=context.stage,
            epoch=context.epoch,
            batch_idx=context.batch_idx,
        ),
        Metric(
            name="scores/q99",
            value=torch.quantile(scores_flat, 0.99).item(),
            stage=context.stage,
            epoch=context.epoch,
            batch_idx=context.batch_idx,
        ),
    ]

    return {"metrics": metrics}
```

`SegMetrics`cuvis_ai_unet.node.seg_metricsmetricevalsegunet[↗](https://github.com/cubert-hyperspectral/cuvis-ai-unet "Open plugin repo")Foreground IoU / Dice / pixel-accuracy over per-pixel class logits.

Foreground IoU / Dice / pixel-accuracy over per-pixel class logits.

Inputs

| Port      | Dtype     | Shape              | Description                                   |
| --------- | --------- | ------------------ | --------------------------------------------- |
| `logits`  | `float32` | `[-1, -1, -1, -1]` | Per-pixel class logits [B, H, W, num_classes] |
| `targets` | `int32`   | `[-1, -1, -1]`     | Integer per-pixel class targets [B, H, W]     |

Outputs

| Port      | Dtype | Shape | Description  |
| --------- | ----- | ----- | ------------ |
| `metrics` | `any` | `any` | list[Metric] |

[View plugin repo (v0.2.2)](https://github.com/cubert-hyperspectral/cuvis-ai-unet/tree/v0.2.2)

`SelectorDiversityMetric`cuvis_ai.node.metricsmetricdim-redevalTrack diversity of channel selection.

#### SelectorDiversityMetric

```python
SelectorDiversityMetric(execution_stages=None, **kwargs)
```

Bases: `Node`

Track diversity of channel selection.

Measures how spread out the selection weights are across channels. Uses Gini coefficient - lower values indicate more diverse selection.

Executes only during validation and test stages.

Source code in `cuvis_ai/node/metrics.py`

```python
def __init__(
    self,
    execution_stages: set[ExecutionStage] | None = None,
    **kwargs,
) -> None:
    name, execution_stages = Node.consume_base_kwargs(
        kwargs, execution_stages or {ExecutionStage.VAL, ExecutionStage.TEST}
    )
    super().__init__(
        name=name,
        execution_stages=execution_stages,
        **kwargs,
    )
```

##### forward

```python
forward(weights, context)
```

Compute diversity metrics for selection weights.

Parameters:

| Name      | Type      | Description                                    | Default    |
| --------- | --------- | ---------------------------------------------- | ---------- |
| `weights` | `Tensor`  | Channel selection weights [n_channels]         | *required* |
| `context` | `Context` | Execution context with stage, epoch, batch_idx | *required* |

Returns:

| Type             | Description                                                     |
| ---------------- | --------------------------------------------------------------- |
| `dict[str, Any]` | Dictionary with "metrics" key containing list of Metric objects |

Source code in `cuvis_ai/node/metrics.py`

```python
def forward(self, weights: Tensor, context: Context) -> dict[str, Any]:
    """Compute diversity metrics for selection weights.

    Parameters
    ----------
    weights : Tensor
        Channel selection weights [n_channels]
    context : Context
        Execution context with stage, epoch, batch_idx

    Returns
    -------
    dict[str, Any]
        Dictionary with "metrics" key containing list of Metric objects
    """
    # Compute variance (measure of spread)
    mean_weight = weights.mean()
    variance = ((weights - mean_weight) ** 2).mean()

    # Compute Gini coefficient (0 = perfect equality, 1 = perfect inequality)
    # Lower Gini = more diverse selection
    sorted_weights, _ = torch.sort(weights)
    n = len(sorted_weights)
    index = torch.arange(1, n + 1, device=weights.device, dtype=weights.dtype)
    gini = (2 * (sorted_weights * index).sum()) / (n * sorted_weights.sum()) - (n + 1) / n

    metrics = [
        Metric(
            name="weight_variance",
            value=variance.item(),
            stage=context.stage,
            epoch=context.epoch,
            batch_idx=context.batch_idx,
        ),
        Metric(
            name="gini_coefficient",
            value=gini.item(),
            stage=context.stage,
            epoch=context.epoch,
            batch_idx=context.batch_idx,
        ),
    ]

    return {"metrics": metrics}
```

`SelectorEntropyMetric`cuvis_ai.node.metricsmetricdim-redevalTrack entropy of channel selection distribution.

#### SelectorEntropyMetric

```python
SelectorEntropyMetric(
    eps=1e-06, execution_stages=None, **kwargs
)
```

Bases: `Node`

Track entropy of channel selection distribution.

Measures the uncertainty/diversity in channel selection weights. Higher entropy indicates more uniform selection (less confident). Lower entropy indicates more peaked selection (more confident).

Executes only during validation and test stages.

Source code in `cuvis_ai/node/metrics.py`

```python
def __init__(
    self,
    eps: float = 1e-6,
    execution_stages: set[ExecutionStage] | None = None,
    **kwargs,
) -> None:
    self.eps = eps
    name, execution_stages = Node.consume_base_kwargs(
        kwargs, execution_stages or {ExecutionStage.VAL, ExecutionStage.TEST}
    )
    super().__init__(
        name=name,
        execution_stages=execution_stages,
        eps=eps,
        **kwargs,
    )
```

##### forward

```python
forward(weights, context)
```

Compute entropy of selection weights.

Parameters:

| Name      | Type      | Description                                    | Default    |
| --------- | --------- | ---------------------------------------------- | ---------- |
| `weights` | `Tensor`  | Channel selection weights [n_channels]         | *required* |
| `context` | `Context` | Execution context with stage, epoch, batch_idx | *required* |

Returns:

| Type             | Description                                                     |
| ---------------- | --------------------------------------------------------------- |
| `dict[str, Any]` | Dictionary with "metrics" key containing list of Metric objects |

Source code in `cuvis_ai/node/metrics.py`

```python
def forward(self, weights: Tensor, context: Context) -> dict[str, Any]:
    """Compute entropy of selection weights.

    Parameters
    ----------
    weights : Tensor
        Channel selection weights [n_channels]
    context : Context
        Execution context with stage, epoch, batch_idx

    Returns
    -------
    dict[str, Any]
        Dictionary with "metrics" key containing list of Metric objects
    """
    # Normalize weights to probabilities
    probs = weights / (weights.sum() + self.eps)

    # Compute entropy: -sum(p * log(p))
    entropy = -(probs * torch.log(probs + self.eps)).sum()

    metrics = [
        Metric(
            name="selector/entropy",
            value=entropy.item(),
            stage=context.stage,
            epoch=context.epoch,
            batch_idx=context.batch_idx,
        ),
    ]

    return {"metrics": metrics}
```

`ValNormalAnomalyMean`cuvis_ai_dinomaly.node.val_anomaly_meanmetricanomevaldinomaly[↗](https://github.com/cubert-hyperspectral/cuvis-ai-dinomaly "Open plugin repo")Running mean of the image-level `anomaly_score` per `(stage, epoch)`.

Running mean of the image-level `anomaly_score` per `(stage, epoch)`.

Inputs

| Port            | Dtype     | Shape  | Description                 |
| --------------- | --------- | ------ | --------------------------- |
| `anomaly_score` | `float32` | `[-1]` | Per-image anomaly score [B] |

Outputs

| Port      | Dtype | Shape | Description                                          |
| --------- | ----- | ----- | ---------------------------------------------------- |
| `metrics` | `any` | `any` | List of Metric objects (running mean_anomaly_normal) |

[View plugin repo (v0.6.3)](https://github.com/cubert-hyperspectral/cuvis-ai-dinomaly/tree/v0.6.3)

`AdaCLIPDetector`cuvis_ai_adaclip.node.adaclip_nodemodelanomimginferlearnmaskrgbtorchadaclip[↗](https://github.com/cubert-hyperspectral/cuvis-ai-adaclip "Open plugin repo")AdaCLIP zero-shot anomaly detector node (plugin version).

AdaCLIP zero-shot anomaly detector node (plugin version).

Inputs

| Port        | Dtype     | Shape             | Description                                            |
| ----------- | --------- | ----------------- | ------------------------------------------------------ |
| `rgb_image` | `float32` | `[-1, -1, -1, 3]` | RGB image [B, H, W, 3] in float32 (0-1 or 0-255 range) |

Outputs

| Port                        | Dtype     | Shape              | Description                                                                                                                                                         |
| --------------------------- | --------- | ------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `scores`                    | `float32` | `[-1, -1, -1, 1]`  | Pixel-level anomaly scores [B, H, W, 1]                                                                                                                             |
| `anomaly_score`             | `float32` | `[-1]`             | Image-level anomaly score [B]                                                                                                                                       |
| `per_layer_scores` optional | `float32` | `[-1, -1, -1, -1]` | Per-layer softmaxed anomaly maps stacked as [B, num_layers\*2, H, W]. Only populated during training when training_aggregation=False. Empty (1×1) tensor otherwise. |
| `image_score_2ch` optional  | `float32` | `[-1, -1]`         | Image-level 2-channel score [B, 2] (softmaxed [normal, anomaly]). Only populated during training when training_aggregation=False.                                   |

[View plugin repo (v0.3.1)](https://github.com/cubert-hyperspectral/cuvis-ai-adaclip/tree/v0.3.1)

`ConcreteChannelMixer`cuvis_ai.node.channel_mixermodeldim-redhsilearnpretorchConcrete/Gumbel-Softmax channel mixer for hyperspectral cubes.

#### ConcreteChannelMixer

```python
ConcreteChannelMixer(
    input_channels,
    output_channels=3,
    tau_start=10.0,
    tau_end=0.1,
    max_epochs=20,
    use_hard_inference=True,
    eps=1e-06,
    **kwargs,
)
```

Bases: `Node`

Concrete/Gumbel-Softmax channel mixer for hyperspectral cubes.

Learns `K` categorical distributions over `T` input bands, and during training uses the Gumbel-Softmax trick to produce differentiable approximate one-hot selection weights that become increasingly peaked as the temperature :math:`\tau` is annealed.

For each output channel :math:`c \in {1, \dots, K}`, we learn logits `L_c in R^T` and sample:

.. math::

```text
w_c = \text{softmax}\left( \frac{L_c + g}{\tau} \right), \quad
g \sim \text{Gumbel}(0, 1)
```

The resulting weights are used to form K-channel RGB-like images:

.. math::

```text
Y[:, :, c] = \sum_{t=1}^T w_c[t] \cdot X[:, :, t]
```

where `X` is the input hyperspectral cube in `[0, 1]`.

Parameters:

| Name                 | Type    | Description                                                                                                                             | Default    |
| -------------------- | ------- | --------------------------------------------------------------------------------------------------------------------------------------- | ---------- |
| `input_channels`     | `int`   | Number of input spectral channels (e.g., 61 for hyperspectral cube).                                                                    | *required* |
| `output_channels`    | `int`   | Number of output channels (default: 3 for RGB/AdaClip compatibility).                                                                   | `3`        |
| `tau_start`          | `float` | Initial temperature for Gumbel-Softmax (default: 10.0).                                                                                 | `10.0`     |
| `tau_end`            | `float` | Final temperature for Gumbel-Softmax (default: 0.1).                                                                                    | `0.1`      |
| `max_epochs`         | `int`   | Number of epochs over which to exponentially anneal :math:\\tau from tau_start to tau_end (default: 20).                                | `20`       |
| `use_hard_inference` | `bool`  | If True, uses hard argmax selection at inference/validation time (one-hot weights). If False, uses softmax over logits (default: True). | `True`     |
| `eps`                | `float` | Small constant for numerical stability (default: 1e-6).                                                                                 | `1e-06`    |

Notes

- During training (`context.stage == 'train'`), the node samples Gumbel noise and uses the Concrete relaxation with the current temperature :math:\`\\tau(\\text{epoch})\`\`.
- During validation/test/inference, it uses deterministic weights without Gumbel noise.
- The node exposes `selection_weights` so that repulsion penalties (e.g., DistinctnessLoss) can be attached in the pipeline.

Source code in `cuvis_ai/node/channel_mixer.py`

```python
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
```

##### get_selection_weights

```python
get_selection_weights(deterministic=True)
```

Return current selection weights without data dependency.

Parameters:

| Name            | Type   | Description                                                                                                                                                | Default |
| --------------- | ------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------- | ------- |
| `deterministic` | `bool` | If True, uses softmax over logits (no Gumbel noise) at a "midpoint" temperature (geometric mean of start/end). If False, uses current logits with tau_end. | `True`  |

Source code in `cuvis_ai/node/channel_mixer.py`

```python
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
```

##### get_selected_bands

```python
get_selected_bands()
```

Return argmax band indices per output channel.

Source code in `cuvis_ai/node/channel_mixer.py`

```python
def get_selected_bands(self) -> Tensor:
    """Return argmax band indices per output channel."""
    with torch.no_grad():
        return torch.argmax(self.logits, dim=-1)
```

##### forward

```python
forward(data, context=None, **_)
```

Apply Concrete/Gumbel-Softmax channel mixing.

Parameters:

| Name      | Type      | Description                                         | Default    |
| --------- | --------- | --------------------------------------------------- | ---------- |
| `data`    | `Tensor`  | Input tensor [B, H, W, C_in] in BHWC format.        | *required* |
| `context` | `Context` | Execution context with stage and epoch information. | `None`     |

Returns:

| Type                | Description                                                                                                  |
| ------------------- | ------------------------------------------------------------------------------------------------------------ |
| `dict[str, Tensor]` | Dictionary with: "rgb": [B, H, W, C_out] RGB-like image. "selection_weights": [C_out, C_in] current weights. |

Source code in `cuvis_ai/node/channel_mixer.py`

```python
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
```

`DeepSVDDProjection`cuvis_ai.node.anomaly.deep_svddmodelanomhsiinferlearntorchProjection head that maps per-pixel features to Deep SVDD embeddings.

#### DeepSVDDProjection

```python
DeepSVDDProjection(
    *,
    in_channels,
    rep_dim=32,
    hidden=128,
    kernel="linear",
    n_rff=2048,
    gamma=None,
    mlp_forward_batch_size=65536,
    **kwargs,
)
```

Bases: `Node`

Projection head that maps per-pixel features to Deep SVDD embeddings.

Source code in `cuvis_ai/node/anomaly/deep_svdd.py`

```python
def __init__(
    self,
    *,
    in_channels: int,
    rep_dim: int = 32,
    hidden: int = 128,
    kernel: str = "linear",
    n_rff: int = 2048,
    gamma: float | None = None,
    mlp_forward_batch_size: int = 65_536,
    **kwargs: Any,
) -> None:
    if in_channels <= 0:
        raise ValueError(f"in_channels must be positive, got {in_channels}")
    self.in_channels = int(in_channels)
    self.rep_dim = int(rep_dim)
    self.hidden = int(hidden)
    self.kernel = str(kernel)
    self.n_rff = int(n_rff)
    self.gamma = None if gamma is None else float(gamma)
    self.mlp_forward_batch_size = max(1, int(mlp_forward_batch_size))

    super().__init__(
        in_channels=self.in_channels,
        rep_dim=self.rep_dim,
        hidden=self.hidden,
        kernel=self.kernel,
        n_rff=self.n_rff,
        gamma=self.gamma,
        mlp_forward_batch_size=self.mlp_forward_batch_size,
        **kwargs,
    )

    # Build projection network eagerly with known in_channels
    self._build_network()
```

##### forward

```python
forward(data, **_)
```

Project BHWC features into a latent embedding space.

Source code in `cuvis_ai/node/anomaly/deep_svdd.py`

```python
def forward(self, data: torch.Tensor, **_: Any) -> dict[str, torch.Tensor]:
    """Project BHWC features into a latent embedding space."""
    B, H, W, C = data.shape
    if C != self.in_channels:
        raise ValueError(f"Expected {self.in_channels} channels, got {C}")

    flat = data.contiguous().reshape(B * H * W, C)

    batch_size = self.mlp_forward_batch_size
    embeddings = []
    for start in range(0, flat.shape[0], batch_size):
        chunk = flat[start : start + batch_size]
        embeddings.append(self.net(chunk))
    z = torch.cat(embeddings, dim=0).reshape(B, H, W, self.rep_dim)

    return {"embeddings": z}
```

`DeepSVDDScores`cuvis_ai.node.anomaly.deep_svddmodelanominferlearntorchConvert Deep SVDD embeddings + center vector into anomaly scores.

#### DeepSVDDScores

Bases: `Node`

Convert Deep SVDD embeddings + center vector into anomaly scores.

##### forward

```python
forward(embeddings, center, **_)
```

Compute anomaly scores as squared distance from center.

Parameters:

| Name         | Type     | Description                                                | Default    |
| ------------ | -------- | ---------------------------------------------------------- | ---------- |
| `embeddings` | `Tensor` | Deep SVDD embeddings [B, H, W, D] from projection network. | *required* |
| `center`     | `Tensor` | Center vector [D] from DeepSVDDCenterTracker.              | *required* |
| `**_`        | `Any`    | Additional unused keyword arguments.                       | `{}`       |

Returns:

| Type                | Description                                                             |
| ------------------- | ----------------------------------------------------------------------- |
| `dict[str, Tensor]` | Dictionary with "scores" key containing squared distances [B, H, W, 1]. |

Source code in `cuvis_ai/node/anomaly/deep_svdd.py`

```python
def forward(
    self, embeddings: torch.Tensor, center: torch.Tensor, **_: Any
) -> dict[str, torch.Tensor]:
    """Compute anomaly scores as squared distance from center.

    Parameters
    ----------
    embeddings : torch.Tensor
        Deep SVDD embeddings [B, H, W, D] from projection network.
    center : torch.Tensor
        Center vector [D] from DeepSVDDCenterTracker.
    **_ : Any
        Additional unused keyword arguments.

    Returns
    -------
    dict[str, torch.Tensor]
        Dictionary with "scores" key containing squared distances [B, H, W, 1].
    """
    scores = ((embeddings - center.view(1, 1, 1, -1)) ** 2).sum(dim=-1, keepdim=True)
    return {"scores": scores}
```

`DinomalyDetector`cuvis_ai_dinomaly.node.dinomaly_detectormodelanomhsilearnrecontorchdinomaly[↗](https://github.com/cubert-hyperspectral/cuvis-ai-dinomaly "Open plugin repo")Pixel-level anomaly detection using Anomalib's `DinomalyModel`.

Pixel-level anomaly detection using Anomalib's `DinomalyModel`.

Inputs

| Port        | Dtype     | Shape              | Description                                                                                                           |
| ----------- | --------- | ------------------ | --------------------------------------------------------------------------------------------------------------------- |
| `rgb_image` | `float32` | `[-1, -1, -1, -1]` | Channel-stacked image [B, H, W, C] in float32 (0–1 or 0–255). C must equal the detector's input_channels (default 3). |

Outputs

| Port                     | Dtype     | Shape             | Description                                           |
| ------------------------ | --------- | ----------------- | ----------------------------------------------------- |
| `scores`                 | `float32` | `[-1, -1, -1, 1]` | Pixel-wise anomaly scores [B, H, W, 1]                |
| `anomaly_score`          | `float32` | `[-1]`            | Image-level anomaly score [B]                         |
| `training_loss` optional | `float32` | `any`             | Scalar Dinomaly training loss (train/val/test stages) |

[View plugin repo (v0.6.3)](https://github.com/cubert-hyperspectral/cuvis-ai-dinomaly/tree/v0.6.3)

`DynUNet`cuvis_ai_unet.node.dynunetmodeldiffhsiinferlearnsegtorchunet[↗](https://github.com/cubert-hyperspectral/cuvis-ai-unet "Open plugin repo")Configurable U-Net for hyperspectral segmentation (2D / factorised 2.5D / 3D).

Configurable U-Net for hyperspectral segmentation (2D / factorised 2.5D / 3D).

Inputs

| Port   | Dtype     | Shape              | Description                           |
| ------ | --------- | ------------------ | ------------------------------------- |
| `data` | `float32` | `[-1, -1, -1, -1]` | Hyperspectral cube [B, H, W, C_bands] |

Outputs

| Port     | Dtype     | Shape              | Description                                   |
| -------- | --------- | ------------------ | --------------------------------------------- |
| `logits` | `float32` | `[-1, -1, -1, -1]` | Per-pixel class logits [B, H, W, num_classes] |

[View plugin repo (v0.2.2)](https://github.com/cubert-hyperspectral/cuvis-ai-unet/tree/v0.2.2)

`GaussianMixtureClusterer`cuvis_ai.node.clustering.gmmmodelclasshsistatefultorchCluster pixel spectra with a Gaussian mixture model.

#### GaussianMixtureClusterer

```python
GaussianMixtureClusterer(
    n_components=3,
    covariance_type="full",
    reg_covar=1e-06,
    max_iter=100,
    n_init=1,
    random_state=0,
    **kwargs,
)
```

Bases: `_StatisticalFitNode`

Cluster pixel spectra with a Gaussian mixture model.

The node is fitted once via `statistical_initialization` (scikit-learn's `GaussianMixture`). The means, mixture weights, and Cholesky factors of the precision matrices fully determine the Gaussian log-probabilities, so they are frozen as torch buffers and the forward pass needs no sklearn.

Only `covariance_type="full"` is supported: the torch forward assumes a `[K, C, C]` Cholesky factor, so `__init__` rejects any other value with `ValueError` rather than failing later at inference.

Parameters:

| Name              | Type    | Description                                                                                                                | Default  |
| ----------------- | ------- | -------------------------------------------------------------------------------------------------------------------------- | -------- |
| `n_components`    | `int`   | Number of mixture components (default: 3).                                                                                 | `3`      |
| `covariance_type` | `str`   | scikit-learn covariance parametrization; only "full" is supported and any other value raises ValueError (default: "full"). | `'full'` |
| `reg_covar`       | `float` | Non-negative regularization added to the covariance diagonals at fit for numerical stability (default: 1e-6).              | `1e-06`  |
| `max_iter`        | `int`   | Maximum EM iterations at fit (default: 100).                                                                               | `100`    |
| `n_init`          | `int`   | Number of seeded EM re-initializations at fit (default: 1).                                                                | `1`      |
| `random_state`    | `int`   | Seed for the sklearn fit, for reproducible parameters (default: 0).                                                        | `0`      |
| `**kwargs`        | `Any`   | Forwarded to \_StatisticalFitNode (max_fit_pixels, fit_seed) and the Node base.                                            | `{}`     |

Attributes:

| Name              | Type     | Description                                                                              |
| ----------------- | -------- | ---------------------------------------------------------------------------------------- |
| `means`           | `Tensor` | Component means, shape [K, C] after fit.                                                 |
| `precisions_chol` | `Tensor` | Cholesky factors of the precision matrices, shape [K, C, C] after fit (full covariance). |
| `weights`         | `Tensor` | Mixture weights, shape [K] after fit; sum to 1.                                          |

Store mixture hyperparameters and register the fitted-state buffers.

Source code in `cuvis_ai/node/clustering/gmm.py`

```python
def __init__(
    self,
    n_components: int = 3,
    covariance_type: str = "full",
    reg_covar: float = 1e-6,
    max_iter: int = 100,
    n_init: int = 1,
    random_state: int = 0,
    **kwargs: Any,
) -> None:
    """Store mixture hyperparameters and register the fitted-state buffers."""
    self.n_components = int(n_components)
    self.covariance_type = str(covariance_type)
    if self.covariance_type != "full":
        raise ValueError(
            "GaussianMixtureClusterer only supports covariance_type='full'; "
            f"got {self.covariance_type!r}. Other parametrizations produce a "
            "differently shaped precisions_cholesky_ that the torch forward "
            "cannot evaluate."
        )
    self.reg_covar = float(reg_covar)
    self.max_iter = int(max_iter)
    self.n_init = int(n_init)
    self.random_state = int(random_state)
    super().__init__(
        n_components=self.n_components,
        covariance_type=self.covariance_type,
        reg_covar=self.reg_covar,
        max_iter=self.max_iter,
        n_init=self.n_init,
        random_state=self.random_state,
        **kwargs,
    )
    self.register_buffer("means", torch.zeros(0, dtype=torch.float32))
    self.register_buffer("precisions_chol", torch.zeros(0, dtype=torch.float32))
    self.register_buffer("weights", torch.zeros(0, dtype=torch.float32))
```

##### forward

```python
forward(cube, **_)
```

Evaluate the mixture posterior for every pixel.

Parameters:

| Name   | Type     | Description                                                      | Default    |
| ------ | -------- | ---------------------------------------------------------------- | ---------- |
| `cube` | `Tensor` | Input hyperspectral cube [B, H, W, C].                           | *required* |
| `**_`  | `Any`    | Additional unused keyword arguments (e.g. the pipeline context). | `{}`       |

Returns:

| Type                | Description                                                                                                                                                                             |
| ------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `dict[str, Tensor]` | class_mask int32 [B, H, W] (argmax component), abundances float32 [B, H, W, K] (responsibilities, sum to 1 over K), and scores float32 [B, H, W, 1] (per-pixel mixture log-likelihood). |

Source code in `cuvis_ai/node/clustering/gmm.py`

```python
@torch.no_grad()
def forward(self, cube: torch.Tensor, **_: Any) -> dict[str, torch.Tensor]:
    """Evaluate the mixture posterior for every pixel.

    Parameters
    ----------
    cube : torch.Tensor
        Input hyperspectral cube ``[B, H, W, C]``.
    **_ : Any
        Additional unused keyword arguments (e.g. the pipeline ``context``).

    Returns
    -------
    dict[str, torch.Tensor]
        ``class_mask`` int32 ``[B, H, W]`` (argmax component),
        ``abundances`` float32 ``[B, H, W, K]`` (responsibilities, sum to 1
        over K), and ``scores`` float32 ``[B, H, W, 1]`` (per-pixel mixture
        log-likelihood).
    """
    self._require_initialized()
    B, H, W, C = cube.shape
    K = self.weights.shape[0]
    flat = cube.reshape(-1, C).to(torch.float32)

    log_weighted = self._estimate_weighted_log_prob(flat)  # [P, K]
    log_norm = torch.logsumexp(log_weighted, dim=1)  # [P]
    log_resp = log_weighted - log_norm.unsqueeze(1)  # [P, K]

    class_mask = log_weighted.argmax(dim=1).reshape(B, H, W).to(torch.int32)
    abundances = torch.exp(log_resp).reshape(B, H, W, K).to(torch.float32)
    scores = log_norm.reshape(B, H, W, 1).to(torch.float32)
    return {"class_mask": class_mask, "abundances": abundances, "scores": scores}
```

`KMeansClusterer`cuvis_ai.node.clustering.kmeansmodelclasshsistatefultorchPartition pixel spectra into `n_clusters` groups by nearest centroid.

#### KMeansClusterer

```python
KMeansClusterer(
    n_clusters=8,
    init="k-means++",
    n_init=10,
    max_iter=300,
    random_state=0,
    **kwargs,
)
```

Bases: `_StatisticalFitNode`

Partition pixel spectra into `n_clusters` groups by nearest centroid.

The node is fitted once via `statistical_initialization` (scikit-learn's `KMeans`); the resulting centroids are stored as a torch buffer. At inference each pixel is assigned to its nearest centroid in Euclidean space, emitting the 0-based cluster id and the distance to that centroid.

Cluster ids are emitted directly in the range `0 .. n_clusters - 1` (no background / `-1` sentinel is used).

Parameters:

| Name           | Type  | Description                                                                     | Default       |
| -------------- | ----- | ------------------------------------------------------------------------------- | ------------- |
| `n_clusters`   | `int` | Number of clusters to fit (default: 8).                                         | `8`           |
| `init`         | `str` | scikit-learn KMeans initialization method (default: "k-means++").               | `'k-means++'` |
| `n_init`       | `int` | Number of seeded re-initializations sklearn runs at fit (default: 10).          | `10`          |
| `max_iter`     | `int` | Maximum Lloyd iterations per run (default: 300).                                | `300`         |
| `random_state` | `int` | Seed for the sklearn fit, for reproducible centroids (default: 0).              | `0`           |
| `**kwargs`     | `Any` | Forwarded to \_StatisticalFitNode (max_fit_pixels, fit_seed) and the Node base. | `{}`          |

Attributes:

| Name        | Type     | Description                                                                                 |
| ----------- | -------- | ------------------------------------------------------------------------------------------- |
| `centroids` | `Tensor` | Fitted cluster centers, shape [n_clusters, C] after fit; a length-0 placeholder before fit. |

Store K-means hyperparameters and register the centroid buffer.

Source code in `cuvis_ai/node/clustering/kmeans.py`

```python
def __init__(
    self,
    n_clusters: int = 8,
    init: str = "k-means++",
    n_init: int = 10,
    max_iter: int = 300,
    random_state: int = 0,
    **kwargs: Any,
) -> None:
    """Store K-means hyperparameters and register the centroid buffer."""
    self.n_clusters = int(n_clusters)
    self.init = str(init)
    self.n_init = int(n_init)
    self.max_iter = int(max_iter)
    self.random_state = int(random_state)
    super().__init__(
        n_clusters=self.n_clusters,
        init=self.init,
        n_init=self.n_init,
        max_iter=self.max_iter,
        random_state=self.random_state,
        **kwargs,
    )
    self.register_buffer("centroids", torch.zeros(0, dtype=torch.float32))
```

##### forward

```python
forward(cube, **_)
```

Assign each pixel to its nearest centroid.

Parameters:

| Name   | Type     | Description                                                      | Default    |
| ------ | -------- | ---------------------------------------------------------------- | ---------- |
| `cube` | `Tensor` | Input hyperspectral cube [B, H, W, C].                           | *required* |
| `**_`  | `Any`    | Additional unused keyword arguments (e.g. the pipeline context). | `{}`       |

Returns:

| Type                | Description                                                                                                          |
| ------------------- | -------------------------------------------------------------------------------------------------------------------- |
| `dict[str, Tensor]` | class_mask int32 [B, H, W] (0-based cluster id) and scores float32 [B, H, W, 1] (distance to the assigned centroid). |

Source code in `cuvis_ai/node/clustering/kmeans.py`

```python
@torch.no_grad()
def forward(self, cube: torch.Tensor, **_: Any) -> dict[str, torch.Tensor]:
    """Assign each pixel to its nearest centroid.

    Parameters
    ----------
    cube : torch.Tensor
        Input hyperspectral cube ``[B, H, W, C]``.
    **_ : Any
        Additional unused keyword arguments (e.g. the pipeline ``context``).

    Returns
    -------
    dict[str, torch.Tensor]
        ``class_mask`` int32 ``[B, H, W]`` (0-based cluster id) and
        ``scores`` float32 ``[B, H, W, 1]`` (distance to the assigned
        centroid).
    """
    self._require_initialized()
    B, H, W, C = cube.shape
    flat = cube.reshape(-1, C).to(torch.float32)
    d = torch.cdist(flat, self.centroids.to(device=flat.device, dtype=flat.dtype))
    class_mask = d.argmin(dim=1).reshape(B, H, W).to(torch.int32)
    scores = d.min(dim=1).values.reshape(B, H, W, 1).to(torch.float32)
    return {"class_mask": class_mask, "scores": scores}
```

`LADGlobal`cuvis_ai.node.anomaly.lad_detectormodelanomhsilearnnumpyLaplacian Anomaly Detector (global), variant 'C' (Cauchy), port-based.

#### LADGlobal

```python
LADGlobal(
    num_channels,
    eps=1e-08,
    normalize_laplacian=True,
    use_numpy_laplacian=True,
    **kwargs,
)
```

Bases: `Node`

Laplacian Anomaly Detector (global), variant 'C' (Cauchy), port-based.

This is the new cuvis.ai v3 implementation of the LAD detector. It follows the same mathematical definition as the legacy v2 `LADGlobal`, but exposes a port-based interface compatible with `CuvisPipeline`, `StatisticalTrainer`, and `GradientTrainer`.

Ports

INPUT_SPECS `data` : float32, shape (-1, -1, -1, -1) Input hyperspectral cube in BHWC format. OUTPUT_SPECS `scores` : float32, shape (-1, -1, -1, 1) Per pixel anomaly scores in BHW1 format.

Parameters:

| Name                  | Type    | Description                                                                                                                                       | Default |
| --------------------- | ------- | ------------------------------------------------------------------------------------------------------------------------------------------------- | ------- |
| `eps`                 | `float` | Small epsilon value for numerical stability in Laplacian construction.                                                                            | `1e-8`  |
| `normalize_laplacian` | `bool`  | If True, applies symmetric normalization: L = D^{-½} (D - A) D^{-½}. If False, uses unnormalized Laplacian: L = D - A.                            | `True`  |
| `use_numpy_laplacian` | `bool`  | If True, constructs the Laplacian matrix using NumPy (float64, 1e-12 eps) for parity with reference implementations. If False, uses pure PyTorch. | `True`  |

Training

After statistical initialization via `statistical_initialization()`, the node can be made trainable by calling `unfreeze()`. This converts the mean `M` and Laplacian `L` buffers to trainable `nn.Parameter` objects, enabling gradient-based fine-tuning.

Example

> > > lad = LADGlobal(num_channels=61) stat_trainer = StatisticalTrainer(pipeline=pipeline, datamodule=datamodule) stat_trainer.fit() # Statistical initialization lad.unfreeze() # Enable gradient training grad_trainer = GradientTrainer(pipeline=pipeline, datamodule=datamodule, ...) grad_trainer.fit() # Gradient-based fine-tuning

Source code in `cuvis_ai/node/anomaly/lad_detector.py`

```python
def __init__(
    self,
    num_channels: int,
    eps: float = 1e-8,
    normalize_laplacian: bool = True,
    use_numpy_laplacian: bool = True,
    **kwargs: Any,
) -> None:
    self.num_channels = int(num_channels)
    self.eps = float(eps)
    self.normalize_laplacian = bool(normalize_laplacian)
    self.use_numpy_laplacian = bool(use_numpy_laplacian)

    super().__init__(
        num_channels=self.num_channels,
        eps=self.eps,
        normalize_laplacian=self.normalize_laplacian,
        use_numpy_laplacian=self.use_numpy_laplacian,
        **kwargs,
    )

    self._welford = WelfordAccumulator(self.num_channels)
    # Model buffers
    self.register_buffer("M", torch.zeros(self.num_channels, dtype=torch.float64))  # (C,)
    self.register_buffer(
        "L", torch.zeros(self.num_channels, self.num_channels, dtype=torch.float64)
    )  # (C, C)
    self._statistically_initialized = False
```

##### statistical_initialization

```python
statistical_initialization(input_stream)
```

Compute global mean M and Laplacian L from a port-based input stream.

Parameters:

| Name           | Type          | Description                                                                                           | Default    |
| -------------- | ------------- | ----------------------------------------------------------------------------------------------------- | ---------- |
| `input_stream` | `InputStream` | Iterator yielding dicts matching INPUT_SPECS. Expected format: {"data": tensor} where tensor is BHWC. | *required* |

Source code in `cuvis_ai/node/anomaly/lad_detector.py`

```python
def statistical_initialization(self, input_stream: InputStream) -> None:
    """Compute global mean M and Laplacian L from a port-based input stream.

    Parameters
    ----------
    input_stream : InputStream
        Iterator yielding dicts matching INPUT_SPECS.
        Expected format: ``{"data": tensor}`` where tensor is BHWC.
    """
    self.reset()

    for batch_data in input_stream:
        x = batch_data.get("data")
        if x is not None:
            self.update(x)

    if self._welford.count <= 0:
        raise RuntimeError("No samples provided to LADGlobal.statistical_initialization()")

    self.finalize()
    self._statistically_initialized = True
```

##### update

```python
update(batch_bhwc)
```

Update running mean statistics from a BHWC batch.

Source code in `cuvis_ai/node/anomaly/lad_detector.py`

```python
@torch.no_grad()
def update(self, batch_bhwc: torch.Tensor) -> None:
    """Update running mean statistics from a BHWC batch."""
    B, H, W, C = batch_bhwc.shape
    X = batch_bhwc.reshape(B * H * W, C)
    if X.shape[0] <= 0:
        return
    self._welford.update(X)
    self._statistically_initialized = False
```

##### finalize

```python
finalize()
```

Finalize mean and Laplacian from accumulated statistics.

Source code in `cuvis_ai/node/anomaly/lad_detector.py`

```python
@torch.no_grad()
def finalize(self) -> None:
    """Finalize mean and Laplacian from accumulated statistics."""
    if self._welford.count <= 0:
        raise RuntimeError("No samples accumulated for LADGlobal.finalize()")

    M = self._welford.mean.to(dtype=torch.float64)
    C = M.shape[0]
    a = M.mean()

    if self.use_numpy_laplacian:
        # NumPy implementation for exact parity with legacy version
        M_np = M.detach().cpu().numpy()
        A_abs = np.abs(M_np[:, None] - M_np[None, :])
        a_np = float(M_np.mean())
        A_np = 1.0 / (1.0 + (A_abs / (a_np + 1e-12)) ** 2)
        np.fill_diagonal(A_np, 0.0)
        D_np = np.diag(A_np.sum(axis=1))
        L_np = D_np - A_np

        if self.normalize_laplacian:
            d_np = np.diag(D_np)
            d_inv_sqrt_np = np.where(d_np > 0, 1.0 / (np.sqrt(d_np) + 1e-12), 0.0)
            D_inv_sqrt_np = np.diag(d_inv_sqrt_np)
            L_np = D_inv_sqrt_np @ L_np @ D_inv_sqrt_np

        L = torch.from_numpy(L_np).to(dtype=torch.float64, device=M.device)
    else:
        Mi = M.view(C, 1)
        Mj = M.view(1, C)
        denom = a + torch.tensor(1e-12, dtype=torch.float64, device=M.device)
        diff = torch.abs(Mi - Mj) / denom
        A = 1.0 / (1.0 + diff.pow(2))
        A.fill_diagonal_(0.0)

        D = torch.diag(A.sum(dim=1))
        L = D - A

        if self.normalize_laplacian:
            d = torch.diag(D)
            d_inv_sqrt = torch.where(
                d > 0,
                1.0 / torch.sqrt(d + torch.tensor(1e-12, dtype=torch.float64, device=M.device)),
                torch.zeros_like(d),
            )
            D_inv_sqrt = torch.diag(d_inv_sqrt)
            L = D_inv_sqrt @ L @ D_inv_sqrt

    self.M = M
    self.L = L
    self._statistically_initialized = True
```

##### reset

```python
reset()
```

Reset all statistics and model parameters to initial state.

Clears the streaming mean accumulator (\_mean_run), sample count (\_count), global mean (M), and Laplacian matrix (L). After reset, the detector must be re-initialized via statistical_initialization() before inference.

Notes

Use this method to re-initialize the detector with different training data or when switching between different spectral distributions.

Source code in `cuvis_ai/node/anomaly/lad_detector.py`

```python
@torch.no_grad()
def reset(self) -> None:
    """Reset all statistics and model parameters to initial state.

    Clears the streaming mean accumulator (_mean_run), sample count (_count),
    global mean (M), and Laplacian matrix (L). After reset, the detector must
    be re-initialized via statistical_initialization() before inference.

    Notes
    -----
    Use this method to re-initialize the detector with different training data
    or when switching between different spectral distributions.
    """
    self._welford.reset()
    self.M.zero_()
    self.L.zero_()
    self._statistically_initialized = False
```

##### forward

```python
forward(data, **_)
```

Compute LAD anomaly scores for a BHWC cube.

Parameters:

| Name   | Type     | Description                  | Default    |
| ------ | -------- | ---------------------------- | ---------- |
| `data` | `Tensor` | Input tensor in BHWC format. | *required* |

Returns:

| Type                | Description                                                  |
| ------------------- | ------------------------------------------------------------ |
| `dict[str, Tensor]` | Dictionary with key "scores" containing BHW1 anomaly scores. |

Source code in `cuvis_ai/node/anomaly/lad_detector.py`

```python
def forward(self, data: torch.Tensor, **_: Any) -> dict[str, torch.Tensor]:
    """Compute LAD anomaly scores for a BHWC cube.

    Parameters
    ----------
    data : torch.Tensor
        Input tensor in BHWC format.

    Returns
    -------
    dict[str, torch.Tensor]
        Dictionary with key ``"scores"`` containing BHW1 anomaly scores.
    """
    if self.M.numel() == 0 or self.L.numel() == 0 or not self._statistically_initialized:
        raise RuntimeError(
            "LADGlobal not finalized. Call statistical_initialization() before forward()."
        )

    B, H, W, C = data.shape
    N = H * W

    X = data.view(B, N, C)

    Xc = X - self.M.to(dtype=X.dtype)
    L = self.L.to(dtype=X.dtype)

    scores = torch.einsum("bnc,cd,bnd->bn", Xc, L, Xc).view(B, H, W).unsqueeze(-1)
    return {"scores": scores}
```

`LearnableChannelMixer`cuvis_ai.node.channel_mixermodeldim-redhsilearnpretorchLearnable channel mixer for hyperspectral data reduction (DRCNN-style).

#### LearnableChannelMixer

```python
LearnableChannelMixer(
    input_channels,
    output_channels,
    leaky_relu_negative_slope=0.01,
    use_bias=True,
    use_activation=True,
    normalize_output=True,
    inference_normalization="batchnorm_sigmoid",
    init_method="xavier",
    eps=1e-06,
    reduction_scheme=None,
    **kwargs,
)
```

Bases: `Node`

Learnable channel mixer for hyperspectral data reduction (DRCNN-style).

This node implements a learnable linear combination layer that reduces the number of spectral channels through spectral pixel-wise 1x1 convolutions. Based on the DRCNN approach, it uses:

- 1x1 convolution (linear combination across spectral dimension)
- Leaky ReLU activation (a=0.01)
- Bias parameters
- Optional PCA-based initialization

The mixer is designed to be trained end-to-end with a downstream model (e.g., AdaClip) while keeping the downstream model frozen. This allows the mixer to learn optimal spectral combinations for the specific task.

Parameters:

| Name                        | Type                                                                | Description                                                                                                                                                                                                                                                                            | Default                                                                                                                                                                                                                                                                                                                                                                                                                   |
| --------------------------- | ------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `input_channels`            | `int`                                                               | Number of input spectral channels (e.g., 61 for hyperspectral cube)                                                                                                                                                                                                                    | *required*                                                                                                                                                                                                                                                                                                                                                                                                                |
| `output_channels`           | `int`                                                               | Number of output channels (e.g., 3 for RGB compatibility)                                                                                                                                                                                                                              | *required*                                                                                                                                                                                                                                                                                                                                                                                                                |
| `leaky_relu_negative_slope` | `float`                                                             | Negative slope for Leaky ReLU activation (default: 0.01, as per DRCNN paper)                                                                                                                                                                                                           | `0.01`                                                                                                                                                                                                                                                                                                                                                                                                                    |
| `use_bias`                  | `bool`                                                              | Whether to use bias parameters (default: True, as per DRCNN paper)                                                                                                                                                                                                                     | `True`                                                                                                                                                                                                                                                                                                                                                                                                                    |
| `use_activation`            | `bool`                                                              | Whether to apply Leaky ReLU activation (default: True, as per DRCNN paper)                                                                                                                                                                                                             | `True`                                                                                                                                                                                                                                                                                                                                                                                                                    |
| `normalize_output`          | `bool`                                                              | Whether to apply output normalization to [0, 1] range (default: True). During training this uses BatchNorm2d + sigmoid.                                                                                                                                                                | `True`                                                                                                                                                                                                                                                                                                                                                                                                                    |
| `inference_normalization`   | `('batchnorm_sigmoid', 'per_frame_minmax', 'sigmoid_only', 'none')` | Inference-time normalization mode used when normalize_output=True. Training always uses batchnorm_sigmoid for consistency.                                                                                                                                                             | `"batchnorm_sigmoid"`                                                                                                                                                                                                                                                                                                                                                                                                     |
| `init_method`               | `('xavier', 'kaiming', 'pca', 'zeros')`                             | Weight initialization method (default: "xavier") "xavier": Xavier/Glorot uniform initialization "kaiming": Kaiming/He uniform initialization "pca": Initialize from PCA components (requires statistical_initialization) "zeros": Zero initialization (weights and bias start at zero) | `"xavier"`                                                                                                                                                                                                                                                                                                                                                                                                                |
| `eps`                       | `float`                                                             | Small constant for numerical stability (default: 1e-6)                                                                                                                                                                                                                                 | `1e-06`                                                                                                                                                                                                                                                                                                                                                                                                                   |
| `reduction_scheme`          | \`list[int]                                                         | None\`                                                                                                                                                                                                                                                                                 | Multi-layer reduction scheme for gradual channel reduction (default: None). If None, uses single-layer reduction (input_channels → output_channels). If provided, must start with input_channels and end with output_channels. Example: [61, 16, 8, 3] means: Layer 1: 61 → 16 channels Layer 2: 16 → 8 channels Layer 3: 8 → 3 channels This matches the DRCNN paper's multi-layer architecture for better optimization. |

Attributes:

| Name         | Type                | Description                                             |
| ------------ | ------------------- | ------------------------------------------------------- |
| `conv`       | `Conv2d`            | 1x1 convolutional layer performing spectral mixing      |
| `activation` | `LeakyReLU or None` | Leaky ReLU activation function (if use_activation=True) |

Examples:

```pycon
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
```

Source code in `cuvis_ai/node/channel_mixer.py`

```python
def __init__(
    self,
    input_channels: int,
    output_channels: int,
    leaky_relu_negative_slope: float = 0.01,
    use_bias: bool = True,
    use_activation: bool = True,
    normalize_output: bool = True,
    inference_normalization: Literal[
        "batchnorm_sigmoid", "per_frame_minmax", "sigmoid_only", "none"
    ] = "batchnorm_sigmoid",
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
    self.inference_normalization = str(inference_normalization)
    self.init_method = init_method
    self.eps = eps
    valid_norm_modes = {"batchnorm_sigmoid", "per_frame_minmax", "sigmoid_only", "none"}
    if self.inference_normalization not in valid_norm_modes:
        raise ValueError(
            f"inference_normalization must be one of {sorted(valid_norm_modes)}, "
            f"got '{self.inference_normalization}'"
        )

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
        inference_normalization=self.inference_normalization,
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

    # Output normalization: BatchNorm + sigmoid replaces per-image min-max
    # BatchNorm tracks running mean/var during training → consistent normalization at eval
    if self.normalize_output:
        self.output_bn = nn.BatchNorm2d(output_channels, affine=True)

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
```

##### requires_initial_fit

```python
requires_initial_fit
```

Whether this node requires statistical initialization.

##### statistical_initialization

```python
statistical_initialization(input_stream)
```

Initialize mixer weights from PCA components.

This method computes PCA on the input data and initializes the mixer weights to the top principal components. This provides a good starting point for gradient-based optimization.

Parameters:

| Name           | Type          | Description                                                                                                                        | Default    |
| -------------- | ------------- | ---------------------------------------------------------------------------------------------------------------------------------- | ---------- |
| `input_stream` | `InputStream` | Iterator yielding dicts matching INPUT_SPECS (port-based format) Expected format: {"data": tensor} where tensor is [B, H, W, C_in] | *required* |

Notes

This method is only used when init_method="pca". For other initialization methods, weights are set in **init**.

Source code in `cuvis_ai/node/channel_mixer.py`

```python
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
```

##### freeze

```python
freeze()
```

Disable gradient-based training of mixer weights.

Source code in `cuvis_ai/node/channel_mixer.py`

```python
def freeze(self) -> None:
    """Disable gradient-based training of mixer weights."""
    for conv in self.convs:
        for param in conv.parameters():
            param.requires_grad = False
    super().freeze()
```

##### unfreeze

```python
unfreeze()
```

Enable gradient-based training of mixer weights.

Source code in `cuvis_ai/node/channel_mixer.py`

```python
def unfreeze(self) -> None:
    """Enable gradient-based training of mixer weights."""
    for conv in self.convs:
        for param in conv.parameters():
            param.requires_grad = True
    super().unfreeze()
```

##### forward

```python
forward(data, context=None, **_)
```

Apply learnable channel mixing to input.

Parameters:

| Name      | Type      | Description                                         | Default    |
| --------- | --------- | --------------------------------------------------- | ---------- |
| `data`    | `Tensor`  | Input tensor [B, H, W, C_in] in BHWC format         | *required* |
| `context` | `Context` | Execution context with epoch, batch_idx, stage info | `None`     |

Returns:

| Type                | Description                                                            |
| ------------------- | ---------------------------------------------------------------------- |
| `dict[str, Tensor]` | Dictionary with "rgb" key containing reduced channels [B, H, W, C_out] |

Source code in `cuvis_ai/node/channel_mixer.py`

```python
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

    # Apply output normalization while still in BCHW format.
    if self.normalize_output:
        norm_mode = "batchnorm_sigmoid"
        if context is not None and context.stage == ExecutionStage.INFERENCE:
            norm_mode = self.inference_normalization

        if norm_mode == "batchnorm_sigmoid":
            mixed = self.output_bn(mixed)  # BatchNorm2d: BCHW -> BCHW
            mixed = torch.sigmoid(mixed)  # Map to (0, 1)
        elif norm_mode == "per_frame_minmax":
            mixed = self._per_frame_minmax_bchw(mixed)
        elif norm_mode == "sigmoid_only":
            mixed = torch.sigmoid(mixed)
        elif norm_mode == "none":
            pass
        else:
            raise RuntimeError(f"Unsupported inference_normalization mode: {norm_mode}")

    # Convert back from BCHW to BHWC
    mixed_bhwc = mixed.permute(0, 2, 3, 1)  # [B, H, W, C_out]

    # DEBUG: Print output info
    if hasattr(self, "_debug") and self._debug:
        print(
            f"[LearnableChannelMixer] Output: shape={mixed_bhwc.shape}, "
            f"min={mixed_bhwc.min().item():.4f}, max={mixed_bhwc.max().item():.4f}, "
            f"mean={mixed_bhwc.mean().item():.4f}, requires_grad={mixed_bhwc.requires_grad}"
        )

    last_layer_weights = self.convs[-1].weight.squeeze(-1).squeeze(-1)
    return {"rgb": mixed_bhwc, "weights": last_layer_weights}
```

`NMFUnmixing`cuvis_ai.node.unmixing.nmfmodelhsistatefultorchBlind unmixing: learn endmembers by NMF, then solve per-pixel abundances.

#### NMFUnmixing

```python
NMFUnmixing(
    n_components=3,
    init="nndsvda",
    beta_loss="frobenius",
    max_iter=300,
    random_state=0,
    **kwargs,
)
```

Bases: `_StatisticalFitNode`

Blind unmixing: learn endmembers by NMF, then solve per-pixel abundances.

During `statistical_initialization` the node fits :class:`sklearn.decomposition.NMF` on the collected training pixels and stores the learned components `[K, C]` as a frozen buffer. At inference it solves `min_{x >= 0} ||A x - b||` per pixel against those frozen endmembers (`A = endmembers.T`) with batched projected-gradient descent, emitting abundances, the learned endmembers, the per-pixel reconstruction residual, and a 1-based argmax class mask.

Parameters:

| Name           | Type  | Description                                                                                                      | Default       |
| -------------- | ----- | ---------------------------------------------------------------------------------------------------------------- | ------------- |
| `n_components` | `int` | Number of endmembers K to learn (default: 3).                                                                    | `3`           |
| `init`         | `str` | sklearn NMF initialization scheme (default: "nndsvda").                                                          | `'nndsvda'`   |
| `beta_loss`    | `str` | sklearn NMF beta-divergence loss (default: "frobenius").                                                         | `'frobenius'` |
| `max_iter`     | `int` | Maximum sklearn NMF solver iterations at fit time (default: 300).                                                | `300`         |
| `random_state` | `int` | Seed for the sklearn NMF solver (default: 0).                                                                    | `0`           |
| `**kwargs`     | `Any` | Forwarded to :class:cuvis_ai.node.\_statistical_fit.\_StatisticalFitNode, including max_fit_pixels and fit_seed. | `{}`          |

Source code in `cuvis_ai/node/unmixing/nmf.py`

```python
def __init__(
    self,
    n_components: int = 3,
    init: str = "nndsvda",
    beta_loss: str = "frobenius",
    max_iter: int = 300,
    random_state: int = 0,
    **kwargs: Any,
) -> None:
    self.n_components = int(n_components)
    self.init = str(init)
    self.beta_loss = str(beta_loss)
    self.max_iter = int(max_iter)
    self.random_state = int(random_state)
    super().__init__(
        n_components=self.n_components,
        init=self.init,
        beta_loss=self.beta_loss,
        max_iter=self.max_iter,
        random_state=self.random_state,
        **kwargs,
    )
    # Placeholder; resized to [K, C] at fit time and on checkpoint reload.
    self.register_buffer("endmembers_buf", torch.zeros(0, dtype=torch.float32))
```

##### forward

```python
forward(cube, **_)
```

Solve per-pixel abundances against the frozen learned endmembers.

Parameters:

| Name   | Type     | Description                      | Default    |
| ------ | -------- | -------------------------------- | ---------- |
| `cube` | `Tensor` | Hyperspectral cube [B, H, W, C]. | *required* |

Returns:

| Type                | Description                                                                                                          |
| ------------------- | -------------------------------------------------------------------------------------------------------------------- |
| `dict[str, Tensor]` | abundances [B, H, W, K], endmembers [K, C], scores (reconstruction residual) [B, H, W, 1], and class_mask [B, H, W]. |

Source code in `cuvis_ai/node/unmixing/nmf.py`

```python
def forward(self, cube: torch.Tensor, **_: Any) -> dict[str, torch.Tensor]:
    """Solve per-pixel abundances against the frozen learned endmembers.

    Parameters
    ----------
    cube : torch.Tensor
        Hyperspectral cube ``[B, H, W, C]``.

    Returns
    -------
    dict[str, torch.Tensor]
        ``abundances`` ``[B, H, W, K]``, ``endmembers`` ``[K, C]``,
        ``scores`` (reconstruction residual) ``[B, H, W, 1]``, and
        ``class_mask`` ``[B, H, W]``.
    """
    self._require_initialized()
    if cube.ndim != 4:
        raise ValueError(f"Expected cube with shape [B, H, W, C], got {tuple(cube.shape)}")

    batch, height, width, channels = cube.shape
    endmembers = self.endmembers_buf.to(dtype=cube.dtype, device=cube.device)  # [K, C]
    components = endmembers.shape[0]
    if endmembers.shape[1] != channels:
        raise ValueError(
            f"Learned endmember channels {endmembers.shape[1]} do not match "
            f"cube channels {channels}."
        )

    a = endmembers.transpose(0, 1)  # [C, K]
    outputs: list[torch.Tensor] = []
    residuals: list[torch.Tensor] = []
    for frame in cube:
        b = frame.reshape(-1, channels)  # [P, C]
        x = nnls_batch(a, b, max_iter=_FORWARD_NNLS_ITERS, tol=1e-6)  # [P, K]
        outputs.append(x.reshape(height, width, components))
        residuals.append(reconstruction_residual(a, x, b).reshape(height, width, 1))

    abundances = torch.stack(outputs, dim=0)  # [B, H, W, K]
    scores = torch.stack(residuals, dim=0)  # [B, H, W, 1]
    class_mask = (abundances.argmax(dim=-1) + 1).to(torch.int32)  # [B, H, W]

    return {
        "abundances": abundances,
        "endmembers": endmembers,
        "scores": scores,
        "class_mask": class_mask,
    }
```

`NNLSUnmixing`cuvis_ai.node.unmixing.nnlsmodelhsitorchUnmix each pixel into non-negative abundances of known endmembers.

#### NNLSUnmixing

```python
NNLSUnmixing(
    max_iter=500, tol=1e-06, min_total=0.0, **kwargs
)
```

Bases: `Node`

Unmix each pixel into non-negative abundances of known endmembers.

Given a hyperspectral cube `[B, H, W, C]` and a set of `K` endmember spectra `[K, C]`, solve `min_{x >= 0} ||A x - b||` for every pixel, where `A = endmembers.T` has shape `[C, K]` and `b` is the pixel spectrum. The solve runs as batched projected-gradient descent, so the node is stateless and runs entirely in torch on the inputs' device.

Only non-negativity (ANC) is enforced, not sum-to-one (ASC): abundances are not constrained to sum to 1 (this is not FCLS).

Parameters:

| Name        | Type    | Description                                                                                                                     | Default |
| ----------- | ------- | ------------------------------------------------------------------------------------------------------------------------------- | ------- |
| `max_iter`  | `int`   | Maximum projected-gradient iterations per forward call (default: 500). Close or collinear endmembers may need more to converge. | `500`   |
| `tol`       | `float` | Early-stop threshold on the per-iteration update norm (default: 1e-6).                                                          | `1e-06` |
| `min_total` | `float` | Pixels whose summed abundance falls below this value are labelled background (class 0) in class_mask (default: 0.0).            | `0.0`   |
| `**kwargs`  | `Any`   | Forwarded to :class:cuvis_ai_core.node.Node.                                                                                    | `{}`    |

Source code in `cuvis_ai/node/unmixing/nnls.py`

```python
def __init__(
    self,
    max_iter: int = 500,
    tol: float = 1e-6,
    min_total: float = 0.0,
    **kwargs: Any,
) -> None:
    self.max_iter = int(max_iter)
    self.tol = float(tol)
    self.min_total = float(min_total)
    super().__init__(
        max_iter=self.max_iter,
        tol=self.tol,
        min_total=self.min_total,
        **kwargs,
    )
```

##### forward

```python
forward(cube, endmembers, **_)
```

Solve per-pixel non-negative least squares against the endmembers.

Parameters:

| Name         | Type     | Description                      | Default    |
| ------------ | -------- | -------------------------------- | ---------- |
| `cube`       | `Tensor` | Hyperspectral cube [B, H, W, C]. | *required* |
| `endmembers` | `Tensor` | Endmember spectra [K, C].        | *required* |

Returns:

| Type                | Description                                                                        |
| ------------------- | ---------------------------------------------------------------------------------- |
| `dict[str, Tensor]` | abundances [B, H, W, K], scores (residual) [B, H, W, 1], and class_mask [B, H, W]. |

Source code in `cuvis_ai/node/unmixing/nnls.py`

```python
def forward(
    self, cube: torch.Tensor, endmembers: torch.Tensor, **_: Any
) -> dict[str, torch.Tensor]:
    """Solve per-pixel non-negative least squares against the endmembers.

    Parameters
    ----------
    cube : torch.Tensor
        Hyperspectral cube ``[B, H, W, C]``.
    endmembers : torch.Tensor
        Endmember spectra ``[K, C]``.

    Returns
    -------
    dict[str, torch.Tensor]
        ``abundances`` ``[B, H, W, K]``, ``scores`` (residual) ``[B, H, W, 1]``,
        and ``class_mask`` ``[B, H, W]``.
    """
    if cube.ndim != 4:
        raise ValueError(f"Expected cube with shape [B, H, W, C], got {tuple(cube.shape)}")
    if endmembers.ndim != 2:
        raise ValueError(
            f"Expected endmembers with shape [K, C], got {tuple(endmembers.shape)}"
        )

    batch, height, width, channels = cube.shape
    components = endmembers.shape[0]
    if endmembers.shape[1] != channels:
        raise ValueError(
            f"Endmember channels {endmembers.shape[1]} do not match cube channels {channels}."
        )

    endmembers = endmembers.to(dtype=cube.dtype, device=cube.device)
    a = endmembers.transpose(0, 1)  # [C, K]
    b = cube.reshape(-1, channels)  # [P, C]

    x = nnls_batch(a, b, max_iter=self.max_iter, tol=self.tol)  # [P, K]
    residual = reconstruction_residual(a, x, b)  # [P]

    abundances = x.reshape(batch, height, width, components)
    scores = residual.reshape(batch, height, width, 1)

    totals = x.sum(dim=1)  # [P]
    class_idx = x.argmax(dim=1) + 1  # [P], 1-based
    class_idx = torch.where(
        totals < self.min_total,
        torch.zeros_like(class_idx),
        class_idx,
    )
    class_mask = class_idx.reshape(batch, height, width).to(torch.int32)

    return {
        "abundances": abundances,
        "scores": scores,
        "class_mask": class_mask,
    }
```

`OSNetExtractor`cuvis_ai_deepeiou.nodemodelbatchedembimginferlearnrgbtorchdeepeiou[↗](https://github.com/cubert-hyperspectral/cuvis-ai-deepeiou "Open plugin repo")OSNet x1.0 feature extractor (512-dim embeddings).

OSNet x1.0 feature extractor (512-dim embeddings).

Inputs

| Port    | Dtype     | Shape             | Description                                      |
| ------- | --------- | ----------------- | ------------------------------------------------ |
| `crops` | `float32` | `[-1, 3, -1, -1]` | Normalized crops [N, 3, crop_h, crop_w] in NCHW. |

Outputs

| Port         | Dtype     | Shape          | Description                         |
| ------------ | --------- | -------------- | ----------------------------------- |
| `embeddings` | `float32` | `[-1, -1, -1]` | L2-normalised embeddings [B, N, D]. |

[View plugin repo (v0.2.2)](https://github.com/cubert-hyperspectral/cuvis-ai-deepeiou/tree/v0.2.2)

`OneClassSVMDetector`cuvis_ai.node.svmmodelanomhsistatefultorchOne-class SVM novelty detector (sklearn fit, pure-torch RBF forward).

#### OneClassSVMDetector

```python
OneClassSVMDetector(
    kernel="rbf",
    nu=0.5,
    gamma="scale",
    chunk_size=65536,
    **kwargs,
)
```

Bases: `_StatisticalFitNode`

One-class SVM novelty detector (sklearn fit, pure-torch RBF forward).

During `statistical_initialization` the node collects background pixels and fits a :class:`sklearn.svm.OneClassSVM`. The fitted estimator is reduced to torch buffers (support vectors, dual coefficients, resolved `gamma` and the offset) so that `forward` can evaluate the RBF decision function in pure torch, chunked over the pixel axis to bound peak memory.

Parameters:

| Name         | Type           | Description                                                                                                                                                                                                 | Default   |
| ------------ | -------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------- |
| `kernel`     | `str`          | Kernel passed to scikit-learn at fit time. Only "rbf" is supported by the torch forward pass; any other value raises at fit (default: "rbf").                                                               | `'rbf'`   |
| `nu`         | `float`        | Upper bound on the fraction of training outliers and lower bound on the fraction of support vectors, in (0, 1\] (default: 0.5).                                                                             | `0.5`     |
| `gamma`      | `str or float` | RBF kernel coefficient. Either a positive float or one of scikit-learn's string presets ("scale" / "auto"); the numeric value sklearn actually resolves is stored and used at inference (default: "scale"). | `'scale'` |
| `chunk_size` | `int`          | Number of pixels scored per chunk in forward. Caps the transient [chunk_size, n_support_vectors] kernel matrix (default: 65536).                                                                            | `65536`   |

Attributes:

| Name              | Type     | Description                                |
| ----------------- | -------- | ------------------------------------------ |
| `support_vectors` | `Tensor` | Fitted support vectors [n_sv, C].          |
| `dual_coef`       | `Tensor` | Signed dual coefficients [n_sv].           |
| `gamma_buf`       | `Tensor` | Resolved scalar RBF gamma as a [1] tensor. |
| `offset_buf`      | `Tensor` | Decision-function offset as a [1] tensor.  |

Examples:

```pycon
>>> from cuvis_ai.node.svm import OneClassSVMDetector
>>> detector = OneClassSVMDetector(nu=0.1, gamma="scale")
>>> # detector.statistical_initialization(background_stream)
>>> # out = detector.forward(cube=cube)
>>> # scores, decisions = out["scores"], out["decisions"]
```

Store hyperparameters and register placeholder fit buffers.

Parameters:

| Name         | Type           | Description                                                                         | Default   |
| ------------ | -------------- | ----------------------------------------------------------------------------------- | --------- |
| `kernel`     | `str`          | scikit-learn kernel; only "rbf" is supported at inference (default: "rbf").         | `'rbf'`   |
| `nu`         | `float`        | One-class SVM nu in (0, 1\] (default: 0.5).                                         | `0.5`     |
| `gamma`      | `str or float` | RBF kernel coefficient or a string preset (default: "scale").                       | `'scale'` |
| `chunk_size` | `int`          | Pixels scored per chunk in forward (default: 65536).                                | `65536`   |
| `**kwargs`   | `Any`          | Forwarded to the statistical-fit base (max_fit_pixels, fit_seed) and the node base. | `{}`      |

Source code in `cuvis_ai/node/svm.py`

```python
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
```

##### forward

```python
forward(cube, **_)
```

Score a cube with the signed RBF decision function, chunked over pixels.

Parameters:

| Name   | Type     | Description                            | Default    |
| ------ | -------- | -------------------------------------- | ---------- |
| `cube` | `Tensor` | Input hyperspectral cube [B, H, W, C]. | *required* |
| `**_`  | `Any`    | Additional unused keyword arguments.   | `{}`       |

Returns:

| Type                | Description                                                                                                                                                                    |
| ------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `dict[str, Tensor]` | Dictionary with: "scores" : signed decision function [B, H, W, 1] (>0 inlier, \<0 outlier). "decisions" : outlier mask [B, H, W, 1] (True where the decision function is < 0). |

Raises:

| Type           | Description                                                                                        |
| -------------- | -------------------------------------------------------------------------------------------------- |
| `RuntimeError` | If the node has not been initialized via statistical_initialization (or a fitted checkpoint load). |

Source code in `cuvis_ai/node/svm.py`

```python
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
```

`RTSAM2BboxPropagation`cuvis_ai_rtsam2.node.rtsam2_streaming_propagationmodelbatchedbboxinferlearnmaskrgbsegstatefultorchtrackvidrtsam2[↗](https://github.com/cubert-hyperspectral/cuvis-ai-rtsam2 "Open plugin repo")RTSAM2 propagation with runtime bbox prompts.

RTSAM2 propagation with runtime bbox prompts.

Inputs

| Port                | Dtype     | Shape            | Description                                                                                               |
| ------------------- | --------- | ---------------- | --------------------------------------------------------------------------------------------------------- |
| `rgb_image`         | `float32` | `[1, -1, -1, 3]` | RGB frame [1,H,W,3] in float32 with values in [0, 1].                                                     |
| `frame_id` optional | `int64`   | `[1]`            | Source frame index [1]. Preserved by upstream sinks.                                                      |
| `bboxes` optional   | `any`     | `any`            | Optional per-frame list of bbox prompt dicts with keys element_id, object_id, x_min, y_min, x_max, y_max. |

Outputs

| Port               | Dtype     | Shape         | Description |
| ------------------ | --------- | ------------- | ----------- |
| `mask`             | `int32`   | `[1, -1, -1]` |             |
| `object_ids`       | `int64`   | `[1, -1]`     |             |
| `detection_scores` | `float32` | `[1, -1]`     |             |

[View plugin repo (v0.3.1)](https://github.com/cubert-hyperspectral/cuvis-ai-rtsam2/tree/v0.3.1)

`RTSAM2MaskPropagation`cuvis_ai_rtsam2.node.rtsam2_streaming_propagationmodelbatchedinferlearnmaskrgbsegstatefultorchtrackvidrtsam2[↗](https://github.com/cubert-hyperspectral/cuvis-ai-rtsam2 "Open plugin repo")RTSAM2 propagation with runtime label-map prompts.

RTSAM2 propagation with runtime label-map prompts.

Inputs

| Port                | Dtype     | Shape            | Description                                                                                                          |
| ------------------- | --------- | ---------------- | -------------------------------------------------------------------------------------------------------------------- |
| `rgb_image`         | `float32` | `[1, -1, -1, 3]` | RGB frame [1,H,W,3] in float32 with values in [0, 1].                                                                |
| `frame_id` optional | `int64`   | `[1]`            | Source frame index [1]. Preserved by upstream sinks.                                                                 |
| `mask` optional     | `int32`   | `[1, -1, -1]`    | Optional int32 label map [1,H,W]. 0=background, each positive label is treated as an object ID prompt on that frame. |

Outputs

| Port               | Dtype     | Shape         | Description |
| ------------------ | --------- | ------------- | ----------- |
| `mask`             | `int32`   | `[1, -1, -1]` |             |
| `object_ids`       | `int64`   | `[1, -1]`     |             |
| `detection_scores` | `float32` | `[1, -1]`     |             |

[View plugin repo (v0.3.1)](https://github.com/cubert-hyperspectral/cuvis-ai-rtsam2/tree/v0.3.1)

`RTSAM2PointExpansion`cuvis_ai_rtsam2.node.rtsam2_point_expansionmodelbatchedimginferkplearnmaskrgbsegtorchrtsam2[↗](https://github.com/cubert-hyperspectral/cuvis-ai-rtsam2 "Open plugin repo")Expand positive/negative click points into one object mask on a single frame.

Expand positive/negative click points into one object mask on a single frame.

Inputs

| Port                | Dtype     | Shape            | Description                                                                                                                                                   |
| ------------------- | --------- | ---------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `rgb_image`         | `float32` | `[1, -1, -1, 3]` | RGB frame [1,H,W,3] in float32 with values in [0, 1].                                                                                                         |
| `points` optional   | `any`     | `any`            | Optional per-frame list of point prompt dicts with keys element_id, x, y, type (type in {positive, negative, neutral}). Positive=object, negative=background. |
| `frame_id` optional | `int64`   | `[1]`            | Optional source frame index [1]; accepted for contract parity, unused.                                                                                        |

Outputs

| Port               | Dtype     | Shape         | Description |
| ------------------ | --------- | ------------- | ----------- |
| `mask`             | `int32`   | `[1, -1, -1]` |             |
| `object_ids`       | `int64`   | `[1, -1]`     |             |
| `detection_scores` | `float32` | `[1, -1]`     |             |

[View plugin repo (v0.3.1)](https://github.com/cubert-hyperspectral/cuvis-ai-rtsam2/tree/v0.3.1)

`RXBase`cuvis_ai.node.anomaly.rx_detectormodelanomhsilearnnumpyBase class for RX anomaly detectors.

#### RXBase

```python
RXBase(eps=1e-06, **kwargs)
```

Bases: `Node`

Base class for RX anomaly detectors.

Source code in `cuvis_ai/node/anomaly/rx_detector.py`

```python
def __init__(self, eps: float = 1e-6, **kwargs) -> None:
    self.eps = eps
    super().__init__(eps=eps, **kwargs)
```

`RXGlobal`cuvis_ai.node.anomaly.rx_detectormodelanomhsilearnnumpyRX anomaly detector with global background statistics.

#### RXGlobal

```python
RXGlobal(
    num_channels, eps=1e-06, cache_inverse=True, **kwargs
)
```

Bases: `RXBase`

RX anomaly detector with global background statistics.

Uses global mean (μ) and covariance (Σ) estimated from training data to compute Mahalanobis distance scores. Supports two-phase training: statistical initialization followed by optional gradient-based fine-tuning.

The detector computes anomaly scores as:

```text
RX(x) = (x - μ)ᵀ Σ⁻¹ (x - μ)
```

where x is a pixel spectrum, μ is the background mean, and Σ is the covariance matrix.

Parameters:

| Name            | Type    | Description                                                                         | Default    |
| --------------- | ------- | ----------------------------------------------------------------------------------- | ---------- |
| `num_channels`  | `int`   | Number of spectral channels in input data                                           | *required* |
| `eps`           | `float` | Small constant added to covariance diagonal for numerical stability (default: 1e-6) | `1e-06`    |
| `cache_inverse` | `bool`  | If True, precompute and cache Σ⁻¹ for faster inference (default: True)              | `True`     |
| `**kwargs`      | `dict`  | Additional arguments passed to Node base class                                      | `{}`       |

Attributes:

| Name                         | Type                  | Description                                                                                  |
| ---------------------------- | --------------------- | -------------------------------------------------------------------------------------------- |
| `mu`                         | `Tensor or Parameter` | Background mean spectrum, shape (C,). Initially a buffer, becomes Parameter after unfreeze() |
| `cov`                        | `Tensor or Parameter` | Background covariance matrix, shape (C, C)                                                   |
| `cov_inv`                    | `Tensor or Parameter` | Cached pseudo-inverse of covariance (if cache_inverse=True)                                  |
| `_statistically_initialized` | `bool`                | Flag indicating whether statistical_initialization() has been called                         |

Examples:

```pycon
>>> from cuvis_ai.node.anomaly.rx_detector import RXGlobal
>>> from cuvis_ai_core.training import StatisticalTrainer
>>>
>>> # Create RX detector
>>> rx = RXGlobal(num_channels=61, eps=1.0e-6)
>>>
>>> # Phase 1: Statistical initialization
>>> stat_trainer = StatisticalTrainer(pipeline=pipeline, datamodule=datamodule)
>>> stat_trainer.fit()  # Computes μ and Σ from training data
>>>
>>> # Inference with frozen statistics
>>> output = rx.forward(data=hyperspectral_cube)
>>> scores = output["scores"]  # [B, H, W, 1]
>>>
>>> # Phase 2: Optional gradient-based fine-tuning
>>> rx.unfreeze()  # Convert buffers to nn.Parameters
>>> # Now μ and Σ can be updated with gradient descent
```

See Also

RXPerBatch : Per-batch RX variant without training MinMaxNormalizer : Recommended preprocessing before RX ScoreToLogit : Convert scores to logits for classification docs/usecases/rx-statistical.md : Complete RX pipeline tutorial

Notes

After statistical_initialization(), mu and cov are stored as buffers (frozen by default). Call unfreeze() to convert them to trainable nn.Parameters for gradient-based optimization.

Source code in `cuvis_ai/node/anomaly/rx_detector.py`

```python
def __init__(
    self, num_channels: int, eps: float = 1e-6, cache_inverse: bool = True, **kwargs
) -> None:
    self.num_channels = int(num_channels)
    self.eps = eps
    self.cache_inverse = cache_inverse
    # Call Node.__init__ directly with all parameters for proper serialization
    # We bypass RXBase.__init__ since it only accepts eps
    # Node.__init__(self, num_channels=self.num_channels, eps=self.eps, cache_inverse=self.cache_inverse)

    super().__init__(
        num_channels=self.num_channels, eps=self.eps, cache_inverse=self.cache_inverse, **kwargs
    )

    # global stats - all stored as buffers initially
    self.register_buffer("mu", torch.zeros(self.num_channels, dtype=torch.float32))  # (C,)
    self.register_buffer(
        "cov", torch.zeros(self.num_channels, self.num_channels, dtype=torch.float32)
    )  # (C,C)
    self.register_buffer(
        "cov_inv", torch.zeros(self.num_channels, self.num_channels, dtype=torch.float32)
    )  # (C,C)
    self._welford = WelfordAccumulator(self.num_channels, track_covariance=True)
    self._statistically_initialized = False
```

##### statistical_initialization

```python
statistical_initialization(input_stream)
```

Initialize mu and Sigma from data iterator.

Parameters:

| Name           | Type          | Description                                                                                                             | Default    |
| -------------- | ------------- | ----------------------------------------------------------------------------------------------------------------------- | ---------- |
| `input_stream` | `InputStream` | Iterator yielding dicts matching INPUT_SPECS (port-based format) Expected format: {"data": tensor} where tensor is BHWC | *required* |

Source code in `cuvis_ai/node/anomaly/rx_detector.py`

```python
def statistical_initialization(self, input_stream: InputStream) -> None:
    """Initialize mu and Sigma from data iterator.

    Parameters
    ----------
    input_stream : InputStream
        Iterator yielding dicts matching INPUT_SPECS (port-based format)
        Expected format: {"data": tensor} where tensor is BHWC
    """
    self.reset()
    for batch_data in input_stream:
        # Extract data from port-based dict
        x = batch_data["data"]
        if x is not None:
            self.update(x)

    if self._welford.count <= 1:
        self._statistically_initialized = False
        raise RuntimeError(
            "RXGlobal.statistical_initialization() received insufficient samples. "
            "Expected at least 2 valid pixels."
        )
    self.finalize()
```

##### update

```python
update(batch_bhwc)
```

Update streaming statistics with a new batch.

Parameters:

| Name         | Type     | Description                                    | Default    |
| ------------ | -------- | ---------------------------------------------- | ---------- |
| `batch_bhwc` | `Tensor` | Input batch in BHWC format, shape (B, H, W, C) | *required* |

Source code in `cuvis_ai/node/anomaly/rx_detector.py`

```python
@torch.no_grad()
def update(self, batch_bhwc: torch.Tensor) -> None:
    """Update streaming statistics with a new batch.

    Parameters
    ----------
    batch_bhwc : torch.Tensor
        Input batch in BHWC format, shape (B, H, W, C)
    """
    X = _flatten_bhwc(batch_bhwc).reshape(-1, batch_bhwc.shape[-1])  # (M,C)
    if X.shape[0] <= 1:
        return
    # Adapt accumulator if actual data channels differ from constructor's num_channels
    # (e.g., upstream SoftChannelSelector preserves all channels instead of reducing)
    if X.shape[1] != self._welford._n_features:
        self._welford = WelfordAccumulator(X.shape[1], track_covariance=True).to(
            device=X.device
        )
    self._welford.update(X)
    self._statistically_initialized = False
```

##### finalize

```python
finalize()
```

Compute final mean and covariance from accumulated streaming statistics.

This method converts the running accumulators (\_mean, \_M2) into the final mean (mu) and covariance (cov) matrices. The covariance is regularized with eps * I for numerical stability, and optionally caches the pseudo-inverse.

Returns:

| Type       | Description                      |
| ---------- | -------------------------------- |
| `RXGlobal` | Returns self for method chaining |

Raises:

| Type         | Description                                                                       |
| ------------ | --------------------------------------------------------------------------------- |
| `ValueError` | If fewer than 2 samples were accumulated (insufficient for covariance estimation) |

Notes

After finalization, mu and cov are stored as buffers (frozen by default). Call unfreeze() to convert them to nn.Parameters for gradient-based training.

Source code in `cuvis_ai/node/anomaly/rx_detector.py`

```python
@torch.no_grad()
def finalize(self) -> "RXGlobal":
    """Compute final mean and covariance from accumulated streaming statistics.

    This method converts the running accumulators (_mean, _M2) into the final
    mean (mu) and covariance (cov) matrices. The covariance is regularized with
    eps * I for numerical stability, and optionally caches the pseudo-inverse.

    Returns
    -------
    RXGlobal
        Returns self for method chaining

    Raises
    ------
    ValueError
        If fewer than 2 samples were accumulated (insufficient for covariance estimation)

    Notes
    -----
    After finalization, mu and cov are stored as buffers (frozen by default).
    Call unfreeze() to convert them to nn.Parameters for gradient-based training.
    """
    if self._welford.count <= 1:
        raise ValueError("Not enough samples to finalize.")
    self.mu = self._welford.mean
    cov = self._welford.cov
    if self.eps > 0:
        cov = cov + self.eps * torch.eye(cov.shape[0], device=cov.device, dtype=cov.dtype)
    self.cov = cov
    if self.cache_inverse:
        self.cov_inv = torch.linalg.pinv(cov)
    else:
        self.cov_inv = torch.empty(0, 0)
    self._statistically_initialized = True
    return self
```

##### reset

```python
reset()
```

Reset all statistics and accumulators to empty state.

Clears mu, cov, cov_inv, and all streaming accumulators (\_mean, \_M2, \_n). After reset, the detector must be re-initialized via statistical_initialization() before it can be used for inference.

Notes

Use this method when you need to re-initialize the detector with different training data or when switching between different dataset distributions.

Source code in `cuvis_ai/node/anomaly/rx_detector.py`

```python
def reset(self) -> None:
    """Reset all statistics and accumulators to empty state.

    Clears mu, cov, cov_inv, and all streaming accumulators (_mean, _M2, _n).
    After reset, the detector must be re-initialized via statistical_initialization()
    before it can be used for inference.

    Notes
    -----
    Use this method when you need to re-initialize the detector with different
    training data or when switching between different dataset distributions.
    """
    self.mu.zero_()
    self.cov.zero_()
    self.cov_inv.zero_()
    self._welford.reset()
    self._statistically_initialized = False
```

##### forward

```python
forward(data, **_)
```

Forward pass computing anomaly scores.

Parameters:

| Name   | Type     | Description                 | Default    |
| ------ | -------- | --------------------------- | ---------- |
| `data` | `Tensor` | Input tensor in BHWC format | *required* |

Returns:

| Type                | Description                                                 |
| ------------------- | ----------------------------------------------------------- |
| `dict[str, Tensor]` | Dictionary with "scores" key containing BHW1 anomaly scores |

Source code in `cuvis_ai/node/anomaly/rx_detector.py`

```python
def forward(self, data: torch.Tensor, **_) -> dict[str, torch.Tensor]:
    """Forward pass computing anomaly scores.

    Parameters
    ----------
    data : torch.Tensor
        Input tensor in BHWC format

    Returns
    -------
    dict[str, torch.Tensor]
        Dictionary with "scores" key containing BHW1 anomaly scores
    """
    if not self._statistically_initialized or self.mu.numel() == 0:
        raise RuntimeError(
            "RXGlobal not initialized. Call statistical_initialization() before inference."
        )
    B, H, W, C = data.shape
    N = H * W
    X = data.view(B, N, C)
    # Convert dtype if needed, but don't change device (assumes everything on same device)
    Xc = X - self.mu.to(X.dtype)
    if self.cov_inv.numel() > 0:
        cov_inv = self.cov_inv.to(X.dtype)
        md2 = torch.einsum("bnc,cd,bnd->bn", Xc, cov_inv, Xc)  # (B,N)
    else:
        md2 = self._quad_form_solve(Xc, self.cov.to(X.dtype))
    scores = md2.view(B, H, W).unsqueeze(-1)  # Add channel dimension (B,H,W,1)
    return {"scores": scores}
```

`RXPerBatch`cuvis_ai.node.anomaly.rx_detectormodelanomhsilearnnumpyComputes μ, Σ per image in the batch on the fly; no fit/finalize.

#### RXPerBatch

```python
RXPerBatch(eps=1e-06, **kwargs)
```

Bases: `RXBase`

Computes μ, Σ per image in the batch on the fly; no fit/finalize.

Source code in `cuvis_ai/node/anomaly/rx_detector.py`

```python
def __init__(self, eps: float = 1e-6, **kwargs) -> None:
    self.eps = eps
    super().__init__(eps=eps, **kwargs)
```

##### forward

```python
forward(data, **_)
```

Forward pass computing per-batch anomaly scores.

Parameters:

| Name   | Type     | Description                 | Default    |
| ------ | -------- | --------------------------- | ---------- |
| `data` | `Tensor` | Input tensor in BHWC format | *required* |

Returns:

| Type                | Description                                                 |
| ------------------- | ----------------------------------------------------------- |
| `dict[str, Tensor]` | Dictionary with "scores" key containing BHW1 anomaly scores |

Source code in `cuvis_ai/node/anomaly/rx_detector.py`

```python
def forward(self, data: torch.Tensor, **_) -> dict[str, torch.Tensor]:
    """Forward pass computing per-batch anomaly scores.

    Parameters
    ----------
    data : torch.Tensor
        Input tensor in BHWC format

    Returns
    -------
    dict[str, torch.Tensor]
        Dictionary with "scores" key containing BHW1 anomaly scores
    """
    B, H, W, C = data.shape
    N = H * W
    X_flat = _flatten_bhwc(data)  # (B,N,C)
    mu = X_flat.mean(1, keepdim=True)  # (B,1,C)
    Xc = X_flat - mu
    cov = torch.matmul(Xc.transpose(1, 2), Xc) / max(N - 1, 1)  # (B,C,C)
    eye = torch.eye(C, device=data.device, dtype=data.dtype).expand(B, C, C)
    cov = cov + self.eps * eye
    md2 = self._quad_form_solve(Xc, cov)  # (B,N)
    scores = md2.view(B, H, W)
    return {"scores": scores.unsqueeze(-1)}
```

`ResNetExtractor`cuvis_ai_deepeiou.nodemodelbatchedembimginferlearnrgbtorchdeepeiou[↗](https://github.com/cubert-hyperspectral/cuvis-ai-deepeiou "Open plugin repo")ResNet-50 feature extractor (2048-dim embeddings).

ResNet-50 feature extractor (2048-dim embeddings).

Inputs

| Port    | Dtype     | Shape             | Description                                      |
| ------- | --------- | ----------------- | ------------------------------------------------ |
| `crops` | `float32` | `[-1, 3, -1, -1]` | Normalized crops [N, 3, crop_h, crop_w] in NCHW. |

Outputs

| Port         | Dtype     | Shape          | Description                         |
| ------------ | --------- | -------------- | ----------------------------------- |
| `embeddings` | `float32` | `[-1, -1, -1]` | L2-normalised embeddings [B, N, D]. |

[View plugin repo (v0.2.2)](https://github.com/cubert-hyperspectral/cuvis-ai-deepeiou/tree/v0.2.2)

`SAM3BboxPropagation`cuvis_ai_sam3.nodemodelbatchedbboxinferlearnmaskrgbsegstatefultorchtrackvidsam3[↗](https://github.com/cubert-hyperspectral/cuvis-ai-sam3 "Open plugin repo")SAM3 streaming propagation with runtime bbox prompts.

SAM3 streaming propagation with runtime bbox prompts.

Inputs

| Port                | Dtype     | Shape            | Description                                                                                               |
| ------------------- | --------- | ---------------- | --------------------------------------------------------------------------------------------------------- |
| `rgb_image`         | `float32` | `[1, -1, -1, 3]` |                                                                                                           |
| `frame_id` optional | `int64`   | `[1]`            | Source frame index [1]. If omitted, local stream index is used.                                           |
| `bboxes` optional   | `any`     | `any`            | Optional per-frame list of bbox prompt dicts with keys element_id, object_id, x_min, y_min, x_max, y_max. |

Outputs

| Port               | Dtype     | Shape         | Description |
| ------------------ | --------- | ------------- | ----------- |
| `mask`             | `int32`   | `[1, -1, -1]` |             |
| `object_ids`       | `int64`   | `[1, -1]`     |             |
| `detection_scores` | `float32` | `[1, -1]`     |             |

[View plugin repo (v0.3.3)](https://github.com/cubert-hyperspectral/cuvis-ai-sam3/tree/v0.3.3)

`SAM3MaskPropagation`cuvis_ai_sam3.nodemodelbatchedinferlearnmaskrgbsegstatefultorchtrackvidsam3[↗](https://github.com/cubert-hyperspectral/cuvis-ai-sam3 "Open plugin repo")SAM3 streaming propagation with runtime label-map prompts.

SAM3 streaming propagation with runtime label-map prompts.

Inputs

| Port                   | Dtype     | Shape            | Description                                                                                                          |
| ---------------------- | --------- | ---------------- | -------------------------------------------------------------------------------------------------------------------- |
| `rgb_image`            | `float32` | `[1, -1, -1, 3]` |                                                                                                                      |
| `frame_id` optional    | `int64`   | `[1]`            | Source frame index [1]. If omitted, local stream index is used.                                                      |
| `mask` optional        | `int32`   | `[1, -1, -1]`    | Optional int32 label map [1,H,W]. 0=background, each positive label is treated as an object ID prompt on that frame. |
| `text_prompt` optional | `any`     | `any`            | Optional text description applied only while injecting the runtime mask prompt on the current frame.                 |

Outputs

| Port               | Dtype     | Shape         | Description |
| ------------------ | --------- | ------------- | ----------- |
| `mask`             | `int32`   | `[1, -1, -1]` |             |
| `object_ids`       | `int64`   | `[1, -1]`     |             |
| `detection_scores` | `float32` | `[1, -1]`     |             |

[View plugin repo (v0.3.3)](https://github.com/cubert-hyperspectral/cuvis-ai-sam3/tree/v0.3.3)

`SAM3PointExpansion`cuvis_ai_sam3.nodemodelbatchedimginferkplearnmaskrgbsegtorchsam3[↗](https://github.com/cubert-hyperspectral/cuvis-ai-sam3 "Open plugin repo")Expand positive/negative click points into one object mask on a single frame.

Expand positive/negative click points into one object mask on a single frame.

Inputs

| Port                | Dtype     | Shape            | Description                                                                                                                                                   |
| ------------------- | --------- | ---------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `rgb_image`         | `float32` | `[1, -1, -1, 3]` | RGB frame [1,H,W,3] in float32 with values in [0,1].                                                                                                          |
| `points` optional   | `any`     | `any`            | Optional per-frame list of point prompt dicts with keys element_id, x, y, type (type in {positive, negative, neutral}). Positive=object, negative=background. |
| `frame_id` optional | `int64`   | `[1]`            | Optional source frame index [1]; keys the image-embedding cache.                                                                                              |

Outputs

| Port               | Dtype     | Shape         | Description |
| ------------------ | --------- | ------------- | ----------- |
| `mask`             | `int32`   | `[1, -1, -1]` |             |
| `object_ids`       | `int64`   | `[1, -1]`     |             |
| `detection_scores` | `float32` | `[1, -1]`     |             |

[View plugin repo (v0.3.3)](https://github.com/cubert-hyperspectral/cuvis-ai-sam3/tree/v0.3.3)

`SAM3PointPropagation`cuvis_ai_sam3.nodemodelbatchedinferkplearnmaskrgbsegstatefultorchtrackvidsam3[↗](https://github.com/cubert-hyperspectral/cuvis-ai-sam3 "Open plugin repo")SAM3 streaming propagation with a point prompt.

SAM3 streaming propagation with a point prompt.

Inputs

| Port                | Dtype     | Shape            | Description                                                     |
| ------------------- | --------- | ---------------- | --------------------------------------------------------------- |
| `rgb_image`         | `float32` | `[1, -1, -1, 3]` |                                                                 |
| `frame_id` optional | `int64`   | `[1]`            | Source frame index [1]. If omitted, local stream index is used. |

Outputs

| Port               | Dtype     | Shape         | Description |
| ------------------ | --------- | ------------- | ----------- |
| `mask`             | `int32`   | `[1, -1, -1]` |             |
| `object_ids`       | `int64`   | `[1, -1]`     |             |
| `detection_scores` | `float32` | `[1, -1]`     |             |

[View plugin repo (v0.3.3)](https://github.com/cubert-hyperspectral/cuvis-ai-sam3/tree/v0.3.3)

`SAM3SegmentEverything`cuvis_ai_sam3.nodemodelbatchedimginferlearnmaskrgbsegtorchsam3[↗](https://github.com/cubert-hyperspectral/cuvis-ai-sam3 "Open plugin repo")Segment everything on one RGB frame using SAM3 point-grid prompting.

Segment everything on one RGB frame using SAM3 point-grid prompting.

Inputs

| Port                | Dtype     | Shape            | Description                                                      |
| ------------------- | --------- | ---------------- | ---------------------------------------------------------------- |
| `rgb_image`         | `float32` | `[1, -1, -1, 3]` | RGB frame [1,H,W,3] in float32 with values in [0,1].             |
| `frame_id` optional | `int64`   | `[1]`            | Optional source frame index [1]. Ignored by this stateless node. |

Outputs

| Port               | Dtype     | Shape         | Description |
| ------------------ | --------- | ------------- | ----------- |
| `mask`             | `int32`   | `[1, -1, -1]` |             |
| `object_ids`       | `int64`   | `[1, -1]`     |             |
| `detection_scores` | `float32` | `[1, -1]`     |             |

[View plugin repo (v0.3.3)](https://github.com/cubert-hyperspectral/cuvis-ai-sam3/tree/v0.3.3)

`SAM3TextPropagation`cuvis_ai_sam3.nodemodelbatchedinferlearnmaskrgbsegstatefultexttorchtrackvidsam3[↗](https://github.com/cubert-hyperspectral/cuvis-ai-sam3 "Open plugin repo")SAM3 streaming propagation with a text/concept prompt.

SAM3 streaming propagation with a text/concept prompt.

Inputs

| Port                   | Dtype     | Shape            | Description                                                     |
| ---------------------- | --------- | ---------------- | --------------------------------------------------------------- |
| `rgb_image`            | `float32` | `[1, -1, -1, 3]` |                                                                 |
| `frame_id` optional    | `int64`   | `[1]`            | Source frame index [1]. If omitted, local stream index is used. |
| `text_prompt` optional | `any`     | `any`            | Optional text prompt applied on the current frame.              |

Outputs

| Port                 | Dtype     | Shape         | Description                                                                                           |
| -------------------- | --------- | ------------- | ----------------------------------------------------------------------------------------------------- |
| `mask`               | `int32`   | `[1, -1, -1]` |                                                                                                       |
| `object_ids`         | `int64`   | `[1, -1]`     |                                                                                                       |
| `detection_scores`   | `float32` | `[1, -1]`     |                                                                                                       |
| `category_ids`       | `int64`   | `[1, -1]`     | Category IDs [1,N], aligned with object_ids.                                                          |
| `category_semantics` | `uint8`   | `[-1]`        | UTF-8 JSON bytes of the cumulative category-id-to-text mapping, for example {"1":"person","2":"car"}. |

[View plugin repo (v0.3.3)](https://github.com/cubert-hyperspectral/cuvis-ai-sam3/tree/v0.3.3)

`SoftChannelSelector`cuvis_ai.node.channel_selectormodeldim-redhsilearnpretorchSoft channel selector with temperature-based Gumbel-Softmax selection.

#### SoftChannelSelector

```python
SoftChannelSelector(
    n_select,
    input_channels,
    init_method="uniform",
    temperature_init=5.0,
    temperature_min=0.1,
    temperature_decay=0.9,
    hard=False,
    eps=1e-06,
    **kwargs,
)
```

Bases: `Node`

Soft channel selector with temperature-based Gumbel-Softmax selection.

This is a **selector** node — it gates/reweights individual channels independently: `output[c] = weight[c] * input[c]` (diagonal operation, preserves channel count).

For cross-channel linear projection that reduces channel count, see :class:`cuvis_ai.node.channel_mixer.ConcreteChannelMixer` or :class:`cuvis_ai.node.channel_mixer.LearnableChannelMixer`.

This node learns to select a subset of input channels using differentiable channel selection with temperature annealing. Supports:

- Statistical initialization (uniform or importance-based)
- Gradient-based optimization with temperature scheduling
- Entropy and diversity regularization
- Hard selection at inference time

Parameters:

| Name                | Type                      | Description                                                    | Default     |
| ------------------- | ------------------------- | -------------------------------------------------------------- | ----------- |
| `n_select`          | `int`                     | Number of channels to select                                   | *required*  |
| `input_channels`    | `int`                     | Number of input channels                                       | *required*  |
| `init_method`       | `('uniform', 'variance')` | Initialization method for channel weights (default: "uniform") | `"uniform"` |
| `temperature_init`  | `float`                   | Initial temperature for Gumbel-Softmax (default: 5.0)          | `5.0`       |
| `temperature_min`   | `float`                   | Minimum temperature (default: 0.1)                             | `0.1`       |
| `temperature_decay` | `float`                   | Temperature decay factor per epoch (default: 0.9)              | `0.9`       |
| `hard`              | `bool`                    | If True, use hard selection at inference (default: False)      | `False`     |
| `eps`               | `float`                   | Small constant for numerical stability (default: 1e-6)         | `1e-06`     |

Attributes:

| Name             | Type                  | Description                                         |
| ---------------- | --------------------- | --------------------------------------------------- |
| `channel_logits` | `Parameter or Tensor` | Unnormalized channel importance scores [n_channels] |
| `temperature`    | `float`               | Current temperature for Gumbel-Softmax              |

Source code in `cuvis_ai/node/channel_selector.py`

```python
def __init__(
    self,
    n_select: int,
    input_channels: int,
    init_method: Literal["uniform", "variance"] = "uniform",
    temperature_init: float = 5.0,
    temperature_min: float = 0.1,
    temperature_decay: float = 0.9,
    hard: bool = False,
    eps: float = 1e-6,
    **kwargs,
) -> None:
    self.n_select = n_select
    self.input_channels = input_channels
    self.init_method = init_method
    self.temperature_init = temperature_init
    self.temperature_min = temperature_min
    self.temperature_decay = temperature_decay
    self.hard = hard
    self.eps = eps

    super().__init__(
        n_select=n_select,
        input_channels=input_channels,
        init_method=init_method,
        temperature_init=temperature_init,
        temperature_min=temperature_min,
        temperature_decay=temperature_decay,
        hard=hard,
        eps=eps,
        **kwargs,
    )

    # Temperature tracking (not a parameter, managed externally)
    self.temperature = temperature_init
    self._n_channels = input_channels

    # Validate selection size
    if self.n_select > self._n_channels:
        raise ValueError(
            f"Cannot select {self.n_select} channels from {self._n_channels} available channels"  # nosec B608
        )

    # Initialize channel logits based on method - always as buffer
    if self.init_method == "uniform":
        # Uniform initialization
        logits = torch.zeros(self._n_channels)
    elif self.init_method == "variance":
        # Random initialization - will be refined with fit if called
        logits = torch.randn(self._n_channels) * 0.01
    else:
        raise ValueError(f"Unknown init_method: {self.init_method}")

    # Store as buffer initially
    self.register_buffer("channel_logits", logits)

    self._statistically_initialized = False
```

##### statistical_initialization

```python
statistical_initialization(input_stream)
```

Initialize channel selection weights from data.

Parameters:

| Name           | Type          | Description                                                                                                             | Default    |
| -------------- | ------------- | ----------------------------------------------------------------------------------------------------------------------- | ---------- |
| `input_stream` | `InputStream` | Iterator yielding dicts matching INPUT_SPECS (port-based format) Expected format: {"data": tensor} where tensor is BHWC | *required* |

Source code in `cuvis_ai/node/channel_selector.py`

```python
def statistical_initialization(self, input_stream: InputStream) -> None:
    """Initialize channel selection weights from data.

    Parameters
    ----------
    input_stream : InputStream
        Iterator yielding dicts matching INPUT_SPECS (port-based format)
        Expected format: {"data": tensor} where tensor is BHWC
    """
    # Collect statistics from first batch to determine n_channels
    first_batch = next(iter(input_stream))
    x = first_batch["data"]

    if x is None:
        raise ValueError("No data provided for selector initialization")

    self._n_channels = x.shape[-1]
    # Keep everything computed here on the data's device so the
    # accumulator and logits match channel_logits' device when the
    # pipeline has been moved to GPU for training.
    device = x.device

    if self.n_select > self._n_channels:
        raise ValueError(
            f"Cannot select {self.n_select} channels from {self._n_channels} available channels"  # nosec B608
        )

    # Initialize channel logits based on method
    if self.init_method == "uniform":
        # Uniform initialization
        logits = torch.zeros(self._n_channels, device=device)
    elif self.init_method == "variance":
        # Importance-based initialization using channel variance
        acc = WelfordAccumulator(self._n_channels).to(device)
        acc.update(x.reshape(-1, x.shape[-1]))
        for batch_data in input_stream:
            x_batch = batch_data["data"]
            if x_batch is not None:
                acc.update(x_batch.reshape(-1, x_batch.shape[-1]))

        variance = acc.var  # [C]

        # Use log variance as initial logits (high variance = high importance)
        logits = torch.log(variance + self.eps)
    else:
        raise ValueError(f"Unknown init_method: {self.init_method}")

    # copy_ writes into the buffer and handles any residual device
    # difference between the computed logits and channel_logits.
    self.channel_logits.data.copy_(logits)
    self._statistically_initialized = True
```

##### update_temperature

```python
update_temperature(epoch=None, step=None)
```

Update temperature with decay schedule.

Parameters:

| Name    | Type  | Description                                       | Default |
| ------- | ----- | ------------------------------------------------- | ------- |
| `epoch` | `int` | Current epoch number (used for per-epoch decay)   | `None`  |
| `step`  | `int` | Current training step (for more granular control) | `None`  |

Source code in `cuvis_ai/node/channel_selector.py`

```python
def update_temperature(self, epoch: int | None = None, step: int | None = None) -> None:
    """Update temperature with decay schedule.

    Parameters
    ----------
    epoch : int, optional
        Current epoch number (used for per-epoch decay)
    step : int, optional
        Current training step (for more granular control)
    """
    if epoch is not None:
        # Exponential decay per epoch
        self.temperature = max(
            self.temperature_min, self.temperature_init * (self.temperature_decay**epoch)
        )
```

##### get_selection_weights

```python
get_selection_weights(hard=None)
```

Get current channel selection weights.

Parameters:

| Name   | Type   | Description                                                   | Default |
| ------ | ------ | ------------------------------------------------------------- | ------- |
| `hard` | `bool` | If True, use hard selection (top-k). If None, uses self.hard. | `None`  |

Returns:

| Type     | Description                                        |
| -------- | -------------------------------------------------- |
| `Tensor` | Selection weights [n_channels] summing to n_select |

Source code in `cuvis_ai/node/channel_selector.py`

```python
def get_selection_weights(self, hard: bool | None = None) -> Tensor:
    """Get current channel selection weights.

    Parameters
    ----------
    hard : bool, optional
        If True, use hard selection (top-k). If None, uses self.hard.

    Returns
    -------
    Tensor
        Selection weights [n_channels] summing to n_select
    """
    if hard is None:
        hard = self.hard and not self.training

    if hard:
        # Hard selection: top-k channels
        _, top_indices = torch.topk(self.channel_logits, self.n_select)
        weights = torch.zeros_like(self.channel_logits)
        weights[top_indices] = 1.0
    else:
        # Soft selection with Gumbel-Softmax
        # First, compute selection probabilities
        probs = F.softmax(self.channel_logits / self.temperature, dim=-1)

        # Scale to sum to n_select instead of 1
        weights = probs * self.n_select

    return weights
```

##### forward

```python
forward(data, **_)
```

Apply soft channel selection to input.

Parameters:

| Name   | Type     | Description               | Default    |
| ------ | -------- | ------------------------- | ---------- |
| `data` | `Tensor` | Input tensor [B, H, W, C] | *required* |

Returns:

| Type                | Description                                                                                                           |
| ------------------- | --------------------------------------------------------------------------------------------------------------------- |
| `dict[str, Tensor]` | Dictionary with "selected" key containing reweighted channels and optional "weights" key containing selection weights |

Source code in `cuvis_ai/node/channel_selector.py`

```python
def forward(self, data: Tensor, **_: Any) -> dict[str, Tensor]:
    """Apply soft channel selection to input.

    Parameters
    ----------
    data : Tensor
        Input tensor [B, H, W, C]

    Returns
    -------
    dict[str, Tensor]
        Dictionary with "selected" key containing reweighted channels
        and optional "weights" key containing selection weights
    """
    # Get selection weights
    weights = self.get_selection_weights()

    # Apply channel-wise weighting: [B, H, W, C] * [C]
    selected = data * weights.view(1, 1, 1, -1)

    # Prepare output dictionary - weights always exposed for loss/metric nodes
    outputs = {"selected": selected, "weights": weights}

    return outputs
```

`SpatialSpectralCNN2D`cuvis_ai_inspecscrap.node.classification.cnn2dmodelclasshsilearncuvis_ai_inspecscrap[↗](https://github.com/cubert-hyperspectral/cuvis-ai-inspecscrap "Open plugin repo")

[View plugin repo (v0.2.4)](https://github.com/cubert-hyperspectral/cuvis-ai-inspecscrap/tree/v0.2.4)

`SpectralAngleMapper`cuvis_ai.node.spectral_angle_mappermodelclasshsinumpystatefulCompute per-pixel spectral angle against one or more reference spectra.

#### SpectralAngleMapper

```python
SpectralAngleMapper(num_channels, eps=1e-12, **kwargs)
```

Bases: `Node`

Compute per-pixel spectral angle against one or more reference spectra.

Source code in `cuvis_ai/node/spectral_angle_mapper.py`

```python
def __init__(self, num_channels: int, eps: float = 1e-12, **kwargs: Any) -> None:
    if int(num_channels) <= 0:
        raise ValueError(f"num_channels must be > 0, got {num_channels}")
    self.num_channels = int(num_channels)
    self.eps = float(eps)
    super().__init__(num_channels=self.num_channels, eps=self.eps, **kwargs)
```

##### forward

```python
forward(cube, spectral_signature, **_)
```

Run spectral-angle scoring for all references.

Source code in `cuvis_ai/node/spectral_angle_mapper.py`

```python
@torch.no_grad()
def forward(
    self,
    cube: torch.Tensor,
    spectral_signature: torch.Tensor,
    **_: Any,
) -> dict[str, torch.Tensor]:
    """Run spectral-angle scoring for all references."""
    ref = spectral_signature.squeeze(1).squeeze(1)  # [N, C]
    channel_count = int(ref.shape[-1])
    ref_mean = ref.mean(dim=-1, keepdim=True)
    ref_norm = ref / (ref_mean + self.eps)

    pixel_mean = cube.mean(dim=-1, keepdim=True)
    cube_norm = cube / (pixel_mean + self.eps)

    ref_expanded = ref_norm.view(1, 1, 1, ref_norm.shape[0], channel_count)
    cube_expanded = cube_norm.unsqueeze(-2)

    dot = (cube_expanded * ref_expanded).sum(dim=-1)
    norms = cube_norm.norm(dim=-1, keepdim=True) * ref_norm.norm(dim=-1).view(1, 1, 1, -1)
    cos_sim = dot / (norms + self.eps)
    scores = torch.acos(cos_sim.clamp(-1.0, 1.0))

    best_scores = scores.amin(dim=-1, keepdim=True)
    identity_mask = scores.argmin(dim=-1).to(torch.int32) + 1

    return {
        "scores": scores,
        "best_scores": best_scores,
        "identity_mask": identity_mask,
    }
```

`SpectralMLPClassifier`cuvis_ai_inspecscrap.node.classification.mlpmodelclasshsilearncuvis_ai_inspecscrap[↗](https://github.com/cubert-hyperspectral/cuvis-ai-inspecscrap "Open plugin repo")

[View plugin repo (v0.2.4)](https://github.com/cubert-hyperspectral/cuvis-ai-inspecscrap/tree/v0.2.4)

`SpectralSpatialCNN3D`cuvis_ai_inspecscrap.node.classification.cnn3dmodelclasshsilearncuvis_ai_inspecscrap[↗](https://github.com/cubert-hyperspectral/cuvis-ai-inspecscrap "Open plugin repo")

[View plugin repo (v0.2.4)](https://github.com/cubert-hyperspectral/cuvis-ai-inspecscrap/tree/v0.2.4)

`SupervisedCIRSelector`cuvis_ai.node.channel_selectormodeldim-redhsilearnpretorchSupervised CIR/NIR band selection with window constraints.

#### SupervisedCIRSelector

```python
SupervisedCIRSelector(
    windows=(
        (840.0, 910.0),
        (650.0, 720.0),
        (500.0, 570.0),
    ),
    score_weights=(1.0, 1.0, 1.0),
    lambda_penalty=0.5,
    **kwargs,
)
```

Bases: `SupervisedSelectorBase`

Supervised CIR/NIR band selection with window constraints.

Windows are typically set to:

```text
- NIR: 840-910 nm
- Red: 650-720 nm
- Green: 500-570 nm
```

The selector chooses one band per window using a supervised score (Fisher + AUC + MI) with an mRMR-style redundancy penalty.

Source code in `cuvis_ai/node/channel_selector.py`

```python
def __init__(
    self,
    windows: Sequence[tuple[float, float]] = ((840.0, 910.0), (650.0, 720.0), (500.0, 570.0)),
    score_weights: tuple[float, float, float] = (1.0, 1.0, 1.0),
    lambda_penalty: float = 0.5,
    **kwargs: Any,
) -> None:
    super().__init__(
        score_weights=score_weights,
        lambda_penalty=lambda_penalty,
        windows=list(windows),
        **kwargs,
    )
    self.windows = list(windows)
```

`SupervisedFullSpectrumSelector`cuvis_ai.node.channel_selectormodeldim-redhsilearnpretorchSupervised selection without window constraints.

#### SupervisedFullSpectrumSelector

```python
SupervisedFullSpectrumSelector(
    score_weights=(1.0, 1.0, 1.0),
    lambda_penalty=0.5,
    **kwargs,
)
```

Bases: `SupervisedSelectorBase`

Supervised selection without window constraints.

Picks the top-3 discriminative bands globally with an mRMR-style redundancy penalty applied over the full spectrum.

Source code in `cuvis_ai/node/channel_selector.py`

```python
def __init__(
    self,
    score_weights: tuple[float, float, float] = (1.0, 1.0, 1.0),
    lambda_penalty: float = 0.5,
    **kwargs: Any,
) -> None:
    super().__init__(score_weights=score_weights, lambda_penalty=lambda_penalty, **kwargs)
```

`SupervisedSelectorBase`cuvis_ai.node.channel_selectormodeldim-redhsilearnpretorchBase class for supervised band selection strategies.

#### SupervisedSelectorBase

```python
SupervisedSelectorBase(
    num_spectral_bands,
    score_weights=(1.0, 1.0, 1.0),
    lambda_penalty=0.5,
    **kwargs,
)
```

Bases: `ChannelSelectorBase`

Base class for supervised band selection strategies.

This class adds an optional `mask` input port and implements common logic for statistical initialization via :meth:`fit`.

The mask is assumed to be binary (0/1), where 1 denotes the positive class (e.g. stone) and 0 denotes the negative class (e.g. lentil/background).

Source code in `cuvis_ai/node/channel_selector.py`

```python
def __init__(
    self,
    num_spectral_bands: int,
    score_weights: tuple[float, float, float] = (1.0, 1.0, 1.0),
    lambda_penalty: float = 0.5,
    **kwargs: Any,
) -> None:
    # Call super().__init__ FIRST so Serializable captures hparams correctly
    super().__init__(
        num_spectral_bands=num_spectral_bands,
        score_weights=score_weights,
        lambda_penalty=lambda_penalty,
        **kwargs,
    )
    # Then set instance attributes
    self.num_spectral_bands = num_spectral_bands
    self.score_weights = score_weights
    self.lambda_penalty = lambda_penalty
    # Initialize buffers with correct shapes (not empty)
    # selected_indices: always 3 for RGB
    # score buffers: num_spectral_bands
    self.register_buffer("selected_indices", torch.zeros(3, dtype=torch.long), persistent=True)
    self.register_buffer(
        "band_scores", torch.zeros(num_spectral_bands, dtype=torch.float32), persistent=True
    )
    self.register_buffer(
        "fisher_scores", torch.zeros(num_spectral_bands, dtype=torch.float32), persistent=True
    )
    self.register_buffer(
        "auc_scores", torch.zeros(num_spectral_bands, dtype=torch.float32), persistent=True
    )
    self.register_buffer(
        "mi_scores", torch.zeros(num_spectral_bands, dtype=torch.float32), persistent=True
    )
    # Use standard instance attribute for initialization tracking
    self._statistically_initialized = False
```

##### requires_initial_fit

```python
requires_initial_fit
```

Whether this node requires statistical initialization from training data.

Returns:

| Type   | Description                                |
| ------ | ------------------------------------------ |
| `bool` | Always True for supervised band selectors. |

##### statistical_initialization

```python
statistical_initialization(input_stream)
```

Initialize band selection using supervised scoring.

Computes Fisher, AUC, and MI scores for each band, delegates to :meth:`_select_bands` for strategy-specific selection, and stores the 3 selected bands.

Parameters:

| Name           | Type          | Description                                            | Default    |
| -------------- | ------------- | ------------------------------------------------------ | ---------- |
| `input_stream` | `InputStream` | Training data stream with cube, mask, and wavelengths. | *required* |

Raises:

| Type         | Description                                       |
| ------------ | ------------------------------------------------- |
| `ValueError` | If band selection doesn't return exactly 3 bands. |

Source code in `cuvis_ai/node/channel_selector.py`

```python
def statistical_initialization(self, input_stream: InputStream) -> None:
    """Initialize band selection using supervised scoring.

    Computes Fisher, AUC, and MI scores for each band, delegates to
    :meth:`_select_bands` for strategy-specific selection, and stores
    the 3 selected bands.

    Parameters
    ----------
    input_stream : InputStream
        Training data stream with cube, mask, and wavelengths.

    Raises
    ------
    ValueError
        If band selection doesn't return exactly 3 bands.
    """
    cubes, masks, wavelengths = self._collect_training_data(input_stream)
    band_scores, fisher_scores, auc_scores, mi_scores = _compute_band_scores_supervised(
        cubes,
        masks,
        wavelengths,
        self.score_weights,
    )
    corr_matrix = _compute_band_correlation_matrix(cubes, len(wavelengths))
    selected_indices = self._select_bands(band_scores, wavelengths, corr_matrix)
    if len(selected_indices) != 3:
        raise ValueError(f"{type(self).__name__} expected 3 bands, got {len(selected_indices)}")
    self._store_scores_and_indices(
        band_scores, fisher_scores, auc_scores, mi_scores, selected_indices
    )
```

##### forward

```python
forward(cube, wavelengths, mask=None, context=None, **_)
```

Generate false-color RGB from selected bands.

Parameters:

| Name          | Type      | Description                                                         | Default    |
| ------------- | --------- | ------------------------------------------------------------------- | ---------- |
| `cube`        | `Tensor`  | Hyperspectral cube [B, H, W, C].                                    | *required* |
| `wavelengths` | `ndarray` | Wavelengths for each channel [C].                                   | *required* |
| `mask`        | `Tensor`  | Ground truth mask (unused in forward, required for initialization). | `None`     |
| `context`     | `Context` | Pipeline execution context (unused).                                | `None`     |
| `**_`         | `Any`     | Additional unused keyword arguments.                                | `{}`       |

Returns:

| Type             | Description                                                        |
| ---------------- | ------------------------------------------------------------------ |
| `dict[str, Any]` | Dictionary with "rgb_image" [B, H, W, 3] and "band_info" metadata. |

Raises:

| Type           | Description                                         |
| -------------- | --------------------------------------------------- |
| `RuntimeError` | If the node has not been statistically initialized. |

Source code in `cuvis_ai/node/channel_selector.py`

```python
def forward(
    self,
    cube: torch.Tensor,
    wavelengths: np.ndarray,
    mask: torch.Tensor | None = None,  # noqa: ARG002
    context: Context | None = None,  # noqa: ARG002
    **_: Any,
) -> dict[str, Any]:
    """Generate false-color RGB from selected bands.

    Parameters
    ----------
    cube : torch.Tensor
        Hyperspectral cube [B, H, W, C].
    wavelengths : np.ndarray
        Wavelengths for each channel [C].
    mask : torch.Tensor, optional
        Ground truth mask (unused in forward, required for initialization).
    context : Context, optional
        Pipeline execution context (unused).
    **_ : Any
        Additional unused keyword arguments.

    Returns
    -------
    dict[str, Any]
        Dictionary with "rgb_image" [B, H, W, 3] and "band_info" metadata.

    Raises
    ------
    RuntimeError
        If the node has not been statistically initialized.
    """
    if not self._statistically_initialized or self.selected_indices.numel() != 3:
        raise RuntimeError(f"{type(self).__name__} not fitted")

    wavelengths_np = np.asarray(wavelengths, dtype=np.float32)
    indices = self.selected_indices.tolist()
    rgb = self._compose_rgb(cube, indices)

    band_info = {
        "strategy": self._strategy_name,
        "band_indices": indices,
        "band_wavelengths_nm": [float(wavelengths_np[i]) for i in indices],
        "score_weights": list(self.score_weights),
        "lambda_penalty": float(self.lambda_penalty),
        **self._extra_band_info(wavelengths_np),
    }
    return {"rgb_image": rgb, "band_info": band_info}
```

`SupervisedWindowedSelector`cuvis_ai.node.channel_selectormodeldim-redhsilearnpretorchSupervised band selection constrained to visible RGB windows.

#### SupervisedWindowedSelector

```python
SupervisedWindowedSelector(
    windows=(
        (440.0, 500.0),
        (500.0, 580.0),
        (610.0, 700.0),
    ),
    score_weights=(1.0, 1.0, 1.0),
    lambda_penalty=0.5,
    **kwargs,
)
```

Bases: `SupervisedSelectorBase`

Supervised band selection constrained to visible RGB windows.

Similar to :class:`HighContrastSelector`, but uses label-driven scores. Default windows:

```text
- Blue: 440-500 nm
- Green: 500-580 nm
- Red: 610-700 nm
```

Source code in `cuvis_ai/node/channel_selector.py`

```python
def __init__(
    self,
    windows: Sequence[tuple[float, float]] = ((440.0, 500.0), (500.0, 580.0), (610.0, 700.0)),
    score_weights: tuple[float, float, float] = (1.0, 1.0, 1.0),
    lambda_penalty: float = 0.5,
    **kwargs: Any,
) -> None:
    super().__init__(
        score_weights=score_weights,
        lambda_penalty=lambda_penalty,
        windows=list(windows),
        **kwargs,
    )
    self.windows = list(windows)
```

`TrainablePCA`cuvis_ai.node.dimensionality_reductionmodeldim-redhsilearnprestatefultorchTrainable PCA node with orthogonality regularization.

#### TrainablePCA

```python
TrainablePCA(
    num_channels,
    n_components,
    whiten=False,
    init_method="svd",
    eps=1e-06,
    **kwargs,
)
```

Bases: `PCA`

Trainable PCA node with orthogonality regularization.

Source code in `cuvis_ai/node/dimensionality_reduction.py`

```python
def __init__(
    self,
    num_channels: int,
    n_components: int,
    whiten: bool = False,
    init_method: Literal["svd", "random"] = "svd",
    eps: float = 1e-6,
    **kwargs,
) -> None:
    self.whiten = whiten
    self.init_method = init_method

    super().__init__(
        num_channels=num_channels,
        n_components=n_components,
        whiten=whiten,
        init_method=init_method,
        eps=eps,
        **kwargs,
    )

    # Buffers for statistical initialization (private to avoid conflicts with output ports)
    self.register_buffer("_mean", torch.empty(num_channels))
    self.register_buffer("_explained_variance", torch.empty(n_components))
    self.register_buffer("_components", torch.empty(n_components, num_channels))

    self._statistically_initialized = False
```

##### statistical_initialization

```python
statistical_initialization(input_stream)
```

Initialize PCA components from data using covariance eigen decomposition.

Source code in `cuvis_ai/node/dimensionality_reduction.py`

```python
def statistical_initialization(self, input_stream: InputStream) -> None:
    """Initialize PCA components from data using covariance eigen decomposition."""
    acc = None
    for batch_data in input_stream:
        x = batch_data["data"]
        if x is not None:
            flat = x.reshape(-1, x.shape[-1])  # [B*H*W, C]
            if acc is None:
                acc = WelfordAccumulator(flat.shape[1], track_covariance=True)
            acc.update(flat)

    if acc is None or acc.count == 0:
        raise ValueError("No data provided for PCA initialization")

    self._mean = acc.mean.to(dtype=torch.float32)  # [C]
    cov = acc.cov.to(torch.float64)  # [C, C]

    # Eigen decomposition on covariance (equivalent to SVD on centered data)
    eigenvalues, eigenvectors = torch.linalg.eigh(cov)
    eigenvalues = eigenvalues.flip(0)
    eigenvectors = eigenvectors.flip(1)

    # Extract top n_components (rows = principal components)
    self._components = eigenvectors[:, : self.n_components].T.float()  # [n_components, C]
    self._explained_variance = eigenvalues[: self.n_components].float()  # [n_components]

    self._statistically_initialized = True
```

##### forward

```python
forward(data, **_)
```

Project data onto statistically initialized global components.

Source code in `cuvis_ai/node/dimensionality_reduction.py`

```python
def forward(self, data: Tensor, **_: Any) -> dict[str, Tensor]:
    """Project data onto statistically initialized global components."""
    if not self._statistically_initialized:
        raise RuntimeError("PCA not initialized. Call statistical_initialization() first.")

    if data.ndim != 4:
        raise ValueError(f"Expected data with shape [B, H, W, C], got {tuple(data.shape)}")

    batch_size, height, width, channels = data.shape
    flat = data.reshape(-1, channels)

    components = (
        self._components.to(data.device)
        if isinstance(self._components, Tensor)
        else self._components
    )
    projected = self._project(flat, self._mean, components)

    if self.whiten:
        explained_variance = self._explained_variance.to(
            device=data.device, dtype=projected.dtype
        )
        scale = 1.0 / torch.sqrt(explained_variance + self.eps)
        projected = projected * scale

    outputs = {
        "projected": projected.reshape(batch_size, height, width, self.n_components),
    }

    if self._explained_variance.numel() > 0:
        outputs["explained_variance_ratio"] = self._variance_ratio(self._explained_variance).to(
            data.device
        )

    if self._components.numel() > 0:
        outputs["components"] = self._components

    return outputs
```

`YOLO26Detection`cuvis_ai_ultralytics.nodemodelbatchedbboxdetimginferlearnrgbtorchultralytics[↗](https://github.com/cubert-hyperspectral/cuvis-ai-ultralytics "Open plugin repo")Run YOLO26 raw tensor inference on a stride-aligned CHW BGR batch.

Run YOLO26 raw tensor inference on a stride-aligned CHW BGR batch.

Inputs

| Port           | Dtype     | Shape             | Description                                          |
| -------------- | --------- | ----------------- | ---------------------------------------------------- |
| `preprocessed` | `float32` | `[-1, 3, -1, -1]` | Channel-first BGR [B, 3, H', W'] from YOLOPreprocess |

Outputs

| Port        | Dtype     | Shape          | Description                |
| ----------- | --------- | -------------- | -------------------------- |
| `raw_preds` | `float32` | `[-1, -1, -1]` | Raw YOLO prediction tensor |

[View plugin repo (v0.1.4)](https://github.com/cubert-hyperspectral/cuvis-ai-ultralytics/tree/v0.1.4)

`_StatisticalFitNode`cuvis_ai.node.\_statistical_fitmodelhsistatefultorchBase for nodes fitted once during statistical initialization.

#### \_StatisticalFitNode

```python
_StatisticalFitNode(
    *, max_fit_pixels=20000, fit_seed=0, **kwargs
)
```

Bases: `Node`

Base for nodes fitted once during statistical initialization.

Subclasses either implement `_fit(pixels)` (and let the default `statistical_initialization` collect the pixel matrix), or override `statistical_initialization` entirely (streaming-moment nodes) while reusing `_require_initialized` / `_reject_if_insufficient` / `_mark_initialized` from this base.

Store the fit-subsample budget and register the persistent fit flag.

Parameters:

| Name             | Type  | Description                                                                                                                                                                        | Default |
| ---------------- | ----- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------- |
| `max_fit_pixels` | `int` | Upper bound on training pixels gathered by \_collect_pixels; 0 disables subsampling. Bounds memory and keeps sklearn fits tractable on full hyperspectral frames (default: 20000). | `20000` |
| `fit_seed`       | `int` | Seed for the subsample permutation, for reproducible fits (default: 0).                                                                                                            | `0`     |

Source code in `cuvis_ai/node/_statistical_fit.py`

```python
def __init__(self, *, max_fit_pixels: int = 20000, fit_seed: int = 0, **kwargs: Any) -> None:
    """Store the fit-subsample budget and register the persistent fit flag.

    Parameters
    ----------
    max_fit_pixels : int, optional
        Upper bound on training pixels gathered by ``_collect_pixels``;
        ``0`` disables subsampling. Bounds memory and keeps sklearn fits
        tractable on full hyperspectral frames (default: 20000).
    fit_seed : int, optional
        Seed for the subsample permutation, for reproducible fits
        (default: 0).
    """
    self.max_fit_pixels = int(max_fit_pixels)
    self.fit_seed = int(fit_seed)
    super().__init__(max_fit_pixels=self.max_fit_pixels, fit_seed=self.fit_seed, **kwargs)
    self.register_buffer("_initialized", torch.zeros(1, dtype=torch.bool))
```

##### is_initialized

```python
is_initialized
```

Whether `statistical_initialization` has successfully fitted state.

##### statistical_initialization

```python
statistical_initialization(input_stream)
```

Collect training pixels, reject empties, fit, and mark initialized.

Subclasses that fit from a raw `[N, C]` matrix implement `_fit`; streaming-moment subclasses override this method instead.

Parameters:

| Name           | Type          | Description                                              | Default    |
| -------------- | ------------- | -------------------------------------------------------- | ---------- |
| `input_stream` | `InputStream` | Iterable of port-keyed batch dicts matching INPUT_SPECS. | *required* |

Source code in `cuvis_ai/node/_statistical_fit.py`

```python
def statistical_initialization(self, input_stream: InputStream) -> None:
    """Collect training pixels, reject empties, fit, and mark initialized.

    Subclasses that fit from a raw ``[N, C]`` matrix implement ``_fit``;
    streaming-moment subclasses override this method instead.

    Parameters
    ----------
    input_stream : InputStream
        Iterable of port-keyed batch dicts matching ``INPUT_SPECS``.
    """
    pixels = self._collect_pixels(input_stream)
    self._reject_if_insufficient(pixels.shape[0])
    self._fit(pixels)
    self._mark_initialized()
```

`OrthogonalityLoss`cuvis_ai.node.lossesregularizerdifftorchtrainOrthogonality regularization loss for TrainablePCA.

#### OrthogonalityLoss

```python
OrthogonalityLoss(weight=1.0, **kwargs)
```

Bases: `LossNode`

Orthogonality regularization loss for TrainablePCA.

Encourages PCA components to remain orthonormal during training. Loss = weight * ||W @ W.T - I||^2_F

Parameters:

| Name     | Type    | Description                                  | Default |
| -------- | ------- | -------------------------------------------- | ------- |
| `weight` | `float` | Weight for orthogonality loss (default: 1.0) | `1.0`   |

Source code in `cuvis_ai/node/losses.py`

```python
def __init__(self, weight: float = 1.0, **kwargs) -> None:
    self.weight = weight

    super().__init__(
        weight=weight,
        **kwargs,
    )
```

##### forward

```python
forward(components, **_)
```

Compute weighted orthogonality loss from PCA components.

Parameters:

| Name         | Type     | Description                                      | Default    |
| ------------ | -------- | ------------------------------------------------ | ---------- |
| `components` | `Tensor` | PCA components matrix [n_components, n_features] | *required* |

Returns:

| Type                | Description                                         |
| ------------------- | --------------------------------------------------- |
| `dict[str, Tensor]` | Dictionary with "loss" key containing weighted loss |

Source code in `cuvis_ai/node/losses.py`

```python
def forward(self, components: Tensor, **_: Any) -> dict[str, Tensor]:
    """Compute weighted orthogonality loss from PCA components.

    Parameters
    ----------
    components : Tensor
        PCA components matrix [n_components, n_features]

    Returns
    -------
    dict[str, Tensor]
        Dictionary with "loss" key containing weighted loss
    """
    # Compute gram matrix: W @ W.T
    gram = components @ components.T

    # Target: identity matrix
    n_components = components.shape[0]
    eye = torch.eye(
        n_components,
        device=components.device,
        dtype=components.dtype,
    )

    # Frobenius norm of difference
    orth_loss = torch.sum((gram - eye) ** 2)

    return {"loss": self.weight * orth_loss}
```

`SelectorDiversityRegularizer`cuvis_ai.node.lossesregularizerdifftorchtrainDiversity regularization for SoftChannelSelector.

#### SelectorDiversityRegularizer

```python
SelectorDiversityRegularizer(weight=0.01, **kwargs)
```

Bases: `LossNode`

Diversity regularization for SoftChannelSelector.

Encourages diverse channel selection by penalizing concentration on few channels. Uses negative variance to encourage spread (higher variance = more diverse).

Parameters:

| Name     | Type    | Description                                         | Default |
| -------- | ------- | --------------------------------------------------- | ------- |
| `weight` | `float` | Weight for diversity regularization (default: 0.01) | `0.01`  |

Source code in `cuvis_ai/node/losses.py`

```python
def __init__(self, weight: float = 0.01, **kwargs) -> None:
    self.weight = weight
    super().__init__(
        weight=weight,
        **kwargs,
    )
```

##### forward

```python
forward(weights, **_)
```

Compute weighted diversity loss from selection weights.

Parameters:

| Name      | Type     | Description                            | Default    |
| --------- | -------- | -------------------------------------- | ---------- |
| `weights` | `Tensor` | Channel selection weights [n_channels] | *required* |

Returns:

| Type                | Description                                         |
| ------------------- | --------------------------------------------------- |
| `dict[str, Tensor]` | Dictionary with "loss" key containing weighted loss |

Source code in `cuvis_ai/node/losses.py`

```python
def forward(self, weights: Tensor, **_: Any) -> dict[str, Tensor]:
    """Compute weighted diversity loss from selection weights.

    Parameters
    ----------
    weights : Tensor
        Channel selection weights [n_channels]

    Returns
    -------
    dict[str, Tensor]
        Dictionary with "loss" key containing weighted loss
    """
    # Compute variance of weights (high variance = diverse selection)
    mean_weight = weights.mean()
    variance = ((weights - mean_weight) ** 2).mean()

    # Return negative variance (minimizing loss = maximizing variance = maximizing diversity)
    diversity_loss = -variance

    return {"loss": self.weight * diversity_loss}
```

`SelectorEntropyRegularizer`cuvis_ai.node.lossesregularizerdifftorchtrainEntropy regularization for SoftChannelSelector.

#### SelectorEntropyRegularizer

```python
SelectorEntropyRegularizer(
    weight=0.01, target_entropy=None, eps=1e-06, **kwargs
)
```

Bases: `LossNode`

Entropy regularization for SoftChannelSelector.

Encourages exploration by penalizing low-entropy (over-confident) selections. Computes entropy from selection weights and applies regularization.

Higher entropy = more uniform selection (encouraged early in training) Lower entropy = more peaked selection (emerges naturally as training progresses)

Parameters:

| Name             | Type    | Description                                                                                                                                                              | Default |
| ---------------- | ------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------- |
| `weight`         | `float` | Weight for entropy regularization (default: 0.01) Positive weight encourages exploration (maximizes entropy) Negative weight encourages exploitation (minimizes entropy) | `0.01`  |
| `target_entropy` | `float` | Target entropy for regularization (default: None, no target) If set, uses squared error: (entropy - target)^2                                                            | `None`  |
| `eps`            | `float` | Small constant for numerical stability (default: 1e-6)                                                                                                                   | `1e-06` |

Source code in `cuvis_ai/node/losses.py`

```python
def __init__(
    self,
    weight: float = 0.01,
    target_entropy: float | None = None,
    eps: float = 1e-6,
    **kwargs,
) -> None:
    self.weight = weight
    self.target_entropy = target_entropy
    self.eps = eps

    super().__init__(
        weight=weight,
        target_entropy=target_entropy,
        eps=eps,
        **kwargs,
    )
```

##### forward

```python
forward(weights, **_)
```

Compute entropy regularization loss from selection weights.

Parameters:

| Name      | Type     | Description                            | Default    |
| --------- | -------- | -------------------------------------- | ---------- |
| `weights` | `Tensor` | Channel selection weights [n_channels] | *required* |

Returns:

| Type                | Description                                               |
| ------------------- | --------------------------------------------------------- |
| `dict[str, Tensor]` | Dictionary with "loss" key containing regularization loss |

Source code in `cuvis_ai/node/losses.py`

```python
def forward(self, weights: Tensor, **_: Any) -> dict[str, Tensor]:
    """Compute entropy regularization loss from selection weights.

    Parameters
    ----------
    weights : Tensor
        Channel selection weights [n_channels]

    Returns
    -------
    dict[str, Tensor]
        Dictionary with "loss" key containing regularization loss
    """
    # Normalize weights to probabilities
    probs = weights / (weights.sum() + self.eps)

    # Compute entropy: -sum(p * log(p))
    entropy = -(probs * torch.log(probs + self.eps)).sum()

    # Compute loss
    if self.target_entropy is not None:
        # Target-based regularization: minimize distance to target
        loss = (entropy - self.target_entropy) ** 2
    else:
        # Simple regularization:
        # maximize (positive weight) or minimize (negative weight) entropy
        loss = -entropy

    # Apply weight
    return {"loss": self.weight * loss}
```

`ClassMapAccumulator`cuvis_ai.node.patch_inferencesinkclassmaskpostScatter chunked patch predictions back into per-frame `[H, W]` class maps (sink).

#### ClassMapAccumulator

```python
ClassMapAccumulator(background_value=-1, **kwargs)
```

Bases: `Node`

Scatter chunked patch predictions back into per-frame `[H, W]` class maps (sink).

Consumes the provenance contract emitted by a patch-tiler data module (not by :class:`PatchSampler`): a tiler streams a frame's pixels through a classifier in batches, each patch tagged with its provenance `(frame_id, y, x)` plus the source frame `height`/`width`. This sink argmaxes the per-batch `logits` and writes each prediction into the right pixel of a per-frame map. After the run the finished maps are read from :attr:`class_maps`.

The run lifecycle is `reset()` (clear maps at the start) -> `forward()` per batch -> `close()` (no external resource; maps stay available via :attr:`class_maps`). One `[H, W]` map is retained per distinct `frame_id` until the next :meth:`reset`, so memory grows with the number of frames in a run; call `reset()` between runs. Per-frame eviction on long streams is out of scope.

Parameters:

| Name               | Type  | Description                                           | Default |
| ------------------ | ----- | ----------------------------------------------------- | ------- |
| `background_value` | `int` | Fill value for pixels no patch wrote to (default -1). | `-1`    |

Store the background fill and start with an empty map set.

Source code in `cuvis_ai/node/patch_inference.py`

```python
def __init__(self, background_value: int = -1, **kwargs: Any) -> None:
    """Store the background fill and start with an empty map set."""
    self.background_value = int(background_value)
    self._maps: dict[int, torch.Tensor] = {}
    super().__init__(background_value=self.background_value, **kwargs)
```

##### class_maps

```python
class_maps
```

Finished per-frame class maps `{frame_id: [H, W] int64}` (background = `background_value`).

##### reset

```python
reset()
```

Clear accumulated maps before a new prediction run (called by the Predictor).

Source code in `cuvis_ai/node/patch_inference.py`

```python
def reset(self) -> None:
    """Clear accumulated maps before a new prediction run (called by the Predictor)."""
    self._maps = {}
```

##### forward

```python
forward(logits, frame_id, y, x, height, width, **_)
```

Argmax the batch's logits and scatter each prediction into its frame's class map.

Parameters:

| Name       | Type     | Description                                                                                                                                            | Default    |
| ---------- | -------- | ------------------------------------------------------------------------------------------------------------------------------------------------------ | ---------- |
| `logits`   | `Tensor` | Per-patch class logits [N, num_classes].                                                                                                               | *required* |
| `frame_id` | `Tensor` | Per-patch provenance \[N\]: source frame id, pixel row/column, and the source frame height/width used to size a frame's map the first time it is seen. | *required* |
| `y`        | `Tensor` | Per-patch provenance \[N\]: source frame id, pixel row/column, and the source frame height/width used to size a frame's map the first time it is seen. | *required* |
| `x`        | `Tensor` | Per-patch provenance \[N\]: source frame id, pixel row/column, and the source frame height/width used to size a frame's map the first time it is seen. | *required* |
| `height`   | `Tensor` | Per-patch provenance \[N\]: source frame id, pixel row/column, and the source frame height/width used to size a frame's map the first time it is seen. | *required* |
| `width`    | `Tensor` | Per-patch provenance \[N\]: source frame id, pixel row/column, and the source frame height/width used to size a frame's map the first time it is seen. | *required* |
| `**_`      | `Any`    | Additional unused keyword arguments (e.g. the pipeline context).                                                                                       | `{}`       |

Returns:

| Type             | Description                                                     |
| ---------------- | --------------------------------------------------------------- |
| `dict[str, Any]` | Empty dict (sink node); results accumulate in :attr:class_maps. |

Raises:

| Type         | Description                                                    |
| ------------ | -------------------------------------------------------------- |
| `IndexError` | If a patch's (y, x) falls outside its frame's (height, width). |

Source code in `cuvis_ai/node/patch_inference.py`

```python
@torch.no_grad()
def forward(
    self,
    logits: torch.Tensor,
    frame_id: torch.Tensor,
    y: torch.Tensor,
    x: torch.Tensor,
    height: torch.Tensor,
    width: torch.Tensor,
    **_: Any,
) -> dict[str, Any]:
    """Argmax the batch's logits and scatter each prediction into its frame's class map.

    Parameters
    ----------
    logits : torch.Tensor
        Per-patch class logits ``[N, num_classes]``.
    frame_id, y, x, height, width : torch.Tensor
        Per-patch provenance ``[N]``: source frame id, pixel row/column, and the source frame
        height/width used to size a frame's map the first time it is seen.
    **_ : Any
        Additional unused keyword arguments (e.g. the pipeline ``context``).

    Returns
    -------
    dict[str, Any]
        Empty dict (sink node); results accumulate in :attr:`class_maps`.

    Raises
    ------
    IndexError
        If a patch's ``(y, x)`` falls outside its frame's ``(height, width)``.
    """
    preds = logits.argmax(dim=-1).to(torch.long).cpu()
    fid_t, y_t, x_t = frame_id.cpu(), y.cpu(), x.cpu()
    h_t, w_t = height.cpu(), width.cpu()
    for fid in torch.unique(fid_t).tolist():
        sel = fid_t == fid
        fid = int(fid)
        if fid not in self._maps:
            h, w = int(h_t[sel][0]), int(w_t[sel][0])
            self._maps[fid] = torch.full((h, w), self.background_value, dtype=torch.long)
        cmap = self._maps[fid]
        h, w = cmap.shape
        ys, xs = y_t[sel], x_t[sel]
        if (ys < 0).any() or (ys >= h).any() or (xs < 0).any() or (xs >= w).any():
            raise IndexError(
                f"patch coordinates out of bounds for frame {fid} of size ({h}, {w})."
            )
        cmap[ys, xs] = preds[sel]
    return {}
```

##### close

```python
close()
```

No external resource to release; maps stay available via :attr:`class_maps`.

Source code in `cuvis_ai/node/patch_inference.py`

```python
def close(self) -> None:
    """No external resource to release; maps stay available via :attr:`class_maps`."""
```

`ClassMapAccumulator`cuvis_ai_inspecscrap.node.class_map_accumulatorsinkclassmaskpostcuvis_ai_inspecscrap[↗](https://github.com/cubert-hyperspectral/cuvis-ai-inspecscrap "Open plugin repo")

[View plugin repo (v0.2.4)](https://github.com/cubert-hyperspectral/cuvis-ai-inspecscrap/tree/v0.2.4)

`CocoTrackBBoxWriter`cuvis_ai.node.json_filesinkbboxmetatrackWrite tracked bbox outputs into COCO tracking JSON.

#### CocoTrackBBoxWriter

```python
CocoTrackBBoxWriter(
    output_json_path,
    category_id_to_name=None,
    write_empty_frames=True,
    atomic_write=True,
    flush_interval=0,
    **kwargs,
)
```

Bases: `_BaseCocoTrackWriter`

Write tracked bbox outputs into COCO tracking JSON.

Source code in `cuvis_ai/node/json_file.py`

```python
def __init__(
    self,
    output_json_path: str,
    category_id_to_name: dict[int, str] | None = None,
    write_empty_frames: bool = True,
    atomic_write: bool = True,
    flush_interval: int = 0,
    **kwargs: Any,
) -> None:
    self.category_id_to_name: dict[int, str] = (
        dict(category_id_to_name) if category_id_to_name is not None else {0: "object"}
    )
    self.write_empty_frames = bool(write_empty_frames)
    self._frames_by_id: dict[int, dict[str, Any]] = {}

    super().__init__(
        output_json_path=output_json_path,
        atomic_write=atomic_write,
        flush_interval=flush_interval,
        category_id_to_name=self.category_id_to_name,
        write_empty_frames=write_empty_frames,
        **kwargs,
    )
```

##### forward

```python
forward(
    frame_id,
    bboxes,
    category_ids,
    confidences,
    track_ids,
    orig_hw,
    context=None,
    **_,
)
```

Store one frame of tracked bounding boxes for later export.

Source code in `cuvis_ai/node/json_file.py`

```python
def forward(
    self,
    frame_id: torch.Tensor,
    bboxes: torch.Tensor,
    category_ids: torch.Tensor,
    confidences: torch.Tensor,
    track_ids: torch.Tensor,
    orig_hw: torch.Tensor,
    context: Context | None = None,  # noqa: ARG002
    **_: Any,
) -> dict[str, Any]:
    """Store one frame of tracked bounding boxes for later export."""
    frame_idx = self._parse_frame_id(frame_id)
    ids_1d = self._parse_vector(category_ids, port_name="category_ids")
    scores_1d = self._parse_vector(confidences, port_name="confidences")
    track_ids_1d = self._parse_vector(track_ids, port_name="track_ids")
    self._validate_alignment(ids_1d, scores_1d, "category_ids", "confidences")
    self._validate_alignment(ids_1d, track_ids_1d, "category_ids", "track_ids")

    h, w = int(orig_hw[0, 0]), int(orig_hw[0, 1])
    boxes_2d = bboxes[0] if bboxes.ndim == 3 else bboxes

    n = int(ids_1d.numel())
    detections: list[dict[str, Any]] = []
    for i in range(n):
        x1, y1, x2, y2 = boxes_2d[i].cpu().tolist()
        bw = float(x2 - x1)
        bh = float(y2 - y1)
        detections.append(
            {
                "category_id": int(ids_1d[i].item()),
                "bbox": [float(x1), float(y1), bw, bh],
                "area": bw * bh,
                "score": float(scores_1d[i].item()),
                "track_id": int(track_ids_1d[i].item()),
            }
        )

    if not detections and not self.write_empty_frames:
        return {}

    self._frames_by_id[frame_idx] = {
        "frame_idx": frame_idx,
        "height": h,
        "width": w,
        "detections": detections,
    }
    self._mark_dirty_and_maybe_flush()
    return {}
```

`CocoTrackMaskWriter`cuvis_ai.node.json_filesinkmaskmetatrackWrite mask tracking outputs as COCO JSON in one of two dialects.

#### CocoTrackMaskWriter

```python
CocoTrackMaskWriter(
    output_json_path,
    dialect="image",
    default_category_name="object",
    write_empty_frames=True,
    atomic_write=True,
    flush_interval=0,
    **kwargs,
)
```

Bases: `_BaseCocoTrackWriter`

Write mask tracking outputs as COCO JSON in one of two dialects.

The default `dialect="image"` emits standard image-keyed COCO: per-frame `images` records plus one annotation per (track, frame) carrying an RLE `segmentation`, `bbox`/`area`, and additive `track_id`/`score` keys — readable by pycocotools and every image-keyed COCO consumer. `dialect="video"` emits the legacy YouTube-VIS-shaped track dialect (top-level `videos`, one annotation per track with per-frame parallel arrays) for external video-COCO tooling.

Source code in `cuvis_ai/node/json_file.py`

```python
def __init__(
    self,
    output_json_path: str,
    dialect: str = "image",
    default_category_name: str = "object",
    write_empty_frames: bool = True,
    atomic_write: bool = True,
    flush_interval: int = 0,
    **kwargs: Any,
) -> None:
    if dialect not in {"image", "video"}:
        raise ValueError("dialect must be one of {'image', 'video'}.")
    if not default_category_name:
        raise ValueError("default_category_name must be a non-empty string.")

    self.dialect = dialect
    self.default_category_name = default_category_name
    self.write_empty_frames = bool(write_empty_frames)
    self._frame_hw_by_id: dict[int, tuple[int, int]] = {}
    self._track_segmentations: dict[int, dict[int, dict[str, Any]]] = {}
    self._track_scores: dict[int, dict[int, float]] = {}
    self._track_bboxes: dict[int, dict[int, list[float]]] = {}
    self._track_areas: dict[int, dict[int, float]] = {}
    self._track_category_ids: dict[int, int] = {}
    self._category_id_to_name: dict[int, str] = {}

    super().__init__(
        output_json_path=output_json_path,
        atomic_write=atomic_write,
        flush_interval=flush_interval,
        dialect=dialect,
        default_category_name=default_category_name,
        write_empty_frames=write_empty_frames,
        **kwargs,
    )
```

##### forward

```python
forward(
    frame_id,
    mask,
    object_ids,
    detection_scores,
    category_ids=None,
    category_semantics=None,
    context=None,
    **_,
)
```

Store one frame of tracked masks and metadata for later JSON export.

Source code in `cuvis_ai/node/json_file.py`

```python
def forward(
    self,
    frame_id: torch.Tensor,
    mask: torch.Tensor,
    object_ids: torch.Tensor,
    detection_scores: torch.Tensor,
    category_ids: torch.Tensor | None = None,
    category_semantics: torch.Tensor | None = None,
    context: Context | None = None,  # noqa: ARG002
    **_: Any,
) -> dict[str, Any]:
    """Store one frame of tracked masks and metadata for later JSON export."""
    frame_idx = self._parse_frame_id(frame_id)
    mask_2d = self._parse_mask(mask)
    ids_1d = self._parse_vector(object_ids, port_name="object_ids")
    scores_1d = self._parse_vector(detection_scores, port_name="detection_scores")
    self._validate_alignment(ids_1d, scores_1d, "object_ids", "detection_scores")
    category_ids_1d: torch.Tensor | None = None
    if category_ids is not None:
        category_ids_1d = self._parse_vector(category_ids, port_name="category_ids")
        self._validate_alignment(ids_1d, category_ids_1d, "object_ids", "category_ids")
    self._update_category_semantics(category_semantics)

    frame_height = int(mask_2d.shape[0])
    frame_width = int(mask_2d.shape[1])

    # Replacing an existing frame should be idempotent.
    self._drop_frame(frame_idx)

    object_ids_list = ids_1d.to(dtype=torch.int64).cpu().tolist()
    detection_scores_list = scores_1d.to(dtype=torch.float32).cpu().tolist()
    category_ids_list = (
        category_ids_1d.to(dtype=torch.int64).cpu().tolist()
        if category_ids_1d is not None
        else [1] * len(object_ids_list)
    )
    score_by_obj_id: dict[int, float] = {
        int(obj_id): float(score)
        for obj_id, score in zip(object_ids_list, detection_scores_list, strict=False)
        if int(obj_id) > 0
    }
    category_by_obj_id: dict[int, int] = {}
    for obj_id, category_id in zip(object_ids_list, category_ids_list, strict=False):
        oid = int(obj_id)
        cid = int(category_id)
        if oid <= 0:
            continue
        if cid <= 0:
            raise ValueError("category_ids must be positive for tracked objects.")
        existing_category_id = self._track_category_ids.get(oid)
        if existing_category_id is not None and existing_category_id != cid:
            raise ValueError(
                f"Track {oid} received conflicting category IDs: "
                f"{existing_category_id} vs {cid}."
            )
        self._track_category_ids.setdefault(oid, cid)
        category_by_obj_id[oid] = cid
        fallback_name = self.default_category_name if cid == 1 else f"category_{cid}"
        self._category_id_to_name.setdefault(cid, fallback_name)
    present_obj_ids = {
        int(obj_id)
        for obj_id in mask_2d.to(dtype=torch.int64).unique().cpu().tolist()
        if int(obj_id) > 0
    }

    export_obj_ids: list[int] = []
    seen_obj_ids: set[int] = set()
    for obj_id in object_ids_list:
        oid = int(obj_id)
        if oid <= 0 or oid not in present_obj_ids or oid in seen_obj_ids:
            continue
        seen_obj_ids.add(oid)
        export_obj_ids.append(oid)

    if not export_obj_ids and not self.write_empty_frames:
        return {}

    self._frame_hw_by_id[frame_idx] = (frame_height, frame_width)

    for oid in export_obj_ids:
        obj_mask = mask_2d.eq(oid)
        if not bool(torch.any(obj_mask)):
            continue

        mask_np = obj_mask.to(dtype=torch.uint8).detach().cpu().numpy()
        rle_json = coco_rle_encode(mask_np)
        bbox = coco_rle_to_bbox(rle_json)
        area = coco_rle_area(rle_json)

        self._track_segmentations.setdefault(oid, {})[frame_idx] = rle_json
        self._track_scores.setdefault(oid, {})[frame_idx] = float(score_by_obj_id.get(oid, 0.0))
        self._track_bboxes.setdefault(oid, {})[frame_idx] = bbox
        self._track_areas.setdefault(oid, {})[frame_idx] = area
        if oid in category_by_obj_id:
            self._track_category_ids.setdefault(oid, category_by_obj_id[oid])

    self._mark_dirty_and_maybe_flush()
    return {}
```

`DetectionCocoJsonNode`cuvis_ai.node.json_filesinkbboxdetmetaWrite frame-wise detections into COCO detection JSON.

#### DetectionCocoJsonNode

```python
DetectionCocoJsonNode(
    output_json_path,
    category_id_to_name=None,
    write_empty_frames=True,
    atomic_write=True,
    flush_interval=0,
    **kwargs,
)
```

Bases: `_BaseJsonWriterNode`

Write frame-wise detections into COCO detection JSON.

Source code in `cuvis_ai/node/json_file.py`

```python
def __init__(
    self,
    output_json_path: str,
    category_id_to_name: dict[int, str] | None = None,
    write_empty_frames: bool = True,
    atomic_write: bool = True,
    flush_interval: int = 0,
    **kwargs: Any,
) -> None:
    self.category_id_to_name: dict[int, str] = (
        dict(category_id_to_name) if category_id_to_name is not None else {0: "person"}
    )
    self.write_empty_frames = bool(write_empty_frames)
    self._frames_by_id: dict[int, dict[str, Any]] = {}

    super().__init__(
        output_json_path=output_json_path,
        atomic_write=atomic_write,
        flush_interval=flush_interval,
        category_id_to_name=self.category_id_to_name,
        write_empty_frames=write_empty_frames,
        **kwargs,
    )
```

##### forward

```python
forward(
    frame_id,
    bboxes,
    category_ids,
    confidences,
    orig_hw,
    context=None,
    **_,
)
```

Store one frame of detections for COCO JSON serialization.

Source code in `cuvis_ai/node/json_file.py`

```python
def forward(
    self,
    frame_id: torch.Tensor,
    bboxes: torch.Tensor,
    category_ids: torch.Tensor,
    confidences: torch.Tensor,
    orig_hw: torch.Tensor,
    context: Context | None = None,  # noqa: ARG002
    **_: Any,
) -> dict[str, Any]:
    """Store one frame of detections for COCO JSON serialization."""
    frame_idx = _BaseCocoTrackWriter._parse_frame_id(frame_id)
    ids_1d = _BaseCocoTrackWriter._parse_vector(category_ids, port_name="category_ids")
    scores_1d = _BaseCocoTrackWriter._parse_vector(confidences, port_name="confidences")
    _BaseCocoTrackWriter._validate_alignment(ids_1d, scores_1d, "category_ids", "confidences")

    h, w = int(orig_hw[0, 0]), int(orig_hw[0, 1])
    boxes_2d = bboxes[0] if bboxes.ndim == 3 else bboxes

    n = int(ids_1d.numel())
    detections: list[dict[str, Any]] = []
    for i in range(n):
        x1, y1, x2, y2 = boxes_2d[i].cpu().tolist()
        bw = float(x2 - x1)
        bh = float(y2 - y1)
        detections.append(
            {
                "category_id": int(ids_1d[i].item()),
                "bbox": [float(x1), float(y1), bw, bh],
                "area": bw * bh,
                "score": float(scores_1d[i].item()),
            }
        )

    if not detections and not self.write_empty_frames:
        return {}

    self._frames_by_id[frame_idx] = {
        "frame_idx": frame_idx,
        "height": h,
        "width": w,
        "detections": detections,
    }
    self._mark_dirty_and_maybe_flush()
    return {}
```

`MontageColumnSink`cuvis_ai_inspecscrap.node.montage_sinksinkimgpostrgbcuvis_ai_inspecscrap[↗](https://github.com/cubert-hyperspectral/cuvis-ai-inspecscrap "Open plugin repo")

[View plugin repo (v0.2.4)](https://github.com/cubert-hyperspectral/cuvis-ai-inspecscrap/tree/v0.2.4)

`NumpyFeatureWriterNode`cuvis_ai.node.numpy_filesinkembmetaSave per-frame feature tensors to `.npy` files.

#### NumpyFeatureWriterNode

```python
NumpyFeatureWriterNode(
    output_dir, prefix="features", **kwargs
)
```

Bases: `Node`

Save per-frame feature tensors to `.npy` files.

Writes one `.npy` file per frame, named `{prefix}_{frame_id:06d}.npy`. Useful for offline analysis, clustering, or evaluation of ReID embeddings.

Parameters:

| Name         | Type  | Description                           | Default      |
| ------------ | ----- | ------------------------------------- | ------------ |
| `output_dir` | `str` | Directory to write .npy files into.   | *required*   |
| `prefix`     | `str` | Filename prefix (default "features"). | `'features'` |

Source code in `cuvis_ai/node/numpy_file.py`

```python
def __init__(
    self,
    output_dir: str,
    prefix: str = "features",
    **kwargs: Any,
) -> None:
    self.output_dir = str(output_dir)
    self.prefix = str(prefix)
    self._dir_created = False
    super().__init__(output_dir=self.output_dir, prefix=self.prefix, **kwargs)
```

##### forward

```python
forward(features, frame_id, **_)
```

Write features to a `.npy` file.

Parameters:

| Name       | Type     | Description                                                   | Default    |
| ---------- | -------- | ------------------------------------------------------------- | ---------- |
| `features` | `Tensor` | [B, N, D] float32. Batch dimension is squeezed before saving. | *required* |
| `frame_id` | `Tensor` | (1,) int64 scalar frame index.                                | *required* |

Returns:

| Type   | Description             |
| ------ | ----------------------- |
| `dict` | Empty dict (sink node). |

Source code in `cuvis_ai/node/numpy_file.py`

```python
@torch.no_grad()
def forward(self, features: Tensor, frame_id: Tensor, **_: Any) -> dict[str, Tensor]:
    """Write features to a ``.npy`` file.

    Parameters
    ----------
    features : Tensor
        ``[B, N, D]`` float32. Batch dimension is squeezed before saving.
    frame_id : Tensor
        ``(1,)`` int64 scalar frame index.

    Returns
    -------
    dict
        Empty dict (sink node).
    """
    out_dir = Path(self.output_dir)
    if not self._dir_created:
        out_dir.mkdir(parents=True, exist_ok=True)
        self._dir_created = True

    fid = int(frame_id.item())
    # Squeeze batch dim: [B, N, D] → [N, D]
    array = features.squeeze(0).cpu().numpy()
    np.save(out_dir / f"{self.prefix}_{fid:06d}.npy", array)

    return {}
```

`PngWriter`cuvis_ai.node.image_filesinkimgrgbWrite RGB frames to PNG files on disk.

#### PngWriter

```python
PngWriter(output_path, compression_level=6, **kwargs)
```

Bases: `Node`

Write RGB frames to PNG files on disk.

A sink node (no output ports): it consumes an `rgb_image` and writes one PNG per frame via :func:`torchvision.io.write_png`, so the final composite image drops straight out of the pipeline. Input is the canonical `[B, H, W, 3]` float32 in `[0, 1]`; it is scaled to `uint8` and written channels-first.

Naming. A single frame with no `frame_id` is written to `output_path` verbatim. With a `frame_id` (per-frame streaming) the index is appended as `{stem}_{frame_id:06d}{suffix}`; a multi-frame batch is written as `{stem}_{i:06d}{suffix}` per frame.

Parameters:

| Name                | Type  | Description                                                            | Default    |
| ------------------- | ----- | ---------------------------------------------------------------------- | ---------- |
| `output_path`       | `str` | Destination PNG path. Its parent directory is created on construction. | *required* |
| `compression_level` | `int` | zlib compression level 0-9 forwarded to write_png (default 6).         | `6`        |

Source code in `cuvis_ai/node/image_file.py`

```python
def __init__(self, output_path: str, compression_level: int = 6, **kwargs: Any) -> None:
    if not 0 <= compression_level <= 9:
        raise ValueError("compression_level must be in [0, 9]")
    self.output_path = Path(output_path)
    self.compression_level = int(compression_level)
    self.output_path.parent.mkdir(parents=True, exist_ok=True)
    super().__init__(
        output_path=str(self.output_path),
        compression_level=self.compression_level,
        **kwargs,
    )
```

##### forward

```python
forward(rgb_image, frame_id=None, **_)
```

Write each RGB frame to a PNG file.

Parameters:

| Name        | Type             | Description                                       | Default    |
| ----------- | ---------------- | ------------------------------------------------- | ---------- |
| `rgb_image` | `Tensor`         | [B, H, W, 3] float32 in [0, 1].                   | *required* |
| `frame_id`  | `Tensor or None` | (B,) int64 frame index used for per-frame naming. | `None`     |

Returns:

| Type   | Description             |
| ------ | ----------------------- |
| `dict` | Empty dict (sink node). |

Source code in `cuvis_ai/node/image_file.py`

```python
@torch.no_grad()
def forward(
    self,
    rgb_image: Tensor,
    frame_id: Tensor | None = None,
    **_: Any,
) -> dict[str, Tensor]:
    """Write each RGB frame to a PNG file.

    Parameters
    ----------
    rgb_image : Tensor
        ``[B, H, W, 3]`` float32 in ``[0, 1]``.
    frame_id : Tensor or None
        ``(B,)`` int64 frame index used for per-frame naming.

    Returns
    -------
    dict
        Empty dict (sink node).
    """
    n_frames = rgb_image.shape[0]
    fid = int(frame_id.reshape(-1)[0].item()) if frame_id is not None else None
    for i in range(n_frames):
        u8 = (rgb_image[i].clamp(0.0, 1.0) * 255.0).round().to(torch.uint8)  # [H, W, 3]
        chw = u8.permute(2, 0, 1).contiguous().cpu()
        write_png(chw, str(self._path_for(i, n_frames, fid)), self.compression_level)
    return {}
```

`TensorBoardMonitorNode`cuvis_ai.node.monitorsinkevalmetaTensorBoard monitoring node for logging artifacts and metrics.

#### TensorBoardMonitorNode

```python
TensorBoardMonitorNode(
    output_dir="./runs",
    run_name=None,
    comment="",
    flush_secs=120,
    **kwargs,
)
```

Bases: `Node`

TensorBoard monitoring node for logging artifacts and metrics.

This is a SINK node that logs visualizations (artifacts) and metrics to TensorBoard. Accepts optional inputs for artifacts and metrics, allowing predecessors to be filtered by execution_stage without causing errors.

Executes during all stages (ALWAYS).

Parameters:

| Name         | Type  | Description                                              | Default    |
| ------------ | ----- | -------------------------------------------------------- | ---------- |
| `output_dir` | `str` | Directory for TensorBoard logs (default: "./runs")       | `'./runs'` |
| `comment`    | `str` | Comment to append to log directory name (default: "")    | `''`       |
| `flush_secs` | `int` | How often to flush pending events to disk (default: 120) | `120`      |

Examples:

```pycon
>>> heatmap_viz = AnomalyHeatmap(cmap='hot', up_to=10)
>>> tensorboard_node = TensorBoardMonitorNode(output_dir="./runs")
>>> graph.connect(
...     (heatmap_viz.artifacts, tensorboard_node.artifacts),
... )
```

Source code in `cuvis_ai/node/monitor.py`

```python
def __init__(
    self,
    output_dir: str = "./runs",
    run_name: str | None = None,
    comment: str = "",
    flush_secs: int = 120,
    **kwargs,
) -> None:
    self.output_dir = Path(output_dir)
    self.run_name = run_name
    self.comment = comment
    self.flush_secs = flush_secs
    self._writer = None
    self._tensorboard_available = False

    super().__init__(
        execution_stages={ExecutionStage.ALWAYS},
        output_dir=str(output_dir),
        run_name=run_name,
        comment=comment,
        flush_secs=flush_secs,
        **kwargs,
    )

    # Check if tensorboard is available

    self._SummaryWriter = SummaryWriter

    # Determine the log directory with run name
    self.log_dir = self._resolve_log_dir()

    # Initialize TensorBoard writer
    self.log_dir.mkdir(parents=True, exist_ok=True)
    self._writer = self._SummaryWriter(
        log_dir=str(self.log_dir),
        comment=self.comment,
        flush_secs=self.flush_secs,
    )
    logger.info(f"TensorBoard writer initialized: {self.log_dir}")
    logger.info(f"To view visualizations, run: uv run tensorboard --logdir={self.output_dir}")
```

##### forward

```python
forward(artifacts=None, metrics=None, context=None)
```

Log artifacts and metrics to TensorBoard.

Parameters:

| Name        | Type             | Description                                                 | Default |
| ----------- | ---------------- | ----------------------------------------------------------- | ------- |
| `context`   | `Context`        | Execution context with stage, epoch, batch_idx, global_step | `None`  |
| `artifacts` | `list[Artifact]` | List of artifacts to log (default: None)                    | `None`  |
| `metrics`   | `list[Metric]`   | List of metrics to log (default: None)                      | `None`  |

Returns:

| Type   | Description                           |
| ------ | ------------------------------------- |
| `dict` | Empty dict (sink node has no outputs) |

Source code in `cuvis_ai/node/monitor.py`

```python
def forward(
    self,
    artifacts: list[Artifact] | None = None,
    metrics: list[Metric] | None = None,
    context: Context | None = None,
) -> dict:
    """Log artifacts and metrics to TensorBoard.

    Parameters
    ----------
    context : Context
        Execution context with stage, epoch, batch_idx, global_step
    artifacts : list[Artifact], optional
        List of artifacts to log (default: None)
    metrics : list[Metric], optional
        List of metrics to log (default: None)

    Returns
    -------
    dict
        Empty dict (sink node has no outputs)
    """
    if context is None:
        context = Context()

    stage = context.stage.value
    step = context.global_step

    # Flatten artifacts if it's a list of lists (variadic port)
    if artifacts is not None:
        if (
            isinstance(artifacts, list)
            and len(artifacts) > 0
            and isinstance(artifacts[0], list)
        ):
            artifacts = [item for sublist in artifacts for item in sublist]

    # Log artifacts
    if artifacts is not None:
        for artifact in artifacts:
            self._log_artifact(artifact, stage, step)
        logger.debug(f"Logged {len(artifacts)} artifacts to TensorBoard at step {step}")

    # Flatten metrics if variadic input provided
    if (
        metrics is not None
        and isinstance(metrics, list)
        and metrics
        and isinstance(metrics[0], list)
    ):
        metrics = [item for sublist in metrics for item in sublist]

    # Log metrics
    if metrics is not None:
        for metric in metrics:
            self._log_metric(metric, stage, step)
        logger.debug(f"Logged {len(metrics)} metrics to TensorBoard at step {step}")

    return {}
```

##### log

```python
log(name, value, step)
```

Log a scalar value to TensorBoard.

This method provides a simple interface for external trainers to log metrics directly, complementing the port-based logging. Used by GradientTrainer to log train/val losses to the same TensorBoard directory as graph metrics and artifacts.

Parameters:

| Name    | Type    | Description                                                  | Default    |
| ------- | ------- | ------------------------------------------------------------ | ---------- |
| `name`  | `str`   | Name/tag for the scalar (e.g., "train/loss", "val/accuracy") | *required* |
| `value` | `float` | Scalar value to log                                          | *required* |
| `step`  | `int`   | Global step number                                           | *required* |

Examples:

```pycon
>>> tensorboard_node = TensorBoardMonitorNode(output_dir="./runs")
>>> # From external trainer
>>> tensorboard_node.log("train/loss", 0.5, step=100)
```

Source code in `cuvis_ai/node/monitor.py`

```python
def log(self, name: str, value: float, step: int) -> None:
    """Log a scalar value to TensorBoard.

    This method provides a simple interface for external trainers
    to log metrics directly, complementing the port-based logging.
    Used by GradientTrainer to log train/val losses to the same
    TensorBoard directory as graph metrics and artifacts.

    Parameters
    ----------
    name : str
        Name/tag for the scalar (e.g., "train/loss", "val/accuracy")
    value : float
        Scalar value to log
    step : int
        Global step number

    Examples
    --------
    >>> tensorboard_node = TensorBoardMonitorNode(output_dir="./runs")
    >>> # From external trainer
    >>> tensorboard_node.log("train/loss", 0.5, step=100)
    """
    self._writer.add_scalar(name, value, step)
```

`ToImage`cuvis_ai.node.videosinkimgWrite incoming RGB frames to individual image files, one file per frame.

#### ToImage

```python
ToImage(
    output_dir,
    filename_pattern="frame_{frame_id:06d}.png",
    frame_rotation=None,
    overlay_title=None,
    **kwargs,
)
```

Bases: `_FrameRenderMixin`, `Node`

Write incoming RGB frames to individual image files, one file per frame.

Mirrors :class:`ToVideoNode` but emits a standalone image per frame instead of an encoded video stream. Each file is written immediately and is complete on disk the moment `forward` returns, so there is no lazy encoder process and no explicit `close()` / finalization step (and none of the fragmented `movflags` playability caveats a streaming video has).

The output name comes from `filename_pattern` with the frame index substituted (the `{frame_id}` field); the image format is inferred from the pattern's file extension (for example `.png` or `.jpg`). When the batch carries a `frame_id` port, that value drives both the filename and the text overlay; otherwise a running per-node counter is used. A pattern without a `{frame_id}` field writes every frame to the same file (last wins).

Parameters:

| Name               | Type  | Description                                                                                                                                                     | Default                                                                                                                      |
| ------------------ | ----- | --------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------- |
| `output_dir`       | `str` | Directory the image files are written to. Created if missing.                                                                                                   | *required*                                                                                                                   |
| `filename_pattern` | `str` | str.format pattern for each file's name, receiving frame_id as a keyword field. The extension selects the image format. Default is "frame\_{frame_id:06d}.png". | `'frame_{frame_id:06d}.png'`                                                                                                 |
| `frame_rotation`   | \`int | None\`                                                                                                                                                          | Optional frame rotation in degrees; same semantics and accepted values as :class:ToVideoNode. Default is None (no rotation). |
| `overlay_title`    | \`str | None\`                                                                                                                                                          | Optional static title rendered at the top center with its own darkened background block. Default is None.                    |

Source code in `cuvis_ai/node/video.py`

```python
def __init__(
    self,
    output_dir: str,
    filename_pattern: str = "frame_{frame_id:06d}.png",
    frame_rotation: int | None = None,
    overlay_title: str | None = None,
    **kwargs: Any,
) -> None:
    if not isinstance(output_dir, str) or not output_dir.strip():
        raise ValueError("output_dir must be a non-empty string")
    if not isinstance(filename_pattern, str) or not filename_pattern.strip():
        raise ValueError("filename_pattern must be a non-empty string")
    if not Path(filename_pattern).suffix:
        raise ValueError(
            "filename_pattern must include an image extension (e.g. '.png', '.jpg')"
        )
    valid_rotations = {None, 0, 90, -90, 180, -180, 270, -270}
    if frame_rotation not in valid_rotations:
        raise ValueError(
            "frame_rotation must be one of: None, 0, 90, -90, 180, -180, 270, -270"
        )

    self.output_dir = Path(output_dir)
    self.filename_pattern = filename_pattern
    self.frame_rotation = self._normalize_rotation(frame_rotation)
    self.overlay_title = (
        None
        if overlay_title is None or not str(overlay_title).strip()
        else str(overlay_title).strip()
    )
    self._frame_counter = 0

    self.output_dir.mkdir(parents=True, exist_ok=True)

    super().__init__(
        output_dir=output_dir,
        filename_pattern=filename_pattern,
        frame_rotation=frame_rotation,
        overlay_title=self.overlay_title,
        **kwargs,
    )
```

##### forward

```python
forward(rgb_image, frame_id=None, context=None, **_)
```

Write each incoming RGB frame to its own image file.

Returns:

| Type   | Description             |
| ------ | ----------------------- |
| `dict` | Empty dict (sink node). |

Source code in `cuvis_ai/node/video.py`

```python
def forward(
    self,
    rgb_image: torch.Tensor,
    frame_id: torch.Tensor | None = None,
    context: Context | None = None,  # noqa: ARG002
    **_: Any,
) -> dict[str, Any]:
    """Write each incoming RGB frame to its own image file.

    Returns
    -------
    dict
        Empty dict (sink node).
    """
    rgb_u8 = self._to_uint8_batch(rgb_image)

    for b, frame in enumerate(rgb_u8):
        self._draw_title_overlay(frame)
        if frame_id is not None and b < len(frame_id):
            fid = int(frame_id[b].item())
            draw_text(frame, 8, 8, f"frame {fid}", (255, 255, 255), scale=2, bg=True)
        else:
            fid = self._frame_counter
        frame = self._rotate_frame(frame)
        self._write_frame(frame, fid)
        self._frame_counter += 1

    return {}
```

`ToVideoNode`cuvis_ai.node.videosinkvidWrite incoming RGB frames directly to a video file via ffmpeg.

#### ToVideoNode

```python
ToVideoNode(
    output_video_path,
    frame_rate=10.0,
    frame_rotation=None,
    video_codec="libx264",
    bitrate="12M",
    overlay_title=None,
    write_mode="full",
    **kwargs,
)
```

Bases: `_FrameRenderMixin`, `Node`

Write incoming RGB frames directly to a video file via ffmpeg.

This node lazily starts a single `ffmpeg` subprocess on the first frame and pipes raw `rgb24` bytes to its stdin; ffmpeg handles encoding, bitrate control, and muxing. `close()` sends EOF and waits for ffmpeg to flush the trailer — callers must invoke it explicitly (e.g. in a `finally` block of the enclosing pipeline driver) to surface encoder errors.

The ffmpeg binary is resolved via `imageio_ffmpeg` by default (bundled with the wheel — no system install needed). Override with the `CUVIS_AI_FFMPEG_BIN` environment variable to point at a custom build (e.g. one with `h264_nvenc` / `vaapi` / `amf` hardware encoders).

Parameters:

| Name                | Type    | Description                                                                                                                                                                                                                                                                                                                                                                                                                                                             | Default                                                                                                                                                                                                                        |
| ------------------- | ------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `output_video_path` | `str`   | Output path for the generated video file (for example .mp4).                                                                                                                                                                                                                                                                                                                                                                                                            | *required*                                                                                                                                                                                                                     |
| `frame_rate`        | `float` | Video frame rate in frames per second. Must be positive. Default is 10.0.                                                                                                                                                                                                                                                                                                                                                                                               | `10.0`                                                                                                                                                                                                                         |
| `frame_rotation`    | \`int   | None\`                                                                                                                                                                                                                                                                                                                                                                                                                                                                  | Optional frame rotation in degrees. Supported values are -90, 90, 180 (and aliases 270, -270, -180). Positive values rotate anticlockwise (counterclockwise), negative values rotate clockwise. Default is None (no rotation). |
| `video_codec`       | `str`   | ffmpeg -c:v codec name (e.g. "libx264", "libx265"). Default is "libx264".                                                                                                                                                                                                                                                                                                                                                                                               | `'libx264'`                                                                                                                                                                                                                    |
| `bitrate`           | `str`   | ffmpeg -b:v target bitrate (e.g. "12M", "8000k"). Default is "12M".                                                                                                                                                                                                                                                                                                                                                                                                     | `'12M'`                                                                                                                                                                                                                        |
| `overlay_title`     | \`str   | None\`                                                                                                                                                                                                                                                                                                                                                                                                                                                                  | Optional static title rendered at the top center with its own slim darkened background block. Default is None.                                                                                                                 |
| `write_mode`        | `str`   | How the mp4 is finalized. "full" (default) writes a standard file with -movflags +faststart (the moov atom is moved to the front on a clean close()); best for a finished run whose driver calls close(), but unreadable until then. "partial" writes a fragmented mp4 (-movflags +frag_keyframe+empty_moov+default_base_moof) that stays playable during recording and after an unclean stop; use it for a streaming / gRPC session with no guaranteed driver close(). | `'full'`                                                                                                                                                                                                                       |

Source code in `cuvis_ai/node/video.py`

```python
def __init__(
    self,
    output_video_path: str,
    frame_rate: float = 10.0,
    frame_rotation: int | None = None,
    video_codec: str = "libx264",
    bitrate: str = "12M",
    overlay_title: str | None = None,
    write_mode: str = "full",
    **kwargs: Any,
) -> None:
    if frame_rate <= 0:
        raise ValueError("frame_rate must be > 0")
    if not isinstance(video_codec, str) or not video_codec.strip():
        raise ValueError("video_codec must be a non-empty string")
    if not isinstance(bitrate, str) or not bitrate.strip():
        raise ValueError("bitrate must be a non-empty string (e.g. '12M', '8000k')")
    if write_mode not in self._WRITE_MODE_MOVFLAGS:
        raise ValueError(
            f"write_mode must be one of {sorted(self._WRITE_MODE_MOVFLAGS)}, got {write_mode!r}"
        )
    valid_rotations = {None, 0, 90, -90, 180, -180, 270, -270}
    if frame_rotation not in valid_rotations:
        raise ValueError(
            "frame_rotation must be one of: None, 0, 90, -90, 180, -180, 270, -270"
        )

    self.output_video_path = Path(output_video_path)
    self.frame_rate = float(frame_rate)
    self.frame_rotation = self._normalize_rotation(frame_rotation)
    self.video_codec = video_codec.strip()
    self.bitrate = bitrate.strip()
    self.write_mode = write_mode
    self.movflags = self._WRITE_MODE_MOVFLAGS[write_mode]
    self.overlay_title = (
        None
        if overlay_title is None or not str(overlay_title).strip()
        else str(overlay_title).strip()
    )
    if self.overlay_title:
        warnings.warn(
            "ToVideoNode renders overlay_title with cv2; it will move to the shared torch "
            "text renderer (cuvis_ai.utils.torch_draw.draw_text) in v1.0.",
            DeprecationWarning,
            stacklevel=2,
        )
    self._proc: subprocess.Popen[bytes] | None = None
    self._frame_size: tuple[int, int] | None = None
    # Records a teardown-time finalize failure (see cleanup()); None until one happens.
    self._finalize_error: str | None = None

    self.output_video_path.parent.mkdir(parents=True, exist_ok=True)

    super().__init__(
        output_video_path=output_video_path,
        frame_rate=frame_rate,
        frame_rotation=frame_rotation,
        video_codec=self.video_codec,
        bitrate=self.bitrate,
        overlay_title=self.overlay_title,
        write_mode=write_mode,
        **kwargs,
    )
```

##### forward

```python
forward(rgb_image, frame_id=None, context=None, **_)
```

Append incoming RGB frames to the configured video file.

Returns:

| Type   | Description             |
| ------ | ----------------------- |
| `dict` | Empty dict (sink node). |

Source code in `cuvis_ai/node/video.py`

```python
def forward(
    self,
    rgb_image: torch.Tensor,
    frame_id: torch.Tensor | None = None,
    context: Context | None = None,  # noqa: ARG002
    **_: Any,
) -> dict[str, Any]:
    """Append incoming RGB frames to the configured video file.

    Returns
    -------
    dict
        Empty dict (sink node).
    """
    rgb_u8 = self._to_uint8_batch(rgb_image)

    for b, frame in enumerate(rgb_u8):
        self._draw_title_overlay(frame)
        if frame_id is not None and b < len(frame_id):
            fid = int(frame_id[b].item())
            draw_text(frame, 8, 8, f"frame {fid}", (255, 255, 255), scale=2, bg=True)
        frame = self._rotate_frame(frame)
        height, width = int(frame.shape[0]), int(frame.shape[1])
        if self._proc is None:
            self._init_ffmpeg(height=height, width=width)
        elif self._frame_size != (height, width):
            raise ValueError(
                f"All frames must share one size. Expected {self._frame_size}, got {(height, width)}"
            )

        assert self._proc is not None and self._proc.stdin is not None
        frame_bytes = np.ascontiguousarray(frame.numpy()).tobytes()
        try:
            self._proc.stdin.write(frame_bytes)
        except (BrokenPipeError, OSError) as exc:
            stderr_text = self._collect_stderr_after_exit()
            returncode = self._proc.poll() if self._proc is not None else None
            raise RuntimeError(
                f"ffmpeg exited during frame write (returncode={returncode}): {stderr_text}"
            ) from exc

    return {}
```

##### close

```python
close()
```

Flush EOF to ffmpeg, wait for mux, and surface any encoder errors.

Idempotent — repeated calls are no-ops. Must be called explicitly by the pipeline driver; do not rely on `__del__` for normal teardown.

Source code in `cuvis_ai/node/video.py`

```python
def close(self) -> None:
    """Flush EOF to ffmpeg, wait for mux, and surface any encoder errors.

    Idempotent — repeated calls are no-ops. Must be called explicitly by the
    pipeline driver; do not rely on ``__del__`` for normal teardown.
    """
    proc = self._proc
    if proc is None:
        return
    self._proc = None

    if proc.stdin is not None:
        try:
            proc.stdin.close()
        except (BrokenPipeError, OSError) as exc:
            logger.debug("ffmpeg stdin close raised during teardown: {}", exc)

    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        proc.kill()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            raise RuntimeError(
                f"ffmpeg did not terminate for {self.output_video_path} (kill after 30s wait)"
            ) from None
        stderr_text = b""
        if proc.stderr is not None:
            try:
                stderr_text = proc.stderr.read() or b""
            except (ValueError, OSError):
                stderr_text = b""
        raise RuntimeError(
            "ffmpeg timed out during mux of "
            f"{self.output_video_path}: {stderr_text.decode('utf-8', errors='replace')}"
        ) from None

    stderr_text = b""
    if proc.stderr is not None:
        try:
            stderr_text = proc.stderr.read() or b""
        except (ValueError, OSError):
            stderr_text = b""

    if proc.returncode != 0:
        raise RuntimeError(
            f"ffmpeg exited with non-zero return code {proc.returncode} "
            f"for {self.output_video_path}: "
            f"{stderr_text.decode('utf-8', errors='replace')}"
        )
```

##### cleanup

```python
cleanup()
```

Finalize the video file when the hosting pipeline is torn down.

A gRPC/session pipeline has no explicit driver `close()` call, so the session-teardown `cleanup()` (invoked by `CuvisPipeline.cleanup` on session close, pipeline replacement, or run stop) is where the ffmpeg trailer gets flushed. `close()` is idempotent, so calling it here in addition to an explicit driver `close()` is safe.

`CuvisPipeline.cleanup` wraps each node's `cleanup()` in a bare try/except that only `logger.warning`s, so a finalize failure here (ffmpeg unable to write the `moov` trailer, leaving an unplayable file) would otherwise be indistinguishable from a benign teardown warning while the run still reports success. Surface it explicitly at ERROR level and record it on `_finalize_error` so callers/tests can detect the truncated output, then re-raise so nothing is silently hidden.

Source code in `cuvis_ai/node/video.py`

```python
def cleanup(self) -> None:
    """Finalize the video file when the hosting pipeline is torn down.

    A gRPC/session pipeline has no explicit driver ``close()`` call, so the
    session-teardown ``cleanup()`` (invoked by ``CuvisPipeline.cleanup`` on
    session close, pipeline replacement, or run stop) is where the ffmpeg
    trailer gets flushed. ``close()`` is idempotent, so calling it here in
    addition to an explicit driver ``close()`` is safe.

    ``CuvisPipeline.cleanup`` wraps each node's ``cleanup()`` in a bare
    try/except that only ``logger.warning``s, so a finalize failure here
    (ffmpeg unable to write the ``moov`` trailer, leaving an unplayable file)
    would otherwise be indistinguishable from a benign teardown warning while
    the run still reports success. Surface it explicitly at ERROR level and
    record it on ``_finalize_error`` so callers/tests can detect the truncated
    output, then re-raise so nothing is silently hidden.
    """
    try:
        self.close()
    except RuntimeError as exc:
        self._finalize_error = str(exc)
        logger.error(
            "ToVideoNode failed to finalize {} at pipeline teardown: {}",
            self.output_video_path,
            exc,
        )
        raise
    finally:
        super().cleanup()
```

`_BaseCocoTrackWriter`cuvis_ai.node.json_filesinkmaskmetatrackShared tensor parsing helpers for tracking writers.

#### \_BaseCocoTrackWriter

```python
_BaseCocoTrackWriter(
    output_json_path,
    atomic_write=True,
    flush_interval=0,
    **kwargs,
)
```

Bases: `_BaseJsonWriterNode`

Shared tensor parsing helpers for tracking writers.

Source code in `cuvis_ai/node/json_file.py`

```python
def __init__(
    self,
    output_json_path: str,
    atomic_write: bool = True,
    flush_interval: int = 0,
    **kwargs: Any,
) -> None:
    if not output_json_path:
        raise ValueError("output_json_path must be a non-empty path.")
    if flush_interval < 0:
        raise ValueError("flush_interval must be >= 0.")

    self.output_json_path = Path(output_json_path)
    self.atomic_write = bool(atomic_write)
    self.flush_interval = int(flush_interval)
    self._dirty = False
    self._frames_since_flush = 0

    self.output_json_path.parent.mkdir(parents=True, exist_ok=True)

    super().__init__(
        output_json_path=output_json_path,
        atomic_write=atomic_write,
        flush_interval=flush_interval,
        **kwargs,
    )
```

`_BaseJsonWriterNode`cuvis_ai.node.json_filesinkmetaShared JSON write lifecycle for sink nodes.

#### \_BaseJsonWriterNode

```python
_BaseJsonWriterNode(
    output_json_path,
    atomic_write=True,
    flush_interval=0,
    **kwargs,
)
```

Bases: `Node`

Shared JSON write lifecycle for sink nodes.

Source code in `cuvis_ai/node/json_file.py`

```python
def __init__(
    self,
    output_json_path: str,
    atomic_write: bool = True,
    flush_interval: int = 0,
    **kwargs: Any,
) -> None:
    if not output_json_path:
        raise ValueError("output_json_path must be a non-empty path.")
    if flush_interval < 0:
        raise ValueError("flush_interval must be >= 0.")

    self.output_json_path = Path(output_json_path)
    self.atomic_write = bool(atomic_write)
    self.flush_interval = int(flush_interval)
    self._dirty = False
    self._frames_since_flush = 0

    self.output_json_path.parent.mkdir(parents=True, exist_ok=True)

    super().__init__(
        output_json_path=output_json_path,
        atomic_write=atomic_write,
        flush_interval=flush_interval,
        **kwargs,
    )
```

##### close

```python
close()
```

Flush pending changes before shutdown.

Source code in `cuvis_ai/node/json_file.py`

```python
def close(self) -> None:
    """Flush pending changes before shutdown."""
    if self._dirty:
        self._flush_json()
```

`AnomalyDataNode`cuvis_ai.node.datasourcehsimetaCU3S data node with binary anomaly label mapping.

#### AnomalyDataNode

```python
AnomalyDataNode(
    normal_class_ids, anomaly_class_ids=None, **kwargs
)
```

Bases: `CU3SDataNode`

CU3S data node with binary anomaly label mapping.

Inherits shared CU3S normalization (cube + wavelengths) and additionally maps multi-class masks to binary anomaly masks: classes in `normal_class_ids` become 0, everything else (or `anomaly_class_ids` when given) becomes 1.

Source code in `cuvis_ai/node/data.py`

```python
def __init__(
    self, normal_class_ids: list[int], anomaly_class_ids: list[int] | None = None, **kwargs
) -> None:
    # Keep node params on the base Node for config/serialization compatibility.
    super().__init__(
        normal_class_ids=normal_class_ids, anomaly_class_ids=anomaly_class_ids, **kwargs
    )
    self._binary_mapper = BinaryAnomalyLabelMapper(
        normal_class_ids=normal_class_ids,
        anomaly_class_ids=anomaly_class_ids,
    )
```

##### forward

```python
forward(
    cube, mask=None, class_mask=None, wavelengths=None, **_
)
```

Apply CU3S normalization and optional binary anomaly mask mapping.

Source code in `cuvis_ai/node/data.py`

```python
def forward(
    self,
    cube: torch.Tensor,
    mask: torch.Tensor | None = None,
    class_mask: torch.Tensor | None = None,
    wavelengths: torch.Tensor | None = None,
    **_: Any,
) -> dict[str, torch.Tensor | np.ndarray]:
    """Apply CU3S normalization and optional binary anomaly mask mapping."""
    result = super().forward(cube=cube, mask=None, wavelengths=wavelengths, **_)

    if mask is not None:
        # Mapper expects channel-last mask: BHW -> BHWC.
        mask_4d = mask.unsqueeze(-1)
        mapped = self._binary_mapper.forward(cube=cube, mask=mask_4d, **_)
        result["mask"] = mapped["mask"]

    if class_mask is not None:
        # Expose the multi-class category-id labels as a port so downstream per-class metrics
        # can read them directly. This is a separate input from the binary `mask`: the data
        # module carries the multi-class ids in their own `class_mask` batch key.
        result["class_mask"] = class_mask.unsqueeze(-1).to(torch.int32)

    return result
```

`BBoxPrompt`cuvis_ai.node.promptssourcebboxinfermetaEmit scheduled runtime bbox prompts plus overlay-friendly debug tensors.

#### BBoxPrompt

```python
BBoxPrompt(json_path, prompt_specs=None, **kwargs)
```

Bases: `Node`

Emit scheduled runtime bbox prompts plus overlay-friendly debug tensors.

Source code in `cuvis_ai/node/prompts.py`

```python
def __init__(
    self,
    json_path: str,
    prompt_specs: Sequence[str] | None = None,
    **kwargs: Any,
) -> None:
    self.json_path = Path(json_path)
    self._prompt_specs = [str(spec) for spec in (prompt_specs or [])]
    self._prompts_by_frame, self._frame_hw_by_id, self._default_hw = load_bbox_prompt_schedule(
        self.json_path,
        self._prompt_specs,
    )
    super().__init__(json_path=str(self.json_path), prompt_specs=self._prompt_specs, **kwargs)
```

##### forward

```python
forward(frame_id, context=None, **_)
```

Emit the scheduled bbox prompt list for `frame_id` or an empty list.

Source code in `cuvis_ai/node/prompts.py`

```python
def forward(
    self,
    frame_id: torch.Tensor,
    context: Context | None = None,  # noqa: ARG002
    **_: Any,
) -> dict[str, torch.Tensor | list[dict[str, float | int]]]:
    """Emit the scheduled bbox prompt list for ``frame_id`` or an empty list."""
    if frame_id is None or frame_id.numel() == 0:
        raise ValueError("BBoxPrompt requires a non-empty frame_id input.")

    current_frame_id = int(frame_id.reshape(-1)[0].item())
    frame_hw = _resolve_frame_hw(
        current_frame_id,
        self._frame_hw_by_id,
        self._default_hw,
        self.json_path,
    )
    prompts = self._prompts_by_frame.get(current_frame_id, [])
    prompts_out = [dict(prompt) for prompt in prompts]

    if prompts_out:
        boxes_xyxy = torch.tensor(
            [
                [prompt["x_min"], prompt["y_min"], prompt["x_max"], prompt["y_max"]]
                for prompt in prompts_out
            ],
            dtype=torch.float32,
        ).unsqueeze(0)
        object_ids = torch.tensor(
            [int(prompt["object_id"]) for prompt in prompts_out],
            dtype=torch.int64,
        ).unsqueeze(0)
    else:
        boxes_xyxy = torch.zeros((1, 0, 4), dtype=torch.float32)
        object_ids = torch.zeros((1, 0), dtype=torch.int64)

    if frame_hw[0] <= 0 or frame_hw[1] <= 0:
        raise ValueError(
            f"Resolved invalid frame size for frame {current_frame_id}: {frame_hw}."
        )

    return {
        "bboxes": prompts_out,
        "prompt_boxes_xyxy": boxes_xyxy,
        "prompt_object_ids": object_ids,
    }
```

`CU3SDataNode`cuvis_ai.node.datasourcehsimetaGeneral-purpose data node for CU3S hyperspectral sequences.

#### CU3SDataNode

Bases: `Node`

General-purpose data node for CU3S hyperspectral sequences.

This node normalizes common CU3S batch inputs for pipelines:

- converts `cube` from uint16 to float32
- passes optional `mask` through unchanged
- extracts 1D `wavelengths` from batched input

##### forward

```python
forward(
    cube, mask=None, wavelengths=None, mesu_index=None, **_
)
```

Normalize CU3S batch data for pipeline consumption.

Source code in `cuvis_ai/node/data.py`

```python
def forward(
    self,
    cube: torch.Tensor,
    mask: torch.Tensor | None = None,
    wavelengths: torch.Tensor | None = None,
    mesu_index: torch.Tensor | None = None,
    **_: Any,
) -> dict[str, torch.Tensor | np.ndarray]:
    """Normalize CU3S batch data for pipeline consumption."""
    result: dict[str, torch.Tensor | np.ndarray] = {"cube": cube.to(torch.float32)}

    # Keep the same behavior as existing data nodes: use first batch entry.
    if wavelengths is not None:
        result["wavelengths"] = wavelengths[0].cpu().numpy()

    if mask is not None:
        result["mask"] = mask

    if mesu_index is not None:
        result["mesu_index"] = mesu_index

    return result
```

`Cu3sDataModule`cuvis_ai_dataloader.data.datamodule_cu3ssourcedata modulecuvis_ai_dataloader[↗](https://github.com/cubert-hyperspectral/cuvis-ai-dataloader "Open plugin repo")Single cu3s recording with optional COCO annotations.

Single cu3s recording with optional COCO annotations.

Data module `cu3s` — pip extras: `cu3s`, `coco`.

[View plugin repo (v0.6.2)](https://github.com/cubert-hyperspectral/cuvis-ai-dataloader/tree/v0.6.2)

`DetectionJsonReader`cuvis_ai.node.json_filesourcebboxdetmetaRead COCO detection JSON and emit tensors per frame.

#### DetectionJsonReader

```python
DetectionJsonReader(json_path, **kwargs)
```

Bases: `Node`

Read COCO detection JSON and emit tensors per frame.

Outputs per call:

- frame_id: int64 [1]
- bboxes: float32 [1, N, 4] (xyxy)
- category_ids: int64 [1, N]
- confidences: float32 [1, N]
- orig_hw: int64 [1, 2]

Source code in `cuvis_ai/node/json_file.py`

```python
def __init__(self, json_path: str, **kwargs: Any) -> None:
    self.json_path = Path(json_path)
    if not self.json_path.exists():
        raise FileNotFoundError(f"JSON not found: {self.json_path}")

    with self.json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    self._images = {int(img["id"]): img for img in data.get("images", [])}
    self._annotations_by_img: dict[int, list[dict[str, Any]]] = {}
    for ann in data.get("annotations", []):
        self._annotations_by_img.setdefault(int(ann["image_id"]), []).append(ann)

    self._frame_ids = sorted(self._images.keys())
    self._cursor = 0

    super().__init__(json_path=str(self.json_path), **kwargs)
```

##### reset

```python
reset()
```

Rewind to the first frame.

Source code in `cuvis_ai/node/json_file.py`

```python
def reset(self) -> None:  # noqa: D401
    """Rewind to the first frame."""
    self._cursor = 0
```

##### forward

```python
forward(context=None, **_)
```

Emit detections for the next frame in the detection JSON stream.

Source code in `cuvis_ai/node/json_file.py`

```python
def forward(self, context: Context | None = None, **_: Any) -> dict[str, Any]:  # noqa: ARG002
    """Emit detections for the next frame in the detection JSON stream."""
    if self._cursor >= len(self._frame_ids):
        raise StopIteration("No more frames in detection JSON")

    frame_id = self._frame_ids[self._cursor]
    self._cursor += 1

    img = self._images[frame_id]
    anns = self._annotations_by_img.get(frame_id, [])

    bboxes = []
    cats = []
    scores = []
    for ann in anns:
        x, y, w, h = ann["bbox"]
        bboxes.append([x, y, x + w, y + h])
        category_id = ann.get("category_id", 0)
        score = ann.get("score", 0.0)
        cats.append(int(category_id) if category_id is not None else 0)
        scores.append(float(score) if score is not None else 0.0)

    bboxes_t = (
        torch.tensor([bboxes], dtype=torch.float32)
        if bboxes
        else torch.empty((1, 0, 4), dtype=torch.float32)
    )
    cats_t = (
        torch.tensor([cats], dtype=torch.int64)
        if cats
        else torch.empty((1, 0), dtype=torch.int64)
    )
    scores_t = (
        torch.tensor([scores], dtype=torch.float32)
        if scores
        else torch.empty((1, 0), dtype=torch.float32)
    )

    h = int(img.get("height", 0))
    w = int(img.get("width", 0))
    orig_hw = torch.tensor([[h, w]], dtype=torch.int64)

    return {
        "frame_id": torch.tensor([frame_id], dtype=torch.int64),
        "bboxes": bboxes_t,
        "category_ids": cats_t,
        "confidences": scores_t,
        "orig_hw": orig_hw,
    }
```

`LentilsAnomalyDataNode`cuvis_ai.node.datasourcehsimetaDeprecated alias of :class:`AnomalyDataNode` (nothing lentils-specific inside).

#### LentilsAnomalyDataNode

```python
LentilsAnomalyDataNode(
    normal_class_ids, anomaly_class_ids=None, **kwargs
)
```

Bases: `AnomalyDataNode`

Deprecated alias of :class:`AnomalyDataNode` (nothing lentils-specific inside).

Kept so saved pipelines referencing the old class name keep loading.

Source code in `cuvis_ai/node/data.py`

```python
def __init__(
    self, normal_class_ids: list[int], anomaly_class_ids: list[int] | None = None, **kwargs
) -> None:
    # Keep node params on the base Node for config/serialization compatibility.
    super().__init__(
        normal_class_ids=normal_class_ids, anomaly_class_ids=anomaly_class_ids, **kwargs
    )
    self._binary_mapper = BinaryAnomalyLabelMapper(
        normal_class_ids=normal_class_ids,
        anomaly_class_ids=anomaly_class_ids,
    )
```

`MaskPrompt`cuvis_ai.node.promptssourceinfermaskmetaEmit a scheduled label-map prompt mask for the requested frame.

#### MaskPrompt

```python
MaskPrompt(json_path, prompt_specs=None, **kwargs)
```

Bases: `Node`

Emit a scheduled label-map prompt mask for the requested frame.

Source code in `cuvis_ai/node/prompts.py`

```python
def __init__(
    self,
    json_path: str,
    prompt_specs: Sequence[str] | None = None,
    **kwargs: Any,
) -> None:
    self.json_path = Path(json_path)
    self._prompt_specs = [str(spec) for spec in (prompt_specs or [])]
    self._masks_by_frame, self._frame_hw_by_id, self._default_hw = load_mask_prompt_schedule(
        self.json_path,
        self._prompt_specs,
    )
    super().__init__(json_path=str(self.json_path), prompt_specs=self._prompt_specs, **kwargs)
```

##### forward

```python
forward(frame_id, context=None, **_)
```

Emit the scheduled prompt label map for `frame_id` or an empty mask.

Source code in `cuvis_ai/node/prompts.py`

```python
def forward(
    self,
    frame_id: torch.Tensor,
    context: Context | None = None,  # noqa: ARG002
    **_: Any,
) -> dict[str, torch.Tensor]:
    """Emit the scheduled prompt label map for ``frame_id`` or an empty mask."""
    if frame_id is None or frame_id.numel() == 0:
        raise ValueError("MaskPrompt requires a non-empty frame_id input.")

    current_frame_id = int(frame_id.reshape(-1)[0].item())
    frame_hw = _resolve_frame_hw(
        current_frame_id,
        self._frame_hw_by_id,
        self._default_hw,
        self.json_path,
        fallback_on_placeholder=True,
    )
    label_map = self._masks_by_frame.get(current_frame_id)

    if label_map is None:
        mask_t = torch.zeros((1, frame_hw[0], frame_hw[1]), dtype=torch.int32)
    else:
        mask_t = (
            torch.from_numpy(np.array(label_map, copy=True)).unsqueeze(0).to(dtype=torch.int32)
        )
    return {"mask": mask_t}
```

`MultiCu3sDataModule`cuvis_ai_dataloader.data.datamodule_cu3s_multisourcedata modulecuvis_ai_dataloader[↗](https://github.com/cubert-hyperspectral/cuvis-ai-dataloader "Open plugin repo")Multiple cu3s recordings over a universe.csv (source, index; optional split).

Multiple cu3s recordings over a universe.csv (source, index; optional split).

Data module `cu3s_multi` — pip extras: `cu3s`, `coco`.

[View plugin repo (v0.6.2)](https://github.com/cubert-hyperspectral/cuvis-ai-dataloader/tree/v0.6.2)

`MultiNpzDataModule`cuvis_ai_dataloader.data.datamodule_npz_multisourcedata modulecuvis_ai_dataloader[↗](https://github.com/cubert-hyperspectral/cuvis-ai-dataloader "Open plugin repo")Per-frame .npz over a universe.csv (source, index, materialized_path), split by a splits.json.

Per-frame .npz over a universe.csv (source, index, materialized_path), split by a splits.json.

Data module `npz_multi` — pip extras: none.

[View plugin repo (v0.6.2)](https://github.com/cubert-hyperspectral/cuvis-ai-dataloader/tree/v0.6.2)

`NpyReader`cuvis_ai.node.numpy_filesourcemetaLoad a `.npy` file once and return the same tensor every forward call.

#### NpyReader

```python
NpyReader(file_path, **kwargs)
```

Bases: `Node`

Load a `.npy` file once and return the same tensor every forward call.

Source code in `cuvis_ai/node/numpy_file.py`

```python
def __init__(self, file_path: str, **kwargs: Any) -> None:
    self.file_path = str(Path(file_path))
    path = Path(self.file_path)
    if not path.exists():
        raise FileNotFoundError(f"NpyReader input file not found: {path}")

    raw = np.load(path, allow_pickle=False)
    padded = _pad_to_bhwc4(np.asarray(raw, dtype=np.float32))
    tensor = torch.from_numpy(np.ascontiguousarray(padded))

    super().__init__(file_path=self.file_path, **kwargs)
    self.register_buffer("_data_buf", tensor, persistent=True)
```

##### forward

```python
forward(frame_id=None, **_)
```

Return cached tensor.

Source code in `cuvis_ai/node/numpy_file.py`

```python
@torch.no_grad()
def forward(
    self,
    frame_id: torch.Tensor | None = None,  # noqa: ARG002
    **_: Any,
) -> dict[str, torch.Tensor]:
    """Return cached tensor."""
    return {"data": self._data_buf}
```

`PointPrompt`cuvis_ai.node.promptssourceinferkpmetaEmit a scheduled list of point prompts for the requested frame.

#### PointPrompt

```python
PointPrompt(points, prompt_frame_id, **kwargs)
```

Bases: `Node`

Emit a scheduled list of point prompts for the requested frame.

Unlike :class:`MaskPrompt` / :class:`BBoxPrompt` (which read prompts from a COCO detection JSON), point prompts are supplied directly as `(x, y, type)` because object selection is interactive, not stored in a detection file. Each point is a pixel coordinate with a `type` of `positive` (object), `negative` (background), or `neutral` (ignored). All points fire on `prompt_frame_id` and address a single object; every other frame emits an empty list.

Configure the point prompts and the frame they fire on.

Args: points: Iterable of `(x, y[, type])` tuples or `{x, y, type, element_id}` dicts in pixel coordinates. `type` defaults to `positive`. prompt_frame_id: Source frame index on which to emit the points.

Source code in `cuvis_ai/node/prompts.py`

```python
def __init__(
    self,
    points: Sequence[Any],
    prompt_frame_id: int,
    **kwargs: Any,
) -> None:
    """Configure the point prompts and the frame they fire on.

    Args:
        points: Iterable of ``(x, y[, type])`` tuples or ``{x, y, type, element_id}``
            dicts in pixel coordinates. ``type`` defaults to ``positive``.
        prompt_frame_id: Source frame index on which to emit the points.
    """
    self._prompt_frame_id = int(prompt_frame_id)
    self._points = self._normalize_points(points)
    super().__init__(
        points=self._points,
        prompt_frame_id=self._prompt_frame_id,
        **kwargs,
    )
```

##### forward

```python
forward(frame_id, context=None, **_)
```

Emit the configured point prompts on `prompt_frame_id`, else an empty list.

Source code in `cuvis_ai/node/prompts.py`

```python
def forward(
    self,
    frame_id: torch.Tensor,
    context: Context | None = None,  # noqa: ARG002
    **_: Any,
) -> dict[str, list[dict[str, Any]]]:
    """Emit the configured point prompts on ``prompt_frame_id``, else an empty list."""
    if frame_id is None or frame_id.numel() == 0:
        raise ValueError("PointPrompt requires a non-empty frame_id input.")
    current_frame_id = int(frame_id.reshape(-1)[0].item())
    if current_frame_id != self._prompt_frame_id:
        return {"points": []}
    return {"points": [dict(point) for point in self._points]}
```

`TextPrompt`cuvis_ai.node.promptssourceinfermetatextEmit a runtime text prompt for the requested frame.

#### TextPrompt

```python
TextPrompt(
    prompt_specs=None, prompt_mode="scheduled", **kwargs
)
```

Bases: `Node`

Emit a runtime text prompt for the requested frame.

Source code in `cuvis_ai/node/prompts.py`

```python
def __init__(
    self,
    prompt_specs: Sequence[str] | None = None,
    prompt_mode: str = "scheduled",
    **kwargs: Any,
) -> None:
    self._prompt_specs = [str(spec) for spec in (prompt_specs or [])]
    self._prompts_by_frame = load_text_prompt_schedule(self._prompt_specs)
    self._prompt_mode = normalize_text_prompt_mode(prompt_mode)
    super().__init__(
        prompt_specs=self._prompt_specs,
        prompt_mode=self._prompt_mode,
        **kwargs,
    )
```

##### forward

```python
forward(frame_id, context=None, **_)
```

Emit the resolved prompt text for `frame_id` or an empty string.

Source code in `cuvis_ai/node/prompts.py`

```python
def forward(
    self,
    frame_id: torch.Tensor,
    context: Context | None = None,  # noqa: ARG002
    **_: Any,
) -> dict[str, str]:
    """Emit the resolved prompt text for ``frame_id`` or an empty string."""
    if frame_id is None or frame_id.numel() == 0:
        raise ValueError("TextPrompt requires a non-empty frame_id input.")

    current_frame_id = int(frame_id.reshape(-1)[0].item())
    return {
        "text_prompt": resolve_text_prompt_for_frame(
            self._prompts_by_frame,
            current_frame_id,
            prompt_mode=self._prompt_mode,
        )
    }
```

`TiffDataNode`cuvis_ai_inspecscrap.node.datasourcehsicuvis_ai_inspecscrap[↗](https://github.com/cubert-hyperspectral/cuvis-ai-inspecscrap "Open plugin repo")

[View plugin repo (v0.2.4)](https://github.com/cubert-hyperspectral/cuvis-ai-inspecscrap/tree/v0.2.4)

`TiffPairedDataModule`cuvis_ai_dataloader.data.datamodule_tiff_pairedsourcedata modulecuvis_ai_dataloader[↗](https://github.com/cubert-hyperspectral/cuvis-ai-dataloader "Open plugin repo")TIFF cubes with paired PNG label images.

TIFF cubes with paired PNG label images.

Data module `tiff_paired` — pip extras: `tiff`.

[View plugin repo (v0.6.2)](https://github.com/cubert-hyperspectral/cuvis-ai-dataloader/tree/v0.6.2)

`TrackingResultsReader`cuvis_ai.node.json_filesourcebboxmetatrackRead tracking results JSON (bbox or mask format) and emit per-frame tensors.

#### TrackingResultsReader

```python
TrackingResultsReader(
    json_path, required_format=None, **kwargs
)
```

Bases: `Node`

Read tracking results JSON (bbox or mask format) and emit per-frame tensors.

Supports two JSON formats:

1. **COCO image dialect** (`coco_bbox`) — `images` + `annotations` with `bbox` and/or RLE-dict `segmentation` fields plus additive `track_id`. Emits `bboxes`, `category_ids`, `confidences`, `track_ids`; when any annotation carries a `segmentation`, also emits the `mask` label map (pixel value = `track_id`, annotation id when track_id is missing/negative; overlaps painted in ascending annotation-id order) and `object_ids`.
1. **Video COCO** (`video_coco`) — `videos` + `annotations` with `segmentations` list of RLE dicts. Emits `mask` label map and `object_ids`.

Optional outputs are `None` when the format doesn't provide them.

**Frame synchronization**: When the optional `frame_id` input is connected (e.g. from `CU3SDataNode.mesu_index`), the reader looks up detections for that specific frame instead of cursor-advancing. This guarantees that the emitted bboxes/masks correspond to the same frame as the cube data. When `frame_id` is not connected, the reader uses the internal cursor (legacy behavior).

Source code in `cuvis_ai/node/json_file.py`

```python
def __init__(
    self,
    json_path: str,
    required_format: str | None = None,
    **kwargs: Any,
) -> None:
    self.json_path = Path(json_path)
    if not self.json_path.exists():
        raise FileNotFoundError(f"JSON not found: {self.json_path}")
    if required_format is not None and required_format not in {"coco_bbox", "video_coco"}:
        raise ValueError(
            "required_format must be one of {'coco_bbox', 'video_coco'} when provided."
        )

    with self.json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    # Detect format and build per-frame lookup
    if "videos" in data and "annotations" in data:
        self._format = "video_coco"
        self._init_video_coco(data)
    elif "images" in data and "annotations" in data:
        self._format = "coco_bbox"
        self._init_coco_bbox(data)
    else:
        raise ValueError(
            f"Unsupported tracking JSON format in {self.json_path}. "
            "Expected COCO bbox (images+annotations) "
            "or video COCO (videos+annotations)."
        )

    self._required_format = required_format
    self._format_mismatch_msg: str | None = None
    if self._required_format is not None and self._format != self._required_format:
        self._format_mismatch_msg = (
            f"Tracking JSON format is '{self._format}', "
            f"but required_format is '{self._required_format}'."
        )

    self._cursor = 0
    logger.info(
        "[TrackingResultsReader] format={}, required_format={}, frames={}, path={}",
        self._format,
        self._required_format,
        len(self._frame_ids),
        self.json_path,
    )

    super().__init__(json_path=str(self.json_path), required_format=required_format, **kwargs)
```

##### num_frames

```python
num_frames
```

Return the number of frames addressable by this reader.

##### format

```python
format
```

Return the detected tracking JSON format identifier.

##### reset

```python
reset()
```

Rewind sequential reads to the first available frame.

Source code in `cuvis_ai/node/json_file.py`

```python
def reset(self) -> None:
    """Rewind sequential reads to the first available frame."""
    self._cursor = 0
```

##### forward

```python
forward(frame_id=None, context=None, **_)
```

Emit tracking tensors for an explicit frame or the next cursor frame.

Source code in `cuvis_ai/node/json_file.py`

```python
def forward(
    self,
    frame_id: torch.Tensor | None = None,
    context: Context | None = None,  # noqa: ARG002
    **_: Any,
) -> dict[str, Any]:
    """Emit tracking tensors for an explicit frame or the next cursor frame."""
    if self._format_mismatch_msg is not None:
        raise ValueError(self._format_mismatch_msg)

    if frame_id is not None:
        # Lookup mode: emit detections for the requested frame
        fid = int(frame_id.item())
    else:
        # Cursor mode (legacy): advance cursor sequentially
        if self._cursor >= len(self._frame_ids):
            raise StopIteration("No more frames in tracking JSON")
        fid = self._frame_ids[self._cursor]
        self._cursor += 1

    if self._format == "coco_bbox":
        return self._emit_coco_bbox(fid)
    else:
        return self._emit_video_coco(fid)
```

`VideoFrameNode`cuvis_ai.node.videosourcestreamvidPassthrough source node that receives RGB frames from the batch.

#### VideoFrameNode

Bases: `Node`

Passthrough source node that receives RGB frames from the batch.

##### forward

```python
forward(rgb_image, frame_id=None, **_)
```

Pass through RGB frames and optional frame IDs from the batch.

Source code in `cuvis_ai/node/video.py`

```python
def forward(
    self,
    rgb_image: torch.Tensor,
    frame_id: torch.Tensor | None = None,
    **_: Any,
) -> dict[str, torch.Tensor]:
    """Pass through RGB frames and optional frame IDs from the batch."""
    result: dict[str, torch.Tensor] = {"rgb_image": rgb_image}
    if frame_id is not None:
        result["frame_id"] = frame_id
    return result
```

`AugmentationCompose`cuvis_ai_augment.node.composetransformaugmasktorchtrainaugment[↗](https://github.com/cubert-hyperspectral/cuvis-ai-augment "Open plugin repo")Applies a sequence of stochastic augmentation transforms to a cube and paired mask during training only.

Applies a sequence of stochastic augmentation transforms to a cube and paired mask during training only.

Inputs

| Port            | Dtype     | Shape              | Description                                |
| --------------- | --------- | ------------------ | ------------------------------------------ |
| `cube`          | `float32` | `[-1, -1, -1, -1]` | Hyperspectral cube [B, H, W, C] in float32 |
| `mask` optional | `int32`   | `[-1, -1, -1]`     | Per-pixel mask [B, H, W] (int32 or bool)   |

Outputs

| Port            | Dtype     | Shape              | Description                 |
| --------------- | --------- | ------------------ | --------------------------- |
| `cube`          | `float32` | `[-1, -1, -1, -1]` | Augmented cube [B, H, W, C] |
| `mask` optional | `int32`   | `[-1, -1, -1]`     | Augmented mask [B, H, W]    |

[View plugin repo (v0.4.2)](https://github.com/cubert-hyperspectral/cuvis-ai-augment/tree/v0.4.2)

`BBoxRoiCropNode`cuvis_ai.node.preprocessorstransformbboxnumpypreDifferentiable bbox cropping via torchvision roi_align.

#### BBoxRoiCropNode

```python
BBoxRoiCropNode(
    output_size=(256, 128), aligned=True, **kwargs
)
```

Bases: `Node`

Differentiable bbox cropping via torchvision roi_align.

Accepts BHWC images and xyxy bboxes, outputs NCHW crops resized to a fixed `output_size`. Padding rows (all coords \<= 0) are filtered out, so the output N equals the number of valid detections.

Parameters:

| Name          | Type              | Description                                    | Default      |
| ------------- | ----------------- | ---------------------------------------------- | ------------ |
| `output_size` | `tuple[int, int]` | Target crop size (H, W) for roi_align.         | `(256, 128)` |
| `aligned`     | `bool`            | Use sub-pixel aligned roi_align (recommended). | `True`       |

Source code in `cuvis_ai/node/preprocessors.py`

```python
def __init__(
    self,
    output_size: tuple[int, int] = (256, 128),
    aligned: bool = True,
    **kwargs: Any,
) -> None:
    self.output_size = tuple(output_size)
    self.aligned = bool(aligned)
    super().__init__(output_size=list(output_size), aligned=aligned, **kwargs)
```

##### forward

```python
forward(images, bboxes, **_)
```

Crop and resize bounding-box regions from images.

Parameters:

| Name     | Type     | Description                                      | Default    |
| -------- | -------- | ------------------------------------------------ | ---------- |
| `images` | `Tensor` | [B, H, W, C] float32, values in [0, 1].          | *required* |
| `bboxes` | `Tensor` | [B, N_padded, 4] float32 xyxy pixel coordinates. | *required* |

Returns:

| Type   | Description                              |
| ------ | ---------------------------------------- |
| `dict` | {"crops": Tensor [N, C, crop_h, crop_w]} |

Source code in `cuvis_ai/node/preprocessors.py`

```python
def forward(self, images: Tensor, bboxes: Tensor, **_: Any) -> dict[str, Tensor]:
    """Crop and resize bounding-box regions from images.

    Parameters
    ----------
    images : Tensor
        ``[B, H, W, C]`` float32, values in [0, 1].
    bboxes : Tensor
        ``[B, N_padded, 4]`` float32 xyxy pixel coordinates.

    Returns
    -------
    dict
        ``{"crops": Tensor [N, C, crop_h, crop_w]}``
    """
    from torchvision.ops import roi_align

    B, _H, _W, C = images.shape
    crop_h, crop_w = self.output_size

    # BHWC → BCHW
    images_bchw = images.permute(0, 3, 1, 2).contiguous()

    # Build batch indices and flatten bboxes
    N_padded = bboxes.shape[1]
    batch_idx = (
        torch.arange(B, device=bboxes.device).unsqueeze(1).expand(B, N_padded).reshape(-1)
    )
    flat_bboxes = bboxes.reshape(-1, 4)  # [B*N_padded, 4]

    # Filter padding rows (all coords <= 0)
    valid_mask = (flat_bboxes > 0).any(dim=1)
    valid_bboxes = flat_bboxes[valid_mask]
    valid_batch_idx = batch_idx[valid_mask]

    N = valid_bboxes.shape[0]
    if N == 0:
        return {
            "crops": torch.empty(0, C, crop_h, crop_w, device=images.device, dtype=images.dtype)
        }

    # Build [N, 5] roi tensor: [batch_index, x1, y1, x2, y2]
    rois = torch.cat([valid_batch_idx.unsqueeze(1).to(valid_bboxes.dtype), valid_bboxes], dim=1)

    crops = roi_align(
        images_bchw,
        rois,
        output_size=self.output_size,
        spatial_scale=1.0,
        aligned=self.aligned,
    )

    return {"crops": crops}
```

`BBoxSpectralExtractor`cuvis_ai.node.spectral_extractortransformembhsinumpyExtract per-bbox spectral signatures with trimmed median/mean and std.

#### BBoxSpectralExtractor

```python
BBoxSpectralExtractor(
    center_crop_scale=0.65,
    min_crop_pixels=4,
    trim_fraction=0.1,
    l2_normalize=True,
    aggregation="median",
    **kwargs,
)
```

Bases: `Node`

Extract per-bbox spectral signatures with trimmed median/mean and std.

Given an HSI cube `[B, H, W, C]` and detection bboxes `[B, N, 4]` (xyxy format), extracts a center-cropped spectral signature for each bbox. Outputs the per-band aggregated signature, per-band std, and a binary validity mask.

Notes

Only the first batch element (`cube[0]`, `bboxes[0]`) is processed. Outputs are always shaped `[1, N, …]`. Feed one frame at a time (`B == 1`).

Source code in `cuvis_ai/node/spectral_extractor.py`

```python
def __init__(
    self,
    center_crop_scale: float = 0.65,
    min_crop_pixels: int = 4,
    trim_fraction: float = 0.10,
    l2_normalize: bool = True,
    aggregation: str = "median",
    **kwargs: Any,
) -> None:
    if not (0.0 < center_crop_scale <= 1.0):
        raise ValueError("center_crop_scale must be in (0.0, 1.0].")
    if min_crop_pixels < 1:
        raise ValueError("min_crop_pixels must be >= 1.")
    if not (0.0 <= trim_fraction < 0.5):
        raise ValueError("trim_fraction must be in [0.0, 0.5).")
    if aggregation not in ("median", "mean"):
        raise ValueError("aggregation must be 'median' or 'mean'.")

    self.center_crop_scale = float(center_crop_scale)
    self.min_crop_pixels = int(min_crop_pixels)
    self.trim_fraction = float(trim_fraction)
    self.l2_normalize = bool(l2_normalize)
    self.aggregation = str(aggregation)

    super().__init__(
        center_crop_scale=center_crop_scale,
        min_crop_pixels=min_crop_pixels,
        trim_fraction=trim_fraction,
        l2_normalize=l2_normalize,
        aggregation=aggregation,
        **kwargs,
    )
```

##### forward

```python
forward(cube, bboxes, context=None, **_)
```

Extract per-bbox spectral signatures. See class docstring for batch semantics.

Source code in `cuvis_ai/node/spectral_extractor.py`

```python
@torch.no_grad()
def forward(
    self,
    cube: torch.Tensor,
    bboxes: torch.Tensor,
    context: Context | None = None,  # noqa: ARG002
    **_: Any,
) -> dict[str, torch.Tensor]:
    """Extract per-bbox spectral signatures. See class docstring for batch semantics."""
    cube_0 = cube[0]  # [H, W, C]
    img_h, img_w, num_channels = (
        int(cube_0.shape[0]),
        int(cube_0.shape[1]),
        int(cube_0.shape[2]),
    )

    num_boxes = int(bboxes.shape[1])

    # Empty detections
    if num_boxes == 0:
        empty_sig = torch.empty((1, 0, num_channels), dtype=torch.float32, device=cube.device)
        empty_valid = torch.empty((1, 0), dtype=torch.int32, device=cube.device)
        return {
            "spectral_signatures": empty_sig,
            "spectral_std": empty_sig.clone(),
            "spectral_valid": empty_valid,
        }

    signatures: list[torch.Tensor] = []
    stds: list[torch.Tensor] = []
    valids: list[int] = []

    for i in range(num_boxes):
        bx1, by1, bx2, by2 = [int(v) for v in bboxes[0, i].round().tolist()]

        cx1, cy1, cx2, cy2 = self._center_crop_bbox(bx1, by1, bx2, by2, img_h, img_w)

        cw = cx2 - cx1
        ch = cy2 - cy1
        if cw <= 0 or ch <= 0:
            # Bbox fully outside image
            zeros = torch.zeros(num_channels, dtype=cube_0.dtype, device=cube_0.device)
            signatures.append(zeros)
            stds.append(zeros.clone())
            valids.append(0)
            continue

        # Gather pixels from crop region: [P, C]
        pixels = cube_0[cy1:cy2, cx1:cx2, :].reshape(-1, num_channels)

        sig, std = self._trimmed_stats(pixels, num_channels)

        is_valid = sig.norm() >= 1e-8
        if is_valid and self.l2_normalize:
            sig_norm = sig.norm()
            if sig_norm >= 1e-8:
                sig = sig / sig_norm

        signatures.append(sig)
        stds.append(std)
        valids.append(1 if is_valid else 0)

    signatures_t = torch.stack(signatures, dim=0).unsqueeze(0)  # [1, N, C]
    stds_t = torch.stack(stds, dim=0).unsqueeze(0)  # [1, N, C]
    valids_t = torch.tensor(valids, dtype=torch.int32, device=cube.device).unsqueeze(
        0
    )  # [1, N]

    return {
        "spectral_signatures": signatures_t.to(torch.float32),
        "spectral_std": stds_t.to(torch.float32),
        "spectral_valid": valids_t,
    }
```

`BandpassByWavelength`cuvis_ai.node.preprocessorstransformhsipreSelect channels by wavelength interval from BHWC tensors.

#### BandpassByWavelength

```python
BandpassByWavelength(
    min_wavelength_nm, max_wavelength_nm=None, **kwargs
)
```

Bases: `Node`

Select channels by wavelength interval from BHWC tensors.

This node filters hyperspectral data by keeping only channels within a specified wavelength range. Wavelengths must be provided via the input port.

Parameters:

| Name                | Type    | Description                                           | Default                                                                                                     |
| ------------------- | ------- | ----------------------------------------------------- | ----------------------------------------------------------------------------------------------------------- |
| `min_wavelength_nm` | `float` | Minimum wavelength (inclusive) to keep, in nanometers | *required*                                                                                                  |
| `max_wavelength_nm` | \`float | None\`                                                | Maximum wavelength (inclusive) to keep. If None, selects all wavelengths = min_wavelength_nm. Default: None |

Examples:

```pycon
>>> # Create bandpass node
>>> bandpass = BandpassByWavelength(
...     min_wavelength_nm=500.0,
...     max_wavelength_nm=700.0,
... )
>>> # Filter cube in BHWC format with wavelengths from input port
>>> wavelengths_tensor = torch.from_numpy(wavelengths).float()
>>> filtered = bandpass.forward(data=cube_bhwc, wavelengths=wavelengths_tensor)["filtered"]
>>>
>>> # For single HWC images, add a batch dimension first:
>>> # filtered = bandpass.forward(data=cube_hwc.unsqueeze(0), wavelengths=wavelengths_tensor)["filtered"]
>>>
>>> # Use with wavelengths from upstream node
>>> pipeline.connect(
...     (data_node.outputs.cube, bandpass.data),
...     (data_node.outputs.wavelengths, bandpass.wavelengths),
... )
```

Source code in `cuvis_ai/node/preprocessors.py`

```python
def __init__(
    self,
    min_wavelength_nm: float,
    max_wavelength_nm: float | None = None,
    **kwargs,
) -> None:
    self.min_wavelength_nm = float(min_wavelength_nm)
    self.max_wavelength_nm = float(max_wavelength_nm) if max_wavelength_nm is not None else None

    super().__init__(
        min_wavelength_nm=self.min_wavelength_nm,
        max_wavelength_nm=self.max_wavelength_nm,
        **kwargs,
    )
```

##### forward

```python
forward(data, wavelengths, **kwargs)
```

Filter cube by wavelength range.

Parameters:

| Name          | Type     | Description                            | Default    |
| ------------- | -------- | -------------------------------------- | ---------- |
| `data`        | `Tensor` | Input hyperspectral cube [B, H, W, C]. | *required* |
| `wavelengths` | `Tensor` | Wavelengths tensor [C] in nanometers.  | *required* |
| `**kwargs`    | `Any`    | Additional keyword arguments (unused). | `{}`       |

Returns:

| Type                | Description                                                                   |
| ------------------- | ----------------------------------------------------------------------------- |
| `dict[str, Tensor]` | Dictionary with "filtered" key containing filtered cube [B, H, W, C_filtered] |

Raises:

| Type         | Description                                                  |
| ------------ | ------------------------------------------------------------ |
| `ValueError` | If no channels are selected by the provided wavelength range |

Source code in `cuvis_ai/node/preprocessors.py`

```python
def forward(self, data: Tensor, wavelengths: Tensor, **kwargs: Any) -> dict[str, Tensor]:
    """Filter cube by wavelength range.

    Parameters
    ----------
    data : Tensor
        Input hyperspectral cube [B, H, W, C].
    wavelengths : Tensor
        Wavelengths tensor [C] in nanometers.
    **kwargs : Any
        Additional keyword arguments (unused).

    Returns
    -------
    dict[str, Tensor]
        Dictionary with "filtered" key containing filtered cube [B, H, W, C_filtered]

    Raises
    ------
    ValueError
        If no channels are selected by the provided wavelength range
    """

    # Create mask for wavelength range
    if self.max_wavelength_nm is None:
        keep_mask = wavelengths >= self.min_wavelength_nm
    else:
        keep_mask = (wavelengths >= self.min_wavelength_nm) & (
            wavelengths <= self.max_wavelength_nm
        )

    if keep_mask.sum().item() == 0:
        raise ValueError("No channels selected by the provided wavelength range")

    # Filter cube
    filtered = data[..., keep_mask]

    return {"filtered": filtered}
```

`BinaryAnomalyLabelMapper`cuvis_ai.node.labelstransformanommasknumpypostConvert multi-class segmentation masks to binary anomaly targets.

#### BinaryAnomalyLabelMapper

```python
BinaryAnomalyLabelMapper(
    normal_class_ids, anomaly_class_ids=None, **kwargs
)
```

Bases: `Node`

Convert multi-class segmentation masks to binary anomaly targets.

Masks are remapped to torch.long tensors with 0 representing normal pixels and 1 indicating anomalies.

Parameters:

| Name                | Type            | Description                                                   | Default                                                                                                                                                                                                                          |
| ------------------- | --------------- | ------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `normal_class_ids`  | `Iterable[int]` | Class IDs that should be considered normal (default: (0, 2)). | *required*                                                                                                                                                                                                                       |
| `anomaly_class_ids` | \`Iterable[int] | None\`                                                        | Explicit anomaly IDs. When None all IDs not in normal_class_ids are treated as anomalies. When provided, only these IDs are treated as anomalies and all others (including those not in normal_class_ids) are treated as normal. |

Source code in `cuvis_ai/node/labels.py`

```python
def __init__(
    self,
    normal_class_ids: Iterable[int],
    anomaly_class_ids: Iterable[int] | None = None,
    **kwargs,
) -> None:
    self.normal_class_ids = tuple(int(c) for c in normal_class_ids)
    self.anomaly_class_ids = (
        tuple(int(c) for c in anomaly_class_ids) if anomaly_class_ids is not None else None
    )

    # Validate that there are no overlaps between normal and anomaly class IDs
    if self.anomaly_class_ids is not None:
        overlap = set(self.normal_class_ids) & set(self.anomaly_class_ids)
        if overlap:
            raise ValueError(
                f"Overlap detected between normal_class_ids and anomaly_class_ids: {overlap}. "
                "Class IDs cannot be both normal and anomaly."
            )

        # Check for gaps in coverage and issue warning
        all_specified_ids = set(self.normal_class_ids) | set(self.anomaly_class_ids)
        max_id = max(all_specified_ids) if all_specified_ids else 0

        # Find gaps (missing class IDs)
        expected_ids = set(range(max_id + 1))
        gaps = expected_ids - all_specified_ids

        if gaps:
            warnings.warn(
                f"Gap detected in class ID coverage. The following class IDs are not specified "
                f"in either normal_class_ids or anomaly_class_ids: {gaps}. "
                f"These will be treated as normal classes. To specify all classes explicitly, "
                f"include them in normal_class_ids or anomaly_class_ids.",
                UserWarning,
                stacklevel=2,
            )
            # Add gaps to normal_class_ids as requested
            self.normal_class_ids = tuple(sorted(set(self.normal_class_ids) | gaps))

    self._target_dtype = torch.long

    super().__init__(
        normal_class_ids=self.normal_class_ids,
        anomaly_class_ids=self.anomaly_class_ids,
        **kwargs,
    )
```

##### forward

```python
forward(cube, mask, **_)
```

Map multi-class labels to binary anomaly labels.

Parameters:

| Name   | Type     | Description                                  | Default    |
| ------ | -------- | -------------------------------------------- | ---------- |
| `cube` | `Tensor` | Features/scores to pass through [B, H, W, C] | *required* |
| `mask` | `Tensor` | Multi-class segmentation masks [B, H, W, 1]  | *required* |

Returns:

| Type                | Description                                                         |
| ------------------- | ------------------------------------------------------------------- |
| `dict[str, Tensor]` | Dictionary with "cube" (pass-through) and "mask" (binary bool) keys |

Source code in `cuvis_ai/node/labels.py`

```python
def forward(self, cube: Tensor, mask: Tensor, **_: Any) -> dict[str, Tensor]:
    """Map multi-class labels to binary anomaly labels.

    Parameters
    ----------
    cube : Tensor
        Features/scores to pass through [B, H, W, C]
    mask : Tensor
        Multi-class segmentation masks [B, H, W, 1]

    Returns
    -------
    dict[str, Tensor]
        Dictionary with "cube" (pass-through) and "mask" (binary bool) keys
    """
    if self.anomaly_class_ids is not None:
        # Explicit anomaly class IDs: only these are anomalies, rest are normal
        mask_anomaly = self._membership_mask(mask, self.anomaly_class_ids)
    else:
        # Original behavior: normal_class_ids are normal, everything else is anomaly
        mask_normal = self._membership_mask(mask, self.normal_class_ids)
        mask_anomaly = ~mask_normal

    mapped = torch.zeros_like(mask, dtype=self._target_dtype, device=mask.device)
    mapped = torch.where(mask_anomaly, torch.ones_like(mapped), mapped)

    # Convert to bool for smaller tensor size
    mapped = mapped.bool()

    return {"cube": cube, "mask": mapped}
```

`BinaryDecider`cuvis_ai.node.deciders.binary_decidertransformclassnumpypostSimple decider node using a static threshold to classify data.

#### BinaryDecider

```python
BinaryDecider(threshold=0.5, **kwargs)
```

Bases: `BinaryDecider`

Simple decider node using a static threshold to classify data.

Accepts logits as input, applies sigmoid transformation to convert to probabilities [0, 1], then applies threshold to produce binary decisions.

Parameters:

| Name        | Type    | Description                                                                                                                 | Default |
| ----------- | ------- | --------------------------------------------------------------------------------------------------------------------------- | ------- |
| `threshold` | `float` | The threshold to use for classification after sigmoid. Values >= threshold are classified as anomalies (True). Default: 0.5 | `0.5`   |

Examples:

```pycon
>>> from cuvis_ai.node.deciders.binary_decider import BinaryDecider
>>> import torch
>>>
>>> # Create decider with default threshold
>>> decider = BinaryDecider(threshold=0.5)
>>>
>>> # Apply to RX anomaly logits
>>> logits = torch.randn(4, 256, 256, 1)  # [B, H, W, C]
>>> output = decider.forward(logits=logits)
>>> decisions = output["decisions"]  # [4, 256, 256, 1] boolean mask
>>>
>>> # Use in pipeline
>>> pipeline.connect(
...     (logit_head.logits, decider.logits),
...     (decider.decisions, visualizer.mask),
... )
```

See Also

QuantileBinaryDecider : Adaptive per-batch thresholding ScoreToLogit : Convert scores to logits before decisioning

Source code in `cuvis_ai/node/deciders/binary_decider.py`

```python
def __init__(self, threshold: float = 0.5, **kwargs) -> None:
    self.threshold = threshold
    # Forward threshold to BaseDecider so Serializable captures it in hparams
    super().__init__(threshold=threshold, **kwargs)
```

##### forward

```python
forward(logits, **_)
```

Apply sigmoid and threshold-based decisioning on channels-last data.

Args: logits: Tensor shaped (B, H, W, C) containing logits.

Returns: Dictionary with "decisions" key containing (B, H, W, 1) decision mask.

Source code in `cuvis_ai/node/deciders/binary_decider.py`

```python
def forward(
    self,
    logits: Tensor,
    **_: Any,
) -> dict[str, Tensor]:
    """Apply sigmoid and threshold-based decisioning on channels-last data.

    Args:
        logits: Tensor shaped (B, H, W, C) containing logits.

    Returns:
        Dictionary with "decisions" key containing (B, H, W, 1) decision mask.
    """

    # Apply sigmoid if needed to convert logits to probabilities
    tensor = torch.sigmoid(logits)

    # Apply threshold to get binary decisions
    decisions = tensor >= self.threshold
    return {"decisions": decisions}
```

`BlobDetector`cuvis_ai.node.blob_detectortransformhsimasksegLocalize bright blobs (e.g. pills, granules, tray compartments) in a cube.

#### BlobDetector

```python
BlobDetector(
    brightness="band_mean",
    threshold_method="otsu",
    threshold=0.5,
    index_wavelengths=None,
    opening_kernel=3,
    closing_kernel=3,
    min_area=5,
    max_area=None,
    keep_largest=None,
    connectivity=8,
    **kwargs,
)
```

Bases: `Node`

Localize bright blobs (e.g. pills, granules, tray compartments) in a cube.

The cube's first frame is reduced to a 2-D brightness image, thresholded into a foreground mask, morphologically cleaned, and labeled into connected components. Components are filtered by area and, optionally, capped to the `keep_largest` biggest so a fixed-layout scene yields a stable blob count.

Parameters:

| Name                | Type                          | Description                                                                                                                                                                                                                                                                                                          | Default       |
| ------------------- | ----------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------- |
| `brightness`        | `str`                         | How to reduce the cube [H, W, C] to a 2-D image: "band_mean" (mean over channels, default), "max" (per-pixel channel max), or "index" (normalized difference (a - b) / (a + b) of the two bands nearest index_wavelengths; falls back to the band mean when wavelengths are unavailable).                            | `'band_mean'` |
| `threshold_method`  | `str`                         | "otsu" (between-class variance, default), "quantile" (keep pixels above the threshold quantile), or "fixed" (keep pixels with min-max-scaled brightness >= threshold).                                                                                                                                               | `'otsu'`      |
| `threshold`         | `float`                       | Quantile in [0, 1] for "quantile" or scaled-brightness cutoff in [0, 1] for "fixed"; ignored for "otsu". Default 0.5.                                                                                                                                                                                                | `0.5`         |
| `index_wavelengths` | `tuple[float, float] or None` | Two wavelengths (nm) for brightness="index".                                                                                                                                                                                                                                                                         | `None`        |
| `opening_kernel`    | `int`                         | Square structuring-element side for morphological opening (speckle removal); 0 or 1 disables. Default 3.                                                                                                                                                                                                             | `3`           |
| `closing_kernel`    | `int`                         | Square side for morphological closing (hole fill); 0 or 1 disables. Default 3.                                                                                                                                                                                                                                       | `3`           |
| `min_area`          | `int`                         | Drop connected components with fewer than this many pixels. Default 5.                                                                                                                                                                                                                                               | `5`           |
| `max_area`          | `int or None`                 | Drop components larger than this many pixels (None disables).                                                                                                                                                                                                                                                        | `None`        |
| `keep_largest`      | `int or None`                 | After area filtering, keep only the keep_largest biggest components (None disables). Stabilizes the blob count across re-scans: a fixed tray of N compartments yields exactly N groups even when a scan sprouts a spurious bright fragment. No effect if fewer than keep_largest components survive the area filter. | `None`        |
| `connectivity`      | `int`                         | 8 (default) or 4 neighborhood for connected components.                                                                                                                                                                                                                                                              | `8`           |

Validate and store the detection hyperparameters.

Source code in `cuvis_ai/node/blob_detector.py`

```python
def __init__(
    self,
    brightness: str = "band_mean",
    threshold_method: str = "otsu",
    threshold: float = 0.5,
    index_wavelengths: tuple[float, float] | None = None,
    opening_kernel: int = 3,
    closing_kernel: int = 3,
    min_area: int = 5,
    max_area: int | None = None,
    keep_largest: int | None = None,
    connectivity: int = 8,
    **kwargs: Any,
) -> None:
    """Validate and store the detection hyperparameters."""
    if brightness not in ("band_mean", "max", "index"):
        raise ValueError("brightness must be 'band_mean', 'max', or 'index'.")
    if threshold_method not in ("otsu", "quantile", "fixed"):
        raise ValueError("threshold_method must be 'otsu', 'quantile', or 'fixed'.")
    if not (0.0 <= threshold <= 1.0):
        raise ValueError("threshold must be in [0, 1].")
    if connectivity not in (4, 8):
        raise ValueError("connectivity must be 4 or 8.")
    if min_area < 1:
        raise ValueError("min_area must be >= 1.")
    if max_area is not None and max_area < min_area:
        raise ValueError("max_area must be >= min_area.")
    if keep_largest is not None and int(keep_largest) < 1:
        raise ValueError("keep_largest must be >= 1 or None.")

    self.brightness = str(brightness)
    self.threshold_method = str(threshold_method)
    self.threshold = float(threshold)
    self.index_wavelengths = (
        None if index_wavelengths is None else tuple(float(w) for w in index_wavelengths)
    )
    self.opening_kernel = int(opening_kernel)
    self.closing_kernel = int(closing_kernel)
    self.min_area = int(min_area)
    self.max_area = None if max_area is None else int(max_area)
    self.keep_largest = None if keep_largest is None else int(keep_largest)
    self.connectivity = int(connectivity)

    super().__init__(
        brightness=self.brightness,
        threshold_method=self.threshold_method,
        threshold=self.threshold,
        index_wavelengths=self.index_wavelengths,
        opening_kernel=self.opening_kernel,
        closing_kernel=self.closing_kernel,
        min_area=self.min_area,
        max_area=self.max_area,
        keep_largest=self.keep_largest,
        connectivity=self.connectivity,
        **kwargs,
    )
```

##### forward

```python
forward(cube, wavelengths=None, **_)
```

Detect blobs in the first frame of `cube`.

Parameters:

| Name          | Type                        | Description                                                      | Default    |
| ------------- | --------------------------- | ---------------------------------------------------------------- | ---------- |
| `cube`        | `Tensor`                    | Hyperspectral cube [B, H, W, C]; only cube[0] is processed.      | *required* |
| `wavelengths` | `ndarray or Tensor or None` | Wavelengths [C] in nanometers, used only by brightness="index".  | `None`     |
| `**_`         | `Any`                       | Additional unused keyword arguments (e.g. the pipeline context). | `{}`       |

Returns:

| Type                | Description                                                                                                                                   |
| ------------------- | --------------------------------------------------------------------------------------------------------------------------------------------- |
| `dict[str, Tensor]` | mask int32 [1, H, W] (blob ids 1..N, 0 background), bboxes float32 [1, N, 4] (xyxy), centroids float32 [1, N, 2] (x, y), and count int32 [1]. |

Source code in `cuvis_ai/node/blob_detector.py`

```python
@torch.no_grad()
def forward(
    self,
    cube: torch.Tensor,
    wavelengths: np.ndarray | torch.Tensor | None = None,
    **_: Any,
) -> dict[str, torch.Tensor]:
    """Detect blobs in the first frame of ``cube``.

    Parameters
    ----------
    cube : torch.Tensor
        Hyperspectral cube ``[B, H, W, C]``; only ``cube[0]`` is processed.
    wavelengths : numpy.ndarray or torch.Tensor or None, optional
        Wavelengths ``[C]`` in nanometers, used only by ``brightness="index"``.
    **_ : Any
        Additional unused keyword arguments (e.g. the pipeline ``context``).

    Returns
    -------
    dict[str, torch.Tensor]
        ``mask`` int32 ``[1, H, W]`` (blob ids 1..N, 0 background),
        ``bboxes`` float32 ``[1, N, 4]`` (xyxy), ``centroids`` float32
        ``[1, N, 2]`` (x, y), and ``count`` int32 ``[1]``.
    """
    cube0 = cube[0]
    fg = self._foreground(cube0, wavelengths)
    if not bool(fg.any()):
        height, width = cube0.shape[0], cube0.shape[1]
        return {
            "mask": torch.zeros((1, height, width), dtype=torch.int32, device=cube.device),
            "bboxes": torch.zeros((1, 0, 4), dtype=torch.float32, device=cube.device),
            "centroids": torch.zeros((1, 0, 2), dtype=torch.float32, device=cube.device),
            "count": torch.zeros((1,), dtype=torch.int32, device=cube.device),
        }

    labels = label_connected_components(fg, connectivity=self.connectivity).to(torch.int64)
    mask, bboxes, centroids, count = self._finalize(labels)
    return {
        "mask": mask.to(device=cube.device, dtype=torch.int32),
        "bboxes": bboxes.to(cube.device),
        "centroids": centroids.to(cube.device),
        "count": torch.tensor([count], dtype=torch.int32, device=cube.device),
    }
```

`BlobMajorityVote`cuvis_ai_inspecscrap.node.deciderstransformclassmaskpostcuvis_ai_inspecscrap[↗](https://github.com/cubert-hyperspectral/cuvis-ai-inspecscrap "Open plugin repo")

[View plugin repo (v0.2.4)](https://github.com/cubert-hyperspectral/cuvis-ai-inspecscrap/tree/v0.2.4)

`CIETristimulusRGBSelector`cuvis_ai.node.channel_selectortransformdim-redhsinumpypreCIE 1931 tristimulus-based RGB rendering.

#### CIETristimulusRGBSelector

```python
CIETristimulusRGBSelector(**kwargs)
```

Bases: `ChannelSelectorBase`

CIE 1931 tristimulus-based RGB rendering.

Converts a hyperspectral cube to sRGB by integrating each pixel's spectrum with the CIE 1931 2-degree standard observer color matching functions (x_bar, y_bar, z_bar), applying a D65 white point normalization, and converting from CIE XYZ to linear sRGB.

Normalization and sRGB gamma are handled by `ChannelSelectorBase` (see `apply_gamma` parameter inherited from the base class).

This produces a faithful (true) RGB rendering and lands closest to the distribution SAM3's Perception Encoder expects.

For wavelengths outside the visible range (approx. >780 nm), the CMFs are zero, so NIR bands do not contribute to the output.

Source code in `cuvis_ai/node/channel_selector.py`

```python
def __init__(self, **kwargs: Any) -> None:
    super().__init__(**kwargs)

    # Static XYZ -> linear sRGB matrix; buffer so .to(device) moves it.
    self.register_buffer(
        "_xyz_to_srgb_matrix",
        torch.from_numpy(self._XYZ_TO_SRGB.astype(np.float32)),
    )
    # Wavelength-dependent CMF integration weights; lazily computed on first forward.
    self.register_buffer("_cmf_weights", None, persistent=False)
    self._cached_wl_key: tuple[float, ...] | None = None
    self._cached_n_visible: int = 0
```

##### forward

```python
forward(cube, wavelengths, context=None, **_)
```

Convert HSI cube to sRGB via CIE 1931 tristimulus integration.

Parameters:

| Name          | Type     | Description                      | Default                             |
| ------------- | -------- | -------------------------------- | ----------------------------------- |
| `cube`        | `Tensor` | Hyperspectral cube [B, H, W, C]. | *required*                          |
| `wavelengths` | \`Tensor | ndarray\`                        | Wavelength array [C] in nanometers. |

Returns:

| Type             | Description                                               |
| ---------------- | --------------------------------------------------------- |
| `dict[str, Any]` | Dictionary with "rgb_image" [B, H, W, 3] and "band_info". |

Source code in `cuvis_ai/node/channel_selector.py`

```python
def forward(
    self,
    cube: torch.Tensor,
    wavelengths: Any,
    context: Context | None = None,  # noqa: ARG002
    **_: Any,
) -> dict[str, Any]:
    """Convert HSI cube to sRGB via CIE 1931 tristimulus integration.

    Parameters
    ----------
    cube : torch.Tensor
        Hyperspectral cube [B, H, W, C].
    wavelengths : torch.Tensor | np.ndarray
        Wavelength array [C] in nanometers.

    Returns
    -------
    dict[str, Any]
        Dictionary with "rgb_image" [B, H, W, 3] and "band_info".
    """
    wavelengths_np = np.asarray(wavelengths, dtype=np.float64).ravel()
    if wavelengths_np.ndim == 0:
        raise ValueError("wavelengths must be a 1-D array")

    # Compute unnormalized linear sRGB, then normalize + gamma via base class.
    rgb = self._normalize_rgb(self._compute_raw_rgb(cube, wavelengths))

    band_info = {
        "strategy": "cie_tristimulus",
        "illuminant": "D65",
        "apply_gamma": self.apply_gamma,
        "sensor_bands_total": len(wavelengths_np),
        "sensor_bands_visible": self._cached_n_visible,
        "wavelength_range_nm": [float(wavelengths_np[0]), float(wavelengths_np[-1])],
    }

    return {"rgb_image": rgb, "band_info": band_info}
```

`CIRSelector`cuvis_ai.node.channel_selectortransformdim-redhsinumpypreColor Infrared (CIR) false color composition.

#### CIRSelector

```python
CIRSelector(
    nir_nm=860.0, red_nm=670.0, green_nm=560.0, **kwargs
)
```

Bases: `ChannelSelectorBase`

Color Infrared (CIR) false color composition.

Maps NIR to Red, Red to Green, Green to Blue for false-color composites. This is useful for highlighting vegetation and certain anomalies.

Parameters:

| Name       | Type    | Description                                    | Default |
| ---------- | ------- | ---------------------------------------------- | ------- |
| `nir_nm`   | `float` | Near-infrared wavelength in nm. Default: 860.0 | `860.0` |
| `red_nm`   | `float` | Red wavelength in nm. Default: 670.0           | `670.0` |
| `green_nm` | `float` | Green wavelength in nm. Default: 560.0         | `560.0` |

Source code in `cuvis_ai/node/channel_selector.py`

```python
def __init__(
    self,
    nir_nm: float = 860.0,
    red_nm: float = 670.0,
    green_nm: float = 560.0,
    **kwargs,
) -> None:
    super().__init__(nir_nm=nir_nm, red_nm=red_nm, green_nm=green_nm, **kwargs)
    self.nir_nm = nir_nm
    self.red_nm = red_nm
    self.green_nm = green_nm
```

##### forward

```python
forward(cube, wavelengths, context=None, **_)
```

Select CIR bands and compose false-color image.

Parameters:

| Name          | Type     | Description                      | Default    |
| ------------- | -------- | -------------------------------- | ---------- |
| `cube`        | `Tensor` | Hyperspectral cube [B, H, W, C]. | *required* |
| `wavelengths` | `Tensor` | Wavelength array [C].            | *required* |

Returns:

| Type             | Description                                       |
| ---------------- | ------------------------------------------------- |
| `dict[str, Any]` | Dictionary with "rgb_image" and "band_info" keys. |

Source code in `cuvis_ai/node/channel_selector.py`

```python
def forward(
    self,
    cube: torch.Tensor,
    wavelengths: Any,
    context: Context | None = None,  # noqa: ARG002
    **_: Any,
) -> dict[str, Any]:
    """Select CIR bands and compose false-color image.

    Parameters
    ----------
    cube : torch.Tensor
        Hyperspectral cube [B, H, W, C].
    wavelengths : torch.Tensor
        Wavelength array [C].

    Returns
    -------
    dict[str, Any]
        Dictionary with "rgb_image" and "band_info" keys.
    """
    wavelengths_np = np.asarray(wavelengths, dtype=np.float32).ravel()
    nir_idx, red_idx, green_idx = self._resolve_band_indices(wavelengths_np)
    indices = [nir_idx, red_idx, green_idx]
    rgb = self._normalize_rgb(self._compute_raw_rgb(cube, wavelengths_np))

    band_info = {
        "strategy": "cir_false_color",
        "band_indices": indices,
        "band_wavelengths_nm": [float(wavelengths_np[i]) for i in indices],
        "target_wavelengths_nm": [self.nir_nm, self.red_nm, self.green_nm],
        "channel_mapping": {"R": "NIR", "G": "Red", "B": "Green"},
    }

    return {"rgb_image": rgb, "band_info": band_info}
```

`CIRedEdgeSelector`cuvis_ai.node.channel_selectortransformdim-redhsinumpypreChlorophyll Index Red Edge renderer.

#### CIRedEdgeSelector

```python
CIRedEdgeSelector(
    red_edge_nm=720.0,
    nir_nm=800.0,
    colormap_min=-1.0,
    colormap_max=1.0,
    eps=1e-06,
    **kwargs,
)
```

Bases: `_VegetationIndexBase`

Chlorophyll Index Red Edge renderer.

Computes `NIR / RedEdge - 1` over bands resolved by nearest sensor wavelength. The raw index map is returned via `index_image` and `rgb_image` carries an HSV colour-mapped render.

Source code in `cuvis_ai/node/channel_selector.py`

```python
def __init__(
    self,
    red_edge_nm: float = 720.0,
    nir_nm: float = 800.0,
    colormap_min: float = -1.0,
    colormap_max: float = 1.0,
    eps: float = 1.0e-6,
    **kwargs: Any,
) -> None:
    super().__init__(
        band_nm={"red_edge": red_edge_nm, "nir": nir_nm},
        colormap_min=colormap_min,
        colormap_max=colormap_max,
        eps=eps,
        red_edge_nm=float(red_edge_nm),
        nir_nm=float(nir_nm),
        **kwargs,
    )
    self.red_edge_nm = float(red_edge_nm)
    self.nir_nm = float(nir_nm)
```

##### index_name

```python
index_name
```

Canonical CIRedEdge strategy name.

`CameraEmulationFalseRGBSelector`cuvis_ai.node.channel_selectortransformdim-redhsinumpypreCamera-emulation false RGB using smooth Gaussian sensitivity curves.

#### CameraEmulationFalseRGBSelector

```python
CameraEmulationFalseRGBSelector(
    r_peak=610.0,
    g_peak=540.0,
    b_peak=460.0,
    r_sigma=40.0,
    g_sigma=35.0,
    b_sigma=30.0,
    **kwargs,
)
```

Bases: `ChannelSelectorBase`

Camera-emulation false RGB using smooth Gaussian sensitivity curves.

Defines three broad, smooth Gaussian weighting curves over the spectral bands that mimic R/G/B camera sensitivity (peaks at configurable wavelengths). The weight matrix W is [3, num_bands], applied as `rgb = W @ spectrum`. Non-negativity is enforced by construction.

This is simple, stable, and requires no training. Good middle ground between single-band selection and learned mapping.

Parameters:

| Name      | Type    | Description                                         | Default |
| --------- | ------- | --------------------------------------------------- | ------- |
| `r_peak`  | `float` | Red channel peak wavelength in nm. Default: 610.0   | `610.0` |
| `g_peak`  | `float` | Green channel peak wavelength in nm. Default: 540.0 | `540.0` |
| `b_peak`  | `float` | Blue channel peak wavelength in nm. Default: 460.0  | `460.0` |
| `r_sigma` | `float` | Red channel Gaussian sigma in nm. Default: 40.0     | `40.0`  |
| `g_sigma` | `float` | Green channel Gaussian sigma in nm. Default: 35.0   | `35.0`  |
| `b_sigma` | `float` | Blue channel Gaussian sigma in nm. Default: 30.0    | `30.0`  |

Source code in `cuvis_ai/node/channel_selector.py`

```python
def __init__(
    self,
    r_peak: float = 610.0,
    g_peak: float = 540.0,
    b_peak: float = 460.0,
    r_sigma: float = 40.0,
    g_sigma: float = 35.0,
    b_sigma: float = 30.0,
    **kwargs: Any,
) -> None:
    super().__init__(
        r_peak=r_peak,
        g_peak=g_peak,
        b_peak=b_peak,
        r_sigma=r_sigma,
        g_sigma=g_sigma,
        b_sigma=b_sigma,
        **kwargs,
    )
    self.peaks = (r_peak, g_peak, b_peak)
    self.sigmas = (r_sigma, g_sigma, b_sigma)

    # Wavelength-dependent Gaussian weights; lazily computed on first forward.
    self.register_buffer("_channel_weights", None, persistent=False)
    self._cached_wl_key: tuple[float, ...] | None = None
```

##### forward

```python
forward(cube, wavelengths, context=None, **_)
```

Convert HSI cube to false RGB using Gaussian camera sensitivity.

Parameters:

| Name          | Type     | Description                      | Default                             |
| ------------- | -------- | -------------------------------- | ----------------------------------- |
| `cube`        | `Tensor` | Hyperspectral cube [B, H, W, C]. | *required*                          |
| `wavelengths` | \`Tensor | ndarray\`                        | Wavelength array [C] in nanometers. |

Returns:

| Type             | Description                                               |
| ---------------- | --------------------------------------------------------- |
| `dict[str, Any]` | Dictionary with "rgb_image" [B, H, W, 3] and "band_info". |

Source code in `cuvis_ai/node/channel_selector.py`

```python
def forward(
    self,
    cube: torch.Tensor,
    wavelengths: Any,
    context: Context | None = None,  # noqa: ARG002
    **_: Any,
) -> dict[str, Any]:
    """Convert HSI cube to false RGB using Gaussian camera sensitivity.

    Parameters
    ----------
    cube : torch.Tensor
        Hyperspectral cube [B, H, W, C].
    wavelengths : torch.Tensor | np.ndarray
        Wavelength array [C] in nanometers.

    Returns
    -------
    dict[str, Any]
        Dictionary with "rgb_image" [B, H, W, 3] and "band_info".
    """
    wavelengths_np = np.asarray(wavelengths, dtype=np.float64).ravel()

    rgb = self._compute_raw_rgb(cube, wavelengths)
    rgb = self._normalize_rgb(rgb)

    band_info = {
        "strategy": "camera_emulation",
        "peaks_nm": {"R": self.peaks[0], "G": self.peaks[1], "B": self.peaks[2]},
        "sigmas_nm": {"R": self.sigmas[0], "G": self.sigmas[1], "B": self.sigmas[2]},
        "sensor_bands_total": len(wavelengths_np),
    }

    return {"rgb_image": rgb, "band_info": band_info}
```

`ChannelNormalizeNode`cuvis_ai.node.preprocessorstransformhsinormnumpyprePer-channel mean/std normalization for NCHW tensors.

#### ChannelNormalizeNode

```python
ChannelNormalizeNode(
    mean=IMAGENET_MEAN, std=IMAGENET_STD, **kwargs
)
```

Bases: `Node`

Per-channel mean/std normalization for NCHW tensors.

Defaults to ImageNet statistics but accepts any per-channel values.

Parameters:

| Name   | Type                | Description       | Default         |
| ------ | ------------------- | ----------------- | --------------- |
| `mean` | `tuple[float, ...]` | Per-channel mean. | `IMAGENET_MEAN` |
| `std`  | `tuple[float, ...]` | Per-channel std.  | `IMAGENET_STD`  |

Source code in `cuvis_ai/node/preprocessors.py`

```python
def __init__(
    self,
    mean: tuple[float, ...] = IMAGENET_MEAN,
    std: tuple[float, ...] = IMAGENET_STD,
    **kwargs: Any,
) -> None:
    self._mean_vals = tuple(float(v) for v in mean)
    self._std_vals = tuple(float(v) for v in std)

    super().__init__(mean=list(self._mean_vals), std=list(self._std_vals), **kwargs)

    # Register as buffers so they auto-move with .to(device)
    self.register_buffer(
        "_mean_buf",
        torch.tensor(self._mean_vals, dtype=torch.float32).view(1, -1, 1, 1),
    )
    self.register_buffer(
        "_std_buf",
        torch.tensor(self._std_vals, dtype=torch.float32).view(1, -1, 1, 1),
    )
```

##### forward

```python
forward(images, **_)
```

Normalize images per channel.

Parameters:

| Name     | Type     | Description           | Default    |
| -------- | -------- | --------------------- | ---------- |
| `images` | `Tensor` | [N, C, H, W] float32. | *required* |

Returns:

| Type   | Description                         |
| ------ | ----------------------------------- |
| `dict` | {"normalized": Tensor [N, C, H, W]} |

Source code in `cuvis_ai/node/preprocessors.py`

```python
def forward(self, images: Tensor, **_: Any) -> dict[str, Tensor]:
    """Normalize images per channel.

    Parameters
    ----------
    images : Tensor
        ``[N, C, H, W]`` float32.

    Returns
    -------
    dict
        ``{"normalized": Tensor [N, C, H, W]}``
    """
    normalized = (images - self._mean_buf) / self._std_buf
    return {"normalized": normalized}
```

`ChannelSelectorBase`cuvis_ai.node.channel_selectortransformdim-redhsinumpypreBase class for hyperspectral band selection strategies.

#### ChannelSelectorBase

```python
ChannelSelectorBase(
    norm_mode=RUNNING,
    apply_gamma=True,
    freeze_running_bounds_after_frames=20,
    running_warmup_frames=_WARMUP_FRAMES,
    **kwargs,
)
```

Bases: `Node`

Base class for hyperspectral band selection strategies.

This base class defines the common input/output ports for band selection nodes and provides shared percentile-based RGB normalization (see module docstring for design rationale).

Subclasses should implement `forward()` and `_compute_raw_rgb()` (the latter is used by `statistical_initialization` and `_running_normalize`).

Parameters:

| Name                                 | Type   | Description                                                                                                                                                 | Default                                                                                                                                               |
| ------------------------------------ | ------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------- |
| `norm_mode`                          | \`str  | NormMode\`                                                                                                                                                  | RGB normalization mode. Default NormMode.RUNNING.                                                                                                     |
| `apply_gamma`                        | `bool` | Apply sRGB gamma curve after normalization. Default True. Lifts midtones so linear [0, 1] values appear natural on standard displays.                       | `True`                                                                                                                                                |
| `freeze_running_bounds_after_frames` | \`int  | None\`                                                                                                                                                      | When norm_mode='running', stop updating running_min/running_max after this many forward calls. None keeps legacy behavior (never freeze). Default 20. |
| `running_warmup_frames`              | `int`  | Number of initial running frames to normalize per-frame while collecting bounds. Set to 0 for fully stable live rendering from the first frame. Default 10. | `_WARMUP_FRAMES`                                                                                                                                      |

Ports

INPUT_SPECS `cube` : float32, shape (-1, -1, -1, -1) Hyperspectral cube in BHWC format. `wavelengths` : float32, shape (-1,) Wavelength array in nanometers. OUTPUT_SPECS `rgb_image` : float32, shape (-1, -1, -1, 3) Composed RGB image in BHWC format (0-1 range). Subclasses that emit a different channel count (e.g. :class:`FixedWavelengthSelector` with `n != 3`) must override `OUTPUT_SPECS` to widen the channel dimension; the base class keeps the tight 3-channel contract so pipeline validation catches accidental mis-wiring of the standard RGB selectors (`FastRGBSelector`, `RangeAverageFalseRGBSelector`, `CIRSelector`, `CIETristimulusRGBSelector`, NDVI variants, …). `band_info` : dict Metadata about selected bands.

Source code in `cuvis_ai/node/channel_selector.py`

```python
def __init__(
    self,
    norm_mode: str | NormMode = NormMode.RUNNING,
    apply_gamma: bool = True,
    freeze_running_bounds_after_frames: int | None = 20,
    running_warmup_frames: int = _WARMUP_FRAMES,
    **kwargs: Any,
) -> None:
    if freeze_running_bounds_after_frames is not None:
        if (
            isinstance(freeze_running_bounds_after_frames, bool)
            or not isinstance(freeze_running_bounds_after_frames, int)
            or freeze_running_bounds_after_frames < 1
        ):
            raise ValueError(
                "freeze_running_bounds_after_frames must be an integer >= 1 or None"
            )
    if (
        isinstance(running_warmup_frames, bool)
        or not isinstance(running_warmup_frames, int)
        or running_warmup_frames < 0
    ):
        raise ValueError("running_warmup_frames must be an integer >= 0")
    super().__init__(
        norm_mode=str(norm_mode) if isinstance(norm_mode, NormMode) else norm_mode,
        apply_gamma=apply_gamma,
        freeze_running_bounds_after_frames=freeze_running_bounds_after_frames,
        running_warmup_frames=running_warmup_frames,
        **kwargs,
    )
    self.norm_mode = NormMode(norm_mode)
    self.apply_gamma = apply_gamma
    self.freeze_running_bounds_after_frames = freeze_running_bounds_after_frames
    self.running_warmup_frames = running_warmup_frames

    # Per-channel [3] running bounds for normalization.
    self.register_buffer("running_min", torch.full((3,), float("nan")))
    self.register_buffer("running_max", torch.full((3,), float("nan")))
    self._norm_frame_count = 0
    self._statistically_initialized = False

    # Only STATISTICAL mode needs the StatisticalTrainer pass; without an override,
    # RUNNING/PER_FRAME would inherit True from the auto-detect because this base
    # implements statistical_initialization. Subclasses with their own initialization
    # (e.g. SupervisedSelectorBase) keep the auto-detect.
    if self.norm_mode == NormMode.STATISTICAL:
        self._requires_initial_fit_override = True
    elif (
        type(self).statistical_initialization is ChannelSelectorBase.statistical_initialization
    ):
        self._requires_initial_fit_override = False
```

##### statistical_initialization

```python
statistical_initialization(input_stream)
```

Compute global percentile bounds across the entire dataset.

Uses `_compute_raw_rgb()` to convert each batch, then accumulates per-channel percentile bounds (min-of-lows, max-of-highs).

Source code in `cuvis_ai/node/channel_selector.py`

```python
def statistical_initialization(self, input_stream: InputStream) -> None:
    """Compute global percentile bounds across the entire dataset.

    Uses ``_compute_raw_rgb()`` to convert each batch, then accumulates
    per-channel percentile bounds (min-of-lows, max-of-highs).
    """
    for batch_data in input_stream:
        raw_rgb = self._compute_raw_rgb(batch_data["cube"], batch_data["wavelengths"])
        flat = raw_rgb.reshape(-1, 3).float()  # quantile() requires float/double
        frame_lo = torch.quantile(flat, self._NORM_QUANTILE_LOW, dim=0)
        frame_hi = torch.quantile(flat, self._NORM_QUANTILE_HIGH, dim=0)

        if torch.isnan(self.running_min).any():
            self.running_min.copy_(frame_lo)
            self.running_max.copy_(frame_hi)
        else:
            torch.minimum(self.running_min, frame_lo, out=self.running_min)
            torch.maximum(self.running_max, frame_hi, out=self.running_max)

    if torch.isnan(self.running_min).any():
        raise RuntimeError(f"{type(self).__name__}.statistical_initialization received no data")
    self._statistically_initialized = True
```

`ClassMapRobustifier`cuvis_ai.node.mask_opstransformclassmasknumpypostsegPer-class morphological cleanup of an integer label map.

#### ClassMapRobustifier

```python
ClassMapRobustifier(
    opening_kernel=0,
    closing_kernel=3,
    min_area=10,
    keep_largest=True,
    background_value=-1,
    **kwargs,
)
```

Bases: `Node`

Per-class morphological cleanup of an integer label map.

Runs :class:`MaskRobustifier` independently on the binary mask of each class present in `class_map` (despeckle + close + min-area / largest-component filter), then repaints the survivors into one label map. Classes are painted in ascending surviving-area order, so a larger class wins any pixel an overlapping smaller class also kept. Pixels removed as speckle become `background_value` (holes); :class:`NearestLabelFill` is the companion node that fills them. The input map is echoed verbatim on the `source` port so the fill node has both the foreground extent and the fallback labels.

Parameters:

| Name               | Type   | Description                                                                                     | Default |
| ------------------ | ------ | ----------------------------------------------------------------------------------------------- | ------- |
| `opening_kernel`   | `int`  | Morphological opening kernel for the internal MaskRobustifier. 0/1 disables opening. Default 0. | `0`     |
| `closing_kernel`   | `int`  | Morphological closing kernel. 0/1 disables. Default 3.                                          | `3`     |
| `min_area`         | `int`  | Drop per-class connected components smaller than this. 0 disables. Default 10.                  | `10`    |
| `keep_largest`     | `bool` | Keep only the largest surviving component of each class. Default True.                          | `True`  |
| `background_value` | `int`  | Label value for unassigned pixels in the output. Default -1.                                    | `-1`    |

Source code in `cuvis_ai/node/mask_ops.py`

```python
def __init__(
    self,
    opening_kernel: int = 0,
    closing_kernel: int = 3,
    min_area: int = 10,
    keep_largest: bool = True,
    background_value: int = -1,
    **kwargs: Any,
) -> None:
    self.opening_kernel = int(opening_kernel)
    self.closing_kernel = int(closing_kernel)
    self.min_area = int(min_area)
    self.keep_largest = bool(keep_largest)
    self.background_value = int(background_value)

    super().__init__(
        opening_kernel=self.opening_kernel,
        closing_kernel=self.closing_kernel,
        min_area=self.min_area,
        keep_largest=self.keep_largest,
        background_value=self.background_value,
        **kwargs,
    )

    # Assigned after super().__init__ so nn.Module is initialised before this
    # submodule is registered. MaskRobustifier validates its own kwargs (so a
    # negative kernel/area raises here at construction).
    self._robust = MaskRobustifier(
        opening_kernel=self.opening_kernel,
        closing_kernel=self.closing_kernel,
        min_area=self.min_area,
        keep_largest=self.keep_largest,
    )
```

##### forward

```python
forward(class_map, **_)
```

Clean each present class with morphology, then repaint area-sorted into one map.

Source code in `cuvis_ai/node/mask_ops.py`

```python
@torch.no_grad()
def forward(self, class_map: torch.Tensor, **_: Any) -> dict[str, torch.Tensor]:
    """Clean each present class with morphology, then repaint area-sorted into one map."""
    bg = self.background_value
    out = torch.full_like(class_map, bg)
    for b in range(class_map.shape[0]):
        pred = class_map[b]
        present = [int(c) for c in torch.unique(pred).tolist() if int(c) != bg]
        surv: dict[int, torch.Tensor] = {}
        for c in present:
            binary = (pred == c).to(torch.int32).unsqueeze(0)
            surv[c] = self._robust.forward(mask=binary)["mask"][0] > 0
        # Larger classes painted last -> they win pixels a smaller class also kept.
        for c in sorted(present, key=lambda cc: int(surv[cc].sum())):
            out[b][surv[c]] = c
    return {"class_map": out, "source": class_map.clone()}
```

`ContinuumRemoval`cuvis_ai.node.pretreatments.continuum_removaltransformhsipretorchPer-pixel continuum (upper convex hull) removal across the spectral axis.

#### ContinuumRemoval

```python
ContinuumRemoval(eps=1e-08, **kwargs)
```

Bases: `Node`

Per-pixel continuum (upper convex hull) removal across the spectral axis.

For every spectrum the upper convex hull over `(wavelength, reflectance)` is computed, linearly interpolated to all bands, and the spectrum is divided by it. Continuum-free regions map to `~1.0` while absorption bands dip below `1.0`, making feature depths comparable across spectra.

The hull is built with a batched Andrew monotone-chain scan that runs entirely in `torch` (vectorized over pixels), so the node stays device-agnostic.

Parameters:

| Name  | Type    | Description                                                                                 | Default |
| ----- | ------- | ------------------------------------------------------------------------------------------- | ------- |
| `eps` | `float` | Lower clamp on the hull before division, guarding against division by zero (default: 1e-8). | `1e-08` |

Source code in `cuvis_ai/node/pretreatments/continuum_removal.py`

```python
def __init__(self, eps: float = 1e-8, **kwargs) -> None:
    self.eps = float(eps)
    super().__init__(eps=self.eps, **kwargs)
```

##### forward

```python
forward(cube, wavelengths, **_)
```

Divide each spectrum by its upper convex hull.

Parameters:

| Name          | Type           | Description                 | Default    |
| ------------- | -------------- | --------------------------- | ---------- |
| `cube`        | `Tensor`       | Input cube in BHWC format.  | *required* |
| `wavelengths` | `array - like` | Band wavelengths, length C. | *required* |

Returns:

| Type                | Description                                                   |
| ------------------- | ------------------------------------------------------------- |
| `dict[str, Tensor]` | {"cube": continuum_removed} with the same shape as the input. |

Source code in `cuvis_ai/node/pretreatments/continuum_removal.py`

```python
def forward(self, cube: torch.Tensor, wavelengths, **_) -> dict[str, torch.Tensor]:
    """Divide each spectrum by its upper convex hull.

    Parameters
    ----------
    cube : torch.Tensor
        Input cube in BHWC format.
    wavelengths : array-like
        Band wavelengths, length ``C``.

    Returns
    -------
    dict[str, torch.Tensor]
        ``{"cube": continuum_removed}`` with the same shape as the input.
    """
    B, H, W, C = cube.shape
    x = torch.as_tensor(np.asarray(wavelengths), device=cube.device).reshape(-1)
    x = x.to(cube.dtype)
    spectra = cube.reshape(-1, C)
    hull = self._upper_hull(x, spectra)
    removed = spectra / hull.clamp_min(self.eps)
    return {"cube": removed.reshape(B, H, W, C)}
```

`Crop`cuvis_ai_augment.node.croptransformhsipretorchaugment[↗](https://github.com/cubert-hyperspectral/cuvis-ai-augment "Open plugin repo")Deterministic fixed-rectangle spatial crop of a cube and optional paired mask, applied identically at every execution stage.

Deterministic fixed-rectangle spatial crop of a cube and optional paired mask, applied identically at every execution stage.

Inputs

| Port            | Dtype     | Shape              | Description                                             |
| --------------- | --------- | ------------------ | ------------------------------------------------------- |
| `data`          | `float32` | `[-1, -1, -1, -1]` | Cube [B, H, W, C] in float32                            |
| `mask` optional | `int32`   | `[-1, -1, -1]`     | Optional per-pixel mask [B, H, W] (cropped identically) |

Outputs

| Port                    | Dtype     | Shape              | Description                                              |
| ----------------------- | --------- | ------------------ | -------------------------------------------------------- |
| `cropped`               | `float32` | `[-1, -1, -1, -1]` | Cropped cube [B, H', W', C]                              |
| `mask_cropped` optional | `int32`   | `[-1, -1, -1]`     | Cropped mask [B, H', W'] (only when a mask is connected) |

[View plugin repo (v0.4.2)](https://github.com/cubert-hyperspectral/cuvis-ai-augment/tree/v0.4.2)

`DecisionToMask`cuvis_ai.node.conversiontransformmaskposttorchCombine binary decisions and identity labels into a single int32 mask.

#### DecisionToMask

Bases: `Node`

Combine binary decisions and identity labels into a single int32 mask.

The output mask keeps per-pixel identity IDs where the decision is True and sets all non-matching pixels to 0.

##### forward

```python
forward(decisions, identity_mask, **_)
```

Apply decisions to identities and return the final segmentation mask.

Source code in `cuvis_ai/node/conversion.py`

```python
@torch.no_grad()
def forward(
    self,
    decisions: torch.Tensor,
    identity_mask: torch.Tensor,
    **_,
) -> dict[str, torch.Tensor]:
    """Apply decisions to identities and return the final segmentation mask."""
    mask = identity_mask.to(torch.int32) * decisions.squeeze(-1).to(torch.int32)
    return {"mask": mask}
```

`DeepEIoUTrack`cuvis_ai_deepeiou.nodetransformbboxinferstatefultorchtrackdeepeiou[↗](https://github.com/cubert-hyperspectral/cuvis-ai-deepeiou "Open plugin repo")DeepEIoU multi-object tracker node.

DeepEIoU multi-object tracker node.

Inputs

| Port                  | Dtype     | Shape         | Description                                                                                |
| --------------------- | --------- | ------------- | ------------------------------------------------------------------------------------------ |
| `bboxes`              | `float32` | `[1, -1, 4]`  | Detection bounding boxes [1, N, 4] xyxy pixel coordinates.                                 |
| `category_ids`        | `int64`   | `[1, -1]`     | Detection category IDs [1, N].                                                             |
| `confidences`         | `float32` | `[1, -1]`     | Detection confidence scores [1, N].                                                        |
| `embeddings` optional | `float32` | `[1, -1, -1]` | Per-detection ReID embeddings [1, N, D]. Optional; if absent, tracker uses EIoU-only mode. |

Outputs

| Port           | Dtype     | Shape        | Description                                                                                  |
| -------------- | --------- | ------------ | -------------------------------------------------------------------------------------------- |
| `bboxes`       | `float32` | `[1, -1, 4]` | Input bboxes pass-through [1, N, 4] xyxy pixel coordinates.                                  |
| `track_ids`    | `int64`   | `[1, -1]`    | Track IDs aligned with input detections [1, N]. -1 for detections not assigned to any track. |
| `confidences`  | `float32` | `[1, -1]`    | Input confidences pass-through [1, N].                                                       |
| `category_ids` | `int64`   | `[1, -1]`    | Input category_ids pass-through [1, N].                                                      |

[View plugin repo (v0.2.2)](https://github.com/cubert-hyperspectral/cuvis-ai-deepeiou/tree/v0.2.2)

`DeepSVDDCenterTracker`cuvis_ai.node.anomaly.deep_svddtransformanomstatefultorchtrainTrack and expose Deep SVDD center statistics with optional logging.

#### DeepSVDDCenterTracker

```python
DeepSVDDCenterTracker(
    *, rep_dim, alpha=0.1, update_in_eval=False, **kwargs
)
```

Bases: `Node`

Track and expose Deep SVDD center statistics with optional logging.

Source code in `cuvis_ai/node/anomaly/deep_svdd.py`

```python
def __init__(
    self, *, rep_dim: int, alpha: float = 0.1, update_in_eval: bool = False, **kwargs
) -> None:
    if rep_dim <= 0:
        raise ValueError(f"rep_dim must be positive, got {rep_dim}")
    if not (0.0 < alpha <= 1.0):
        raise ValueError("alpha must be in (0, 1]")
    self.rep_dim = int(rep_dim)
    self.alpha = float(alpha)
    self.update_in_eval = bool(update_in_eval)

    super().__init__(
        rep_dim=self.rep_dim, alpha=self.alpha, update_in_eval=self.update_in_eval, **kwargs
    )

    # Pre-allocate buffer with known dimensions
    self.register_buffer(
        "_tracked_center", torch.zeros(rep_dim, dtype=torch.get_default_dtype())
    )
```

##### requires_initial_fit

```python
requires_initial_fit
```

Whether this node requires statistical initialization from training data.

Returns:

| Type   | Description                                     |
| ------ | ----------------------------------------------- |
| `bool` | Always True for center tracking initialization. |

##### statistical_initialization

```python
statistical_initialization(input_stream)
```

Initialize the Deep SVDD center from training embeddings.

Computes the mean embedding across all training samples to initialize the hypersphere center.

Parameters:

| Name           | Type          | Description                                        | Default    |
| -------------- | ------------- | -------------------------------------------------- | ---------- |
| `input_stream` | `InputStream` | Training data stream with embeddings [B, H, W, D]. | *required* |

Raises:

| Type           | Description                                              |
| -------------- | -------------------------------------------------------- |
| `RuntimeError` | If no embeddings are received from the input stream.     |
| `ValueError`   | If embedding dimensions don't match initialized rep_dim. |

Source code in `cuvis_ai/node/anomaly/deep_svdd.py`

```python
def statistical_initialization(self, input_stream: InputStream) -> None:
    """Initialize the Deep SVDD center from training embeddings.

    Computes the mean embedding across all training samples to initialize
    the hypersphere center.

    Parameters
    ----------
    input_stream : InputStream
        Training data stream with embeddings [B, H, W, D].

    Raises
    ------
    RuntimeError
        If no embeddings are received from the input stream.
    ValueError
        If embedding dimensions don't match initialized rep_dim.
    """
    total = None
    count = 0
    for batch in input_stream:
        embeddings = batch.get("embeddings")
        if embeddings is None:
            embeddings = batch.get("data")
        if embeddings is None:
            continue

        # Validate dimensions
        if embeddings.shape[-1] != self.rep_dim:
            raise ValueError(
                f"Embedding dimension mismatch: expected {self.rep_dim}, got {embeddings.shape[-1]}"
            )

        flat = embeddings.reshape(-1, embeddings.shape[-1])
        batch_sum = flat.sum(dim=0)
        total = batch_sum if total is None else total + batch_sum
        count += flat.shape[0]

    if total is None or count == 0:
        raise RuntimeError(
            "DeepSVDDCenterTracker.statistical_initialization() received no embeddings"
        )

    self._tracked_center.copy_((total / count).detach())
    self._statistically_initialized = True
```

##### forward

```python
forward(embeddings, context=None, **_)
```

Track and output the Deep SVDD center with exponential moving average.

Updates the center using EMA during training (and optionally during eval), then outputs the current center and center norm metric.

Parameters:

| Name         | Type      | Description                                             | Default    |
| ------------ | --------- | ------------------------------------------------------- | ---------- |
| `embeddings` | `Tensor`  | Deep SVDD embeddings [B, H, W, D].                      | *required* |
| `context`    | `Context` | Execution context determining whether to update center. | `None`     |
| `**_`        | `Any`     | Additional unused keyword arguments.                    | `{}`       |

Returns:

| Type             | Description                                                                                                         |
| ---------------- | ------------------------------------------------------------------------------------------------------------------- |
| `dict[str, Any]` | Dictionary with: "center" : torch.Tensor [D] - Current tracked center "metrics" : list[Metric] - Center norm metric |

Raises:

| Type           | Description                                              |
| -------------- | -------------------------------------------------------- |
| `RuntimeError` | If statistical_initialization() has not been called.     |
| `ValueError`   | If embedding dimensions don't match initialized rep_dim. |

Source code in `cuvis_ai/node/anomaly/deep_svdd.py`

```python
def forward(
    self, embeddings: torch.Tensor, context: Context | None = None, **_: Any
) -> dict[str, Any]:
    """Track and output the Deep SVDD center with exponential moving average.

    Updates the center using EMA during training (and optionally during eval),
    then outputs the current center and center norm metric.

    Parameters
    ----------
    embeddings : torch.Tensor
        Deep SVDD embeddings [B, H, W, D].
    context : Context, optional
        Execution context determining whether to update center.
    **_ : Any
        Additional unused keyword arguments.

    Returns
    -------
    dict[str, Any]
        Dictionary with:

        - "center" : torch.Tensor [D] - Current tracked center
        - "metrics" : list[Metric] - Center norm metric

    Raises
    ------
    RuntimeError
        If statistical_initialization() has not been called.
    ValueError
        If embedding dimensions don't match initialized rep_dim.
    """
    if not self._statistically_initialized:
        raise RuntimeError(
            "DeepSVDDCenterTracker requires statistical_initialization() before forward()"
        )

    # Validate dimensions
    if embeddings.shape[-1] != self.rep_dim:
        raise ValueError(
            f"Embedding dimension mismatch: expected {self.rep_dim}, got {embeddings.shape[-1]}"
        )

    batch_mean = embeddings.mean(dim=(0, 1, 2)).detach()
    should_update = context is None or context.stage is ExecutionStage.TRAIN
    if not should_update and self.update_in_eval and context is not None:
        should_update = context.stage in {ExecutionStage.VAL, ExecutionStage.TEST}

    if should_update:
        self._tracked_center.copy_(
            (1.0 - self.alpha) * self._tracked_center + self.alpha * batch_mean
        )

    metrics = []
    center_cpu = self._tracked_center.detach().cpu()
    metrics.append(
        Metric(
            name="deepsvdd_center/norm",
            value=float(center_cpu.norm().item()),
            stage=context.stage if context else ExecutionStage.INFERENCE,
            epoch=context.epoch if context else 0,
            batch_idx=context.batch_idx if context else 0,
        )
    )

    center_value = self._tracked_center.detach().clone()
    return {"center": center_value, "metrics": metrics}
```

`DisplayNormalizer`cuvis_ai.node.normalizationtransformnormnumpypreApply sRGB gamma companding (IEC 61966-2-1) to a `[0, 1]` BHWC tensor.

#### DisplayNormalizer

```python
DisplayNormalizer(*args, **kwargs)
```

Bases: `_ScoreNormalizerBase`

Apply sRGB gamma companding (IEC 61966-2-1) to a `[0, 1]` BHWC tensor.

The stateless display-encoding companion to :class:`PercentileNormalizer`: chain it after the normalizer on the false-RGB display path (`selector -> PercentileNormalizer -> DisplayNormalizer`) to lift midtones so images look natural on standard displays. ML / n-channel paths skip it.

Ports

INPUT_SPECS `data` : float32, shape (-1, -1, -1, -1), BHWC tensor, values in `[0, 1]`. OUTPUT_SPECS `normalized` : float32, shape (-1, -1, -1, -1), sRGB gamma-encoded, `[0, 1]`.

Source code in `cuvis_ai/node/normalization.py`

```python
def __init__(self, *args, **kwargs) -> None:
    super().__init__(*args, **kwargs)
```

`EVI2Selector`cuvis_ai.node.channel_selectortransformdim-redhsinumpypreTwo-band Enhanced Vegetation Index renderer.

#### EVI2Selector

```python
EVI2Selector(
    red_nm=660.0,
    nir_nm=800.0,
    colormap_min=-1.0,
    colormap_max=1.0,
    eps=1e-06,
    **kwargs,
)
```

Bases: `_VegetationIndexBase`

Two-band Enhanced Vegetation Index renderer.

Computes `2.5 * (NIR - Red) / (NIR + 2.4 * Red + 1)` over bands resolved by nearest sensor wavelength. The raw index map is returned via `index_image` and `rgb_image` carries an HSV colour-mapped render.

The additive `+1` constant is only meaningful for reflectance in [0, 1]; feed reflectance-calibrated cubes, not raw radiance/DN.

Source code in `cuvis_ai/node/channel_selector.py`

```python
def __init__(
    self,
    red_nm: float = 660.0,
    nir_nm: float = 800.0,
    colormap_min: float = -1.0,
    colormap_max: float = 1.0,
    eps: float = 1.0e-6,
    **kwargs: Any,
) -> None:
    super().__init__(
        band_nm={"red": red_nm, "nir": nir_nm},
        colormap_min=colormap_min,
        colormap_max=colormap_max,
        eps=eps,
        red_nm=float(red_nm),
        nir_nm=float(nir_nm),
        **kwargs,
    )
    self.red_nm = float(red_nm)
    self.nir_nm = float(nir_nm)
```

##### index_name

```python
index_name
```

Canonical EVI2 strategy name.

`EVISelector`cuvis_ai.node.channel_selectortransformdim-redhsinumpypreEnhanced Vegetation Index renderer.

#### EVISelector

```python
EVISelector(
    blue_nm=460.0,
    red_nm=660.0,
    nir_nm=800.0,
    colormap_min=-1.0,
    colormap_max=1.0,
    eps=1e-06,
    **kwargs,
)
```

Bases: `_VegetationIndexBase`

Enhanced Vegetation Index renderer.

Computes `2.5 * (NIR - Red) / (NIR + 6 * Red - 7.5 * Blue + 1)` over bands resolved by nearest sensor wavelength. The raw index map is returned via `index_image` and `rgb_image` carries an HSV colour-mapped render.

The additive `+1` constant is only meaningful for reflectance in [0, 1]; feed reflectance-calibrated cubes, not raw radiance/DN.

Source code in `cuvis_ai/node/channel_selector.py`

```python
def __init__(
    self,
    blue_nm: float = 460.0,
    red_nm: float = 660.0,
    nir_nm: float = 800.0,
    colormap_min: float = -1.0,
    colormap_max: float = 1.0,
    eps: float = 1.0e-6,
    **kwargs: Any,
) -> None:
    super().__init__(
        band_nm={"blue": blue_nm, "red": red_nm, "nir": nir_nm},
        colormap_min=colormap_min,
        colormap_max=colormap_max,
        eps=eps,
        blue_nm=float(blue_nm),
        red_nm=float(red_nm),
        nir_nm=float(nir_nm),
        **kwargs,
    )
    self.blue_nm = float(blue_nm)
    self.red_nm = float(red_nm)
    self.nir_nm = float(nir_nm)
```

##### index_name

```python
index_name
```

Canonical EVI strategy name.

`FastRGBSelector`cuvis_ai.node.channel_selectortransformdim-redhsinumpyprecuvis-next parity FastRGB renderer.

#### FastRGBSelector

```python
FastRGBSelector(
    red_range=(580.0, 650.0),
    green_range=(500.0, 580.0),
    blue_range=(420.0, 500.0),
    normalization_strength=0.75,
    **kwargs,
)
```

Bases: `ChannelSelectorBase`

cuvis-next parity FastRGB renderer.

This selector mirrors the cuvis `fast_rgb` user-plugin behavior:

- Per-channel contiguous spectral range averaging.
- Dynamic per-frame normalization by global RGB mean when enabled.
- Static reflectance-style scaling when normalization is disabled.
- 8-bit quantization before returning float RGB in [0, 1].

Source code in `cuvis_ai/node/channel_selector.py`

```python
def __init__(
    self,
    red_range: tuple[float, float] = (580.0, 650.0),
    green_range: tuple[float, float] = (500.0, 580.0),
    blue_range: tuple[float, float] = (420.0, 500.0),
    normalization_strength: float = 0.75,
    **kwargs: Any,
) -> None:
    for name, rng in {
        "red_range": red_range,
        "green_range": green_range,
        "blue_range": blue_range,
    }.items():
        if len(rng) != 2 or rng[0] > rng[1]:
            raise ValueError(f"{name} must be (min_nm, max_nm) with min_nm <= max_nm")

    # FastRGB has its own scaling path; disable base normalization/gamma.
    kwargs.pop("norm_mode", None)
    kwargs.pop("apply_gamma", None)
    super().__init__(
        norm_mode=NormMode.PER_FRAME,
        apply_gamma=False,
        red_range=red_range,
        green_range=green_range,
        blue_range=blue_range,
        normalization_strength=float(normalization_strength),
        **kwargs,
    )
    self.red_range = red_range
    self.green_range = green_range
    self.blue_range = blue_range
    self.normalization_strength = float(normalization_strength)

    self.register_buffer(
        "_ranges",
        torch.tensor(
            [
                [red_range[0], red_range[1]],
                [green_range[0], green_range[1]],
                [blue_range[0], blue_range[1]],
            ],
            dtype=torch.float32,
        ),
    )
    self.register_buffer("_channel_bounds", None, persistent=False)
    self.register_buffer("_channel_valid", None, persistent=False)
    self._cached_wl_key: tuple[float, ...] | None = None
```

##### forward

```python
forward(cube, wavelengths, context=None, **_)
```

Render fast_rgb output with cuvis-next parity scaling.

Source code in `cuvis_ai/node/channel_selector.py`

```python
def forward(
    self,
    cube: torch.Tensor,
    wavelengths: Any,
    context: Context | None = None,  # noqa: ARG002
    **_: Any,
) -> dict[str, Any]:
    """Render fast_rgb output with cuvis-next parity scaling."""
    wavelengths_t = self._prepare_wavelengths_tensor(wavelengths, device=cube.device)
    raw_rgb = self._compute_raw_rgb(cube, wavelengths_t)
    rgb, factor = self._fast_rgb_scale(raw_rgb)

    channel_indices: list[list[int]] = []
    missing_channels: list[str] = []
    channel_names = ["red", "green", "blue"]
    for c in range(3):
        if bool(self._channel_valid[c].item()):
            low = int(self._channel_bounds[c, 0].item())
            high = int(self._channel_bounds[c, 1].item())
            channel_indices.append(list(range(low, high + 1)))
        else:
            channel_indices.append([])
            missing_channels.append(channel_names[c])

    band_info = {
        "strategy": "fast_rgb",
        "band_indices": channel_indices,
        "band_wavelengths_nm": [wavelengths_t[idxs].tolist() for idxs in channel_indices],
        "ranges_nm": {
            "red": [float(self.red_range[0]), float(self.red_range[1])],
            "green": [float(self.green_range[0]), float(self.green_range[1])],
            "blue": [float(self.blue_range[0]), float(self.blue_range[1])],
        },
        "aggregation": "mean",
        "normalization_strength": float(self.normalization_strength),
        "applied_scale_factor": float(factor),
        "missing_channels": missing_channels,
    }
    return {"rgb_image": rgb, "band_info": band_info}
```

`FixedWavelengthSelector`cuvis_ai.node.channel_selectortransformdim-redhsinumpypreFixed wavelength band selection — picks the nearest band for each target wavelength.

#### FixedWavelengthSelector

```python
FixedWavelengthSelector(
    target_wavelengths=(650.0, 550.0, 450.0),
    normalize_output=True,
    **kwargs,
)
```

Bases: `ChannelSelectorBase`

Fixed wavelength band selection — picks the nearest band for each target wavelength.

For the standard 3-channel case (default) this produces a "true color-ish" RGB image. For `n > 3` target wavelengths the node stacks `n` bands into a `[B, H, W, n]` output — useful for multi-channel hyperspectral models (e.g. a 6-channel VIS+SWIR input to a Dinomaly detector).

Parameters:

| Name                 | Type                | Description                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  | Default                 |
| -------------------- | ------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------- |
| `target_wavelengths` | `tuple[float, ...]` | Target wavelengths in nanometers, in the order they should be stacked. Must contain at least one wavelength. Default: (650.0, 550.0, 450.0) — standard false-RGB.                                                                                                                                                                                                                                                                                                                                                                                                                                                            | `(650.0, 550.0, 450.0)` |
| `normalize_output`   | `bool`              | If True (default), apply selector normalization to produce a 0–1 output. Normalization (running bounds + optional sRGB gamma) is only available for the 3-channel case (len(target_wavelengths) == 3). For n != 3 the bands are stacked raw regardless of this flag; a single warning is emitted at construction so callers know to pass normalize_output=False to silence it. norm_mode settings of running / statistical rely on 3-element running buffers and a hard reshape(-1, 3) and are therefore rejected at construction for n != 3 — pass norm_mode="per_frame" (the only mode the n-channel path supports today). | `True`                  |

Ports

OUTPUT_SPECS `rgb_image` : float32, shape `(-1, -1, -1, -1)` Stacked selected bands `[B, H, W, len(target_wavelengths)]`. Port name kept as `rgb_image` for graph compatibility with downstream consumers. For the 3-channel default the output is the normalised RGB image; for `n != 3` it is the raw stacked bands. `band_info` : dict See `forward` for keys.

Source code in `cuvis_ai/node/channel_selector.py`

```python
def __init__(
    self,
    target_wavelengths: tuple[float, ...] = (650.0, 550.0, 450.0),
    normalize_output: bool = True,
    **kwargs: Any,
) -> None:
    target_wavelengths = tuple(float(w) for w in target_wavelengths)
    if len(target_wavelengths) < 1:
        raise ValueError(
            "FixedWavelengthSelector: target_wavelengths must contain at least one wavelength"
        )
    n = len(target_wavelengths)

    # The base class's running/statistical normalisation machinery assumes a
    # 3-channel output: running_min/running_max are 3-element buffers, and
    # statistical_initialization / _running_normalize do `reshape(-1, 3)`.
    # For n != 3 those modes either silently mix unrelated channels (e.g.
    # n=6 reshapes into (-1, 3) and pairs c0+c3, c1+c4, c2+c5 per row), or
    # raise RuntimeError on non-divisible totals (n=4). Reject them at
    # construction so the failure mode is obvious. ``per_frame`` (which
    # operates only on the forward path's branch below) is the only mode
    # supported for the n != 3 case today.
    norm_mode_arg = kwargs.get("norm_mode", NormMode.RUNNING)
    norm_mode_resolved = NormMode(
        str(norm_mode_arg) if isinstance(norm_mode_arg, NormMode) else norm_mode_arg
    )
    if n != 3 and norm_mode_resolved in (NormMode.STATISTICAL, NormMode.RUNNING):
        raise ValueError(
            f"FixedWavelengthSelector: norm_mode={norm_mode_resolved.value!r} requires "
            f"exactly 3 target wavelengths (got n={n}). The running/statistical paths "
            f"rely on 3-element buffers and reshape(-1, 3). Pass "
            f"norm_mode='per_frame' (the only mode supported for n != 3) or use "
            f"len(target_wavelengths) == 3."
        )

    super().__init__(
        target_wavelengths=target_wavelengths,
        normalize_output=normalize_output,
        **kwargs,
    )
    self.target_wavelengths = target_wavelengths
    self.normalize_output = bool(normalize_output)

    # Whether the forward path will actually normalise. The band_info flag
    # downstream MUST mirror this — `normalize_output=True` with n != 3
    # silently returns raw bands, so any consumer that trusts the flag
    # would skip a normalisation it still needs.
    self._effective_normalize_output = self.normalize_output and n == 3

    # Warn ONCE at construction (not per forward call — n and
    # normalize_output are both known here and don't change later).
    if self.normalize_output and n != 3:
        logger.warning(
            "FixedWavelengthSelector: normalize_output=True is only supported for "
            "3-channel selection; got {} target wavelengths. The forward path will "
            "return raw stacked bands; pass normalize_output=False to suppress this "
            "warning.",
            n,
        )
```

##### forward

```python
forward(cube, wavelengths, context=None, **_)
```

Select bands and compose RGB image.

Parameters:

| Name          | Type     | Description                      | Default    |
| ------------- | -------- | -------------------------------- | ---------- |
| `cube`        | `Tensor` | Hyperspectral cube [B, H, W, C]. | *required* |
| `wavelengths` | `Tensor` | Wavelength array [C].            | *required* |

Returns:

| Type             | Description                                       |
| ---------------- | ------------------------------------------------- |
| `dict[str, Any]` | Dictionary with "rgb_image" and "band_info" keys. |

Source code in `cuvis_ai/node/channel_selector.py`

```python
def forward(
    self,
    cube: torch.Tensor,
    wavelengths: Any,
    context: Context | None = None,  # noqa: ARG002
    **_: Any,
) -> dict[str, Any]:
    """Select bands and compose RGB image.

    Parameters
    ----------
    cube : torch.Tensor
        Hyperspectral cube [B, H, W, C].
    wavelengths : torch.Tensor
        Wavelength array [C].

    Returns
    -------
    dict[str, Any]
        Dictionary with "rgb_image" and "band_info" keys.
    """
    wavelengths_np = np.asarray(wavelengths, dtype=np.float32)

    # Find nearest bands
    indices = [self._nearest_band_index(wavelengths_np, nm) for nm in self.target_wavelengths]

    # 3-channel path goes through the (running / statistical / per-frame)
    # normalisation machinery in ChannelSelectorBase. For n != 3 we always
    # stack raw bands — the construction-time guard guarantees we only see
    # `per_frame` mode here, and the n != 3 + normalize_output=True warning
    # was emitted once in __init__.
    if self._effective_normalize_output:
        rgb = self._compose_rgb(cube, indices)
    else:
        bands = [cube[..., idx] for idx in indices]
        rgb = torch.stack(bands, dim=-1)

    n = len(self.target_wavelengths)
    band_info = {
        "strategy": "baseline_false_rgb" if n == 3 else "stacked_bands",
        "band_indices": indices,
        "band_wavelengths_nm": [float(wavelengths_np[i]) for i in indices],
        "target_wavelengths_nm": list(self.target_wavelengths),
        # Mirror what actually happened — `normalize_output=True` with
        # n != 3 returns raw bands, so this MUST be False there.
        "normalized_output": self._effective_normalize_output,
    }

    return {"rgb_image": rgb, "band_info": band_info}
```

`GNDVISelector`cuvis_ai.node.channel_selectortransformdim-redhsinumpypreGreen Normalized Difference Vegetation Index renderer.

#### GNDVISelector

```python
GNDVISelector(
    nir_nm=800.0,
    green_nm=550.0,
    colormap_min=-1.0,
    colormap_max=1.0,
    eps=1e-06,
    **kwargs,
)
```

Bases: `_ColormappedNormalizedDifferenceSelector`

Green Normalized Difference Vegetation Index renderer.

Computes `(CUBE(nir_nm) - CUBE(green_nm)) / (CUBE(nir_nm) + CUBE(green_nm))`. Bands are resolved by nearest available sensor wavelength. The raw index map is returned via `index_image` and `rgb_image` carries an HSV colour-mapped render.

Source code in `cuvis_ai/node/channel_selector.py`

```python
def __init__(
    self,
    nir_nm: float = 800.0,
    green_nm: float = 550.0,
    colormap_min: float = -1.0,
    colormap_max: float = 1.0,
    eps: float = 1.0e-6,
    **kwargs: Any,
) -> None:
    super().__init__(
        primary_nm=nir_nm,
        secondary_nm=green_nm,
        colormap_min=colormap_min,
        colormap_max=colormap_max,
        eps=eps,
        nir_nm=float(nir_nm),
        green_nm=float(green_nm),
        **kwargs,
    )
    self.nir_nm = float(nir_nm)
    self.green_nm = float(green_nm)
```

##### index_name

```python
index_name
```

Canonical GNDVI strategy name.

##### primary_label

```python
primary_label
```

GNDVI primary operand label.

##### secondary_label

```python
secondary_label
```

GNDVI secondary operand label.

`HighContrastSelector`cuvis_ai.node.channel_selectortransformdim-redhsinumpypreData-driven band selection using spatial variance + Laplacian energy.

#### HighContrastSelector

```python
HighContrastSelector(
    windows=((440, 500), (500, 580), (610, 700)),
    alpha=0.1,
    **kwargs,
)
```

Bases: `ChannelSelectorBase`

Data-driven band selection using spatial variance + Laplacian energy.

For each wavelength window, selects the band with the highest score based on: score = variance + alpha * Laplacian_energy

This produces "high contrast" images that may work better for visual anomaly detection.

Parameters:

| Name      | Type                            | Description                                                                                                           | Default                                |
| --------- | ------------------------------- | --------------------------------------------------------------------------------------------------------------------- | -------------------------------------- |
| `windows` | `Sequence[tuple[float, float]]` | Wavelength windows for Blue, Green, Red channels. Default: ((440, 500), (500, 580), (610, 700)) for visible spectrum. | `((440, 500), (500, 580), (610, 700))` |
| `alpha`   | `float`                         | Weight for Laplacian energy term. Default: 0.1                                                                        | `0.1`                                  |

Source code in `cuvis_ai/node/channel_selector.py`

```python
def __init__(
    self,
    windows: Sequence[tuple[float, float]] = ((440, 500), (500, 580), (610, 700)),
    alpha: float = 0.1,
    **kwargs,
) -> None:
    super().__init__(windows=windows, alpha=alpha, **kwargs)
    self.windows = list(windows)
    self.alpha = alpha
```

##### forward

```python
forward(cube, wavelengths, context=None, **_)
```

Select high-contrast bands and compose RGB image.

Parameters:

| Name          | Type     | Description                      | Default    |
| ------------- | -------- | -------------------------------- | ---------- |
| `cube`        | `Tensor` | Hyperspectral cube [B, H, W, C]. | *required* |
| `wavelengths` | `Tensor` | Wavelength array [C].            | *required* |

Returns:

| Type             | Description                                       |
| ---------------- | ------------------------------------------------- |
| `dict[str, Any]` | Dictionary with "rgb_image" and "band_info" keys. |

Source code in `cuvis_ai/node/channel_selector.py`

```python
def forward(
    self,
    cube: torch.Tensor,
    wavelengths: Any,
    context: Context | None = None,  # noqa: ARG002
    **_: Any,
) -> dict[str, Any]:
    """Select high-contrast bands and compose RGB image.

    Parameters
    ----------
    cube : torch.Tensor
        Hyperspectral cube [B, H, W, C].
    wavelengths : torch.Tensor
        Wavelength array [C].

    Returns
    -------
    dict[str, Any]
        Dictionary with "rgb_image" and "band_info" keys.
    """
    wavelengths_np = np.asarray(wavelengths, dtype=np.float32)
    selected_indices = self._select_indices(cube, wavelengths_np)
    rgb = self._compose_rgb(cube, selected_indices)

    band_info = {
        "strategy": "high_contrast",
        "band_indices": selected_indices,
        "band_wavelengths_nm": [float(wavelengths_np[i]) for i in selected_indices],
        "windows_nm": [[float(s), float(e)] for s, e in self.windows],
        "alpha": self.alpha,
    }

    return {"rgb_image": rgb, "band_info": band_info}
```

`IdentityNormalizer`cuvis_ai.node.normalizationtransformnormnumpypreNo-op normalizer; preserves incoming scores.

#### IdentityNormalizer

```python
IdentityNormalizer(**kwargs)
```

Bases: `_ScoreNormalizerBase`

No-op normalizer; preserves incoming scores.

Source code in `cuvis_ai/node/normalization.py`

```python
def __init__(self, **kwargs) -> None:
    super().__init__(**kwargs)
```

`ImageConcatenator`cuvis_ai.node.compositingtransformimgrgbConcatenate several RGB frames into one side-by-side or stacked strip.

#### ImageConcatenator

```python
ImageConcatenator(
    axis="horizontal",
    gap=0,
    bg_color=(1.0, 1.0, 1.0),
    align="center",
    **kwargs,
)
```

Bases: `Node`

Concatenate several RGB frames into one side-by-side or stacked strip.

A fan-in node: connect any number of `rgb_image` sources to the single `images` port and they are concatenated in **connection order** (the order the edges were added with `pipeline.connect`). Frames may differ on the cross axis (height for a horizontal strip, width for a vertical one); each is padded to the common size with `bg_color` and aligned per `align`. An optional `gap` inserts a `bg_color` separator between frames. The whole batch is concatenated together, so every source must share the same batch size.

Parameters:

| Name       | Type                         | Description                                                                                                                           | Default           |
| ---------- | ---------------------------- | ------------------------------------------------------------------------------------------------------------------------------------- | ----------------- |
| `axis`     | `str`                        | "horizontal" places frames left-to-right (pads heights); "vertical" stacks them top-to-bottom (pads widths). Default "horizontal".    | `'horizontal'`    |
| `gap`      | `int`                        | Width (horizontal) or height (vertical) in pixels of a bg_color separator inserted between adjacent frames. 0 disables it. Default 0. | `0`               |
| `bg_color` | `tuple[float, float, float]` | RGB in [0, 1] used for padding and gaps. Default white (1, 1, 1).                                                                     | `(1.0, 1.0, 1.0)` |
| `align`    | `str`                        | Cross-axis placement of a smaller frame: "start" (top / left), "center", or "end" (bottom / right). Default "center".                 | `'center'`        |

Source code in `cuvis_ai/node/compositing.py`

```python
def __init__(
    self,
    axis: str = "horizontal",
    gap: int = 0,
    bg_color: tuple[float, float, float] = (1.0, 1.0, 1.0),
    align: str = "center",
    **kwargs: Any,
) -> None:
    if axis not in _AXES:
        raise ValueError(f"axis must be one of {_AXES}, got {axis!r}")
    if gap < 0:
        raise ValueError("gap must be >= 0")
    if len(bg_color) != 3:
        raise ValueError("bg_color must have 3 channels")
    if align not in _ALIGNS:
        raise ValueError(f"align must be one of {_ALIGNS}, got {align!r}")

    self.axis = axis
    self.gap = int(gap)
    self.bg_color = tuple(float(c) for c in bg_color)
    self.align = align

    super().__init__(
        axis=self.axis,
        gap=self.gap,
        bg_color=self.bg_color,
        align=self.align,
        **kwargs,
    )
```

`InsetComposer`cuvis_ai.node.compositingtransformimgnumpyprePaste a fixed-size inset frame into a corner of a larger base frame.

#### InsetComposer

```python
InsetComposer(
    corner="top-right",
    margin_px=16,
    border_px=2,
    border_color=(1.0, 1.0, 1.0),
    **kwargs,
)
```

Bases: `Node`

Paste a fixed-size inset frame into a corner of a larger base frame.

Picture-in-picture compositor. The inset is expected to already be at its final pixel size (e.g. produced by :class:`ROIZoomNode`); this node only places it onto the base, optionally with a coloured border. When `valid == 0` for a frame the base passes through untouched, so the inset never lies about a stale ROI.

Parameters:

| Name           | Type                         | Description                                                                         | Default           |
| -------------- | ---------------------------- | ----------------------------------------------------------------------------------- | ----------------- |
| `corner`       | `str`                        | One of "top-left", "top-right", "bottom-left", "bottom-right". Default "top-right". | `'top-right'`     |
| `margin_px`    | `int`                        | Distance in pixels between the inset and the closest base edges. Default 16.        | `16`              |
| `border_px`    | `int`                        | Border thickness in pixels. 0 disables the border. Default 2.                       | `2`               |
| `border_color` | `tuple[float, float, float]` | Border RGB in [0, 1]. Default white (1, 1, 1).                                      | `(1.0, 1.0, 1.0)` |

Source code in `cuvis_ai/node/compositing.py`

```python
def __init__(
    self,
    corner: str = "top-right",
    margin_px: int = 16,
    border_px: int = 2,
    border_color: tuple[float, float, float] = (1.0, 1.0, 1.0),
    **kwargs: Any,
) -> None:
    if corner not in _CORNERS:
        raise ValueError(f"corner must be one of {_CORNERS}, got {corner!r}")
    if margin_px < 0:
        raise ValueError("margin_px must be >= 0")
    if border_px < 0:
        raise ValueError("border_px must be >= 0")
    if len(border_color) != 3:
        raise ValueError("border_color must have 3 channels")

    self.corner = corner
    self.margin_px = int(margin_px)
    self.border_px = int(border_px)
    self.border_color = tuple(float(c) for c in border_color)

    super().__init__(
        corner=self.corner,
        margin_px=self.margin_px,
        border_px=self.border_px,
        border_color=self.border_color,
        **kwargs,
    )
```

`IntensityThresholdSegmenter`cuvis_ai.node.segmentation.intensitytransformhsimaskposttorchSegment foreground by thresholding a per-pixel reduced intensity.

#### IntensityThresholdSegmenter

```python
IntensityThresholdSegmenter(
    low=0.0,
    high=1.0,
    reduction="mean",
    band_index=0,
    **kwargs,
)
```

Bases: `Node`

Segment foreground by thresholding a per-pixel reduced intensity.

The cube is collapsed over its channel axis to a single per-pixel intensity using the chosen `reduction`, then pixels whose intensity lies inside the closed interval `[low, high]` are marked as foreground (`1`); all other pixels are background (`0`).

Parameters:

| Name         | Type    | Description                                                                                                                                                        | Default  |
| ------------ | ------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ | -------- |
| `low`        | `float` | Inclusive lower bound of the foreground intensity interval. Default: 0.0.                                                                                          | `0.0`    |
| `high`       | `float` | Inclusive upper bound of the foreground intensity interval. Default: 1.0.                                                                                          | `1.0`    |
| `reduction`  | `str`   | How to collapse the channel axis to a scalar intensity. One of "mean" (channel mean), "max" (channel max), or "band" (single band at band_index). Default: "mean". | `'mean'` |
| `band_index` | `int`   | Channel index used when reduction == "band". Default: 0.                                                                                                           | `0`      |

Examples:

```pycon
>>> seg = IntensityThresholdSegmenter(low=0.2, high=0.8, reduction="mean")
>>> cube = torch.rand(2, 8, 8, 16)
>>> seg.forward(cube=cube)["mask"].shape
torch.Size([2, 8, 8])
```

Source code in `cuvis_ai/node/segmentation/intensity.py`

```python
def __init__(
    self,
    low: float = 0.0,
    high: float = 1.0,
    reduction: str = "mean",
    band_index: int = 0,
    **kwargs: Any,
) -> None:
    if reduction not in self._VALID_REDUCTIONS:
        raise ValueError(
            f"reduction must be one of {self._VALID_REDUCTIONS}, got {reduction!r}"
        )
    if float(low) > float(high):
        raise ValueError(f"low must be <= high, got low={low}, high={high}")
    self.low = float(low)
    self.high = float(high)
    self.reduction = reduction
    self.band_index = int(band_index)
    super().__init__(
        low=self.low,
        high=self.high,
        reduction=self.reduction,
        band_index=self.band_index,
        **kwargs,
    )
```

##### forward

```python
forward(cube, **_)
```

Reduce the cube over channels and threshold it into a foreground mask.

Parameters:

| Name   | Type     | Description                            | Default    |
| ------ | -------- | -------------------------------------- | ---------- |
| `cube` | `Tensor` | Input hyperspectral cube [B, H, W, C]. | *required* |

Returns:

| Type                | Description                                            |
| ------------------- | ------------------------------------------------------ |
| `dict[str, Tensor]` | {"mask": Tensor [B, H, W]} of int32 foreground labels. |

Source code in `cuvis_ai/node/segmentation/intensity.py`

```python
@torch.no_grad()
def forward(self, cube: Tensor, **_: Any) -> dict[str, Tensor]:
    """Reduce the cube over channels and threshold it into a foreground mask.

    Parameters
    ----------
    cube : Tensor
        Input hyperspectral cube ``[B, H, W, C]``.

    Returns
    -------
    dict[str, Tensor]
        ``{"mask": Tensor [B, H, W]}`` of int32 foreground labels.
    """
    if self.reduction == "mean":
        intensity = cube.mean(dim=-1)
    elif self.reduction == "max":
        intensity = cube.amax(dim=-1)
    else:  # "band"
        intensity = cube[..., self.band_index]

    mask = ((intensity >= self.low) & (intensity <= self.high)).to(torch.int32)
    return {"mask": mask}
```

`InverseFrequencyClassWeights`cuvis_ai_inspecscrap.node.lossestransformclasstraincuvis_ai_inspecscrap[↗](https://github.com/cubert-hyperspectral/cuvis-ai-inspecscrap "Open plugin repo")

[View plugin repo (v0.2.4)](https://github.com/cubert-hyperspectral/cuvis-ai-inspecscrap/tree/v0.2.4)

`LabelOffset`cuvis_ai.node.mask_opstransformclassmasksegAdd a constant offset to every label in an integer label map.

#### LabelOffset

```python
LabelOffset(offset=1, **kwargs)
```

Bases: `Node`

Add a constant offset to every label in an integer label map.

Mainly used to lift a 0-based dense label map (e.g. a `KMeansClusterer` / `GaussianMixtureClusterer` `class_mask`, where cluster ids run `0..k-1`) to 1-based ids before :class:`MajorityVoteByBlob`, which treats `0` as background in both its vote and its output. Without the shift a cluster-`0` region would be dropped as background and collide with the unassigned label.

Parameters:

| Name     | Type  | Description                            | Default |
| -------- | ----- | -------------------------------------- | ------- |
| `offset` | `int` | Value added to every label. Default 1. | `1`     |

Source code in `cuvis_ai/node/mask_ops.py`

```python
def __init__(self, offset: int = 1, **kwargs: Any) -> None:
    self.offset = int(offset)
    super().__init__(offset=self.offset, **kwargs)
```

##### forward

```python
forward(class_map, **_)
```

Return the label map with `offset` added to every element.

Source code in `cuvis_ai/node/mask_ops.py`

```python
@torch.no_grad()
def forward(self, class_map: torch.Tensor, **_: Any) -> dict[str, torch.Tensor]:
    """Return the label map with `offset` added to every element."""
    return {"class_map": (class_map.to(torch.int32) + self.offset).to(torch.int32)}
```

`LabelOverlay`cuvis_ai.node.compositingtransformimgrgbAlpha-blend a colourised label map onto an RGB image on its foreground pixels.

#### LabelOverlay

```python
LabelOverlay(
    alpha=0.55, background_color=(0.0, 0.0, 0.0), **kwargs
)
```

Bases: `Node`

Alpha-blend a colourised label map onto an RGB image on its foreground pixels.

A pixel is "foreground" when its `label_rgb` differs from `background_color`; background pixels keep the original RGB. Returns a single blended frame, so several overlays can be montaged column-by-column.

Parameters:

| Name               | Type                         | Description                                                                   | Default           |
| ------------------ | ---------------------------- | ----------------------------------------------------------------------------- | ----------------- |
| `alpha`            | `float`                      | Blend factor for the label colour over the base image (default 0.55).         | `0.55`            |
| `background_color` | `tuple[float, float, float]` | Label-map background colour in [0, 1]; pixels equal to it are left unblended. | `(0.0, 0.0, 0.0)` |

Source code in `cuvis_ai/node/compositing.py`

```python
def __init__(
    self,
    alpha: float = 0.55,
    background_color: tuple[float, float, float] = (0.0, 0.0, 0.0),
    **kwargs: Any,
) -> None:
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1]; got {alpha}")
    if len(background_color) != 3:
        raise ValueError("background_color must have 3 channels")
    self.alpha = float(alpha)
    self.background_color = tuple(float(c) for c in background_color)
    super().__init__(alpha=self.alpha, background_color=self.background_color, **kwargs)
```

##### forward

```python
forward(rgb_image, label_rgb, **_)
```

Blend `label_rgb` onto `rgb_image` where the label is non-background.

Source code in `cuvis_ai/node/compositing.py`

```python
@torch.no_grad()
def forward(
    self, rgb_image: torch.Tensor, label_rgb: torch.Tensor, **_: Any
) -> dict[str, torch.Tensor]:
    """Blend ``label_rgb`` onto ``rgb_image`` where the label is non-background."""
    if not torch.is_floating_point(label_rgb):
        label_rgb = label_rgb.to(torch.float32) / 255.0
    label_rgb = label_rgb.to(device=rgb_image.device, dtype=rgb_image.dtype)
    if rgb_image.shape != label_rgb.shape:
        raise ValueError(
            f"rgb_image {tuple(rgb_image.shape)} != label_rgb {tuple(label_rgb.shape)}"
        )
    bg = torch.tensor(self.background_color, device=rgb_image.device, dtype=rgb_image.dtype)
    fg_mask = (label_rgb - bg).abs().sum(dim=-1, keepdim=True) > 1e-6
    blend_weight = self.alpha * fg_mask.to(rgb_image.dtype)
    frame = (1.0 - blend_weight) * rgb_image + blend_weight * label_rgb
    return {"frame": frame.clamp(0.0, 1.0)}
```

`LegendStrip`cuvis_ai.node.compositingtransformimgrgbAppend a horizontal class-colour legend strip below the input frame.

#### LegendStrip

```python
LegendStrip(
    entries,
    n_columns=6,
    tile_height_px=22,
    swatch_width_px=28,
    text_padding_px=6,
    background_color=(0.08, 0.08, 0.08),
    text_color=(240, 240, 240),
    dim_text_color=(110, 110, 110),
    font_size=12,
    **kwargs,
)
```

Bases: `Node`

Append a horizontal class-colour legend strip below the input frame.

Each `(label, rgb)` entry renders as a swatch plus its text label, wrapped over `n_columns`. When the optional `label_rgb` mask is connected, the legend appends a connected-component instance count `(N)` per class for the current frame and dims rows whose count is zero. The legend is built from an explicit `entries` list of `(label, rgb)` rows.

Parameters:

| Name               | Type                                     | Description                                                        | Default              |
| ------------------ | ---------------------------------------- | ------------------------------------------------------------------ | -------------------- |
| `entries`          | `list[tuple[str, tuple[int, int, int]]]` | Ordered (label, (r, g, b)) legend rows; colours in 0-255.          | *required*           |
| `n_columns`        | `int`                                    | Number of legend columns before wrapping to a new row (default 6). | `6`                  |
| `tile_height_px`   | `int`                                    | Legend layout dimensions in pixels.                                | `22`                 |
| `swatch_width_px`  | `int`                                    | Legend layout dimensions in pixels.                                | `22`                 |
| `text_padding_px`  | `int`                                    | Legend layout dimensions in pixels.                                | `22`                 |
| `font_size`        | `int`                                    | Legend layout dimensions in pixels.                                | `22`                 |
| `background_color` | `tuple[float, float, float]`             | Strip background colour in [0, 1].                                 | `(0.08, 0.08, 0.08)` |
| `text_color`       | `tuple[int, int, int]`                   | Text colour for present / zero-count rows, in 0-255.               | `(240, 240, 240)`    |
| `dim_text_color`   | `tuple[int, int, int]`                   | Text colour for present / zero-count rows, in 0-255.               | `(240, 240, 240)`    |

Source code in `cuvis_ai/node/compositing.py`

```python
def __init__(
    self,
    entries: list[tuple[str, tuple[int, int, int]]],
    n_columns: int = 6,
    tile_height_px: int = 22,
    swatch_width_px: int = 28,
    text_padding_px: int = 6,
    background_color: tuple[float, float, float] = (0.08, 0.08, 0.08),
    text_color: tuple[int, int, int] = (240, 240, 240),
    dim_text_color: tuple[int, int, int] = (110, 110, 110),
    font_size: int = 12,
    **kwargs: Any,
) -> None:
    if not entries:
        raise ValueError("entries must be a non-empty list of (label, (r, g, b))")
    self._entries = [(str(name), tuple(int(c) for c in rgb)) for name, rgb in entries]
    self.n_columns = int(n_columns)
    self.tile_height_px = int(tile_height_px)
    self.swatch_width_px = int(swatch_width_px)
    self.text_padding_px = int(text_padding_px)
    self.background_color = tuple(float(c) for c in background_color)
    self.text_color = tuple(int(c) for c in text_color)
    self.dim_text_color = tuple(int(c) for c in dim_text_color)
    self.font_size = int(font_size)
    super().__init__(
        entries=[[name, list(rgb)] for name, rgb in self._entries],
        n_columns=self.n_columns,
        tile_height_px=self.tile_height_px,
        swatch_width_px=self.swatch_width_px,
        text_padding_px=self.text_padding_px,
        background_color=list(self.background_color),
        text_color=list(self.text_color),
        dim_text_color=list(self.dim_text_color),
        font_size=self.font_size,
        **kwargs,
    )
    try:
        self._font = ImageFont.truetype("arial.ttf", self.font_size)
    except OSError:
        self._font = ImageFont.load_default()
    n_rows = (len(self._entries) + self.n_columns - 1) // self.n_columns
    self._legend_h = n_rows * self.tile_height_px + 2 * self.text_padding_px
```

##### forward

```python
forward(frame, label_rgb=None, **_)
```

Append the legend strip below `frame`; optionally count instances per class.

Source code in `cuvis_ai/node/compositing.py`

```python
@torch.no_grad()
def forward(
    self, frame: torch.Tensor, label_rgb: torch.Tensor | None = None, **_: Any
) -> dict[str, torch.Tensor]:
    """Append the legend strip below ``frame``; optionally count instances per class."""
    b, _, w, _ = frame.shape
    counts: list[int] | None = None
    if label_rgb is not None:
        lab = label_rgb[0].detach().cpu()
        if torch.is_floating_point(lab):
            arr_u8 = (lab.clamp(0.0, 1.0) * 255.0).round().to(torch.uint8).numpy()
        else:
            arr_u8 = lab.to(torch.uint8).numpy()
        counts = [self._count_instances(arr_u8, color) for _, color in self._entries]
    strip = self._render_strip(w, counts).to(device=frame.device, dtype=frame.dtype)
    strip = strip.unsqueeze(0).expand(b, -1, -1, -1)
    return {"frame": torch.cat([frame, strip], dim=1).clamp(0.0, 1.0)}
```

`LegendStripNode`cuvis_ai_inspecscrap.node.legendtransformvidcuvis_ai_inspecscrap[↗](https://github.com/cubert-hyperspectral/cuvis-ai-inspecscrap "Open plugin repo")

[View plugin repo (v0.2.4)](https://github.com/cubert-hyperspectral/cuvis-ai-inspecscrap/tree/v0.2.4)

`Logarithm`cuvis_ai.node.pretreatments.logarithmtransformhsipretorchElement-wise logarithm of the cube.

#### Logarithm

```python
Logarithm(mode='log10', negate=False, eps=1e-08, **kwargs)
```

Bases: `Node`

Element-wise logarithm of the cube.

Computes `log10(x)` (default) or `ln(x)` after clamping the input to a small positive floor so non-positive values do not produce `-inf` or `nan`. With `negate=True` the sign is flipped, yielding true absorbance `-log10(R)` from reflectance.

Parameters:

| Name     | Type    | Description                                                           | Default   |
| -------- | ------- | --------------------------------------------------------------------- | --------- |
| `mode`   | `str`   | "log10" (default) for base-10, or "ln" for the natural log.           | `'log10'` |
| `negate` | `bool`  | Negate the result so reflectance maps to absorbance (default: False). | `False`   |
| `eps`    | `float` | Lower clamp applied before the logarithm (default: 1e-8).             | `1e-08`   |

Source code in `cuvis_ai/node/pretreatments/logarithm.py`

```python
def __init__(
    self, mode: str = "log10", negate: bool = False, eps: float = 1e-8, **kwargs
) -> None:
    self.mode = str(mode)
    if self.mode not in self._MODES:
        raise ValueError(f"mode must be one of {list(self._MODES)}, got {self.mode!r}")
    self.negate = bool(negate)
    self.eps = float(eps)
    super().__init__(mode=self.mode, negate=self.negate, eps=self.eps, **kwargs)
```

##### forward

```python
forward(cube, **_)
```

Apply the configured logarithm to the cube.

Parameters:

| Name   | Type     | Description                | Default    |
| ------ | -------- | -------------------------- | ---------- |
| `cube` | `Tensor` | Input cube in BHWC format. | *required* |

Returns:

| Type                | Description                                                 |
| ------------------- | ----------------------------------------------------------- |
| `dict[str, Tensor]` | {"cube": log_transformed} with the same shape as the input. |

Source code in `cuvis_ai/node/pretreatments/logarithm.py`

```python
def forward(self, cube: torch.Tensor, **_) -> dict[str, torch.Tensor]:
    """Apply the configured logarithm to the cube.

    Parameters
    ----------
    cube : torch.Tensor
        Input cube in BHWC format.

    Returns
    -------
    dict[str, torch.Tensor]
        ``{"cube": log_transformed}`` with the same shape as the input.
    """
    clamped = cube.clamp_min(self.eps)
    result = torch.log(clamped) if self.mode == "ln" else torch.log10(clamped)
    if self.negate:
        result = -result
    return {"cube": result}
```

`MCARISelector`cuvis_ai.node.channel_selectortransformdim-redhsinumpypreModified Chlorophyll Absorption in Reflectance Index renderer.

#### MCARISelector

```python
MCARISelector(
    green_nm=550.0,
    red_nm=670.0,
    red_edge_nm=700.0,
    colormap_min=-1.0,
    colormap_max=1.0,
    eps=1e-06,
    **kwargs,
)
```

Bases: `_VegetationIndexBase`

Modified Chlorophyll Absorption in Reflectance Index renderer.

Computes `((RE - Red) - 0.2 * (RE - Green)) * (RE / Red)` over bands resolved by nearest sensor wavelength, where `RE` is the red-edge band. The raw index map is returned via `index_image` and `rgb_image` carries an HSV colour-mapped render.

Source code in `cuvis_ai/node/channel_selector.py`

```python
def __init__(
    self,
    green_nm: float = 550.0,
    red_nm: float = 670.0,
    red_edge_nm: float = 700.0,
    colormap_min: float = -1.0,
    colormap_max: float = 1.0,
    eps: float = 1.0e-6,
    **kwargs: Any,
) -> None:
    super().__init__(
        band_nm={"green": green_nm, "red": red_nm, "red_edge": red_edge_nm},
        colormap_min=colormap_min,
        colormap_max=colormap_max,
        eps=eps,
        green_nm=float(green_nm),
        red_nm=float(red_nm),
        red_edge_nm=float(red_edge_nm),
        **kwargs,
    )
    self.green_nm = float(green_nm)
    self.red_nm = float(red_nm)
    self.red_edge_nm = float(red_edge_nm)
```

##### index_name

```python
index_name
```

Canonical MCARI strategy name.

`MSAVISelector`cuvis_ai.node.channel_selectortransformdim-redhsinumpypreModified Soil Adjusted Vegetation Index renderer.

#### MSAVISelector

```python
MSAVISelector(
    red_nm=660.0,
    nir_nm=800.0,
    colormap_min=-1.0,
    colormap_max=1.0,
    eps=1e-06,
    **kwargs,
)
```

Bases: `_VegetationIndexBase`

Modified Soil Adjusted Vegetation Index renderer.

Computes `0.5 * (2*NIR + 1 - sqrt((2*NIR + 1)^2 - 8*(NIR - Red)))` over bands resolved by nearest sensor wavelength. The raw index map is returned via `index_image` and `rgb_image` carries an HSV colour-mapped render.

The additive `+1` constants are only meaningful for reflectance in [0, 1]; feed reflectance-calibrated cubes, not raw radiance/DN.

Source code in `cuvis_ai/node/channel_selector.py`

```python
def __init__(
    self,
    red_nm: float = 660.0,
    nir_nm: float = 800.0,
    colormap_min: float = -1.0,
    colormap_max: float = 1.0,
    eps: float = 1.0e-6,
    **kwargs: Any,
) -> None:
    super().__init__(
        band_nm={"red": red_nm, "nir": nir_nm},
        colormap_min=colormap_min,
        colormap_max=colormap_max,
        eps=eps,
        red_nm=float(red_nm),
        nir_nm=float(nir_nm),
        **kwargs,
    )
    self.red_nm = float(red_nm)
    self.nir_nm = float(nir_nm)
```

##### index_name

```python
index_name
```

Canonical MSAVI strategy name.

`MajorityVoteByBlob`cuvis_ai.node.mask_opstransformclassmasksegAssign each blob the majority per-pixel label found inside it.

#### MajorityVoteByBlob

Bases: `Node`

Assign each blob the majority per-pixel label found inside it.

Per-pixel classifiers (e.g. a Spectral Angle Mapper) produce noisy labels when reference spectra are close together. Voting within each detected blob denoises that into a single robust label per object: for every blob id in `blob_mask` (1..N), the most frequent nonzero `identity_mask` value over that blob's pixels becomes the blob's label; blobs with no labelled pixels stay `0`, as does the background.

##### forward

```python
forward(identity_mask, blob_mask, **_)
```

Paint each blob with the majority label of its pixels.

Parameters:

| Name            | Type     | Description                                                      | Default    |
| --------------- | -------- | ---------------------------------------------------------------- | ---------- |
| `identity_mask` | `Tensor` | Per-pixel labels [1, H, W] (int32); 0 is unassigned.             | *required* |
| `blob_mask`     | `Tensor` | Blob label map [1, H, W] (int32); ids 1..N, 0 background.        | *required* |
| `**_`           | `Any`    | Additional unused keyword arguments (e.g. the pipeline context). | `{}`       |

Returns:

| Type                | Description                                                                              |
| ------------------- | ---------------------------------------------------------------------------------------- |
| `dict[str, Tensor]` | mask int32 [1, H, W] with each blob painted its majority label and background left at 0. |

Source code in `cuvis_ai/node/mask_ops.py`

```python
@torch.no_grad()
def forward(
    self, identity_mask: torch.Tensor, blob_mask: torch.Tensor, **_: Any
) -> dict[str, torch.Tensor]:
    """Paint each blob with the majority label of its pixels.

    Parameters
    ----------
    identity_mask : torch.Tensor
        Per-pixel labels ``[1, H, W]`` (int32); ``0`` is unassigned.
    blob_mask : torch.Tensor
        Blob label map ``[1, H, W]`` (int32); ids ``1..N``, ``0`` background.
    **_ : Any
        Additional unused keyword arguments (e.g. the pipeline ``context``).

    Returns
    -------
    dict[str, torch.Tensor]
        ``mask`` int32 ``[1, H, W]`` with each blob painted its majority
        label and background left at ``0``.
    """
    ident = identity_mask[0].to(torch.int64)
    blobs = blob_mask[0].to(torch.int64)
    out = torch.zeros_like(blobs, dtype=torch.int32)

    for blob_id in torch.unique(blobs).tolist():
        if blob_id == 0:
            continue
        region = blobs == blob_id
        votes = ident[region]
        votes = votes[votes > 0]
        if votes.numel() == 0:
            continue
        majority = int(torch.bincount(votes).argmax())
        out[region] = majority
    return {"mask": out.unsqueeze(0)}
```

`MaskRobustifier`cuvis_ai.node.mask_opstransformmasknumpypostsegClean a binary/labelled mask with morphology + largest-component filter.

#### MaskRobustifier

```python
MaskRobustifier(
    opening_kernel=0,
    closing_kernel=3,
    min_area=10,
    keep_largest=True,
    **kwargs,
)
```

Bases: `Node`

Clean a binary/labelled mask with morphology + largest-component filter.

Applies morphological opening (remove speckle), then closing (fill small holes), optionally drops connected components below `min_area` pixels, and optionally keeps only the single largest component.

Output is an int32 mask with the same spatial shape as the input; non-zero values are preserved where the original mask was non-zero and survives the cleanup.

Parameters:

| Name             | Type   | Description                                                                                                                                                                                                                                             | Default |
| ---------------- | ------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------- |
| `opening_kernel` | `int`  | Side length of the square structuring element used for cv2.MORPH_OPEN. 0 or 1 disables opening. Default 0 (disabled); opening is aggressive enough to erase narrow real detections, so the default is off and callers enable it explicitly when needed. | `0`     |
| `closing_kernel` | `int`  | Side length for cv2.MORPH_CLOSE. 0/1 disables closing. Default 3.                                                                                                                                                                                       | `3`     |
| `min_area`       | `int`  | Drop connected components with fewer than this many pixels. 0 disables the filter. Default 10 (kills singleton/doubleton speckle while preserving small compact detections).                                                                            | `10`    |
| `keep_largest`   | `bool` | If True, keep only the single largest surviving component. Default True.                                                                                                                                                                                | `True`  |

Source code in `cuvis_ai/node/mask_ops.py`

```python
def __init__(
    self,
    opening_kernel: int = 0,
    closing_kernel: int = 3,
    min_area: int = 10,
    keep_largest: bool = True,
    **kwargs: Any,
) -> None:
    if opening_kernel < 0:
        raise ValueError("opening_kernel must be >= 0")
    if closing_kernel < 0:
        raise ValueError("closing_kernel must be >= 0")
    if min_area < 0:
        raise ValueError("min_area must be >= 0")

    self.opening_kernel = int(opening_kernel)
    self.closing_kernel = int(closing_kernel)
    self.min_area = int(min_area)
    self.keep_largest = bool(keep_largest)

    super().__init__(
        opening_kernel=self.opening_kernel,
        closing_kernel=self.closing_kernel,
        min_area=self.min_area,
        keep_largest=self.keep_largest,
        **kwargs,
    )
```

`MaskToBBoxKalman`cuvis_ai.node.mask_opstransformbboxmasknumpypoststatefultrackMask -> bounding box with constant-velocity Kalman smoothing.

#### MaskToBBoxKalman

```python
MaskToBBoxKalman(
    padding_fraction=0.2,
    min_size_px=96,
    min_hits=3,
    max_predict_frames=20,
    process_noise=0.01,
    measurement_noise=1.0,
    **kwargs,
)
```

Bases: `Node`

Mask -> bounding box with constant-velocity Kalman smoothing.

Each frame the bbox tight to the non-zero extent of the mask (with padding) is used as a measurement to update an 8-state Kalman filter (cx, cy, w, h, vx, vy, vw, vh). When the mask is empty the filter is stepped in prediction-only mode, so the downstream ROI stays pinned to a plausible location for a few frames rather than vanishing.

A warm-up of `min_hits` consecutive measurement frames is required before the track is confirmed; hits during the warm-up never leak to downstream consumers (`valid=0`), and a single missed frame during warm-up resets the hit counter. This suppresses isolated false-positive detections that would otherwise briefly pop the inset into view.

Output `valid` encodes track state per frame:

- `1` - measurement used this frame on a confirmed track.
- `2` - predicted only (mask empty on a confirmed track, within budget).
- `0` - unconfirmed warm-up, no track, or post-drop.

Parameters:

| Name                 | Type    | Description                                                                                                                                                                                                                | Default |
| -------------------- | ------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------- |
| `padding_fraction`   | `float` | Fractional padding applied to the measurement bbox before it is fed to the filter. 0.2 adds 10% on each side. Default 0.2.                                                                                                 | `0.2`   |
| `min_size_px`        | `int`   | Lower bound on the output bbox edge length (post-Kalman). Small measurements are expanded around the centre. Default 96.                                                                                                   | `96`    |
| `min_hits`           | `int`   | Number of consecutive measurement frames required to confirm a new track. Missed frames during warm-up reset the hit counter back to zero, so transient false positives never graduate. Default 3; 1 disables the warm-up. | `3`     |
| `max_predict_frames` | `int`   | After this many consecutive empty frames (on a confirmed track) the track is dropped and subsequent empty frames emit valid=0 until a new measurement. Default 20.                                                         | `20`    |
| `process_noise`      | `float` | Scalar multiplier for the Kalman process-noise covariance.                                                                                                                                                                 | `0.01`  |
| `measurement_noise`  | `float` | Scalar multiplier for the Kalman measurement-noise covariance.                                                                                                                                                             | `1.0`   |

Source code in `cuvis_ai/node/mask_ops.py`

```python
def __init__(
    self,
    padding_fraction: float = 0.2,
    min_size_px: int = 96,
    min_hits: int = 3,
    max_predict_frames: int = 20,
    process_noise: float = 1e-2,
    measurement_noise: float = 1.0,
    **kwargs: Any,
) -> None:
    if padding_fraction < 0:
        raise ValueError("padding_fraction must be >= 0")
    if min_size_px < 1:
        raise ValueError("min_size_px must be >= 1")
    if min_hits < 1:
        raise ValueError("min_hits must be >= 1")
    if max_predict_frames < 0:
        raise ValueError("max_predict_frames must be >= 0")

    self.padding_fraction = float(padding_fraction)
    self.min_size_px = int(min_size_px)
    self.min_hits = int(min_hits)
    self.max_predict_frames = int(max_predict_frames)
    self.process_noise = float(process_noise)
    self.measurement_noise = float(measurement_noise)

    super().__init__(
        padding_fraction=self.padding_fraction,
        min_size_px=self.min_size_px,
        min_hits=self.min_hits,
        max_predict_frames=self.max_predict_frames,
        process_noise=self.process_noise,
        measurement_noise=self.measurement_noise,
        **kwargs,
    )

    self._kf: cv2.KalmanFilter | None = None
    self._missed = 0
    self._hits = 0
    self._has_track = False
    self._confirmed = False
```

`MaskedMeanSpectrum`cuvis_ai.node.spectral_extractortransformembhsinumpyPer-frame mean spectrum of a hyperspectral cube over a binary mask.

#### MaskedMeanSpectrum

Bases: `Node`

Per-frame mean spectrum of a hyperspectral cube over a binary mask.

For each frame, averages cube values at pixels where `mask > 0` and emits the resulting `[C]` spectrum. When the mask is empty for a given frame the output is a zero vector and `valid` is `0`.

`MeanCenter`cuvis_ai.node.pretreatments.scalingtransformhsinormpretorchSubtract a globally-fitted per-channel mean from the cube.

#### MeanCenter

```python
MeanCenter(**kwargs)
```

Bases: `_StatisticalFitNode`

Subtract a globally-fitted per-channel mean from the cube.

During `statistical_initialization` every training pixel is streamed through a Welford accumulator to compute the exact per-channel mean over the full dataset; `forward` then subtracts that mean from each spectrum.

Notes

The fitted `mean_c` is registered as a persistent buffer so a checkpointed node reloads ready for inference.

Source code in `cuvis_ai/node/pretreatments/scaling.py`

```python
def __init__(self, **kwargs) -> None:
    super().__init__(**kwargs)
    self.register_buffer("mean_c", torch.zeros(0, dtype=torch.float32))
    self._welford: WelfordAccumulator | None = None
```

##### statistical_initialization

```python
statistical_initialization(input_stream)
```

Fit the per-channel mean from the training stream via Welford.

Parameters:

| Name           | Type          | Description                                              | Default    |
| -------------- | ------------- | -------------------------------------------------------- | ---------- |
| `input_stream` | `InputStream` | Iterable of port-keyed batch dicts matching INPUT_SPECS. | *required* |

Source code in `cuvis_ai/node/pretreatments/scaling.py`

```python
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
```

##### forward

```python
forward(cube, **_)
```

Subtract the fitted per-channel mean from the cube.

Parameters:

| Name   | Type     | Description                | Default    |
| ------ | -------- | -------------------------- | ---------- |
| `cube` | `Tensor` | Input cube in BHWC format. | *required* |

Returns:

| Type                | Description                                         |
| ------------------- | --------------------------------------------------- |
| `dict[str, Tensor]` | {"cube": centred} with the same shape as the input. |

Source code in `cuvis_ai/node/pretreatments/scaling.py`

```python
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
```

`MinMaxNormalizer`cuvis_ai.node.normalizationtransformnormnumpypreMin-max normalization per sample and channel (keeps gradients).

#### MinMaxNormalizer

```python
MinMaxNormalizer(
    eps=1e-06,
    use_running_stats=True,
    max_initialization_frames=None,
    **kwargs,
)
```

Bases: `_ScoreNormalizerBase`

Min-max normalization per sample and channel (keeps gradients).

Scales data to [0, 1] range using (x - min) / (max - min) transformation. Can operate in two modes:

1. **Per-sample normalization** (use_running_stats=False): min/max computed per batch
1. **Global normalization** (use_running_stats=True): uses running statistics from statistical initialization

Parameters:

| Name                        | Type          | Description                                                                                                                                                                                                                    | Default |
| --------------------------- | ------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------- |
| `eps`                       | `float`       | Small constant for numerical stability, prevents division by zero (default: 1e-6)                                                                                                                                              | `1e-06` |
| `use_running_stats`         | `bool`        | If True, use global min/max from statistical_initialization(). If False, compute min/max per batch during forward pass (default: True)                                                                                         | `True`  |
| `max_initialization_frames` | `int or None` | Cap on the number of frames (batch elements) consumed by statistical_initialization(): the final batch is sliced to the cap and the input stream is not iterated further. None uses the entire training stream (default: None) | `None`  |
| `**kwargs`                  | `dict`        | Additional arguments passed to Node base class                                                                                                                                                                                 | `{}`    |

Attributes:

| Name          | Type     | Description                                                     |
| ------------- | -------- | --------------------------------------------------------------- |
| `running_min` | `Tensor` | Global minimum value computed during statistical initialization |
| `running_max` | `Tensor` | Global maximum value computed during statistical initialization |

Examples:

```pycon
>>> from cuvis_ai.node.normalization import MinMaxNormalizer
>>> from cuvis_ai_core.training import StatisticalTrainer
>>> import torch
>>>
>>> # Mode 1: Global normalization with statistical initialization
>>> normalizer = MinMaxNormalizer(eps=1.0e-6, use_running_stats=True)
>>> stat_trainer = StatisticalTrainer(pipeline=pipeline, datamodule=datamodule)
>>> stat_trainer.fit()  # Computes global min/max from training data
>>>
>>> # Inference uses global statistics
>>> output = normalizer.forward(data=hyperspectral_cube)
>>> normalized = output["normalized"]  # [B, H, W, C], values in [0, 1]
>>>
>>> # Mode 2: Per-sample normalization (no initialization required)
>>> normalizer_local = MinMaxNormalizer(use_running_stats=False)
>>> output = normalizer_local.forward(data=hyperspectral_cube)
>>> # Each sample normalized independently using its own min/max
```

See Also

ZScoreNormalizer : Z-score standardization SigmoidNormalizer : Sigmoid-based normalization docs/usecases/rx-statistical.md : RX pipeline with MinMaxNormalizer

Notes

Global normalization (use_running_stats=True) is recommended for RX detectors to ensure consistent scaling between training and inference. Per-sample normalization can be useful for real-time processing when training data is unavailable.

Source code in `cuvis_ai/node/normalization.py`

```python
def __init__(
    self,
    eps: float = 1e-6,
    use_running_stats: bool = True,
    max_initialization_frames: int | None = None,
    **kwargs,
) -> None:
    if max_initialization_frames is not None and max_initialization_frames <= 0:
        raise ValueError("max_initialization_frames must be None or a positive integer.")
    self.eps = float(eps)
    self.use_running_stats = use_running_stats
    self.max_initialization_frames = max_initialization_frames
    super().__init__(
        eps=eps,
        use_running_stats=use_running_stats,
        max_initialization_frames=max_initialization_frames,
        **kwargs,
    )

    # Running statistics for global normalization
    self.register_buffer("running_min", torch.tensor(float("nan")))
    self.register_buffer("running_max", torch.tensor(float("nan")))

    # Only require initialization when running stats are requested
    self._requires_initial_fit_override = self.use_running_stats
```

##### statistical_initialization

```python
statistical_initialization(input_stream)
```

Compute global min/max from data iterator.

Consumes at most `max_initialization_frames` frames when the cap is set: the final batch is sliced to the cap and the stream is not iterated further.

Parameters:

| Name           | Type          | Description                                                                                                                        | Default    |
| -------------- | ------------- | ---------------------------------------------------------------------------------------------------------------------------------- | ---------- |
| `input_stream` | `InputStream` | Iterator yielding dicts matching INPUT_SPECS (port-based format) Expected format: {"data": tensor} where tensor is the scores/data | *required* |

Source code in `cuvis_ai/node/normalization.py`

```python
def statistical_initialization(self, input_stream) -> None:
    """Compute global min/max from data iterator.

    Consumes at most ``max_initialization_frames`` frames when the cap is set: the
    final batch is sliced to the cap and the stream is not iterated further.

    Parameters
    ----------
    input_stream : InputStream
        Iterator yielding dicts matching INPUT_SPECS (port-based format)
        Expected format: {"data": tensor} where tensor is the scores/data
    """
    # Reset previous running statistics before recomputing.
    self.running_min.fill_(float("nan"))
    self.running_max.fill_(float("nan"))
    self._statistically_initialized = False

    all_mins = []
    all_maxs = []
    cap = self.max_initialization_frames
    frames_seen = 0

    for batch_data in input_stream:
        # Extract data from port-based dict
        x = batch_data.get("data")
        if x is not None:
            if cap is not None:
                x = x[: cap - frames_seen]
            frames_seen += x.shape[0]
            # Flatten spatial dimensions
            flat = x.reshape(x.shape[0], -1)
            batch_min = flat.min()
            batch_max = flat.max()
            all_mins.append(batch_min)
            all_maxs.append(batch_max)
            if cap is not None and frames_seen >= cap:
                break

    if not all_mins:
        raise RuntimeError(
            "MinMaxNormalizer.statistical_initialization() did not receive any data."
        )

    self.running_min.copy_(torch.stack(all_mins).min())
    self.running_max.copy_(torch.stack(all_maxs).max())
    self._statistically_initialized = True
```

`MultiRangeSlicer`cuvis_ai.node.deciders.multi_range_decidertransformclassposttorchBucket a per-pixel score map into ordered class indices by edges.

#### MultiRangeSlicer

```python
MultiRangeSlicer(edges=None, right=False, **kwargs)
```

Bases: `Node`

Bucket a per-pixel score map into ordered class indices by edges.

Each pixel score is assigned the index of the half-open range it falls into, delegating to :func:`torch.bucketize`. With `edges = [e0, e1, ..., e_{k-1}]` the output index is `0` for scores below the first edge and `k` for scores at/above the last edge.

Convention

`torch.bucketize(x, edges, right=False)` is equivalent to `numpy.digitize(x, edges, right=True)` (and `right=True` corresponds to `numpy.digitize(..., right=False)`). The `right` flag selects whether a value exactly equal to an edge falls into the lower or upper bucket: with `right=False` (the default here) an edge value goes to the upper bucket, matching `numpy.digitize(..., right=True)`.

Parameters:

| Name    | Type          | Description                                                                                                  | Default |
| ------- | ------------- | ------------------------------------------------------------------------------------------------------------ | ------- |
| `edges` | `list[float]` | Monotonically increasing bucket boundaries. Default: [0.25, 0.5, 0.75].                                      | `None`  |
| `right` | `bool`        | Passed through to :func:torch.bucketize. Controls edge-equality behavior as described above. Default: False. | `False` |

Examples:

```pycon
>>> slicer = MultiRangeSlicer(edges=[0.25, 0.5, 0.75])
>>> scores = torch.tensor([[[[0.1], [0.3], [0.6], [0.9]]]])
>>> slicer.forward(scores=scores)["class_mask"]
tensor([[[0, 1, 2, 3]]], dtype=torch.int32)
```

Source code in `cuvis_ai/node/deciders/multi_range_decider.py`

```python
def __init__(
    self,
    edges: list[float] | None = None,
    right: bool = False,
    **kwargs: Any,
) -> None:
    self.edges = [float(e) for e in (edges if edges is not None else [0.25, 0.5, 0.75])]
    if any(b <= a for a, b in zip(self.edges, self.edges[1:], strict=False)):
        raise ValueError(f"edges must be strictly increasing, got {self.edges}")
    self.right = bool(right)
    super().__init__(edges=self.edges, right=self.right, **kwargs)
```

##### forward

```python
forward(scores, **_)
```

Slice the score map into ordered bucket indices.

Parameters:

| Name     | Type     | Description                       | Default    |
| -------- | -------- | --------------------------------- | ---------- |
| `scores` | `Tensor` | Per-pixel score map [B, H, W, 1]. | *required* |

Returns:

| Type                | Description                                               |
| ------------------- | --------------------------------------------------------- |
| `dict[str, Tensor]` | {"class_mask": Tensor [B, H, W]} of int32 bucket indices. |

Source code in `cuvis_ai/node/deciders/multi_range_decider.py`

```python
@torch.no_grad()
def forward(self, scores: Tensor, **_: Any) -> dict[str, Tensor]:
    """Slice the score map into ordered bucket indices.

    Parameters
    ----------
    scores : Tensor
        Per-pixel score map ``[B, H, W, 1]``.

    Returns
    -------
    dict[str, Tensor]
        ``{"class_mask": Tensor [B, H, W]}`` of int32 bucket indices.
    """
    boundaries = torch.tensor(self.edges, dtype=scores.dtype, device=scores.device)
    idx = torch.bucketize(scores.squeeze(-1), boundaries, right=self.right)
    return {"class_mask": idx.to(torch.int32)}
```

`NBRSelector`cuvis_ai.node.channel_selectortransformdim-redhsinumpypreNormalized Burn Ratio renderer.

#### NBRSelector

```python
NBRSelector(
    nir_nm=850.0,
    swir_nm=2200.0,
    colormap_min=-1.0,
    colormap_max=1.0,
    eps=1e-06,
    **kwargs,
)
```

Bases: `_ColormappedNormalizedDifferenceSelector`

Normalized Burn Ratio renderer.

Computes `(CUBE(nir_nm) - CUBE(swir_nm)) / (CUBE(nir_nm) + CUBE(swir_nm))`. Bands are resolved by nearest available sensor wavelength. The raw index map is returned via `index_image` and `rgb_image` carries an HSV colour-mapped render.

Source code in `cuvis_ai/node/channel_selector.py`

```python
def __init__(
    self,
    nir_nm: float = 850.0,
    swir_nm: float = 2200.0,
    colormap_min: float = -1.0,
    colormap_max: float = 1.0,
    eps: float = 1.0e-6,
    **kwargs: Any,
) -> None:
    super().__init__(
        primary_nm=nir_nm,
        secondary_nm=swir_nm,
        colormap_min=colormap_min,
        colormap_max=colormap_max,
        eps=eps,
        nir_nm=float(nir_nm),
        swir_nm=float(swir_nm),
        **kwargs,
    )
    self.nir_nm = float(nir_nm)
    self.swir_nm = float(swir_nm)
```

##### index_name

```python
index_name
```

Canonical NBR strategy name.

##### primary_label

```python
primary_label
```

NBR primary operand label.

##### secondary_label

```python
secondary_label
```

NBR secondary operand label.

`NDRESelector`cuvis_ai.node.channel_selectortransformdim-redhsinumpypreNormalized Difference Red Edge index renderer.

#### NDRESelector

```python
NDRESelector(
    nir_nm=800.0,
    red_edge_nm=720.0,
    colormap_min=-1.0,
    colormap_max=1.0,
    eps=1e-06,
    **kwargs,
)
```

Bases: `_ColormappedNormalizedDifferenceSelector`

Normalized Difference Red Edge index renderer.

Computes `(CUBE(nir_nm) - CUBE(red_edge_nm)) / (CUBE(nir_nm) + CUBE(red_edge_nm))`. Bands are resolved by nearest available sensor wavelength. The raw index map is returned via `index_image` and `rgb_image` carries an HSV colour-mapped render.

Source code in `cuvis_ai/node/channel_selector.py`

```python
def __init__(
    self,
    nir_nm: float = 800.0,
    red_edge_nm: float = 720.0,
    colormap_min: float = -1.0,
    colormap_max: float = 1.0,
    eps: float = 1.0e-6,
    **kwargs: Any,
) -> None:
    super().__init__(
        primary_nm=nir_nm,
        secondary_nm=red_edge_nm,
        colormap_min=colormap_min,
        colormap_max=colormap_max,
        eps=eps,
        nir_nm=float(nir_nm),
        red_edge_nm=float(red_edge_nm),
        **kwargs,
    )
    self.nir_nm = float(nir_nm)
    self.red_edge_nm = float(red_edge_nm)
```

##### index_name

```python
index_name
```

Canonical NDRE strategy name.

##### primary_label

```python
primary_label
```

NDRE primary operand label.

##### secondary_label

```python
secondary_label
```

NDRE secondary operand label.

`NDVISelector`cuvis_ai.node.channel_selectortransformdim-redhsinumpypreNormalized Difference Vegetation Index renderer.

#### NDVISelector

```python
NDVISelector(
    nir_nm=827.0,
    red_nm=668.0,
    colormap_min=-0.7,
    colormap_max=0.5,
    eps=1e-06,
    **kwargs,
)
```

Bases: `_NormalizedDifferenceIndexBase`

Normalized Difference Vegetation Index renderer.

Computes:

`(CUBE(nir_nm) - CUBE(red_nm)) / (CUBE(nir_nm) + CUBE(red_nm))`

Bands are resolved by nearest available sensor wavelength. The raw NDVI map is returned via `index_image` and `rgb_image` contains a colour-mapped render. The scalar NDVI image is mapped with the HSV-style colormap used by the `Blood_OXY` plugin XML.

Source code in `cuvis_ai/node/channel_selector.py`

```python
def __init__(
    self,
    nir_nm: float = 827.0,
    red_nm: float = 668.0,
    colormap_min: float = -0.7,
    colormap_max: float = 0.5,
    eps: float = 1.0e-6,
    **kwargs: Any,
) -> None:
    if colormap_max <= colormap_min:
        raise ValueError("colormap_max must be greater than colormap_min")
    kwargs.setdefault("norm_mode", NormMode.PER_FRAME)
    kwargs.setdefault("apply_gamma", False)
    super().__init__(
        primary_nm=nir_nm,
        secondary_nm=red_nm,
        eps=eps,
        nir_nm=float(nir_nm),
        red_nm=float(red_nm),
        colormap_min=float(colormap_min),
        colormap_max=float(colormap_max),
        **kwargs,
    )
    self.nir_nm = float(nir_nm)
    self.red_nm = float(red_nm)
    self.colormap = "hsv"
    self.colormap_min = float(colormap_min)
    self.colormap_max = float(colormap_max)
    self._colormap_range = self.colormap_max - self.colormap_min
```

##### index_name

```python
index_name
```

Canonical NDVI strategy name.

##### primary_label

```python
primary_label
```

NDVI primary operand label.

##### secondary_label

```python
secondary_label
```

NDVI secondary operand label.

##### forward

```python
forward(cube, wavelengths, context=None, **_)
```

Compute NDVI plus colour-mapped RGB output.

Source code in `cuvis_ai/node/channel_selector.py`

```python
def forward(
    self,
    cube: torch.Tensor,
    wavelengths: Any,
    context: Context | None = None,  # noqa: ARG002
    **_: Any,
) -> dict[str, Any]:
    """Compute NDVI plus colour-mapped RGB output."""
    result = super().forward(cube=cube, wavelengths=wavelengths, context=context, **_)
    result["band_info"].update(
        {
            "rendering": f"{self.colormap}_colormap",
            "colormap": self.colormap,
            "colormap_min": self.colormap_min,
            "colormap_max": self.colormap_max,
        }
    )
    return result
```

`NDWISelector`cuvis_ai.node.channel_selectortransformdim-redhsinumpypreNormalized Difference Water Index renderer.

#### NDWISelector

```python
NDWISelector(
    green_nm=560.0,
    nir_nm=860.0,
    colormap_min=-1.0,
    colormap_max=1.0,
    eps=1e-06,
    **kwargs,
)
```

Bases: `_ColormappedNormalizedDifferenceSelector`

Normalized Difference Water Index renderer.

Computes `(CUBE(green_nm) - CUBE(nir_nm)) / (CUBE(green_nm) + CUBE(nir_nm))`. Bands are resolved by nearest available sensor wavelength. The raw index map is returned via `index_image` and `rgb_image` carries an HSV colour-mapped render.

Source code in `cuvis_ai/node/channel_selector.py`

```python
def __init__(
    self,
    green_nm: float = 560.0,
    nir_nm: float = 860.0,
    colormap_min: float = -1.0,
    colormap_max: float = 1.0,
    eps: float = 1.0e-6,
    **kwargs: Any,
) -> None:
    super().__init__(
        primary_nm=green_nm,
        secondary_nm=nir_nm,
        colormap_min=colormap_min,
        colormap_max=colormap_max,
        eps=eps,
        green_nm=float(green_nm),
        nir_nm=float(nir_nm),
        **kwargs,
    )
    self.green_nm = float(green_nm)
    self.nir_nm = float(nir_nm)
```

##### index_name

```python
index_name
```

Canonical NDWI strategy name.

##### primary_label

```python
primary_label
```

NDWI primary operand label.

##### secondary_label

```python
secondary_label
```

NDWI secondary operand label.

`NearestLabelFill`cuvis_ai.node.mask_opstransformclassmaskpostsegFill morphology-removed gaps in a label map with the nearest surviving label.

#### NearestLabelFill

```python
NearestLabelFill(background_value=-1, **kwargs)
```

Bases: `Node`

Fill morphology-removed gaps in a label map with the nearest surviving label.

After per-class morphology (:class:`ClassMapRobustifier`) some pixels that were labelled in the original map are dropped to `background_value`. This node repaints every such gap with the label of its nearest surviving pixel, found by iterative single-pixel dilation (8-connected / Chebyshev nearest; ties resolved toward the larger class id). Gaps no label can reach -- e.g. a class wiped out entirely by an area filter -- fall back to the original label on the `source` port. The foreground to fill is `source != background_value`.

Parameters:

| Name               | Type  | Description                                                                    | Default |
| ------------------ | ----- | ------------------------------------------------------------------------------ | ------- |
| `background_value` | `int` | Label value treated as "unassigned" in both inputs and the output. Default -1. | `-1`    |

Source code in `cuvis_ai/node/mask_ops.py`

```python
def __init__(self, background_value: int = -1, **kwargs: Any) -> None:
    self.background_value = int(background_value)
    super().__init__(background_value=self.background_value, **kwargs)
```

##### forward

```python
forward(class_map, source, **_)
```

Grow surviving labels into the gaps; fall back to `source` where unreachable.

Source code in `cuvis_ai/node/mask_ops.py`

```python
@torch.no_grad()
def forward(
    self, class_map: torch.Tensor, source: torch.Tensor, **_: Any
) -> dict[str, torch.Tensor]:
    """Grow surviving labels into the gaps; fall back to ``source`` where unreachable."""
    bg = self.background_value
    neg = -1.0e9  # any class id (>= 0) beats this in the max-pool, so unknown never wins
    out = class_map.clone()
    target = source != bg
    known = out != bg
    remaining = target & ~known
    while bool(remaining.any()):
        filled = out.to(torch.float32).masked_fill(~known, neg)
        cand = F.max_pool2d(filled.unsqueeze(1), kernel_size=3, stride=1, padding=1).squeeze(1)
        known_f = known.to(torch.float32).unsqueeze(1)
        reach = F.max_pool2d(known_f, kernel_size=3, stride=1, padding=1).squeeze(1) > 0
        newly = remaining & reach
        if not bool(newly.any()):
            break
        out[newly] = cand[newly].round().to(out.dtype)
        known = out != bg
        remaining = target & ~known
    still = target & (out == bg)
    out[still] = source[still]
    return {"class_map": out}
```

`OcclusionNodeBase`cuvis_ai.node.occlusiontransformaugnumpystochtrainBase class for synthetic occlusion from tracking masks.

#### OcclusionNodeBase

```python
OcclusionNodeBase(
    tracking_json_path,
    track_ids,
    occlusion_start_frame,
    occlusion_end_frame,
    **kwargs,
)
```

Bases: `Node`, `ABC`

Base class for synthetic occlusion from tracking masks.

Source code in `cuvis_ai/node/occlusion.py`

```python
def __init__(
    self,
    tracking_json_path: str,
    track_ids: list[int],
    occlusion_start_frame: int,
    occlusion_end_frame: int,
    **kwargs,
) -> None:
    path = Path(tracking_json_path)
    if not path.is_file():
        raise FileNotFoundError(f"Tracking JSON not found: {tracking_json_path}")

    data = json.loads(path.read_text(encoding="utf-8"))
    track_id_set = set(track_ids)

    self._masks_by_frame: dict[int, list[dict]] = {}
    for ann in data.get("annotations", []):
        tid = ann.get("track_id")
        if tid not in track_id_set:
            continue
        fid = int(ann["image_id"])
        if fid < occlusion_start_frame or fid > occlusion_end_frame:
            continue
        seg = ann.get("segmentation")
        if seg is None or not isinstance(seg, dict):
            continue
        entry = {
            "track_id": int(tid),
            "bbox": ann["bbox"],
            "segmentation": seg,
        }
        self._masks_by_frame.setdefault(fid, []).append(entry)

    self.occlusion_start_frame = int(occlusion_start_frame)
    self.occlusion_end_frame = int(occlusion_end_frame)

    n_frames = len(self._masks_by_frame)
    n_annots = sum(len(v) for v in self._masks_by_frame.values())
    logger.info(
        "OcclusionNode: loaded {} annotations across {} frames for tracks {} (range [{}, {}])",
        n_annots,
        n_frames,
        track_ids,
        occlusion_start_frame,
        occlusion_end_frame,
    )

    super().__init__(
        tracking_json_path=tracking_json_path,
        track_ids=track_ids,
        occlusion_start_frame=occlusion_start_frame,
        occlusion_end_frame=occlusion_end_frame,
        **kwargs,
    )
```

##### forward

```python
forward(rgb_image, frame_id, **_)
```

Conditionally occlude an RGB batch using tracking-derived masks.

Source code in `cuvis_ai/node/occlusion.py`

```python
@torch.no_grad()
def forward(
    self,
    rgb_image: torch.Tensor,
    frame_id: torch.Tensor,
    **_,
) -> dict[str, torch.Tensor]:
    """Conditionally occlude an RGB batch using tracking-derived masks."""
    return self._forward_tensor(data=rgb_image, output_key="rgb_image", frame_id=frame_id)
```

`PCA`cuvis_ai.node.dimensionality_reductiontransformdim-redhsipretorchProject each frame independently onto its principal components.

#### PCA

```python
PCA(n_components, eps=1e-06, **kwargs)
```

Bases: `Node`

Project each frame independently onto its principal components.

Source code in `cuvis_ai/node/dimensionality_reduction.py`

```python
def __init__(
    self,
    n_components: int,
    eps: float = 1e-6,
    **kwargs,
) -> None:
    self.n_components = int(n_components)
    self.eps = float(eps)

    super().__init__(n_components=self.n_components, eps=self.eps, **kwargs)
```

##### forward

```python
forward(data, **_)
```

Fit PCA independently on each frame and return the per-frame projection.

Source code in `cuvis_ai/node/dimensionality_reduction.py`

```python
def forward(self, data: Tensor, **_: Any) -> dict[str, Tensor]:
    """Fit PCA independently on each frame and return the per-frame projection."""
    if data.ndim != 4:
        raise ValueError(f"Expected data with shape [B, H, W, C], got {tuple(data.shape)}")
    if data.shape[0] == 0:
        raise ValueError("PCA requires a non-empty batch.")

    projected_frames: list[Tensor] = []
    explained_variance_ratio: Tensor | None = None
    components: Tensor | None = None

    for frame in data:
        frame_components, mean, eigenvalues = self._fit_frame(frame)
        flat = frame.reshape(-1, frame.shape[-1]).to(dtype=torch.float32)
        projected = self._project(flat, mean, frame_components).reshape(
            frame.shape[0],
            frame.shape[1],
            self.n_components,
        )

        projected_frames.append(projected.to(dtype=torch.float32))
        explained_variance_ratio = self._variance_ratio(eigenvalues).to(device=data.device)
        components = frame_components.to(device=data.device)

    assert explained_variance_ratio is not None
    assert components is not None

    return {
        "projected": torch.stack(projected_frames, dim=0),
        "explained_variance_ratio": explained_variance_ratio,
        "components": components,
    }
```

`PRISelector`cuvis_ai.node.channel_selectortransformdim-redhsinumpyprePhotochemical Reflectance Index renderer.

#### PRISelector

```python
PRISelector(
    band1_nm=531.0,
    band2_nm=570.0,
    colormap_min=-1.0,
    colormap_max=1.0,
    eps=1e-06,
    **kwargs,
)
```

Bases: `_VegetationIndexBase`

Photochemical Reflectance Index renderer.

Computes `(R531 - R570) / (R531 + R570)` over bands resolved by nearest sensor wavelength. The raw index map is returned via `index_image` and `rgb_image` carries an HSV colour-mapped render.

Source code in `cuvis_ai/node/channel_selector.py`

```python
def __init__(
    self,
    band1_nm: float = 531.0,
    band2_nm: float = 570.0,
    colormap_min: float = -1.0,
    colormap_max: float = 1.0,
    eps: float = 1.0e-6,
    **kwargs: Any,
) -> None:
    super().__init__(
        band_nm={"band1": band1_nm, "band2": band2_nm},
        colormap_min=colormap_min,
        colormap_max=colormap_max,
        eps=eps,
        band1_nm=float(band1_nm),
        band2_nm=float(band2_nm),
        **kwargs,
    )
    self.band1_nm = float(band1_nm)
    self.band2_nm = float(band2_nm)
```

##### index_name

```python
index_name
```

Canonical PRI strategy name.

`PatchSampler`cuvis_ai.node.patch_inferencetransformhsiprestochExtract labeled center-pixel patches from a cube and an integer target map.

#### PatchSampler

```python
PatchSampler(
    patch_size=7,
    samples_per_frame=256,
    class_balanced=True,
    ignore_index=-100,
    mode="train",
    max_per_frame=None,
    **kwargs,
)
```

Bases: `Node`

Extract labeled center-pixel patches from a cube and an integer target map.

For each frame, gather the pixels whose target is not `ignore_index` and, around each, cut a `patch_size` x `patch_size` window (reflect-padded at borders), emitting `patches` `[N, P, P, C]` and integer `labels` `[N]`. `patch_size=1` yields single-pixel spectra; larger odd sizes yield spatial-spectral patches.

`mode="train"` draws `samples_per_frame` center pixels per frame (class-balanced by default, with replacement); `mode="eval"` takes every labeled pixel, optionally strided down to `max_per_frame` for dense scoring. Sampling uses torch's global RNG, so seeding torch makes a run reproducible while still drawing fresh patches each call.

Parameters:

| Name                | Type          | Description                                                                         | Default   |
| ------------------- | ------------- | ----------------------------------------------------------------------------------- | --------- |
| `patch_size`        | `int`         | Side of the square window; must be a positive odd integer. Default 7.               | `7`       |
| `samples_per_frame` | `int`         | Center pixels drawn per frame in mode="train". Default 256.                         | `256`     |
| `class_balanced`    | `bool`        | In mode="train", draw an equal share per present class. Default True.               | `True`    |
| `ignore_index`      | `int`         | Target value marking pixels to skip (never sampled). Default -100.                  | `-100`    |
| `mode`              | `str`         | "train" (random per-frame sample) or "eval" (every labeled pixel). Default "train". | `'train'` |
| `max_per_frame`     | `int or None` | In mode="eval", strided cap on labeled pixels per frame (None keeps all).           | `None`    |

Validate and store the sampling hyperparameters.

Source code in `cuvis_ai/node/patch_inference.py`

```python
def __init__(
    self,
    patch_size: int = 7,
    samples_per_frame: int = 256,
    class_balanced: bool = True,
    ignore_index: int = -100,
    mode: str = "train",
    max_per_frame: int | None = None,
    **kwargs: Any,
) -> None:
    """Validate and store the sampling hyperparameters."""
    if patch_size < 1 or patch_size % 2 == 0:
        raise ValueError(f"patch_size must be a positive odd int; got {patch_size}")
    if mode not in ("train", "eval"):
        raise ValueError(f"mode must be 'train' or 'eval'; got {mode!r}")
    self.patch_size = int(patch_size)
    self.samples_per_frame = int(samples_per_frame)
    self.class_balanced = bool(class_balanced)
    self.ignore_index = int(ignore_index)
    self.mode = mode
    self.max_per_frame = None if max_per_frame is None else int(max_per_frame)
    super().__init__(
        patch_size=self.patch_size,
        samples_per_frame=self.samples_per_frame,
        class_balanced=self.class_balanced,
        ignore_index=self.ignore_index,
        mode=self.mode,
        max_per_frame=self.max_per_frame,
        **kwargs,
    )
```

##### forward

```python
forward(cube, targets, **_)
```

Sample patches and labels across the batch's frames.

Parameters:

| Name      | Type     | Description                                                                 | Default    |
| --------- | -------- | --------------------------------------------------------------------------- | ---------- |
| `cube`    | `Tensor` | Hyperspectral cube [B, H, W, C].                                            | *required* |
| `targets` | `Tensor` | Per-pixel integer class targets [B, H, W]; ignore_index pixels are skipped. | *required* |
| `**_`     | `Any`    | Additional unused keyword arguments (e.g. the pipeline context).            | `{}`       |

Returns:

| Type                | Description                                                                                                  |
| ------------------- | ------------------------------------------------------------------------------------------------------------ |
| `dict[str, Tensor]` | patches float32 [N, P, P, C] and labels int64 [N]; both empty when no labeled pixel is present in the batch. |

Source code in `cuvis_ai/node/patch_inference.py`

```python
@torch.no_grad()
def forward(
    self, cube: torch.Tensor, targets: torch.Tensor, **_: Any
) -> dict[str, torch.Tensor]:
    """Sample patches and labels across the batch's frames.

    Parameters
    ----------
    cube : torch.Tensor
        Hyperspectral cube ``[B, H, W, C]``.
    targets : torch.Tensor
        Per-pixel integer class targets ``[B, H, W]``; ``ignore_index`` pixels are skipped.
    **_ : Any
        Additional unused keyword arguments (e.g. the pipeline ``context``).

    Returns
    -------
    dict[str, torch.Tensor]
        ``patches`` float32 ``[N, P, P, C]`` and ``labels`` int64 ``[N]``; both empty when no
        labeled pixel is present in the batch.
    """
    b, _, _, c = cube.shape
    p = self.patch_size
    r = p // 2
    out_patches: list[torch.Tensor] = []
    out_labels: list[torch.Tensor] = []
    for i in range(b):
        cube_i = cube[i]  # [H,W,C]
        tgt_i = targets[i]  # [H,W]
        coords = (tgt_i != self.ignore_index).nonzero(as_tuple=False)  # [M,2]
        if coords.shape[0] == 0:
            continue
        labels_i = tgt_i[coords[:, 0], coords[:, 1]]  # [M]
        if self.mode == "train":
            sel = self._sample_indices(labels_i)
        else:
            sel = torch.arange(coords.shape[0])
            if self.max_per_frame and sel.numel() > self.max_per_frame:
                step = math.ceil(sel.numel() / self.max_per_frame)
                sel = sel[::step]
        ci = coords[sel]  # [N,2]
        lab = labels_i[sel]  # [N]
        if r > 0:
            padded = F.pad(cube_i.permute(2, 0, 1).unsqueeze(0), (r, r, r, r), mode="reflect")[
                0
            ].permute(1, 2, 0)  # [H+2r, W+2r, C]
        else:
            padded = cube_i
        rows = ci[:, 0].unsqueeze(1) + torch.arange(p, device=ci.device)  # [N,P]
        cols = ci[:, 1].unsqueeze(1) + torch.arange(p, device=ci.device)  # [N,P]
        patches = padded[rows[:, :, None], cols[:, None, :], :]  # [N,P,P,C]
        out_patches.append(patches)
        out_labels.append(lab)

    if not out_patches:
        return {
            "patches": cube.new_zeros((0, p, p, c)),
            "labels": targets.new_zeros((0,), dtype=torch.int64),
        }
    return {"patches": torch.cat(out_patches), "labels": torch.cat(out_labels).to(torch.int64)}
```

`PatchSampler`cuvis_ai_inspecscrap.node.patch_samplertransformhsiprecuvis_ai_inspecscrap[↗](https://github.com/cubert-hyperspectral/cuvis-ai-inspecscrap "Open plugin repo")

[View plugin repo (v0.2.4)](https://github.com/cubert-hyperspectral/cuvis-ai-inspecscrap/tree/v0.2.4)

`PerChannelStandardizer`cuvis_ai_inspecscrap.node.normalizationtransformhsinormprecuvis_ai_inspecscrap[↗](https://github.com/cubert-hyperspectral/cuvis-ai-inspecscrap "Open plugin repo")

[View plugin repo (v0.2.4)](https://github.com/cubert-hyperspectral/cuvis-ai-inspecscrap/tree/v0.2.4)

`PerPixelUnitNorm`cuvis_ai.node.normalizationtransformnormnumpyprePer-pixel mean-centering and L2 normalization across channels.

#### PerPixelUnitNorm

```python
PerPixelUnitNorm(eps=1e-08, **kwargs)
```

Bases: `_ScoreNormalizerBase`

Per-pixel mean-centering and L2 normalization across channels.

Source code in `cuvis_ai/node/normalization.py`

```python
def __init__(self, eps: float = 1e-8, **kwargs) -> None:
    self.eps = float(eps)
    super().__init__(eps=self.eps, **kwargs)
```

##### forward

```python
forward(data, **_)
```

Normalize BHWC tensors per pixel.

Source code in `cuvis_ai/node/normalization.py`

```python
def forward(self, data: Tensor, **_: Any) -> dict[str, Tensor]:
    """Normalize BHWC tensors per pixel."""
    normalized = self._normalize(data)
    return {"normalized": normalized}
```

`PercentileNormalizer`cuvis_ai.node.normalizationtransformnormnumpyprePer-channel normalization to `[0, 1]` for BHWC data of any channel count.

#### PercentileNormalizer

```python
PercentileNormalizer(
    n_channels,
    norm_mode=RUNNING,
    freeze_running_bounds_after_frames=20,
    running_warmup_frames=10,
    quantile_low=0.005,
    quantile_high=0.995,
    eps=1e-08,
    **kwargs,
)
```

Bases: `_ScoreNormalizerBase`

Per-channel normalization to `[0, 1]` for BHWC data of any channel count.

Extracted from `ChannelSelectorBase` so band selection and display normalization are separate, composable steps. Operates on any channel count `C` (fixed at construction via `n_channels`). Does **not** apply sRGB gamma; chain :class:`DisplayNormalizer` after it for the false-RGB display path. ML / n-channel callers use this node alone.

Modes (`norm_mode`):

- `per_frame`: per-batch, per-channel absolute min/max; no inter-frame state.
- `statistical`: global percentile bounds precomputed via `StatisticalTrainer`.
- `running` (default): the first `running_warmup_frames` frames use per-frame percentile normalization while accumulating global percentile bounds (min-of-lows, max-of-highs); afterwards those bounds are used, frozen after `freeze_running_bounds_after_frames` calls. Bounds update on every call including inference, which live false-RGB video relies on; the freeze guards late drift.

Parameters:

| Name                                 | Type    | Description                                                                           | Default                                                                                            |
| ------------------------------------ | ------- | ------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------- |
| `n_channels`                         | `int`   | Channel count C of the input. Sizes the per-channel bound buffers.                    | *required*                                                                                         |
| `norm_mode`                          | \`str   | NormMode\`                                                                            | Normalization mode. Default running.                                                               |
| `freeze_running_bounds_after_frames` | \`int   | None\`                                                                                | Stop updating running bounds after this many calls. Default 20; None keeps unbounded accumulation. |
| `running_warmup_frames`              | `int`   | Frames to normalize per-frame while accumulating bounds. Default 10.                  | `10`                                                                                               |
| `quantile_low`                       | `float` | Percentile bounds (fractions) for running / statistical modes. Default 0.005 / 0.995. | `0.005`                                                                                            |
| `quantile_high`                      | `float` | Percentile bounds (fractions) for running / statistical modes. Default 0.005 / 0.995. | `0.005`                                                                                            |
| `eps`                                | `float` | Floor for the (max - min) denominator. Default 1e-8.                                  | `1e-08`                                                                                            |

Ports

INPUT_SPECS `data` : float32, shape (-1, -1, -1, -1), BHWC tensor with `C == n_channels`. OUTPUT_SPECS `normalized` : float32, shape (-1, -1, -1, -1), same shape, values in `[0, 1]`.

Source code in `cuvis_ai/node/normalization.py`

```python
def __init__(
    self,
    n_channels: int,
    norm_mode: str | NormMode = NormMode.RUNNING,
    freeze_running_bounds_after_frames: int | None = 20,
    running_warmup_frames: int = 10,
    quantile_low: float = 0.005,
    quantile_high: float = 0.995,
    eps: float = 1e-8,
    **kwargs: Any,
) -> None:
    if isinstance(n_channels, bool) or not isinstance(n_channels, int) or n_channels < 1:
        raise ValueError("PercentileNormalizer: n_channels must be an integer >= 1")
    norm_mode = NormMode(str(norm_mode) if isinstance(norm_mode, NormMode) else norm_mode)
    if freeze_running_bounds_after_frames is not None and (
        isinstance(freeze_running_bounds_after_frames, bool)
        or not isinstance(freeze_running_bounds_after_frames, int)
        or freeze_running_bounds_after_frames < 1
    ):
        raise ValueError("freeze_running_bounds_after_frames must be an integer >= 1 or None")
    if (
        isinstance(running_warmup_frames, bool)
        or not isinstance(running_warmup_frames, int)
        or running_warmup_frames < 0
    ):
        raise ValueError("running_warmup_frames must be an integer >= 0")
    if not 0.0 <= float(quantile_low) < float(quantile_high) <= 1.0:
        raise ValueError("PercentileNormalizer: require 0 <= quantile_low < quantile_high <= 1")

    self.n_channels = int(n_channels)
    self.norm_mode = norm_mode
    self.freeze_running_bounds_after_frames = freeze_running_bounds_after_frames
    self.running_warmup_frames = int(running_warmup_frames)
    self.quantile_low = float(quantile_low)
    self.quantile_high = float(quantile_high)
    self.eps = float(eps)

    super().__init__(
        n_channels=self.n_channels,
        norm_mode=str(norm_mode),
        freeze_running_bounds_after_frames=freeze_running_bounds_after_frames,
        running_warmup_frames=self.running_warmup_frames,
        quantile_low=self.quantile_low,
        quantile_high=self.quantile_high,
        eps=self.eps,
        **kwargs,
    )

    # Per-channel bounds + frame counter persist in state_dict (so warmup /
    # freeze survive a reload) but are deliberately NOT in TRAINABLE_BUFFERS:
    # they are fitted display statistics, and a gradient-learned bound could
    # violate lo < hi and turn normalization into an unconstrained transform.
    self.register_buffer("running_min", torch.full((self.n_channels,), float("nan")))
    self.register_buffer("running_max", torch.full((self.n_channels,), float("nan")))
    self.register_buffer("_norm_frame_count", torch.zeros((), dtype=torch.long))

    # Only the statistical path needs a fit pass; running / per_frame do not.
    self._requires_initial_fit_override = self.norm_mode == NormMode.STATISTICAL
```

##### statistical_initialization

```python
statistical_initialization(input_stream)
```

Accumulate global per-channel percentile bounds across the dataset.

Preserves the established min-of-batch-lows / max-of-batch-highs accumulation (batch-order sensitive); a true streaming percentile is a deliberate follow-up, not changed here.

Source code in `cuvis_ai/node/normalization.py`

```python
def statistical_initialization(self, input_stream: InputStream) -> None:
    """Accumulate global per-channel percentile bounds across the dataset.

    Preserves the established min-of-batch-lows / max-of-batch-highs
    accumulation (batch-order sensitive); a true streaming percentile is a
    deliberate follow-up, not changed here.
    """
    for batch_data in input_stream:
        data = batch_data["data"]
        flat = data.reshape(-1, self.n_channels).float()
        frame_lo = torch.quantile(flat, self.quantile_low, dim=0)
        frame_hi = torch.quantile(flat, self.quantile_high, dim=0)
        if torch.isnan(self.running_min).any():
            self.running_min.copy_(frame_lo)
            self.running_max.copy_(frame_hi)
        else:
            torch.minimum(self.running_min, frame_lo, out=self.running_min)
            torch.maximum(self.running_max, frame_hi, out=self.running_max)
    if torch.isnan(self.running_min).any():
        raise RuntimeError("PercentileNormalizer.statistical_initialization received no data")
```

`PoissonCubeOcclusionNode`cuvis_ai.node.occlusiontransformaugnumpystochtrainDeprecated alias of PoissonOcclusionNode with cube-only ports.

#### PoissonCubeOcclusionNode

```python
PoissonCubeOcclusionNode(
    tracking_json_path,
    track_ids,
    occlusion_start_frame,
    occlusion_end_frame,
    fill_color="poisson",
    *,
    input_key=None,
    max_iter=1000,
    tol=1e-06,
    occlusion_shape="bbox",
    bbox_mode="static",
    static_bbox_scale=1.2,
    static_bbox_padding_px=0,
    static_full_width_x=False,
    **kwargs,
)
```

Bases: `PoissonOcclusionNode`

Deprecated alias of PoissonOcclusionNode with cube-only ports.

Source code in `cuvis_ai/node/occlusion.py`

```python
def __init__(
    self,
    tracking_json_path: str,
    track_ids: list[int],
    occlusion_start_frame: int,
    occlusion_end_frame: int,
    fill_color: tuple[float, float, float] | str = "poisson",
    *,
    input_key: str | None = None,
    max_iter: int = 1000,
    tol: float = 1e-6,
    occlusion_shape: str = "bbox",
    bbox_mode: str = "static",
    static_bbox_scale: float = 1.2,
    static_bbox_padding_px: int = 0,
    static_full_width_x: bool = False,
    **kwargs,
) -> None:
    if occlusion_shape not in self._VALID_SHAPES:
        raise ValueError(
            f"occlusion_shape must be one of {self._VALID_SHAPES}, got '{occlusion_shape}'"
        )
    if bbox_mode not in self._VALID_BBOX_MODES:
        raise ValueError(
            f"bbox_mode must be one of {self._VALID_BBOX_MODES}, got '{bbox_mode}'"
        )
    if static_bbox_scale <= 0:
        raise ValueError("static_bbox_scale must be > 0")
    if static_bbox_padding_px < 0:
        raise ValueError("static_bbox_padding_px must be >= 0")
    if int(max_iter) <= 0:
        raise ValueError("max_iter must be > 0")
    if float(tol) <= 0:
        raise ValueError("tol must be > 0")
    if input_key is not None and input_key not in {"rgb_image", "cube"}:
        raise ValueError("input_key must be 'rgb_image', 'cube', or None")

    self._use_poisson_fill = False
    if isinstance(fill_color, str):
        if fill_color != "poisson":
            raise ValueError("fill_color string must be exactly 'poisson'")
        self.fill_color: tuple[float, float, float] | str = fill_color
        self._use_poisson_fill = True
    else:
        parsed_fill = tuple(float(c) for c in fill_color)
        if len(parsed_fill) != 3:
            raise ValueError("fill_color tuple must have exactly 3 values")
        if any(c < 0.0 or c > 1.0 for c in parsed_fill):
            raise ValueError("fill_color tuple values must be in [0, 1]")
        self.fill_color = parsed_fill

    self.max_iter = int(max_iter)
    self.tol = float(tol)
    self.input_key = input_key
    self.occlusion_shape = occlusion_shape
    self.bbox_mode = bbox_mode
    self.static_bbox_scale = float(static_bbox_scale)
    self.static_bbox_padding_px = int(static_bbox_padding_px)
    self.static_full_width_x = bool(static_full_width_x)

    super().__init__(
        tracking_json_path=tracking_json_path,
        track_ids=track_ids,
        occlusion_start_frame=occlusion_start_frame,
        occlusion_end_frame=occlusion_end_frame,
        fill_color=self.fill_color,
        input_key=self.input_key,
        max_iter=self.max_iter,
        tol=self.tol,
        occlusion_shape=occlusion_shape,
        bbox_mode=bbox_mode,
        static_bbox_scale=static_bbox_scale,
        static_bbox_padding_px=static_bbox_padding_px,
        static_full_width_x=static_full_width_x,
        **kwargs,
    )

    self._static_bboxes_by_track: dict[int, list[float]] = {}
    if self.occlusion_shape == "bbox" and self.bbox_mode == "static":
        self._static_bboxes_by_track = self._build_static_bboxes_by_track()
        logger.info(
            "PoissonOcclusionNode static bboxes: {} tracks (scale={}, padding_px={})",
            len(self._static_bboxes_by_track),
            self.static_bbox_scale,
            self.static_bbox_padding_px,
        )
```

##### forward

```python
forward(cube, frame_id, **_)
```

Apply cube-only occlusion using the parent implementation.

Source code in `cuvis_ai/node/occlusion.py`

```python
@torch.no_grad()
def forward(
    self,
    cube: torch.Tensor,
    frame_id: torch.Tensor,
    **_,
) -> dict[str, torch.Tensor]:
    """Apply cube-only occlusion using the parent implementation."""
    return super().forward(frame_id=frame_id, cube=cube)
```

`PoissonOcclusionNode`cuvis_ai.node.occlusiontransformaugnumpystochtrainPure-PyTorch occlusion node for either RGB frames or hyperspectral cubes.

#### PoissonOcclusionNode

```python
PoissonOcclusionNode(
    tracking_json_path,
    track_ids,
    occlusion_start_frame,
    occlusion_end_frame,
    fill_color="poisson",
    *,
    input_key=None,
    max_iter=1000,
    tol=1e-06,
    occlusion_shape="bbox",
    bbox_mode="static",
    static_bbox_scale=1.2,
    static_bbox_padding_px=0,
    static_full_width_x=False,
    **kwargs,
)
```

Bases: `OcclusionNodeBase`

Pure-PyTorch occlusion node for either RGB frames or hyperspectral cubes.

Source code in `cuvis_ai/node/occlusion.py`

```python
def __init__(
    self,
    tracking_json_path: str,
    track_ids: list[int],
    occlusion_start_frame: int,
    occlusion_end_frame: int,
    fill_color: tuple[float, float, float] | str = "poisson",
    *,
    input_key: str | None = None,
    max_iter: int = 1000,
    tol: float = 1e-6,
    occlusion_shape: str = "bbox",
    bbox_mode: str = "static",
    static_bbox_scale: float = 1.2,
    static_bbox_padding_px: int = 0,
    static_full_width_x: bool = False,
    **kwargs,
) -> None:
    if occlusion_shape not in self._VALID_SHAPES:
        raise ValueError(
            f"occlusion_shape must be one of {self._VALID_SHAPES}, got '{occlusion_shape}'"
        )
    if bbox_mode not in self._VALID_BBOX_MODES:
        raise ValueError(
            f"bbox_mode must be one of {self._VALID_BBOX_MODES}, got '{bbox_mode}'"
        )
    if static_bbox_scale <= 0:
        raise ValueError("static_bbox_scale must be > 0")
    if static_bbox_padding_px < 0:
        raise ValueError("static_bbox_padding_px must be >= 0")
    if int(max_iter) <= 0:
        raise ValueError("max_iter must be > 0")
    if float(tol) <= 0:
        raise ValueError("tol must be > 0")
    if input_key is not None and input_key not in {"rgb_image", "cube"}:
        raise ValueError("input_key must be 'rgb_image', 'cube', or None")

    self._use_poisson_fill = False
    if isinstance(fill_color, str):
        if fill_color != "poisson":
            raise ValueError("fill_color string must be exactly 'poisson'")
        self.fill_color: tuple[float, float, float] | str = fill_color
        self._use_poisson_fill = True
    else:
        parsed_fill = tuple(float(c) for c in fill_color)
        if len(parsed_fill) != 3:
            raise ValueError("fill_color tuple must have exactly 3 values")
        if any(c < 0.0 or c > 1.0 for c in parsed_fill):
            raise ValueError("fill_color tuple values must be in [0, 1]")
        self.fill_color = parsed_fill

    self.max_iter = int(max_iter)
    self.tol = float(tol)
    self.input_key = input_key
    self.occlusion_shape = occlusion_shape
    self.bbox_mode = bbox_mode
    self.static_bbox_scale = float(static_bbox_scale)
    self.static_bbox_padding_px = int(static_bbox_padding_px)
    self.static_full_width_x = bool(static_full_width_x)

    super().__init__(
        tracking_json_path=tracking_json_path,
        track_ids=track_ids,
        occlusion_start_frame=occlusion_start_frame,
        occlusion_end_frame=occlusion_end_frame,
        fill_color=self.fill_color,
        input_key=self.input_key,
        max_iter=self.max_iter,
        tol=self.tol,
        occlusion_shape=occlusion_shape,
        bbox_mode=bbox_mode,
        static_bbox_scale=static_bbox_scale,
        static_bbox_padding_px=static_bbox_padding_px,
        static_full_width_x=static_full_width_x,
        **kwargs,
    )

    self._static_bboxes_by_track: dict[int, list[float]] = {}
    if self.occlusion_shape == "bbox" and self.bbox_mode == "static":
        self._static_bboxes_by_track = self._build_static_bboxes_by_track()
        logger.info(
            "PoissonOcclusionNode static bboxes: {} tracks (scale={}, padding_px={})",
            len(self._static_bboxes_by_track),
            self.static_bbox_scale,
            self.static_bbox_padding_px,
        )
```

##### forward

```python
forward(frame_id, rgb_image=None, cube=None, **_)
```

Occlude either the provided RGB batch or cube batch for the current frame.

Source code in `cuvis_ai/node/occlusion.py`

```python
@torch.no_grad()
def forward(
    self,
    frame_id: torch.Tensor,
    rgb_image: torch.Tensor | None = None,
    cube: torch.Tensor | None = None,
    **_,
) -> dict[str, torch.Tensor]:
    """Occlude either the provided RGB batch or cube batch for the current frame."""
    if self.input_key == "rgb_image":
        if rgb_image is None:
            raise ValueError(
                "PoissonOcclusionNode configured for rgb_image but none was provided"
            )
        return self._forward_tensor(data=rgb_image, output_key="rgb_image", frame_id=frame_id)

    if self.input_key == "cube":
        if cube is None:
            raise ValueError("PoissonOcclusionNode configured for cube but none was provided")
        return self._forward_tensor(data=cube, output_key="cube", frame_id=frame_id)

    if (rgb_image is None) and (cube is None):
        raise ValueError("PoissonOcclusionNode requires exactly one input: rgb_image or cube")
    if (rgb_image is not None) and (cube is not None):
        raise ValueError("PoissonOcclusionNode accepts either rgb_image or cube, not both")

    if rgb_image is not None:
        return self._forward_tensor(data=rgb_image, output_key="rgb_image", frame_id=frame_id)

    assert cube is not None
    return self._forward_tensor(data=cube, output_key="cube", frame_id=frame_id)
```

`QuantileBinaryDecider`cuvis_ai.node.deciders.binary_decidertransformclassnumpypostQuantile-based thresholding node operating on BHWC logits or scores.

#### QuantileBinaryDecider

```python
QuantileBinaryDecider(
    quantile=0.995, reduce_dims=None, **kwargs
)
```

Bases: `BinaryDecider`

Quantile-based thresholding node operating on BHWC logits or scores.

This decider computes a tensor-valued threshold per batch item using the requested quantile over one or more non-batch dimensions, then produces a binary mask where values greater than or equal to that threshold are marked as anomalies. Useful for adaptive thresholding when score distributions vary across batches.

Parameters:

| Name          | Type            | Description                                                                                                                                                                             | Default                                                                                                                                                                                                             |
| ------------- | --------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `quantile`    | `float`         | Quantile in the closed interval [0, 1] used for the threshold computation (default: 0.995). Higher values (e.g., 0.99, 0.995) are typical for anomaly detection to capture rare events. | `0.995`                                                                                                                                                                                                             |
| `reduce_dims` | \`Sequence[int] | None\`                                                                                                                                                                                  | Axes (relative to the input tensor) over which to compute the quantile. When None (default), all non-batch dimensions (H, W, C) are reduced. For per-channel thresholds, use reduce_dims=[1, 2] (reduce H, W only). |

Examples:

```pycon
>>> from cuvis_ai.node.deciders.binary_decider import QuantileBinaryDecider
>>> import torch
>>>
>>> # Create quantile-based decider (99.5th percentile)
>>> decider = QuantileBinaryDecider(quantile=0.995)
>>>
>>> # Apply to anomaly scores
>>> scores = torch.randn(4, 256, 256, 1)  # [B, H, W, C]
>>> output = decider.forward(logits=scores)
>>> decisions = output["decisions"]  # [4, 256, 256, 1] boolean mask
>>>
>>> # Per-channel thresholding (reduce H, W only)
>>> decider_perchannel = QuantileBinaryDecider(
...     quantile=0.99,
...     reduce_dims=[1, 2],  # Compute threshold per channel
... )
```

See Also

BinaryDecider : Fixed threshold decisioning

Source code in `cuvis_ai/node/deciders/binary_decider.py`

```python
def __init__(
    self,
    quantile: float = 0.995,
    reduce_dims: Sequence[int] | None = None,
    **kwargs,
) -> None:
    self._validate_quantile(quantile)
    self.quantile = float(quantile)
    self.reduce_dims = (
        tuple(int(dim) for dim in reduce_dims) if reduce_dims is not None else None
    )
    # Forward init params so Serializable records them for config serialization
    super().__init__(quantile=self.quantile, reduce_dims=self.reduce_dims, **kwargs)
```

##### forward

```python
forward(logits, **_)
```

Apply quantile-based thresholding to produce binary decisions.

Computes per-batch thresholds using the specified quantile over reduce_dims, then classifies values >= threshold as anomalies.

Parameters:

| Name     | Type     | Description                                        | Default    |
| -------- | -------- | -------------------------------------------------- | ---------- |
| `logits` | `Tensor` | Input logits or anomaly scores, shape (B, H, W, C) | *required* |

Returns:

| Type                | Description                                                                          |
| ------------------- | ------------------------------------------------------------------------------------ |
| `dict[str, Tensor]` | Dictionary containing: "decisions" : Tensor Binary decision mask, shape (B, H, W, 1) |

Source code in `cuvis_ai/node/deciders/binary_decider.py`

```python
def forward(self, logits: Tensor, **_: Any) -> dict[str, Tensor]:
    """Apply quantile-based thresholding to produce binary decisions.

    Computes per-batch thresholds using the specified quantile over reduce_dims,
    then classifies values >= threshold as anomalies.

    Parameters
    ----------
    logits : Tensor
        Input logits or anomaly scores, shape (B, H, W, C)

    Returns
    -------
    dict[str, Tensor]
        Dictionary containing:

        - "decisions" : Tensor
            Binary decision mask, shape (B, H, W, 1)
    """
    tensor = logits
    dims = resolve_reduce_dims(self.reduce_dims, tensor.dim())

    if len(dims) == 1:
        threshold = torch.quantile(
            tensor,
            self.quantile,
            dim=dims[0],
            keepdim=True,
        )
    else:
        tensor_ndim = tensor.dim()
        dims_to_keep = tuple(i for i in range(tensor_ndim) if i not in dims)
        new_order = (*dims_to_keep, *dims)
        permuted = tensor.permute(new_order)
        sizes_keep = [permuted.size(i) for i in range(len(dims_to_keep))]
        flattened = permuted.reshape(*sizes_keep, -1)
        threshold_flat = torch.quantile(
            flattened,
            self.quantile,
            dim=len(dims_to_keep),
            keepdim=True,
        )
        threshold_permuted = threshold_flat.reshape(
            *sizes_keep,
            *([1] * len(dims)),
        )
        inverse_order = [0] * tensor_ndim
        for original_idx, permuted_idx in enumerate(new_order):
            inverse_order[permuted_idx] = original_idx
        threshold = threshold_permuted.permute(*inverse_order)

    decisions = (tensor >= threshold).to(torch.bool)
    return {"decisions": decisions}
```

`ROIZoomNode`cuvis_ai.node.compositingtransformimgnumpypreCrop a region defined by a bbox and resize it to a fixed output frame.

#### ROIZoomNode

```python
ROIZoomNode(
    zoom_height=320,
    zoom_width=320,
    bg_color=(0.0, 0.0, 0.0),
    **kwargs,
)
```

Bases: `Node`

Crop a region defined by a bbox and resize it to a fixed output frame.

Emits one RGB frame per input frame at `(zoom_height, zoom_width)`. When `valid` is provided and equals `0` for a frame, the output is a solid `bg_color` frame.

Parameters:

| Name          | Type                         | Description                                            | Default           |
| ------------- | ---------------------------- | ------------------------------------------------------ | ----------------- |
| `zoom_height` | `int`                        | Output frame dimensions in pixels. Defaults 320 x 320. | `320`             |
| `zoom_width`  | `int`                        | Output frame dimensions in pixels. Defaults 320 x 320. | `320`             |
| `bg_color`    | `tuple[float, float, float]` | Background RGB (in [0, 1]) used when valid == 0.       | `(0.0, 0.0, 0.0)` |

Source code in `cuvis_ai/node/compositing.py`

```python
def __init__(
    self,
    zoom_height: int = 320,
    zoom_width: int = 320,
    bg_color: tuple[float, float, float] = (0.0, 0.0, 0.0),
    **kwargs: Any,
) -> None:
    if zoom_height < 8 or zoom_width < 8:
        raise ValueError("zoom dimensions must be >= 8 px")
    if len(bg_color) != 3:
        raise ValueError("bg_color must have 3 channels")

    self.zoom_height = int(zoom_height)
    self.zoom_width = int(zoom_width)
    self.bg_color = tuple(float(c) for c in bg_color)

    super().__init__(
        zoom_height=self.zoom_height,
        zoom_width=self.zoom_width,
        bg_color=self.bg_color,
        **kwargs,
    )
```

`RangeAverageFalseRGBSelector`cuvis_ai.node.channel_selectortransformdim-redhsinumpypreRange-based false RGB selection by averaging bands per channel.

#### RangeAverageFalseRGBSelector

```python
RangeAverageFalseRGBSelector(
    red_range=(580.0, 650.0),
    green_range=(500.0, 580.0),
    blue_range=(420.0, 500.0),
    **kwargs,
)
```

Bases: `ChannelSelectorBase`

Range-based false RGB selection by averaging bands per channel.

For each output channel (R/G/B), all spectral bands within the configured wavelength range are averaged per pixel. Channels with no matching bands are filled with zeros.

Parameters:

| Name          | Type                  | Description                                                 | Default          |
| ------------- | --------------------- | ----------------------------------------------------------- | ---------------- |
| `red_range`   | `tuple[float, float]` | Inclusive wavelength range for red channel in nanometers.   | `(580.0, 650.0)` |
| `green_range` | `tuple[float, float]` | Inclusive wavelength range for green channel in nanometers. | `(500.0, 580.0)` |
| `blue_range`  | `tuple[float, float]` | Inclusive wavelength range for blue channel in nanometers.  | `(420.0, 500.0)` |

Source code in `cuvis_ai/node/channel_selector.py`

```python
def __init__(
    self,
    red_range: tuple[float, float] = (580.0, 650.0),
    green_range: tuple[float, float] = (500.0, 580.0),
    blue_range: tuple[float, float] = (420.0, 500.0),
    **kwargs: Any,
) -> None:
    for name, rng in {
        "red_range": red_range,
        "green_range": green_range,
        "blue_range": blue_range,
    }.items():
        if len(rng) != 2 or rng[0] > rng[1]:
            raise ValueError(f"{name} must be (min_nm, max_nm) with min_nm <= max_nm")

    super().__init__(
        red_range=red_range, green_range=green_range, blue_range=blue_range, **kwargs
    )
    self.red_range = red_range
    self.green_range = green_range
    self.blue_range = blue_range

    # Static channel range boundaries [3, 2]; buffer so .to(device) moves it.
    self.register_buffer(
        "_ranges",
        torch.tensor(
            [
                [red_range[0], red_range[1]],
                [green_range[0], green_range[1]],
                [blue_range[0], blue_range[1]],
            ],
            dtype=torch.float32,
        ),
    )
    # Wavelength-dependent channel weights; lazily computed on first forward.
    self.register_buffer("_avg_weights", None, persistent=False)
    self.register_buffer("_avg_mask", None, persistent=False)
    self._cached_wl_key: tuple[float, ...] | None = None
```

##### forward

```python
forward(cube, wavelengths, context=None, **_)
```

Average spectral bands inside RGB ranges and compose normalized RGB.

Source code in `cuvis_ai/node/channel_selector.py`

```python
def forward(
    self,
    cube: torch.Tensor,
    wavelengths: Any,
    context: Context | None = None,  # noqa: ARG002
    **_: Any,
) -> dict[str, Any]:
    """Average spectral bands inside RGB ranges and compose normalized RGB."""
    self._ensure_weights(wavelengths, cube.device)
    wavelengths_t = self._prepare_wavelengths_tensor(wavelengths, cube.device)

    # Vectorized channel averaging:
    # cube [B,H,W,C] and weights [3,C] -> rgb [B,H,W,3]
    rgb = self._compute_raw_rgb(cube, wavelengths)
    rgb = self._normalize_rgb(rgb)

    channel_indices = [
        torch.where(self._avg_mask[i])[0].tolist() for i in range(self._avg_mask.shape[0])
    ]
    channel_names = ["red", "green", "blue"]
    missing_channels = [
        channel_names[i] for i, indices in enumerate(channel_indices) if len(indices) == 0
    ]

    band_info = {
        "strategy": "range_average_false_rgb",
        "band_indices": channel_indices,  # [R, G, B]
        "band_wavelengths_nm": [wavelengths_t[idxs].tolist() for idxs in channel_indices],
        "ranges_nm": {
            "red": [float(self.red_range[0]), float(self.red_range[1])],
            "green": [float(self.green_range[0]), float(self.green_range[1])],
            "blue": [float(self.blue_range[0]), float(self.blue_range[1])],
        },
        "aggregation": "mean",
        "missing_channels": missing_channels,
    }
    return {"rgb_image": rgb, "band_info": band_info}
```

`RgbLabelToClassIndex`cuvis_ai_inspecscrap.node.labelstransformclassmaskprecuvis_ai_inspecscrap[↗](https://github.com/cubert-hyperspectral/cuvis-ai-inspecscrap "Open plugin repo")

[View plugin repo (v0.2.4)](https://github.com/cubert-hyperspectral/cuvis-ai-inspecscrap/tree/v0.2.4)

`SAVISelector`cuvis_ai.node.channel_selectortransformdim-redhsinumpypreSoil Adjusted Vegetation Index renderer.

#### SAVISelector

```python
SAVISelector(
    red_nm=660.0,
    nir_nm=800.0,
    soil_factor=0.5,
    colormap_min=-1.0,
    colormap_max=1.0,
    eps=1e-06,
    **kwargs,
)
```

Bases: `_VegetationIndexBase`

Soil Adjusted Vegetation Index renderer.

Computes `(1 + L) * (NIR - Red) / (NIR + Red + L)` over bands resolved by nearest sensor wavelength, where `L` is the soil-brightness correction. The raw index map is returned via `index_image` and `rgb_image` carries an HSV colour-mapped render.

The additive `L` constant is only meaningful for reflectance in [0, 1]; feed reflectance-calibrated cubes, not raw radiance/DN.

Source code in `cuvis_ai/node/channel_selector.py`

```python
def __init__(
    self,
    red_nm: float = 660.0,
    nir_nm: float = 800.0,
    soil_factor: float = 0.5,
    colormap_min: float = -1.0,
    colormap_max: float = 1.0,
    eps: float = 1.0e-6,
    **kwargs: Any,
) -> None:
    super().__init__(
        band_nm={"red": red_nm, "nir": nir_nm},
        colormap_min=colormap_min,
        colormap_max=colormap_max,
        eps=eps,
        red_nm=float(red_nm),
        nir_nm=float(nir_nm),
        soil_factor=float(soil_factor),
        **kwargs,
    )
    self.red_nm = float(red_nm)
    self.nir_nm = float(nir_nm)
    self.soil_factor = float(soil_factor)
```

##### index_name

```python
index_name
```

Canonical SAVI strategy name.

`SNVCorrection`cuvis_ai.node.pretreatments.snvtransformhsinormpretorchStandard Normal Variate correction along the spectral axis.

#### SNVCorrection

```python
SNVCorrection(eps=1e-08, **kwargs)
```

Bases: `Node`

Standard Normal Variate correction along the spectral axis.

For every spectrum the per-band mean is subtracted and the result divided by the per-band standard deviation, removing multiplicative scatter and additive baseline effects on a pixel-by-pixel basis. Stateless: no fitting required.

Parameters:

| Name  | Type    | Description                                                                                                            | Default |
| ----- | ------- | ---------------------------------------------------------------------------------------------------------------------- | ------- |
| `eps` | `float` | Lower clamp on the per-spectrum standard deviation, guarding against division by zero on flat spectra (default: 1e-8). | `1e-08` |

Source code in `cuvis_ai/node/pretreatments/snv.py`

```python
def __init__(self, eps: float = 1e-8, **kwargs) -> None:
    self.eps = float(eps)
    super().__init__(eps=self.eps, **kwargs)
```

##### forward

```python
forward(cube, **_)
```

Apply per-spectrum mean centring and unit-variance scaling.

Parameters:

| Name   | Type     | Description                | Default    |
| ------ | -------- | -------------------------- | ---------- |
| `cube` | `Tensor` | Input cube in BHWC format. | *required* |

Returns:

| Type                | Description                                           |
| ------------------- | ----------------------------------------------------- |
| `dict[str, Tensor]` | {"cube": corrected} with the same shape as the input. |

Source code in `cuvis_ai/node/pretreatments/snv.py`

```python
def forward(self, cube: torch.Tensor, **_) -> dict[str, torch.Tensor]:
    """Apply per-spectrum mean centring and unit-variance scaling.

    Parameters
    ----------
    cube : torch.Tensor
        Input cube in BHWC format.

    Returns
    -------
    dict[str, torch.Tensor]
        ``{"cube": corrected}`` with the same shape as the input.
    """
    mean = cube.mean(dim=-1, keepdim=True)
    std = cube.std(dim=-1, keepdim=True).clamp_min(self.eps)
    return {"cube": (cube - mean) / std}
```

`SaturatedPixelDetector`cuvis_ai.node.preprocessorstransformhsipretorchFlag pixels whose channels reach the sensor saturation value.

#### SaturatedPixelDetector

```python
SaturatedPixelDetector(
    saturation_value=1.0, mask_threshold=0.0, **kwargs
)
```

Bases: `Node`

Flag pixels whose channels reach the sensor saturation value.

For every pixel the node computes the fraction of bands that sit at or above `saturation_value` and exposes it as a per-pixel `scores` map. A boolean `decisions` mask marks pixels whose saturated-band fraction exceeds `mask_threshold`.

Parameters:

| Name               | Type    | Description                                                                                                                                                | Default |
| ------------------ | ------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------- | ------- |
| `saturation_value` | `float` | Reflectance/intensity level at which a band counts as saturated. Bands with cube >= saturation_value are saturated. Default: 1.0                           | `1.0`   |
| `mask_threshold`   | `float` | Threshold on the saturated-band fraction. Pixels with scores > mask_threshold are flagged in decisions. Default: 0.0 (any saturated band flags the pixel). | `0.0`   |

Examples:

```pycon
>>> detector = SaturatedPixelDetector(saturation_value=1.0, mask_threshold=0.0)
>>> cube = torch.rand(2, 8, 8, 16)
>>> out = detector.forward(cube=cube)
>>> out["scores"].shape, out["decisions"].shape
(torch.Size([2, 8, 8, 1]), torch.Size([2, 8, 8, 1]))
```

Source code in `cuvis_ai/node/preprocessors.py`

```python
def __init__(
    self,
    saturation_value: float = 1.0,
    mask_threshold: float = 0.0,
    **kwargs: Any,
) -> None:
    self.saturation_value = float(saturation_value)
    self.mask_threshold = float(mask_threshold)
    super().__init__(
        saturation_value=self.saturation_value,
        mask_threshold=self.mask_threshold,
        **kwargs,
    )
```

##### forward

```python
forward(cube, **_)
```

Compute the per-pixel saturated-band fraction and saturation mask.

Parameters:

| Name   | Type     | Description                            | Default    |
| ------ | -------- | -------------------------------------- | ---------- |
| `cube` | `Tensor` | Input hyperspectral cube [B, H, W, C]. | *required* |

Returns:

| Type                | Description                                                        |
| ------------------- | ------------------------------------------------------------------ |
| `dict[str, Tensor]` | {"scores": Tensor [B, H, W, 1], "decisions": Tensor [B, H, W, 1]}. |

Source code in `cuvis_ai/node/preprocessors.py`

```python
@torch.no_grad()
def forward(self, cube: Tensor, **_: Any) -> dict[str, Tensor]:
    """Compute the per-pixel saturated-band fraction and saturation mask.

    Parameters
    ----------
    cube : Tensor
        Input hyperspectral cube ``[B, H, W, C]``.

    Returns
    -------
    dict[str, Tensor]
        ``{"scores": Tensor [B, H, W, 1], "decisions": Tensor [B, H, W, 1]}``.
    """
    scores = (cube >= self.saturation_value).float().mean(dim=-1, keepdim=True)
    decisions = scores > self.mask_threshold
    return {"scores": scores, "decisions": decisions}
```

`SavitzkyGolay`cuvis_ai.node.pretreatments.savitzky_golaytransformhsipretorchSavitzky-Golay smoothing / derivative filter over the spectral axis.

#### SavitzkyGolay

```python
SavitzkyGolay(
    window_length=11,
    polyorder=2,
    deriv=0,
    delta=1.0,
    mode="nearest",
    **kwargs,
)
```

Bases: `Node`

Savitzky-Golay smoothing / derivative filter over the spectral axis.

A polynomial of degree `polyorder` is least-squares fitted within a sliding window of `window_length` bands; the fitted value (or its `deriv`-th derivative) replaces the centre band. Coefficients are built once via :func:`scipy.signal.savgol_coeffs` and applied with a single `conv1d` along the channel axis.

Notes

The SciPy coefficients are convolution-oriented, while `torch` `conv1d` is a cross-correlation; the kernel is therefore stored already flipped so the result matches `scipy.signal.savgol_filter(..., mode="nearest")` on the interior. This filter does not reproduce SciPy's `mode="interp"` boundary handling.

Sample spacing (`deriv > 0` only)

A Savitzky-Golay kernel is a fixed convolution, so it assumes uniform band spacing. When the optional `wavelengths` port is connected, the effective spacing is taken from it (the median of the band steps) and the derivative is rescaled accordingly, so the `delta` parameter is only used as a fallback when `wavelengths` is absent. If the bands are not uniformly spaced, a single kernel cannot be exact and a warning is emitted; use :class:`~cuvis_ai.node.pretreatments.spectral_derivative.SpectralDerivative` for a coordinate-aware derivative.

Parameters:

| Name            | Type    | Description                                                                                                  | Default     |
| --------------- | ------- | ------------------------------------------------------------------------------------------------------------ | ----------- |
| `window_length` | `int`   | Length of the filter window in bands; must be odd (default: 11).                                             | `11`        |
| `polyorder`     | `int`   | Order of the fitted polynomial; must be less than window_length (default: 2).                                | `2`         |
| `deriv`         | `int`   | Order of the derivative to compute; 0 smooths (default: 0).                                                  | `0`         |
| `delta`         | `float` | Fallback sample spacing in nm, used for deriv > 0 when the wavelengths port is not connected (default: 1.0). | `1.0`       |
| `mode`          | `str`   | Boundary handling: "nearest" (replicate), "mirror" (reflect), or "constant" (zero pad) (default: "nearest"). | `'nearest'` |

Source code in `cuvis_ai/node/pretreatments/savitzky_golay.py`

```python
def __init__(
    self,
    window_length: int = 11,
    polyorder: int = 2,
    deriv: int = 0,
    delta: float = 1.0,
    mode: str = "nearest",
    **kwargs,
) -> None:
    self.window_length = int(window_length)
    if self.window_length % 2 == 0:
        raise ValueError(f"window_length must be odd, got {self.window_length}")
    self.polyorder = int(polyorder)
    self.deriv = int(deriv)
    self.delta = float(delta)
    self.mode = str(mode)
    if self.mode not in _PAD_MODE:
        raise ValueError(f"mode must be one of {sorted(_PAD_MODE)}, got {self.mode!r}")
    super().__init__(
        window_length=self.window_length,
        polyorder=self.polyorder,
        deriv=self.deriv,
        delta=self.delta,
        mode=self.mode,
        **kwargs,
    )
    self._pad_mode = _PAD_MODE[self.mode]
    self._pad = self.window_length // 2
    # savgol_coeffs are convolution-oriented; flip for conv1d cross-correlation.
    coefs = savgol_coeffs(
        self.window_length, self.polyorder, deriv=self.deriv, delta=self.delta
    )
    kernel = torch.tensor(coefs, dtype=torch.float32).flip(0).view(1, 1, -1)
    self.register_buffer("coefs", kernel)
```

##### forward

```python
forward(cube, wavelengths=None, **_)
```

Apply the Savitzky-Golay filter along the spectral axis.

Parameters:

| Name          | Type           | Description                                                                                                                                                                             | Default    |
| ------------- | -------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------- |
| `cube`        | `Tensor`       | Input cube in BHWC format.                                                                                                                                                              | *required* |
| `wavelengths` | `array - like` | Band wavelengths in nm. For deriv > 0 the derivative is rescaled to the median band spacing taken from this array (overriding the delta parameter); ignored for smoothing (deriv == 0). | `None`     |

Returns:

| Type                | Description                                          |
| ------------------- | ---------------------------------------------------- |
| `dict[str, Tensor]` | {"cube": filtered} with the same shape as the input. |

Source code in `cuvis_ai/node/pretreatments/savitzky_golay.py`

```python
def forward(self, cube: torch.Tensor, wavelengths=None, **_) -> dict[str, torch.Tensor]:
    """Apply the Savitzky-Golay filter along the spectral axis.

    Parameters
    ----------
    cube : torch.Tensor
        Input cube in BHWC format.
    wavelengths : array-like, optional
        Band wavelengths in nm. For ``deriv > 0`` the derivative is rescaled
        to the median band spacing taken from this array (overriding the
        ``delta`` parameter); ignored for smoothing (``deriv == 0``).

    Returns
    -------
    dict[str, torch.Tensor]
        ``{"cube": filtered}`` with the same shape as the input.
    """
    B, H, W, C = cube.shape
    signal = cube.reshape(B * H * W, 1, C)
    if self._pad_mode == "constant":
        padded = F.pad(signal, (self._pad, self._pad), mode="constant", value=0.0)
    else:
        padded = F.pad(signal, (self._pad, self._pad), mode=self._pad_mode)
    kernel = self.coefs.to(dtype=signal.dtype)
    filtered = F.conv1d(padded, kernel).reshape(B, H, W, C)
    if self.deriv > 0 and wavelengths is not None:
        filtered = filtered * self._spacing_rescale(wavelengths, cube)
    return {"cube": filtered}
```

`ScoreToLogit`cuvis_ai.node.conversiontransformmaskposttorchTrainable head that converts RX scores to anomaly logits.

#### ScoreToLogit

```python
ScoreToLogit(init_scale=1.0, init_bias=0.0, **kwargs)
```

Bases: `Node`

Trainable head that converts RX scores to anomaly logits.

This node takes RX anomaly scores (typically Mahalanobis distances) and applies a learned affine transformation to produce logits suitable for binary classification with BCEWithLogitsLoss.

The transformation is: logit = scale * (score - bias)

Parameters:

| Name         | Type    | Description                                      | Default |
| ------------ | ------- | ------------------------------------------------ | ------- |
| `init_scale` | `float` | Initial value for the scale parameter            | `1.0`   |
| `init_bias`  | `float` | Initial value for the bias parameter (threshold) | `0.0`   |

Attributes:

| Name    | Type                  | Description                                            |
| ------- | --------------------- | ------------------------------------------------------ |
| `scale` | `Parameter or Tensor` | Scale factor applied to scores                         |
| `bias`  | `Parameter or Tensor` | Bias (threshold) subtracted from scores before scaling |

Examples:

```pycon
>>> # After RX detector
>>> rx = RXGlobal(eps=1e-6)
>>> logit_head = ScoreToLogit(init_scale=1.0, init_bias=5.0)
>>> logit_head.unfreeze()  # Enable gradient training
>>> graph.connect(rx.scores, logit_head.scores)
```

Source code in `cuvis_ai/node/conversion.py`

```python
def __init__(
    self,
    init_scale: float = 1.0,
    init_bias: float = 0.0,
    **kwargs,
) -> None:
    self.init_scale = init_scale
    self.init_bias = init_bias

    super().__init__(
        init_scale=init_scale,
        init_bias=init_bias,
        **kwargs,
    )

    # Initialize as buffers (frozen by default)
    self.register_buffer("scale", torch.tensor(init_scale, dtype=torch.float32))
    self.register_buffer("bias", torch.tensor(init_bias, dtype=torch.float32))

    self._welford = WelfordAccumulator(1)
    # Allow using the head with the provided init_scale/init_bias without forcing a fit()
    self._statistically_initialized = True
```

##### statistical_initialization

```python
statistical_initialization(input_stream)
```

Initialize bias from statistics of RX scores using streaming approach.

Uses Welford's algorithm for numerically stable online computation of mean and standard deviation, similar to RXGlobal.

Parameters:

| Name           | Type          | Description                                                                                                                        | Default    |
| -------------- | ------------- | ---------------------------------------------------------------------------------------------------------------------------------- | ---------- |
| `input_stream` | `InputStream` | Iterator yielding dicts matching INPUT_SPECS (port-based format) Expected format: {"scores": tensor} where tensor is the RX scores | *required* |

Source code in `cuvis_ai/node/conversion.py`

```python
def statistical_initialization(self, input_stream) -> None:
    """Initialize bias from statistics of RX scores using streaming approach.

    Uses Welford's algorithm for numerically stable online computation of
    mean and standard deviation, similar to RXGlobal.

    Parameters
    ----------
    input_stream : InputStream
        Iterator yielding dicts matching INPUT_SPECS (port-based format)
        Expected format: {"scores": tensor} where tensor is the RX scores
    """
    self.reset()
    for batch_data in input_stream:
        # Extract scores from port-based dict
        scores = batch_data.get("scores")
        if scores is not None:
            self.update(scores)

    if self._welford.count <= 1:
        self._statistically_initialized = False
        raise RuntimeError(
            "ScoreToLogit.statistical_initialization() received insufficient samples. "
            "Expected at least 2 score values."
        )
    self.finalize()
```

##### update

```python
update(scores)
```

Update running statistics with a batch of scores.

Parameters:

| Name     | Type     | Description                       | Default    |
| -------- | -------- | --------------------------------- | ---------- |
| `scores` | `Tensor` | Batch of RX scores in BHWC format | *required* |

Source code in `cuvis_ai/node/conversion.py`

```python
@torch.no_grad()
def update(self, scores: torch.Tensor) -> None:
    """Update running statistics with a batch of scores.

    Parameters
    ----------
    scores : torch.Tensor
        Batch of RX scores in BHWC format
    """
    X = scores.flatten()
    if X.shape[0] <= 1:
        return
    self._welford.update(X)
    self._statistically_initialized = False
```

##### finalize

```python
finalize()
```

Finalize statistics and set bias to mean + 2\*std.

This threshold (mean + 2\*std) is a common heuristic for anomaly detection, capturing ~95% of normal data under Gaussian assumption.

Source code in `cuvis_ai/node/conversion.py`

```python
@torch.no_grad()
def finalize(self) -> None:
    """Finalize statistics and set bias to mean + 2*std.

    This threshold (mean + 2*std) is a common heuristic for anomaly detection,
    capturing ~95% of normal data under Gaussian assumption.
    """
    if self._welford.count <= 1:
        raise ValueError("Not enough samples to finalize ScoreToLogit statistics.")

    mean = self._welford.mean.squeeze()
    std = self._welford.std.squeeze()

    # Set bias to mean + 2*std (threshold for anomalies)
    self.bias = mean + 2.0 * std
    self._statistically_initialized = True
```

##### reset

```python
reset()
```

Reset all statistics and accumulators.

Source code in `cuvis_ai/node/conversion.py`

```python
def reset(self) -> None:
    """Reset all statistics and accumulators."""
    self._welford.reset()
    # Keep explicit init_scale/init_bias usable in inference-only runs.
    # Statistical flows still transition through update()/finalize().
    self._statistically_initialized = True
```

##### forward

```python
forward(scores, **_)
```

Transform RX scores to logits.

Parameters:

| Name     | Type     | Description                             | Default    |
| -------- | -------- | --------------------------------------- | ---------- |
| `scores` | `Tensor` | Input RX scores with shape (B, H, W, 1) | *required* |

Returns:

| Type                | Description                                                |
| ------------------- | ---------------------------------------------------------- |
| `dict[str, Tensor]` | Dictionary with "logits" key containing transformed scores |

Source code in `cuvis_ai/node/conversion.py`

```python
def forward(self, scores: torch.Tensor, **_) -> dict[str, torch.Tensor]:
    """Transform RX scores to logits.

    Parameters
    ----------
    scores : torch.Tensor
        Input RX scores with shape (B, H, W, 1)

    Returns
    -------
    dict[str, torch.Tensor]
        Dictionary with "logits" key containing transformed scores
    """

    if not self._statistically_initialized:
        raise RuntimeError(
            "ScoreToLogit not initialized. Call statistical_initialization() before forward()."
        )
    # Apply affine transformation: logit = scale * (score - bias)
    logits = self.scale * (scores - self.bias)

    return {"logits": logits}
```

##### get_threshold

```python
get_threshold()
```

Get the current anomaly threshold (bias value).

Returns:

| Type    | Description             |
| ------- | ----------------------- |
| `float` | Current threshold value |

Source code in `cuvis_ai/node/conversion.py`

```python
def get_threshold(self) -> float:
    """Get the current anomaly threshold (bias value).

    Returns
    -------
    float
        Current threshold value
    """
    return self.bias.item()
```

##### set_threshold

```python
set_threshold(threshold)
```

Set the anomaly threshold (bias value).

Parameters:

| Name        | Type    | Description         | Default    |
| ----------- | ------- | ------------------- | ---------- |
| `threshold` | `float` | New threshold value | *required* |

Source code in `cuvis_ai/node/conversion.py`

```python
def set_threshold(self, threshold: float) -> None:
    """Set the anomaly threshold (bias value).

    Parameters
    ----------
    threshold : float
        New threshold value
    """
    with torch.no_grad():
        self.bias.fill_(threshold)
```

##### predict_anomalies

```python
predict_anomalies(logits)
```

Convert logits to binary anomaly predictions.

Parameters:

| Name     | Type     | Description                                  | Default    |
| -------- | -------- | -------------------------------------------- | ---------- |
| `logits` | `Tensor` | Logits from forward pass, shape (B, H, W, 1) | *required* |

Returns:

| Type     | Description                                                  |
| -------- | ------------------------------------------------------------ |
| `Tensor` | Binary predictions (0=normal, 1=anomaly), shape (B, H, W, 1) |

Source code in `cuvis_ai/node/conversion.py`

```python
def predict_anomalies(self, logits: torch.Tensor) -> torch.Tensor:
    """Convert logits to binary anomaly predictions.

    Parameters
    ----------
    logits : torch.Tensor
        Logits from forward pass, shape (B, H, W, 1)

    Returns
    -------
    torch.Tensor
        Binary predictions (0=normal, 1=anomaly), shape (B, H, W, 1)
    """
    return (logits > 0).float()
```

`SegmentationAnomalyScore`cuvis_ai_unet.node.scoringtransformevalsegtorchunet[↗](https://github.com/cubert-hyperspectral/cuvis-ai-unet "Open plugin repo")Adapt segmentation logits to anomaly-detection score maps + a per-image score.

Adapt segmentation logits to anomaly-detection score maps + a per-image score.

Inputs

| Port               | Dtype     | Shape              | Description                                                                                       |
| ------------------ | --------- | ------------------ | ------------------------------------------------------------------------------------------------- |
| `logits`           | `float32` | `[-1, -1, -1, -1]` | Per-pixel class logits [B, H, W, num_classes]                                                     |
| `targets` optional | `int32`   | `[-1, -1, -1]`     | Optional integer class mask [B, H, W]; when connected, emits targets_bool for a paired AUROC node |

Outputs

| Port                    | Dtype     | Shape             | Description                                                        |
| ----------------------- | --------- | ----------------- | ------------------------------------------------------------------ |
| `scores`                | `float32` | `[-1, -1, -1, 1]` | Foreground-probability map [B, H, W, 1]                            |
| `anomaly_score`         | `float32` | `[-1]`            | Per-image score: mean of the top-`top_frac` score pixels [B]       |
| `targets_bool` optional | `bool`    | `[-1, -1, -1, 1]` | Boolean foreground mask [B, H, W, 1] (only when targets connected) |

[View plugin repo (v0.2.2)](https://github.com/cubert-hyperspectral/cuvis-ai-unet/tree/v0.2.2)

`ShapeMorphology`cuvis_ai.node.morphologytransformmasktorchPer-object shape descriptors from a binary or labeled mask.

#### ShapeMorphology

```python
ShapeMorphology(
    properties=None,
    max_objects=256,
    connectivity=8,
    binarize=True,
    **kwargs,
)
```

Bases: `Node`

Per-object shape descriptors from a binary or labeled mask.

For each batch element the mask is reduced to integer instance labels (connected components when `binarize` is True, otherwise the input is taken to already carry labels) and a fixed set of geometric descriptors is computed per object. Descriptors follow the `skimage.measure.regionprops` conventions:

- `area` - pixel count of the region.
- `centroid_y` / `centroid_x` - mean pixel coordinates.
- `major_axis` / `minor_axis` - `4 * sqrt(lambda)` of the two covariance eigenvalues (descending).
- `eccentricity` - `sqrt(1 - lambda_2 / lambda_1)`.
- `orientation` - `0.5 * atan2(2 * cov_xy, cov_yy - cov_xx)`.
- `bbox_area` - area of the axis-aligned bounding box.

Rows are padded to `max_objects`; `valid` marks the first `N` rows True (real objects) and the remainder False (padding).

Parameters:

| Name           | Type          | Description                                                                                                                                                                                          | Default |
| -------------- | ------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------- |
| `properties`   | `list of str` | Descriptor names to emit, in order along the P axis. Defaults to all supported descriptors.                                                                                                          | `None`  |
| `max_objects`  | `int`         | Number of object rows in the output; objects beyond this are dropped and fewer objects are zero-padded. Default 256.                                                                                 | `256`   |
| `connectivity` | `int`         | Pixel connectivity (4 or 8) for connected-component labeling when binarize is True. Default 8.                                                                                                       | `8`     |
| `binarize`     | `bool`        | If True, treat any nonzero pixel as foreground and run connected- component labeling. If False, treat the input as an already-labeled mask and use its nonzero values as instance ids. Default True. | `True`  |

Source code in `cuvis_ai/node/morphology.py`

```python
def __init__(
    self,
    properties: list[str] | None = None,
    max_objects: int = 256,
    connectivity: int = 8,
    binarize: bool = True,
    **kwargs: Any,
) -> None:
    if properties is None:
        properties = list(_SUPPORTED_PROPERTIES)
    unknown = [p for p in properties if p not in _SUPPORTED_PROPERTIES]
    if unknown:
        raise ValueError(f"unknown properties: {unknown}; supported: {_SUPPORTED_PROPERTIES}")
    if len(properties) == 0:
        raise ValueError("properties must list at least one descriptor")
    if max_objects < 1:
        raise ValueError("max_objects must be >= 1")
    if connectivity not in (4, 8):
        raise ValueError("connectivity must be 4 or 8")

    self.properties = list(properties)
    self.max_objects = int(max_objects)
    self.connectivity = int(connectivity)
    self.binarize = bool(binarize)

    super().__init__(
        properties=self.properties,
        max_objects=self.max_objects,
        connectivity=self.connectivity,
        binarize=self.binarize,
        **kwargs,
    )
```

##### forward

```python
forward(mask, **_)
```

Label objects and emit per-object descriptors, padded to `max_objects`.

Source code in `cuvis_ai/node/morphology.py`

```python
@torch.no_grad()
def forward(self, mask: torch.Tensor, **_: Any) -> dict[str, torch.Tensor]:
    """Label objects and emit per-object descriptors, padded to ``max_objects``."""
    device = mask.device
    b, h, w = mask.shape
    p = len(self.properties)

    identity = torch.zeros((b, h, w), dtype=torch.int32, device=device)
    properties = torch.zeros((b, self.max_objects, p), dtype=torch.float32, device=device)
    valid = torch.zeros((b, self.max_objects), dtype=torch.bool, device=device)

    for i in range(b):
        labels = self._labels_for_frame(mask[i]).to(device)
        num_objects = int(labels.max().item())
        n_keep = min(num_objects, self.max_objects)

        # Drop labels beyond max_objects from the emitted identity map.
        if num_objects > self.max_objects:
            labels = torch.where(labels > self.max_objects, torch.zeros_like(labels), labels)
        identity[i] = labels.to(torch.int32)

        if num_objects == 0:
            continue

        desc = self._descriptors_for_frame(labels, num_objects, device)
        properties[i, :n_keep] = desc[:n_keep]
        valid[i, :n_keep] = True

    return {
        "identity_mask": identity,
        "properties": properties,
        "valid": valid,
    }
```

`SigmoidNormalizer`cuvis_ai.node.normalizationtransformnormnumpypreMedian-centered sigmoid squashing per sample and channel.

#### SigmoidNormalizer

```python
SigmoidNormalizer(std_floor=1e-06, **kwargs)
```

Bases: `_ScoreNormalizerBase`

Median-centered sigmoid squashing per sample and channel.

Applies sigmoid transformation centered at the median with standard deviation scaling:

```text
sigmoid((x - median) / std)
```

Produces values in [0, 1] range with median mapped to 0.5.

Parameters:

| Name        | Type    | Description                                                                      | Default |
| ----------- | ------- | -------------------------------------------------------------------------------- | ------- |
| `std_floor` | `float` | Minimum standard deviation threshold to prevent division by zero (default: 1e-6) | `1e-06` |
| `**kwargs`  | `dict`  | Additional arguments passed to Node base class                                   | `{}`    |

Examples:

```pycon
>>> from cuvis_ai.node.normalization import SigmoidNormalizer
>>> import torch
>>>
>>> # Create sigmoid normalizer
>>> normalizer = SigmoidNormalizer(std_floor=1.0e-6)
>>>
>>> # Apply to hyperspectral data
>>> data = torch.randn(4, 256, 256, 61)  # [B, H, W, C]
>>> output = normalizer.forward(data=data)
>>> normalized = output["normalized"]  # [4, 256, 256, 61], values in [0, 1]
```

See Also

MinMaxNormalizer : Min-max scaling to [0, 1] ZScoreNormalizer : Z-score standardization

Notes

Sigmoid normalization is robust to outliers because extreme values are squashed asymptotically to 0 or 1. This makes it suitable for data with heavy-tailed distributions or sporadic anomalies.

Source code in `cuvis_ai/node/normalization.py`

```python
def __init__(self, std_floor: float = 1e-6, **kwargs) -> None:
    self.std_floor = float(std_floor)
    super().__init__(std_floor=std_floor, **kwargs)
```

`SigmoidTransform`cuvis_ai.node.normalizationtransformposttorchApplies sigmoid transformation to convert logits to probabilities [0,1].

#### SigmoidTransform

```python
SigmoidTransform(**kwargs)
```

Bases: `Node`

Applies sigmoid transformation to convert logits to probabilities [0,1].

General-purpose sigmoid node for converting raw scores/logits to probability space. Useful for visualization or downstream nodes that expect bounded [0,1] values.

Examples:

```pycon
>>> sigmoid = SigmoidTransform()
>>> # Route logits to both loss (raw) and visualization (sigmoid)
>>> graph.connect(
...     (rx.scores, loss_node.predictions),  # Raw logits to loss
...     (rx.scores, sigmoid.data),           # Logits to sigmoid
...     (sigmoid.transformed, viz.scores),   # Probabilities to viz
... )
```

Source code in `cuvis_ai/node/normalization.py`

```python
def __init__(self, **kwargs) -> None:
    super().__init__(**kwargs)
```

##### forward

```python
forward(data, **_)
```

Apply sigmoid transformation.

Parameters:

| Name   | Type     | Description  | Default    |
| ------ | -------- | ------------ | ---------- |
| `data` | `Tensor` | Input tensor | *required* |

Returns:

| Type                | Description                                                 |
| ------------------- | ----------------------------------------------------------- |
| `dict[str, Tensor]` | Dictionary with "transformed" key containing sigmoid output |

Source code in `cuvis_ai/node/normalization.py`

```python
def forward(self, data: Tensor, **_: Any) -> dict[str, Tensor]:
    """Apply sigmoid transformation.

    Parameters
    ----------
    data : Tensor
        Input tensor

    Returns
    -------
    dict[str, Tensor]
        Dictionary with "transformed" key containing sigmoid output
    """
    return {"transformed": torch.sigmoid(data)}
```

`SignaturesToReferences`cuvis_ai.node.spectral_extractortransformclasshsiTreat each object's signature as its own Spectral Angle Mapper reference.

#### SignaturesToReferences

```python
SignaturesToReferences(normalize=None, **kwargs)
```

Bases: `Node`

Treat each object's signature as its own Spectral Angle Mapper reference.

Reshapes per-object signatures `[1, N, C]` (as produced by :class:`SpectralSignatureExtractor`) into reference spectra `[N, 1, 1, C]` for :class:`~cuvis_ai.node.spectral_angle_mapper.SpectralAngleMapper` -- one reference per object, in object-id order, so reference `k` corresponds to object `k`. Use it when every detected object is known a priori to be a distinct material: each object's signature becomes a reference, the mapper scores every pixel against all references, and each object should recover its own pixels. The reference count follows the number of objects (no fixed reference count, no clustering).

Parameters:

| Name        | Type          | Description                                                                                                                          | Default |
| ----------- | ------------- | ------------------------------------------------------------------------------------------------------------------------------------ | ------- |
| `normalize` | `str or None` | Optional reference normalization: "unit_mean", "l2", or None (default; the Spectral Angle Mapper mean-normalizes internally anyway). | `None`  |

Validate the optional normalization mode and store it.

Source code in `cuvis_ai/node/spectral_extractor.py`

```python
def __init__(self, normalize: str | None = None, **kwargs: Any) -> None:
    """Validate the optional normalization mode and store it."""
    if normalize not in (None, "unit_mean", "l2"):
        raise ValueError("normalize must be None, 'unit_mean', or 'l2'.")
    self.normalize = normalize
    super().__init__(normalize=self.normalize, **kwargs)
```

##### forward

```python
forward(signatures, **_)
```

Reshape per-object signatures into one reference spectrum per object.

Parameters:

| Name         | Type     | Description                                                      | Default    |
| ------------ | -------- | ---------------------------------------------------------------- | ---------- |
| `signatures` | `Tensor` | Per-object signatures [1, N, C] (float32).                       | *required* |
| `**_`        | `Any`    | Additional unused keyword arguments (e.g. the pipeline context). | `{}`       |

Returns:

| Type                | Description                                                |
| ------------------- | ---------------------------------------------------------- |
| `dict[str, Tensor]` | spectral_signature float32 [N, 1, 1, C] reference spectra. |

Source code in `cuvis_ai/node/spectral_extractor.py`

```python
@torch.no_grad()
def forward(self, signatures: torch.Tensor, **_: Any) -> dict[str, torch.Tensor]:
    """Reshape per-object signatures into one reference spectrum per object.

    Parameters
    ----------
    signatures : torch.Tensor
        Per-object signatures ``[1, N, C]`` (float32).
    **_ : Any
        Additional unused keyword arguments (e.g. the pipeline ``context``).

    Returns
    -------
    dict[str, torch.Tensor]
        ``spectral_signature`` float32 ``[N, 1, 1, C]`` reference spectra.
    """
    sig = signatures[0].to(torch.float32)  # [N, C]
    num_refs, channels = int(sig.shape[0]), int(sig.shape[1])
    if self.normalize == "unit_mean":
        sig = sig / sig.mean(dim=1, keepdim=True).clamp_min(1e-8)
    elif self.normalize == "l2":
        sig = sig / sig.norm(dim=1, keepdim=True).clamp_min(1e-8)
    return {"spectral_signature": sig.view(num_refs, 1, 1, channels)}
```

`SolidOcclusionNode`cuvis_ai.node.occlusiontransformaugnumpystochtrainDeprecated alias of PoissonOcclusionNode.

#### SolidOcclusionNode

```python
SolidOcclusionNode(
    tracking_json_path,
    track_ids,
    occlusion_start_frame,
    occlusion_end_frame,
    fill_color="poisson",
    *,
    input_key=None,
    max_iter=1000,
    tol=1e-06,
    occlusion_shape="bbox",
    bbox_mode="static",
    static_bbox_scale=1.2,
    static_bbox_padding_px=0,
    static_full_width_x=False,
    **kwargs,
)
```

Bases: `PoissonOcclusionNode`

Deprecated alias of PoissonOcclusionNode.

Source code in `cuvis_ai/node/occlusion.py`

```python
def __init__(
    self,
    tracking_json_path: str,
    track_ids: list[int],
    occlusion_start_frame: int,
    occlusion_end_frame: int,
    fill_color: tuple[float, float, float] | str = "poisson",
    *,
    input_key: str | None = None,
    max_iter: int = 1000,
    tol: float = 1e-6,
    occlusion_shape: str = "bbox",
    bbox_mode: str = "static",
    static_bbox_scale: float = 1.2,
    static_bbox_padding_px: int = 0,
    static_full_width_x: bool = False,
    **kwargs,
) -> None:
    if occlusion_shape not in self._VALID_SHAPES:
        raise ValueError(
            f"occlusion_shape must be one of {self._VALID_SHAPES}, got '{occlusion_shape}'"
        )
    if bbox_mode not in self._VALID_BBOX_MODES:
        raise ValueError(
            f"bbox_mode must be one of {self._VALID_BBOX_MODES}, got '{bbox_mode}'"
        )
    if static_bbox_scale <= 0:
        raise ValueError("static_bbox_scale must be > 0")
    if static_bbox_padding_px < 0:
        raise ValueError("static_bbox_padding_px must be >= 0")
    if int(max_iter) <= 0:
        raise ValueError("max_iter must be > 0")
    if float(tol) <= 0:
        raise ValueError("tol must be > 0")
    if input_key is not None and input_key not in {"rgb_image", "cube"}:
        raise ValueError("input_key must be 'rgb_image', 'cube', or None")

    self._use_poisson_fill = False
    if isinstance(fill_color, str):
        if fill_color != "poisson":
            raise ValueError("fill_color string must be exactly 'poisson'")
        self.fill_color: tuple[float, float, float] | str = fill_color
        self._use_poisson_fill = True
    else:
        parsed_fill = tuple(float(c) for c in fill_color)
        if len(parsed_fill) != 3:
            raise ValueError("fill_color tuple must have exactly 3 values")
        if any(c < 0.0 or c > 1.0 for c in parsed_fill):
            raise ValueError("fill_color tuple values must be in [0, 1]")
        self.fill_color = parsed_fill

    self.max_iter = int(max_iter)
    self.tol = float(tol)
    self.input_key = input_key
    self.occlusion_shape = occlusion_shape
    self.bbox_mode = bbox_mode
    self.static_bbox_scale = float(static_bbox_scale)
    self.static_bbox_padding_px = int(static_bbox_padding_px)
    self.static_full_width_x = bool(static_full_width_x)

    super().__init__(
        tracking_json_path=tracking_json_path,
        track_ids=track_ids,
        occlusion_start_frame=occlusion_start_frame,
        occlusion_end_frame=occlusion_end_frame,
        fill_color=self.fill_color,
        input_key=self.input_key,
        max_iter=self.max_iter,
        tol=self.tol,
        occlusion_shape=occlusion_shape,
        bbox_mode=bbox_mode,
        static_bbox_scale=static_bbox_scale,
        static_bbox_padding_px=static_bbox_padding_px,
        static_full_width_x=static_full_width_x,
        **kwargs,
    )

    self._static_bboxes_by_track: dict[int, list[float]] = {}
    if self.occlusion_shape == "bbox" and self.bbox_mode == "static":
        self._static_bboxes_by_track = self._build_static_bboxes_by_track()
        logger.info(
            "PoissonOcclusionNode static bboxes: {} tracks (scale={}, padding_px={})",
            len(self._static_bboxes_by_track),
            self.static_bbox_scale,
            self.static_bbox_padding_px,
        )
```

`SpatialRotateNode`cuvis_ai.node.preprocessorstransformimgnumpypreRotate spatial dimensions of cubes, masks, and RGB images.

#### SpatialRotateNode

```python
SpatialRotateNode(rotation=None, **kwargs)
```

Bases: `Node`

Rotate spatial dimensions of cubes, masks, and RGB images.

Applies a fixed rotation (90, -90, or 180 degrees) to the H and W dimensions of all provided inputs. Wavelengths pass through unchanged.

Place immediately after a data node so all downstream consumers see correctly oriented data.

Parameters:

| Name       | Type  | Description | Default                                                                                                  |
| ---------- | ----- | ----------- | -------------------------------------------------------------------------------------------------------- |
| `rotation` | \`int | None\`      | Rotation in degrees. Supported: 90, -90, 180 (and aliases 270, -270, -180). None or 0 means passthrough. |

Source code in `cuvis_ai/node/preprocessors.py`

```python
def __init__(self, rotation: int | None = None, **kwargs: Any) -> None:
    if rotation not in self._VALID_ROTATIONS:
        raise ValueError(
            f"rotation must be one of {sorted(r for r in self._VALID_ROTATIONS if r is not None)}"
            f" or None, got {rotation}"
        )
    self.rotation = self._normalize(rotation)
    super().__init__(rotation=rotation, **kwargs)
```

##### forward

```python
forward(cube, mask=None, rgb_image=None, **_)
```

Apply the configured rotation to the cube, mask, and rgb_image tensors.

Source code in `cuvis_ai/node/preprocessors.py`

```python
@torch.no_grad()
def forward(
    self,
    cube: Tensor,
    mask: Tensor | None = None,
    rgb_image: Tensor | None = None,
    **_: Any,
) -> dict[str, Tensor]:
    """Apply the configured rotation to the cube, mask, and rgb_image tensors."""
    k = {None: 0, 90: 1, -90: -1, 180: 2}[self.rotation]

    result: dict[str, Tensor] = {}
    result["cube"] = torch.rot90(cube, k=k, dims=(1, 2)).contiguous() if k else cube
    if mask is not None:
        result["mask"] = torch.rot90(mask, k=k, dims=(1, 2)).contiguous() if k else mask
    if rgb_image is not None:
        result["rgb_image"] = (
            torch.rot90(rgb_image, k=k, dims=(1, 2)).contiguous() if k else rgb_image
        )
    return result
```

`SpectralDerivative`cuvis_ai.node.pretreatments.spectral_derivativetransformhsipretorchFirst- or second-order spectral derivative along the band axis.

#### SpectralDerivative

```python
SpectralDerivative(order=1, **kwargs)
```

Bases: `Node`

First- or second-order spectral derivative along the band axis.

Derivatives suppress additive/multiplicative baseline effects and sharpen absorption features. The derivative is taken with respect to wavelength (nanometers), honouring non-uniform band spacing via the `wavelengths` port.

Parameters:

| Name    | Type  | Description                                                  | Default |
| ------- | ----- | ------------------------------------------------------------ | ------- |
| `order` | `int` | Derivative order; 1 (default) or 2 (gradient applied twice). | `1`     |

Source code in `cuvis_ai/node/pretreatments/spectral_derivative.py`

```python
def __init__(self, order: int = 1, **kwargs) -> None:
    self.order = int(order)
    super().__init__(order=self.order, **kwargs)
```

##### forward

```python
forward(cube, wavelengths, **_)
```

Differentiate each spectrum with respect to wavelength.

Parameters:

| Name          | Type           | Description                 | Default    |
| ------------- | -------------- | --------------------------- | ---------- |
| `cube`        | `Tensor`       | Input cube in BHWC format.  | *required* |
| `wavelengths` | `array - like` | Band wavelengths, length C. | *required* |

Returns:

| Type                | Description                                            |
| ------------------- | ------------------------------------------------------ |
| `dict[str, Tensor]` | {"cube": derivative} with the same shape as the input. |

Source code in `cuvis_ai/node/pretreatments/spectral_derivative.py`

```python
def forward(self, cube: torch.Tensor, wavelengths, **_) -> dict[str, torch.Tensor]:
    """Differentiate each spectrum with respect to wavelength.

    Parameters
    ----------
    cube : torch.Tensor
        Input cube in BHWC format.
    wavelengths : array-like
        Band wavelengths, length ``C``.

    Returns
    -------
    dict[str, torch.Tensor]
        ``{"cube": derivative}`` with the same shape as the input.
    """
    x = torch.as_tensor(np.asarray(wavelengths), device=cube.device).reshape(-1)
    x = x.to(cube.dtype)
    out = cube
    for _step in range(self.order):
        out = torch.gradient(out, spacing=(x,), dim=-1)[0]
    return {"cube": out}
```

`SpectralSignatureExtractor`cuvis_ai.node.spectral_extractortransformembhsinumpyExtract per-object spectral signatures from SAM-style label masks.

#### SpectralSignatureExtractor

```python
SpectralSignatureExtractor(
    trim_fraction=0.1,
    min_mask_pixels=10,
    zero_norm_threshold=1e-08,
    **kwargs,
)
```

Bases: `Node`

Extract per-object spectral signatures from SAM-style label masks.

Notes

Only the first batch element of `cube` is processed; `mask` is required to be `[1, H, W]` by `INPUT_SPECS`. Outputs are always shaped `[1, N, C]`. Feed one frame at a time.

Source code in `cuvis_ai/node/spectral_extractor.py`

```python
def __init__(
    self,
    trim_fraction: float = 0.1,
    min_mask_pixels: int = 10,
    zero_norm_threshold: float = 1e-8,
    **kwargs: Any,
) -> None:
    if not (0.0 <= trim_fraction < 0.5):
        raise ValueError("trim_fraction must be in [0.0, 0.5).")
    if min_mask_pixels < 1:
        raise ValueError("min_mask_pixels must be >= 1.")
    if zero_norm_threshold < 0.0:
        raise ValueError("zero_norm_threshold must be non-negative.")

    self.trim_fraction = float(trim_fraction)
    self.min_mask_pixels = int(min_mask_pixels)
    self.zero_norm_threshold = float(zero_norm_threshold)

    super().__init__(
        trim_fraction=trim_fraction,
        min_mask_pixels=min_mask_pixels,
        zero_norm_threshold=zero_norm_threshold,
        **kwargs,
    )
```

##### forward

```python
forward(
    cube,
    mask,
    object_ids=None,
    wavelengths=None,
    context=None,
    **_,
)
```

Extract per-object signatures. See class docstring for batch semantics.

Source code in `cuvis_ai/node/spectral_extractor.py`

```python
def forward(
    self,
    cube: torch.Tensor,
    mask: torch.Tensor,
    object_ids: torch.Tensor | None = None,
    wavelengths: np.ndarray | torch.Tensor | None = None,  # noqa: ARG002
    context: Context | None = None,  # noqa: ARG002
    **_: Any,
) -> dict[str, torch.Tensor]:
    """Extract per-object signatures. See class docstring for batch semantics."""
    cube_0 = cube[0]
    height, width, num_channels = (
        int(cube_0.shape[0]),
        int(cube_0.shape[1]),
        int(cube_0.shape[2]),
    )

    mask_2d = self._parse_mask(mask).to(device=cube_0.device)
    mask_2d = self._resize_mask_if_needed(mask_2d, height=height, width=width)

    parsed_ids = self._parse_object_ids(object_ids)
    if parsed_ids is None:
        resolved_ids = torch.unique(mask_2d[mask_2d != 0], sorted=True).to(torch.int64)
    else:
        resolved_ids = parsed_ids.to(device=cube_0.device, dtype=torch.int64)

    if resolved_ids.numel() == 0:
        empty = torch.empty((1, 0, num_channels), dtype=cube_0.dtype, device=cube_0.device)
        return {"signatures": empty, "signatures_std": empty.clone()}

    signatures: list[torch.Tensor] = []
    signatures_std: list[torch.Tensor] = []
    for obj_id in resolved_ids.tolist():
        obj_mask = mask_2d == int(obj_id)
        if not bool(obj_mask.any()):
            zeros = torch.zeros(num_channels, dtype=cube_0.dtype, device=cube_0.device)
            signatures.append(zeros)
            signatures_std.append(zeros.clone())
            continue

        pixels = cube_0[obj_mask]
        mean, std = self._trimmed_stats(pixels, num_channels=num_channels)
        signatures.append(mean)
        signatures_std.append(std)

    signatures_t = torch.stack(signatures, dim=0).unsqueeze(0)
    signatures_std_t = torch.stack(signatures_std, dim=0).unsqueeze(0)
    return {
        "signatures": signatures_t.to(torch.float32),
        "signatures_std": signatures_std_t.to(torch.float32),
    }
```

`TitleOverlay`cuvis_ai.node.compositingtransformimgrgbBurn a text caption into the top-left of each RGB frame, over a translucent box.

#### TitleOverlay

```python
TitleOverlay(
    text="",
    font_size=20,
    pad_px=8,
    text_color=(255, 255, 255),
    box_color=(0, 0, 0),
    box_alpha=0.5,
    **kwargs,
)
```

Bases: `Node`

Burn a text caption into the top-left of each RGB frame, over a translucent box.

The caption comes from one of three places, in priority order: the per-frame `caption` input port (a `list[str]`, one entry per frame, so a DataModule can title each montage column), the `text` argument to :meth:`forward`, or the constructor `text` default. Drawn with PIL over a semi-transparent box so it stays legible on any background.

Parameters:

| Name         | Type                   | Description                                                                | Default           |
| ------------ | ---------------------- | -------------------------------------------------------------------------- | ----------------- |
| `text`       | `str`                  | Default caption drawn into every frame when no per-frame caption is wired. | `''`              |
| `font_size`  | `int`                  | Caption font size in points (default 20).                                  | `20`              |
| `pad_px`     | `int`                  | Inset of the caption box from the top-left corner (default 8).             | `8`               |
| `text_color` | `tuple[int, int, int]` | Text and box RGB colours in 0-255.                                         | `(255, 255, 255)` |
| `box_color`  | `tuple[int, int, int]` | Text and box RGB colours in 0-255.                                         | `(255, 255, 255)` |
| `box_alpha`  | `float`                | Opacity of the box behind the text (default 0.5).                          | `0.5`             |

Source code in `cuvis_ai/node/compositing.py`

```python
def __init__(
    self,
    text: str = "",
    font_size: int = 20,
    pad_px: int = 8,
    text_color: tuple[int, int, int] = (255, 255, 255),
    box_color: tuple[int, int, int] = (0, 0, 0),
    box_alpha: float = 0.5,
    **kwargs: Any,
) -> None:
    if not 0.0 <= box_alpha <= 1.0:
        raise ValueError(f"box_alpha must be in [0, 1]; got {box_alpha}")
    self.text = str(text)
    self.pad_px = int(pad_px)
    self.text_color = tuple(int(c) for c in text_color)
    self.box_color = tuple(int(c) for c in box_color)
    self.box_alpha = float(box_alpha)
    super().__init__(
        text=self.text,
        font_size=int(font_size),
        pad_px=self.pad_px,
        text_color=list(self.text_color),
        box_color=list(self.box_color),
        box_alpha=self.box_alpha,
        **kwargs,
    )
    try:
        self._font = ImageFont.truetype("arial.ttf", int(font_size))
    except OSError:
        self._font = ImageFont.load_default()
```

##### forward

```python
forward(frame, caption=None, text=None, **_)
```

Caption each frame from the per-frame `caption` port, `text`, or the default.

Parameters:

| Name      | Type                | Description                                                                                                          | Default    |
| --------- | ------------------- | -------------------------------------------------------------------------------------------------------------------- | ---------- |
| `frame`   | `Tensor`            | RGB frames [B, H, W, 3] in [0, 1].                                                                                   | *required* |
| `caption` | `list[str] or None` | Per-frame captions, one per batch element; takes priority over text and the constructor default. Must have length B. | `None`     |
| `text`    | `str or None`       | Single caption applied to every frame, overriding the constructor default.                                           | `None`     |
| `**_`     | `Any`               | Additional unused keyword arguments (e.g. the pipeline context).                                                     | `{}`       |

Returns:

| Type                | Description                                                                                         |
| ------------------- | --------------------------------------------------------------------------------------------------- |
| `dict[str, Tensor]` | frame float32 [B, H, W, 3] with each caption drawn in; an empty caption leaves its frame unchanged. |

Source code in `cuvis_ai/node/compositing.py`

```python
@torch.no_grad()
def forward(
    self,
    frame: torch.Tensor,
    caption: list[str] | None = None,
    text: str | None = None,
    **_: Any,
) -> dict[str, torch.Tensor]:
    """Caption each frame from the per-frame ``caption`` port, ``text``, or the default.

    Parameters
    ----------
    frame : torch.Tensor
        RGB frames ``[B, H, W, 3]`` in ``[0, 1]``.
    caption : list[str] or None, optional
        Per-frame captions, one per batch element; takes priority over ``text`` and the
        constructor default. Must have length ``B``.
    text : str or None, optional
        Single caption applied to every frame, overriding the constructor default.
    **_ : Any
        Additional unused keyword arguments (e.g. the pipeline ``context``).

    Returns
    -------
    dict[str, torch.Tensor]
        ``frame`` float32 ``[B, H, W, 3]`` with each caption drawn in; an empty caption
        leaves its frame unchanged.
    """
    batch = frame.shape[0]
    if caption is not None:
        if len(caption) != batch:
            raise ValueError(
                f"caption has {len(caption)} entries but the batch has {batch} frames."
            )
        labels = [str(c) for c in caption]
    else:
        single = self.text if text is None else str(text)
        labels = [single] * batch
    out = torch.empty_like(frame)
    for i in range(batch):
        clamped = frame[i].clamp(0.0, 1.0)
        if not labels[i].strip():
            # An empty caption is a no-op: draw no box and pass the frame through.
            out[i] = clamped
            continue
        arr = (clamped * 255).round().to(torch.uint8).cpu().numpy()
        out[i] = torch.from_numpy(self._draw(arr, labels[i])).to(frame.device, frame.dtype)
    return {"frame": out}
```

`TopKIndices`cuvis_ai.node.channel_selectortransformdim-redhsinumpypreUtility node that surfaces the top-k channel indices from selector weights.

#### TopKIndices

```python
TopKIndices(k, **kwargs)
```

Bases: `Node`

Utility node that surfaces the top-k channel indices from selector weights.

This node extracts the indices of the top-k weighted channels from a selector's weight vector. Useful for introspection and reporting which channels were selected.

Parameters:

| Name | Type  | Description                     | Default    |
| ---- | ----- | ------------------------------- | ---------- |
| `k`  | `int` | Number of top indices to return | *required* |

Attributes:

| Name | Type  | Description                     |
| ---- | ----- | ------------------------------- |
| `k`  | `int` | Number of top indices to return |

Source code in `cuvis_ai/node/channel_selector.py`

```python
def __init__(self, k: int, **kwargs: Any) -> None:
    self.k = int(k)

    # Extract Node base parameters from kwargs to avoid duplication
    name = kwargs.pop("name", None)
    execution_stages = kwargs.pop("execution_stages", None)

    super().__init__(
        name=name,
        execution_stages=execution_stages,
        k=self.k,
        **kwargs,
    )
```

##### forward

```python
forward(weights, **_)
```

Return the indices of the top-k weighted channels.

Parameters:

| Name      | Type     | Description                            | Default    |
| --------- | -------- | -------------------------------------- | ---------- |
| `weights` | `Tensor` | Channel selection weights [n_channels] | *required* |

Returns:

| Type                | Description                                            |
| ------------------- | ------------------------------------------------------ |
| `dict[str, Tensor]` | Dictionary with "indices" key containing top-k indices |

Source code in `cuvis_ai/node/channel_selector.py`

```python
def forward(self, weights: torch.Tensor, **_: Any) -> dict[str, torch.Tensor]:
    """Return the indices of the top-k weighted channels.

    Parameters
    ----------
    weights : torch.Tensor
        Channel selection weights [n_channels]

    Returns
    -------
    dict[str, torch.Tensor]
        Dictionary with "indices" key containing top-k indices
    """
    top_k = min(self.k, weights.shape[-1]) if weights.numel() else 0
    if top_k == 0:
        return {"indices": torch.zeros(0, dtype=torch.int64, device=weights.device)}

    _, indices = torch.topk(weights, top_k)
    return {"indices": indices}
```

`TwoStageBinaryDecider`cuvis_ai.node.deciders.two_stage_decidertransformclassnumpypostTwo-stage binary decider: image-level gate + pixel quantile mask.

#### TwoStageBinaryDecider

```python
TwoStageBinaryDecider(
    image_threshold=0.5,
    top_k_fraction=0.001,
    quantile=0.995,
    reduce_dims=None,
    **kwargs,
)
```

Bases: `BinaryDecider`

Two-stage binary decider: image-level gate + pixel quantile mask.

Source code in `cuvis_ai/node/deciders/two_stage_decider.py`

```python
def __init__(
    self,
    image_threshold: float = 0.5,
    top_k_fraction: float = 0.001,
    quantile: float = 0.995,
    reduce_dims: Sequence[int] | None = None,
    **kwargs,
) -> None:
    if not 0.0 <= image_threshold <= 1.0:
        raise ValueError("image_threshold must be within [0, 1]")
    if not 0.0 < top_k_fraction <= 1.0:
        raise ValueError("top_k_fraction must be in (0, 1]")
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must be within [0, 1]")

    self.image_threshold = float(image_threshold)
    self.top_k_fraction = float(top_k_fraction)
    self.quantile = float(quantile)
    self.reduce_dims = (
        tuple(int(dim) for dim in reduce_dims) if reduce_dims is not None else None
    )
    super().__init__(
        image_threshold=self.image_threshold,
        top_k_fraction=self.top_k_fraction,
        quantile=self.quantile,
        reduce_dims=self.reduce_dims,
        **kwargs,
    )
```

##### forward

```python
forward(logits, **_)
```

Apply two-stage binary decision: image-level gate + pixel quantile.

Stage 1: Compute image-level anomaly score from top-k pixel scores. If below threshold, return blank mask (no anomalies).

Stage 2: For images passing the gate, apply pixel-level quantile thresholding to create binary anomaly mask.

Parameters:

| Name     | Type     | Description                                  | Default    |
| -------- | -------- | -------------------------------------------- | ---------- |
| `logits` | `Tensor` | Anomaly scores [B, H, W, C] or [B, H, W, 1]. | *required* |
| `**_`    | `Any`    | Additional unused keyword arguments.         | `{}`       |

Returns:

| Type                | Description                                                           |
| ------------------- | --------------------------------------------------------------------- |
| `dict[str, Tensor]` | Dictionary with "decisions" key containing binary masks [B, H, W, 1]. |

Notes

The image-level score is computed as the mean of the top-k% highest pixel scores. For multi-channel inputs, the max across channels is used for each pixel.

Source code in `cuvis_ai/node/deciders/two_stage_decider.py`

```python
def forward(self, logits: Tensor, **_: Any) -> dict[str, Tensor]:
    """Apply two-stage binary decision: image-level gate + pixel quantile.

    Stage 1: Compute image-level anomaly score from top-k pixel scores.
    If below threshold, return blank mask (no anomalies).

    Stage 2: For images passing the gate, apply pixel-level quantile
    thresholding to create binary anomaly mask.

    Parameters
    ----------
    logits : Tensor
        Anomaly scores [B, H, W, C] or [B, H, W, 1].
    **_ : Any
        Additional unused keyword arguments.

    Returns
    -------
    dict[str, Tensor]
        Dictionary with "decisions" key containing binary masks [B, H, W, 1].

    Notes
    -----
    The image-level score is computed as the mean of the top-k% highest
    pixel scores. For multi-channel inputs, the max across channels is
    used for each pixel.
    """
    tensor = logits
    bsz = tensor.shape[0]

    # DEBUG: Log input tensor stats
    logger.debug(
        f"TwoStageDecider input: shape={tensor.shape}, device={tensor.device}, "
        f"dtype={tensor.dtype}, min={tensor.min().item():.6f}, "
        f"max={tensor.max().item():.6f}, mean={tensor.mean().item():.6f}"
    )

    decisions = []
    for b in range(bsz):
        scores = tensor[b]  # [H, W, C]
        # Reduce to per-pixel max for image score
        if scores.dim() == 3:
            pixel_scores = scores.max(dim=-1)[0]
        else:
            pixel_scores = scores
        flat = pixel_scores.reshape(-1)
        k = max(
            1,
            int(
                torch.ceil(
                    torch.tensor(flat.numel() * self.top_k_fraction, dtype=torch.float32)
                ).item()
            ),
        )
        topk_vals, _ = torch.topk(flat, k)
        image_score = topk_vals.mean().item()  # Convert to Python float for comparison

        # DEBUG: Log intermediate computation values
        logger.debug(
            f"TwoStageDecider[batch={b}]: k={k}, topk_min={topk_vals.min().item():.6f}, "
            f"topk_max={topk_vals.max().item():.6f}, image_score={image_score:.6f}"
        )

        # Stage 1: Image-level gate
        if image_score < self.image_threshold:
            # Gate failed: return blank mask
            logger.debug(
                f"TwoStageDecider: image_score={image_score:.6f} < threshold={self.image_threshold:.6f}, "
                f"returning blank mask"
            )
            decisions.append(
                torch.zeros((*pixel_scores.shape, 1), dtype=torch.bool, device=tensor.device)
            )
            continue

        # Stage 2: Gate passed, apply pixel-level quantile thresholding
        logger.debug(
            f"TwoStageDecider: image_score={image_score:.6f} >= threshold={self.image_threshold:.6f}, "
            f"applying quantile thresholding (q={self.quantile})"
        )
        # Compute quantile threshold: reduce over all dimensions to get scalar per batch item
        # This matches QuantileBinaryDecider behavior: for [B, H, W, C] it reduces over (H, W, C)
        # For single batch item [H, W, C], we reduce over all dims (0, 1, 2)
        threshold = torch.quantile(scores, self.quantile)

        # Apply threshold: for multi-channel scores, take max across channels first
        if scores.dim() == 3:  # [H, W, C]
            # Take max across channels to get per-pixel score, then threshold
            pixel_scores = scores.max(dim=-1, keepdim=False)[0]  # [H, W]
            binary_map = (pixel_scores >= threshold).unsqueeze(-1).to(torch.bool)  # [H, W, 1]
        else:  # [H, W] - single channel
            binary_map = (scores >= threshold).unsqueeze(-1).to(torch.bool)  # [H, W, 1]

        decisions.append(binary_map)

    return {"decisions": torch.stack(decisions, dim=0)}
```

`UnitVarianceScaling`cuvis_ai.node.pretreatments.scalingtransformhsinormpretorchDivide the cube by a globally-fitted per-channel standard deviation.

#### UnitVarianceScaling

```python
UnitVarianceScaling(**kwargs)
```

Bases: `_StatisticalFitNode`

Divide the cube by a globally-fitted per-channel standard deviation.

During `statistical_initialization` every training pixel is streamed through a Welford accumulator to compute the exact per-channel standard deviation (sample, `ddof=1`) over the full dataset; `forward` then divides each spectrum by that standard deviation.

Notes

The fitted `std_c` is registered as a persistent buffer so a checkpointed node reloads ready for inference.

Source code in `cuvis_ai/node/pretreatments/scaling.py`

```python
def __init__(self, **kwargs) -> None:
    super().__init__(**kwargs)
    self.register_buffer("std_c", torch.zeros(0, dtype=torch.float32))
    self._welford: WelfordAccumulator | None = None
```

##### statistical_initialization

```python
statistical_initialization(input_stream)
```

Fit the per-channel standard deviation from the stream via Welford.

Parameters:

| Name           | Type          | Description                                              | Default    |
| -------------- | ------------- | -------------------------------------------------------- | ---------- |
| `input_stream` | `InputStream` | Iterable of port-keyed batch dicts matching INPUT_SPECS. | *required* |

Source code in `cuvis_ai/node/pretreatments/scaling.py`

```python
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
```

##### forward

```python
forward(cube, **_)
```

Divide the cube by the fitted per-channel standard deviation.

Parameters:

| Name   | Type     | Description                | Default    |
| ------ | -------- | -------------------------- | ---------- |
| `cube` | `Tensor` | Input cube in BHWC format. | *required* |

Returns:

| Type                | Description                                        |
| ------------------- | -------------------------------------------------- |
| `dict[str, Tensor]` | {"cube": scaled} with the same shape as the input. |

Source code in `cuvis_ai/node/pretreatments/scaling.py`

```python
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
```

`WaferSegmentation`cuvis_ai_wafer_thickness.node.wafer_segmentationtransformhsimasknumpysegtorchwafer_thickness[↗](https://github.com/cubert-hyperspectral/cuvis-ai-wafer-thickness "Open plugin repo")Segment a wafer from a flat background by spectral contrast.

Segment a wafer from a flat background by spectral contrast.

Inputs

| Port          | Dtype     | Shape              | Description                                  |
| ------------- | --------- | ------------------ | -------------------------------------------- |
| `cube`        | `float32` | `[-1, -1, -1, -1]` | Hyperspectral reflectance cube [B, H, W, C]. |
| `wavelengths` | `int32`   | `[-1]`             | Wavelengths [C] in nm.                       |

Outputs

| Port   | Dtype   | Shape          | Description                             |
| ------ | ------- | -------------- | --------------------------------------- |
| `mask` | `int32` | `[-1, -1, -1]` | Wafer mask [B, H, W]; non-zero = wafer. |

[View plugin repo (v0.3.1)](https://github.com/cubert-hyperspectral/cuvis-ai-wafer-thickness/tree/v0.3.1)

`WaferThickness`cuvis_ai_wafer_thickness.node.wafer_thicknesstransformhsinumpyregtorchwafer_thickness[↗](https://github.com/cubert-hyperspectral/cuvis-ai-wafer-thickness "Open plugin repo")Per-pixel film thickness [nm] from interference-order peak positions.

Per-pixel film thickness [nm] from interference-order peak positions.

Inputs

| Port            | Dtype     | Shape              | Description                                                               |
| --------------- | --------- | ------------------ | ------------------------------------------------------------------------- |
| `cube`          | `float32` | `[-1, -1, -1, -1]` | Hyperspectral reflectance cube [B, H, W, C].                              |
| `wavelengths`   | `int32`   | `[-1]`             | Wavelengths [C] in nm.                                                    |
| `mask` optional | `int32`   | `[-1, -1, -1]`     | Wafer mask [B, H, W]; non-zero = analyse. Optional: omitted = all pixels. |

Outputs

| Port          | Dtype     | Shape          | Description                                                                                                     |
| ------------- | --------- | -------------- | --------------------------------------------------------------------------------------------------------------- |
| `thickness`   | `float32` | `[-1, -1, -1]` | Per-pixel film thickness [B, H, W] in nm; NaN outside mask.                                                     |
| `uncertainty` | `float32` | `[-1, -1, -1]` | Std of per-order thickness estimates [B, H, W] in nm (lower is better; 0 for a single order); NaN outside mask. |

[View plugin repo (v0.3.1)](https://github.com/cubert-hyperspectral/cuvis-ai-wafer-thickness/tree/v0.3.1)

`YOLOPostprocess`cuvis_ai_ultralytics.nodetransformbboxdetposttorchultralytics[↗](https://github.com/cubert-hyperspectral/cuvis-ai-ultralytics "Open plugin repo")Ultralytics NMS + box scaling postprocess for YOLO raw tensors.

Ultralytics NMS + box scaling postprocess for YOLO raw tensors.

Inputs

| Port             | Dtype     | Shape          | Description     |
| ---------------- | --------- | -------------- | --------------- |
| `raw_preds`      | `float32` | `[-1, -1, -1]` | Raw YOLO tensor |
| `model_input_hw` | `int64`   | `[-1, 2]`      | Model H,W       |
| `orig_hw`        | `int64`   | `[-1, 2]`      | Original H,W    |

Outputs

| Port           | Dtype     | Shape         | Description           |
| -------------- | --------- | ------------- | --------------------- |
| `bboxes`       | `float32` | `[-1, -1, 4]` | Final boxes [B, N, 4] |
| `category_ids` | `int64`   | `[-1, -1]`    | Class ids [B, N]      |
| `confidences`  | `float32` | `[-1, -1]`    | Scores [B, N]         |

[View plugin repo (v0.1.4)](https://github.com/cubert-hyperspectral/cuvis-ai-ultralytics/tree/v0.1.4)

`YOLOPreprocess`cuvis_ai_ultralytics.nodetransformimgprergbtorchultralytics[↗](https://github.com/cubert-hyperspectral/cuvis-ai-ultralytics "Open plugin repo")Convert RGB images to stride-aligned channel-first BGR tensors for YOLO.

Convert RGB images to stride-aligned channel-first BGR tensors for YOLO.

Inputs

| Port        | Dtype     | Shape             | Description                      |
| ----------- | --------- | ----------------- | -------------------------------- |
| `rgb_image` | `float32` | `[-1, -1, -1, 3]` | RGB image [B, H, W, 3] in [0, 1] |

Outputs

| Port             | Dtype     | Shape             | Description                                                |
| ---------------- | --------- | ----------------- | ---------------------------------------------------------- |
| `preprocessed`   | `float32` | `[-1, 3, -1, -1]` | Channel-first BGR [B, 3, H', W'] stride-aligned, in [0, 1] |
| `model_input_hw` | `int64`   | `[-1, 2]`         | Padded model input [H', W'] per sample                     |
| `orig_hw`        | `int64`   | `[-1, 2]`         | Original image [H, W] per sample before preprocessing      |

[View plugin repo (v0.1.4)](https://github.com/cubert-hyperspectral/cuvis-ai-ultralytics/tree/v0.1.4)

`ZScoreNormalizer`cuvis_ai.node.normalizationtransformnormnumpypreZ-score (standardization) normalization along specified dimensions.

#### ZScoreNormalizer

```python
ZScoreNormalizer(
    dims=None, eps=1e-06, keepdim=True, **kwargs
)
```

Bases: `_ScoreNormalizerBase`

Z-score (standardization) normalization along specified dimensions.

Computes: (x - mean) / (std + eps) along specified dims. Per-sample normalization with no statistical initialization required.

Parameters:

| Name      | Type        | Description                                                                   | Default |
| --------- | ----------- | ----------------------------------------------------------------------------- | ------- |
| `dims`    | `list[int]` | Dimensions to compute statistics over (default: [1,2] for H,W in BHWC format) | `None`  |
| `eps`     | `float`     | Small constant for numerical stability (default: 1e-6)                        | `1e-06` |
| `keepdim` | `bool`      | Whether to keep reduced dimensions (default: True)                            | `True`  |

Examples:

```pycon
>>> # Normalize over spatial dimensions (H, W)
>>> zscore = ZScoreNormalizer(dims=[1, 2])
>>>
>>> # Normalize over all spatial and channel dimensions
>>> zscore_all = ZScoreNormalizer(dims=[1, 2, 3])
```

Source code in `cuvis_ai/node/normalization.py`

```python
def __init__(
    self, dims: list[int] | None = None, eps: float = 1e-6, keepdim: bool = True, **kwargs
) -> None:
    self.dims = dims if dims is not None else [1, 2]
    self.eps = float(eps)
    self.keepdim = keepdim
    super().__init__(dims=self.dims, eps=eps, keepdim=keepdim, **kwargs)
```

`ZScoreNormalizerGlobal`cuvis_ai.node.anomaly.deep_svddtransformhsinormnumpyprePort-based Deep SVDD z-score normalizer for BHWC cubes.

#### ZScoreNormalizerGlobal

```python
ZScoreNormalizerGlobal(
    *, num_channels, eps=1e-08, **kwargs
)
```

Bases: `Node`

Port-based Deep SVDD z-score normalizer for BHWC cubes.

Source code in `cuvis_ai/node/anomaly/deep_svdd.py`

```python
def __init__(
    self,
    *,
    num_channels: int,
    eps: float = 1e-8,
    **kwargs: Any,
) -> None:
    if num_channels <= 0:
        raise ValueError(f"num_channels must be positive, got {num_channels}")
    self.num_channels = int(num_channels)
    self.eps = float(eps)

    super().__init__(
        num_channels=self.num_channels,
        eps=self.eps,
        **kwargs,
    )

    # Pre-allocate buffers with known dimensions
    self.register_buffer(
        "zscore_mean", torch.zeros(num_channels, dtype=torch.get_default_dtype())
    )
    self.register_buffer(
        "zscore_std", torch.ones(num_channels, dtype=torch.get_default_dtype())
    )
```

##### requires_initial_fit

```python
requires_initial_fit
```

Whether this node requires statistical initialization from training data.

Returns:

| Type   | Description                            |
| ------ | -------------------------------------- |
| `bool` | Always True for Z-score normalization. |

##### statistical_initialization

```python
statistical_initialization(input_stream)
```

Estimate per-band z-score statistics from the provided stream.

Source code in `cuvis_ai/node/anomaly/deep_svdd.py`

```python
def statistical_initialization(self, input_stream: InputStream) -> None:
    """Estimate per-band z-score statistics from the provided stream."""
    acc = WelfordAccumulator(self.num_channels)
    for batch in input_stream:
        data = batch.get("data")
        if data is None:
            continue
        data = data.contiguous()

        B, H, W, C = data.shape
        if C != self.num_channels:
            raise ValueError(f"Channel mismatch: expected {self.num_channels}, got {C}")
        acc.update(data.reshape(B * H * W, C))

    if acc.count == 0:
        raise RuntimeError(
            "DeepSVDDEncoder.statistical_initialization() did not receive any data"
        )

    self.zscore_mean.copy_(acc.mean)
    self.zscore_std.copy_(acc.std + self.eps)
    self._statistically_initialized = True
```

##### forward

```python
forward(data, **_)
```

Apply per-channel Z-score normalization.

Parameters:

| Name   | Type     | Description                          | Default    |
| ------ | -------- | ------------------------------------ | ---------- |
| `data` | `Tensor` | Input feature tensor [B, H, W, C].   | *required* |
| `**_`  | `Any`    | Additional unused keyword arguments. | `{}`       |

Returns:

| Type                | Description                                                                       |
| ------------------- | --------------------------------------------------------------------------------- |
| `dict[str, Tensor]` | Dictionary with "normalized" key containing Z-score normalized data [B, H, W, C]. |

Raises:

| Type           | Description                                                    |
| -------------- | -------------------------------------------------------------- |
| `RuntimeError` | If statistical_initialization() has not been called.           |
| `ValueError`   | If input channel count doesn't match initialized num_channels. |

Source code in `cuvis_ai/node/anomaly/deep_svdd.py`

```python
def forward(self, data: torch.Tensor, **_: Any) -> dict[str, torch.Tensor]:
    """Apply per-channel Z-score normalization.

    Parameters
    ----------
    data : torch.Tensor
        Input feature tensor [B, H, W, C].
    **_ : Any
        Additional unused keyword arguments.

    Returns
    -------
    dict[str, torch.Tensor]
        Dictionary with "normalized" key containing Z-score normalized data [B, H, W, C].

    Raises
    ------
    RuntimeError
        If statistical_initialization() has not been called.
    ValueError
        If input channel count doesn't match initialized num_channels.
    """
    if not self._statistically_initialized:
        raise RuntimeError(
            "DeepSVDDEncoder requires statistical_initialization() before forward()"
        )

    B, H, W, C = data.shape
    if C != self.num_channels:
        raise ValueError(f"Channel mismatch: expected {self.num_channels}, got {C}")

    flat = data.contiguous().reshape(B * H * W, C)
    mean = self.zscore_mean.to(dtype=flat.dtype, copy=False).unsqueeze(0)
    std = self.zscore_std.to(dtype=flat.dtype, copy=False).unsqueeze(0)
    flat = (flat - mean) / std

    normalized = flat.reshape(B, H, W, C)
    return {"normalized": normalized}
```

`_ColormappedNormalizedDifferenceSelector`cuvis_ai.node.channel_selectortransformdim-redhsinumpypreTwo-band normalized-difference selector with an HSV-colormap RGB render.

#### \_ColormappedNormalizedDifferenceSelector

```python
_ColormappedNormalizedDifferenceSelector(
    primary_nm,
    secondary_nm,
    colormap_min=-1.0,
    colormap_max=1.0,
    eps=1e-06,
    **kwargs,
)
```

Bases: `_NormalizedDifferenceIndexBase`, `ABC`

Two-band normalized-difference selector with an HSV-colormap RGB render.

Abstract base: concrete subclasses set the primary/secondary default wavelengths and implement the semantic `index_name` / `primary_label` / `secondary_label` properties. The scalar index image is mapped to `rgb_image` with the same Blood_OXY-style HSV colormap path that :class:`NDVISelector` uses, controlled by `colormap_min` / `colormap_max`.

Source code in `cuvis_ai/node/channel_selector.py`

```python
def __init__(
    self,
    primary_nm: float,
    secondary_nm: float,
    colormap_min: float = -1.0,
    colormap_max: float = 1.0,
    eps: float = 1.0e-6,
    **kwargs: Any,
) -> None:
    if colormap_max <= colormap_min:
        raise ValueError("colormap_max must be greater than colormap_min")
    kwargs.setdefault("norm_mode", NormMode.PER_FRAME)
    kwargs.setdefault("apply_gamma", False)
    super().__init__(
        primary_nm=primary_nm,
        secondary_nm=secondary_nm,
        eps=eps,
        colormap_min=float(colormap_min),
        colormap_max=float(colormap_max),
        **kwargs,
    )
    self.colormap = "hsv"
    self.colormap_min = float(colormap_min)
    self.colormap_max = float(colormap_max)
    self._colormap_range = self.colormap_max - self.colormap_min
```

##### index_name

```python
index_name
```

Canonical strategy / index name.

##### primary_label

```python
primary_label
```

Semantic label for the first operand.

##### secondary_label

```python
secondary_label
```

Semantic label for the second operand.

##### forward

```python
forward(cube, wavelengths, context=None, **_)
```

Compute the normalized-difference index plus a colour-mapped RGB output.

Source code in `cuvis_ai/node/channel_selector.py`

```python
def forward(
    self,
    cube: torch.Tensor,
    wavelengths: Any,
    context: Context | None = None,
    **_: Any,
) -> dict[str, Any]:
    """Compute the normalized-difference index plus a colour-mapped RGB output."""
    result = super().forward(cube=cube, wavelengths=wavelengths, context=context, **_)
    result["band_info"].update(
        {
            "rendering": f"{self.colormap}_colormap",
            "colormap": self.colormap,
            "colormap_min": self.colormap_min,
            "colormap_max": self.colormap_max,
        }
    )
    return result
```

`_NormalizedDifferenceIndexBase`cuvis_ai.node.channel_selectortransformdim-redhsinumpypreAbstract base for two-band normalized-difference indices.

#### \_NormalizedDifferenceIndexBase

```python
_NormalizedDifferenceIndexBase(
    primary_nm,
    secondary_nm,
    eps=1e-06,
    band_tolerance_nm=50.0,
    **kwargs,
)
```

Bases: `ChannelSelectorBase`, `ABC`

Abstract base for two-band normalized-difference indices.

Subclasses define the semantic labels and strategy name, while this base handles nearest-band resolution, stable normalized-difference computation, and delegates RGB rendering to subclasses.

Source code in `cuvis_ai/node/channel_selector.py`

```python
def __init__(
    self,
    primary_nm: float,
    secondary_nm: float,
    eps: float = 1.0e-6,
    band_tolerance_nm: float = 50.0,
    **kwargs: Any,
) -> None:
    if eps < 0:
        raise ValueError("eps must be >= 0")

    super().__init__(
        primary_nm=float(primary_nm),
        secondary_nm=float(secondary_nm),
        eps=float(eps),
        band_tolerance_nm=float(band_tolerance_nm),
        **kwargs,
    )
    self.primary_nm = float(primary_nm)
    self.secondary_nm = float(secondary_nm)
    self.eps = float(eps)
    self.band_tolerance_nm = float(band_tolerance_nm)
    self._bands_warned: set[str] = set()
```

##### index_name

```python
index_name
```

Canonical strategy / index name.

##### primary_label

```python
primary_label
```

Semantic label for the first operand.

##### secondary_label

```python
secondary_label
```

Semantic label for the second operand.

##### forward

```python
forward(cube, wavelengths, context=None, **_)
```

Compute raw index image plus RGB render.

Source code in `cuvis_ai/node/channel_selector.py`

```python
def forward(
    self,
    cube: torch.Tensor,
    wavelengths: Any,
    context: Context | None = None,  # noqa: ARG002
    **_: Any,
) -> dict[str, Any]:
    """Compute raw index image plus RGB render."""
    wavelengths_np, primary_idx, secondary_idx = self._resolve_band_indices(wavelengths)
    self._warn_bands_out_of_tolerance(
        wavelengths_np,
        [self.primary_label, self.secondary_label],
        [self.primary_nm, self.secondary_nm],
        [primary_idx, secondary_idx],
    )
    index_image = self._compute_index_image(cube, wavelengths_np)
    rgb = self._render_rgb_from_index(index_image)

    band_info = {
        "strategy": self.index_name,
        "band_labels": [self.primary_label, self.secondary_label],
        "band_indices": [primary_idx, secondary_idx],
        "requested_wavelengths_nm": [self.primary_nm, self.secondary_nm],
        "resolved_wavelengths_nm": [
            float(wavelengths_np[primary_idx]),
            float(wavelengths_np[secondary_idx]),
        ],
    }
    return {
        "index_image": index_image,
        "rgb_image": rgb,
        "band_info": band_info,
    }
```

`_ScoreNormalizerBase`cuvis_ai.node.normalizationtransformnormnumpypreBase class for BHWC normalization nodes.

#### \_ScoreNormalizerBase

```python
_ScoreNormalizerBase(*args, **kwargs)
```

Bases: `Node`

Base class for BHWC normalization nodes.

Notes

All normalization nodes in this module expect inputs in BHWC format ([batch, height, width, channels]). Callers are responsible for adding a batch dimension when working with HWC tensors (use `x.unsqueeze(0)`).

Most subclasses are differentiable, but some keep fitted statistical state (e.g. :class:`MinMaxNormalizer` with running stats, :class:`PercentileNormalizer`) and update it under `no_grad` during `forward`.

Source code in `cuvis_ai/node/normalization.py`

```python
def __init__(self, *args, **kwargs) -> None:
    super().__init__(*args, **kwargs)
```

##### forward

```python
forward(data, **_)
```

Normalize input data (BHWC only).

Parameters:

| Name   | Type     | Description                              | Default    |
| ------ | -------- | ---------------------------------------- | ---------- |
| `data` | `Tensor` | Input tensor in BHWC format [B, H, W, C] | *required* |

Returns:

| Type                | Description                                                   |
| ------------------- | ------------------------------------------------------------- |
| `dict[str, Tensor]` | Dictionary with "normalized" key containing normalized tensor |

Source code in `cuvis_ai/node/normalization.py`

```python
def forward(self, data: Tensor, **_: Any) -> dict[str, Tensor]:
    """Normalize input data (BHWC only).

    Parameters
    ----------
    data : Tensor
        Input tensor in BHWC format [B, H, W, C]

    Returns
    -------
    dict[str, Tensor]
        Dictionary with "normalized" key containing normalized tensor
    """
    normalized = self._normalize(data)
    return {"normalized": normalized}
```

`_VegetationIndexBase`cuvis_ai.node.channel_selectortransformdim-redhsinumpypreAbstract base for multi-band vegetation indices with custom formulas.

#### \_VegetationIndexBase

```python
_VegetationIndexBase(
    band_nm,
    colormap_min=-1.0,
    colormap_max=1.0,
    eps=1e-06,
    band_tolerance_nm=50.0,
    **kwargs,
)
```

Bases: `ChannelSelectorBase`, `ABC`

Abstract base for multi-band vegetation indices with custom formulas.

Resolves a set of named bands (each a defaulted, tunable wavelength hparam) to their nearest sensor wavelengths, computes a scalar `index_image` from a subclass-provided formula, and renders `rgb_image` with the same Blood_OXY HSV colormap path used by :class:`NDVISelector`. Denominators in subclass formulas should be clamped with `self.eps` to avoid divide-by-zero.

Source code in `cuvis_ai/node/channel_selector.py`

```python
def __init__(
    self,
    band_nm: dict[str, float],
    colormap_min: float = -1.0,
    colormap_max: float = 1.0,
    eps: float = 1.0e-6,
    band_tolerance_nm: float = 50.0,
    **kwargs: Any,
) -> None:
    if eps < 0:
        raise ValueError("eps must be >= 0")
    if colormap_max <= colormap_min:
        raise ValueError("colormap_max must be greater than colormap_min")
    kwargs.setdefault("norm_mode", NormMode.PER_FRAME)
    kwargs.setdefault("apply_gamma", False)
    super().__init__(
        colormap_min=float(colormap_min),
        colormap_max=float(colormap_max),
        eps=float(eps),
        band_tolerance_nm=float(band_tolerance_nm),
        **kwargs,
    )
    self.band_nm = {name: float(nm) for name, nm in band_nm.items()}
    self.eps = float(eps)
    self.band_tolerance_nm = float(band_tolerance_nm)
    self._bands_warned: set[str] = set()
    self.colormap = "hsv"
    self.colormap_min = float(colormap_min)
    self.colormap_max = float(colormap_max)
    self._colormap_range = self.colormap_max - self.colormap_min
```

##### index_name

```python
index_name
```

Canonical strategy / index name.

##### forward

```python
forward(cube, wavelengths, context=None, **_)
```

Compute the vegetation index plus a colour-mapped RGB output.

Source code in `cuvis_ai/node/channel_selector.py`

```python
def forward(
    self,
    cube: torch.Tensor,
    wavelengths: Any,
    context: Context | None = None,  # noqa: ARG002
    **_: Any,
) -> dict[str, Any]:
    """Compute the vegetation index plus a colour-mapped RGB output."""
    wavelengths_np, indices = self._resolve_band_indices(wavelengths)
    band_labels = list(indices.keys())
    band_indices = [indices[name] for name in band_labels]
    self._warn_bands_out_of_tolerance(
        wavelengths_np,
        band_labels,
        [self.band_nm[name] for name in band_labels],
        band_indices,
    )
    index_image = self._compute_index_image(cube, wavelengths_np)
    rgb = self._render_rgb_from_index(index_image)
    band_info = {
        "strategy": self.index_name,
        "band_labels": band_labels,
        "band_indices": band_indices,
        "requested_wavelengths_nm": [self.band_nm[name] for name in band_labels],
        "resolved_wavelengths_nm": [
            float(wavelengths_np[indices[name]]) for name in band_labels
        ],
        "rendering": f"{self.colormap}_colormap",
        "colormap": self.colormap,
        "colormap_min": self.colormap_min,
        "colormap_max": self.colormap_max,
    }
    return {
        "index_image": index_image,
        "rgb_image": rgb,
        "band_info": band_info,
    }
```

`DensePatchDataModule`cuvis_ai_inspecscrap.data.dense_patch_datamoduleunspecifieddata modulecuvis_ai_inspecscrap[↗](https://github.com/cubert-hyperspectral/cuvis-ai-inspecscrap "Open plugin repo")

Data module `metal_scrap_dense_patch` — pip extras: `tiff`.

[View plugin repo (v0.2.4)](https://github.com/cubert-hyperspectral/cuvis-ai-inspecscrap/tree/v0.2.4)

`MetalScrapDataModule`cuvis_ai_inspecscrap.data.metal_scrap_datamoduleunspecifieddata modulecuvis_ai_inspecscrap[↗](https://github.com/cubert-hyperspectral/cuvis-ai-inspecscrap "Open plugin repo")

Data module `metal_scrap` — pip extras: `tiff`.

[View plugin repo (v0.2.4)](https://github.com/cubert-hyperspectral/cuvis-ai-inspecscrap/tree/v0.2.4)

`MetalScrapPatchDataModule`cuvis_ai_inspecscrap.data.metal_scrap_patch_datamoduleunspecifieddata modulecuvis_ai_inspecscrap[↗](https://github.com/cubert-hyperspectral/cuvis-ai-inspecscrap "Open plugin repo")

Data module `metal_scrap_patch` — pip extras: `tiff`.

[View plugin repo (v0.2.4)](https://github.com/cubert-hyperspectral/cuvis-ai-inspecscrap/tree/v0.2.4)

`MontageColumnDataModule`cuvis_ai_inspecscrap.data.montage_column_datamoduleunspecifieddata modulecuvis_ai_inspecscrap[↗](https://github.com/cubert-hyperspectral/cuvis-ai-inspecscrap "Open plugin repo")

Data module `metal_scrap_montage_columns` — pip extras: none.

[View plugin repo (v0.2.4)](https://github.com/cubert-hyperspectral/cuvis-ai-inspecscrap/tree/v0.2.4)

`AnomalyMask`cuvis_ai.node.anomaly_visualizationvisualizeranommaskVisualize anomaly detection with GT and predicted masks.

#### AnomalyMask

```python
AnomalyMask(channel, up_to=None, **kwargs)
```

Bases: `Node`

Visualize anomaly detection with GT and predicted masks.

Creates side-by-side visualizations showing ground truth masks, predicted masks, and overlay comparisons on hyperspectral cube images. The overlay shows:

- Green: True Positives (correct anomaly detection)
- Red: False Positives (false alarms)
- Yellow: False Negatives (missed anomalies)

Also displays IoU and other metrics. Returns a list of Artifact objects for logging to monitoring systems.

Executes during validation and inference stages.

Parameters:

| Name      | Type  | Description                                                                    | Default    |
| --------- | ----- | ------------------------------------------------------------------------------ | ---------- |
| `channel` | `int` | Channel index to use for cube visualization (required)                         | *required* |
| `up_to`   | `int` | Maximum number of images to visualize. If None, visualizes all (default: None) | `None`     |

Examples:

```pycon
>>> decider = BinaryDecider(threshold=0.2)
>>> viz_mask = AnomalyMask(channel=30, up_to=5)
>>> tensorboard_node = TensorBoardMonitorNode(output_dir="./runs")
>>> graph.connect(
...     (logit_head.logits, decider.data),
...     (decider.decisions, viz_mask.decisions),
...     (data_node.mask, viz_mask.mask),
...     (data_node.cube, viz_mask.cube),
...     (viz_mask.artifacts, tensorboard_node.artifacts),
... )
```

Source code in `cuvis_ai/node/anomaly_visualization.py`

```python
def __init__(self, channel: int, up_to: int | None = None, **kwargs) -> None:
    self.channel = channel
    self.up_to = up_to

    super().__init__(
        execution_stages={ExecutionStage.VAL, ExecutionStage.TEST, ExecutionStage.INFERENCE},
        channel=channel,
        up_to=up_to,
        **kwargs,
    )
```

##### forward

```python
forward(decisions, cube, context, mask=None, scores=None)
```

Create anomaly mask visualizations with GT/pred comparison.

Parameters:

| Name        | Type      | Description                                    | Default                                           |
| ----------- | --------- | ---------------------------------------------- | ------------------------------------------------- |
| `decisions` | `Tensor`  | Binary anomaly decisions [B, H, W, 1]          | *required*                                        |
| `mask`      | \`Tensor  | None\`                                         | Ground truth anomaly mask [B, H, W, 1] (optional) |
| `cube`      | `Tensor`  | Original cube [B, H, W, C] for visualization   | *required*                                        |
| `context`   | `Context` | Execution context with stage, epoch, batch_idx | *required*                                        |

Returns:

| Type   | Description                                                         |
| ------ | ------------------------------------------------------------------- |
| `dict` | Dictionary with "artifacts" key containing list of Artifact objects |

Source code in `cuvis_ai/node/anomaly_visualization.py`

```python
def forward(
    self,
    decisions: torch.Tensor,
    cube: torch.Tensor,
    context: Context,
    mask: torch.Tensor | None = None,
    scores: torch.Tensor | None = None,
) -> dict:
    """Create anomaly mask visualizations with GT/pred comparison.

    Parameters
    ----------
    decisions : torch.Tensor
        Binary anomaly decisions [B, H, W, 1]
    mask : torch.Tensor | None
        Ground truth anomaly mask [B, H, W, 1] (optional)
    cube : torch.Tensor
        Original cube [B, H, W, C] for visualization
    context : Context
        Execution context with stage, epoch, batch_idx

    Returns
    -------
    dict
        Dictionary with "artifacts" key containing list of Artifact objects
    """
    # Extract context information
    stage = context.stage.value
    epoch = context.epoch
    batch_idx = context.batch_idx

    # Use decisions directly (already binary)
    pred_mask = decisions.float()

    # Convert to numpy and squeeze channel dimension
    pred_mask_np = pred_mask.detach().cpu().numpy().squeeze(-1)  # [B, H, W]
    cube_np = cube.detach().cpu().numpy()  # [B, H, W, C]

    # Determine if we should use ground truth
    # Skip GT comparison if: mask not provided, inference stage, or mask is all zeros
    use_gt = (
        mask is not None and context.stage != ExecutionStage.INFERENCE and mask.any().item()
    )

    # Process ground truth mask if available
    gt_mask_np = None
    batch_iou = None
    if use_gt:
        gt_mask_np = mask.detach().cpu().numpy().squeeze(-1)  # [B, H, W]

        # Add binary mask assertion
        unique_values = np.unique(gt_mask_np)
        if not np.all(np.isin(unique_values, [0, 1, True, False])):
            raise ValueError(
                f"AnomalyMask expects binary masks with only values {{0, 1}}. "
                f"Found unique values: {unique_values}. "
                f"Ensure LentilsAnomolyDataNode is configured with anomaly_class_ids "
                f"to convert multi-class masks to binary."
            )

        # Compute batch-level IoU (matches AnomalyDetectionMetrics computation)
        batch_gt = gt_mask_np > 0.5  # [B, H, W] bool
        batch_pred = pred_mask_np > 0.5  # [B, H, W] bool
        batch_tp = np.logical_and(batch_pred, batch_gt).sum()
        batch_fp = np.logical_and(batch_pred, ~batch_gt).sum()
        batch_fn = np.logical_and(~batch_pred, batch_gt).sum()
        batch_iou = batch_tp / (batch_tp + batch_fp + batch_fn + 1e-8)

    # Determine how many images to visualize from this batch
    batch_size = pred_mask_np.shape[0]
    up_to_batch = batch_size if self.up_to is None else min(batch_size, self.up_to)

    # List to collect artifacts
    artifacts = []

    # Loop through each image in the batch up to the limit
    for i in range(up_to_batch):
        # Get predicted mask for this image
        pred = pred_mask_np[i] > 0.5  # [H, W] bool

        # Get cube channel for visualization
        cube_img = cube_np[i]  # [H, W, C]
        cube_channel = cube_img[:, :, self.channel]

        # Normalize cube channel to [0, 1] for display
        cube_norm = (cube_channel - cube_channel.min()) / (
            cube_channel.max() - cube_channel.min() + 1e-8
        )

        if use_gt:
            # Mode A: Full comparison with ground truth
            assert gt_mask_np is not None, "gt_mask_np should not be None when use_gt is True"
            gt = gt_mask_np[i] > 0.5  # [H, W] bool

            # Compute confusion matrix
            tp = np.logical_and(pred, gt)  # True Positives
            fp = np.logical_and(pred, ~gt)  # False Positives
            fn = np.logical_and(~pred, gt)  # False Negatives
            # Compute metrics
            tp_count = tp.sum()
            fp_count = fp.sum()
            fn_count = fn.sum()

            precision = tp_count / (tp_count + fp_count + 1e-8)
            recall = tp_count / (tp_count + fn_count + 1e-8)
            iou = tp_count / (tp_count + fp_count + fn_count + 1e-8)

            # Create figure with 3 subplots
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))

            # Subplot 1: Ground truth mask
            axes[0].imshow(gt, cmap="gray", aspect="auto")
            axes[0].set_title("Ground Truth Mask")
            axes[0].set_xlabel("Width")
            axes[0].set_ylabel("Height")

            # Subplot 2: Cube with TP/FP/FN overlay
            per_image_ap = None
            if scores is not None:
                raw_scores = scores[i, ..., 0]
                probs = torch.sigmoid(raw_scores).flatten()
                target_tensor = mask[i, ..., 0].flatten().to(dtype=torch.long)
                if probs.numel() == target_tensor.numel():
                    per_image_ap = binary_average_precision(probs, target_tensor).item()

            axes[1].imshow(cube_norm, cmap="gray", aspect="auto")

            # Create color overlay
            overlay = np.zeros((*gt.shape, 4))
            overlay[tp] = [0, 1, 0, 0.6]  # Green: True Positives
            overlay[fp] = [1, 0, 0, 0.6]  # Red: False Positives
            overlay[fn] = [1, 1, 0, 0.6]  # Yellow: False Negatives
            # TN pixels remain transparent (no overlay)

            overlay_title = f"Overlay (Channel {self.channel}) - IoU: {iou:.3f}"
            if per_image_ap is not None:
                overlay_title += f" | AP: {per_image_ap:.3f}"
            overlay_title += "\nGreen=TP, Red=FP, Yellow=FN"

            axes[1].imshow(overlay, aspect="auto")
            axes[1].set_title(overlay_title)
            axes[1].set_xlabel("Width")
            axes[1].set_ylabel("Height")

            # Subplot 3: Predicted mask with metrics in title
            axes[2].imshow(pred, cmap="gray", aspect="auto")

            # Add metrics as title (smaller font)
            metrics_title = (
                f"Predicted Mask\nIoU: {iou:.4f} | Prec: {precision:.4f} | Rec: {recall:.4f}"
            )
            if per_image_ap is not None:
                metrics_title += f" | AP: {per_image_ap:.4f}"
            metrics_title += f"\nBatch IoU: {batch_iou:.4f} (all {batch_size} imgs) | Ch: {self.channel}/{cube_img.shape[2]}"
            axes[2].set_title(metrics_title, fontsize=9)
            axes[2].set_xlabel("Width")
            axes[2].set_ylabel("Height")

            log_msg = f"Created anomaly mask artifact ({i + 1}/{up_to_batch}): IoU: {iou:.3f}"
        else:
            # Mode B: Prediction-only visualization (no ground truth)
            # Create figure with 2 subplots
            fig, axes = plt.subplots(1, 2, figsize=(12, 6))

            # Subplot 1: Cube with predicted overlay
            axes[0].imshow(cube_norm, cmap="gray", aspect="auto")

            # Create prediction overlay (cyan for predicted anomalies)
            overlay = np.zeros((*pred.shape, 4))
            overlay[pred] = [0, 1, 1, 0.6]  # Cyan: Predicted anomalies

            axes[0].imshow(overlay, aspect="auto")
            axes[0].set_title(
                f"Prediction Overlay (Channel {self.channel})\nCyan=Predicted Anomalies"
            )
            axes[0].set_xlabel("Width")
            axes[0].set_ylabel("Height")

            # Subplot 2: Predicted mask
            axes[1].imshow(pred, cmap="gray", aspect="auto")
            axes[1].set_title("Predicted Mask")
            axes[1].set_xlabel("Width")
            axes[1].set_ylabel("Height")

            # Add statistics as text
            pred_pixels = pred.sum()
            total_pixels = pred.size
            pred_ratio = pred_pixels / total_pixels

            stats_text = (
                f"Prediction Stats:\n"
                f"Anomaly pixels: {pred_pixels}\n"
                f"Total pixels: {total_pixels}\n"
                f"Anomaly ratio: {pred_ratio:.4f}\n"
                f"\n"
                f"Channel: {self.channel}/{cube_img.shape[2]}\n"
                f"\n"
                f"Mode: Inference/No GT"
            )

            fig.text(
                0.98,
                0.5,
                stats_text,
                ha="left",
                va="center",
                bbox={
                    "boxstyle": "round",
                    "facecolor": "lightblue",
                    "alpha": 0.5,
                },
                fontfamily="monospace",
            )

            log_msg = (
                f"Created anomaly mask artifact ({i + 1}/{up_to_batch}): prediction-only mode"
            )

        # Add main title with epoch/batch info
        fig.suptitle(
            f"Anomaly Mask Visualization - {stage} E{epoch} B{batch_idx} Img{i}",
            fontsize=14,
            fontweight="bold",
        )

        plt.tight_layout()

        # Convert figure to numpy array (RGB format)
        img_array = fig_to_array(fig, dpi=150)

        # Create Artifact object
        artifact = Artifact(
            name=f"anomaly_mask_img{i:02d}",
            stage=context.stage,
            epoch=context.epoch,
            batch_idx=context.batch_idx,
            value=img_array,
            el_id=i,
            desc=f"Anomaly mask for {stage} epoch {epoch}, batch {batch_idx}, image {i}",
            type=ArtifactType.IMAGE,
        )
        artifacts.append(artifact)

        logger.info(log_msg)

        plt.close(fig)

    # Return artifacts
    return {"artifacts": artifacts}
```

`BBoxesOverlayNode`cuvis_ai.node.anomaly_visualizationvisualizerbboxrgbTorch-only bounding-box overlay renderer for YOLO-style detections.

#### BBoxesOverlayNode

```python
BBoxesOverlayNode(
    line_thickness=2,
    draw_labels=False,
    draw_sparklines=False,
    sparkline_height=24,
    hide_untracked=False,
    **kwargs,
)
```

Bases: `Node`

Torch-only bounding-box overlay renderer for YOLO-style detections.

Source code in `cuvis_ai/node/anomaly_visualization.py`

```python
def __init__(
    self,
    line_thickness: int = 2,
    draw_labels: bool = False,
    draw_sparklines: bool = False,
    sparkline_height: int = 24,
    hide_untracked: bool = False,
    **kwargs,
) -> None:
    self.line_thickness = int(line_thickness)
    self.draw_labels = bool(draw_labels)
    self.draw_sparklines = bool(draw_sparklines)
    self.sparkline_height = int(sparkline_height)
    self.hide_untracked = bool(hide_untracked)
    super().__init__(
        line_thickness=line_thickness,
        draw_labels=draw_labels,
        draw_sparklines=draw_sparklines,
        sparkline_height=sparkline_height,
        hide_untracked=hide_untracked,
        **kwargs,
    )
```

##### forward

```python
forward(
    rgb_image,
    bboxes,
    category_ids,
    frame_id=None,
    confidences=None,
    spectral_signatures=None,
    **_,
)
```

Overlay bbox edges with deterministic per-class colors.

Source code in `cuvis_ai/node/anomaly_visualization.py`

```python
@torch.no_grad()
def forward(
    self,
    rgb_image: torch.Tensor,
    bboxes: torch.Tensor,
    category_ids: torch.Tensor,
    frame_id: torch.Tensor | None = None,
    confidences: torch.Tensor | None = None,  # noqa: ARG002
    spectral_signatures: torch.Tensor | None = None,
    **_,
) -> dict[str, torch.Tensor]:
    """Overlay bbox edges with deterministic per-class colors."""
    # Optionally filter out untracked detections (category_id / track_id < 0).
    if self.hide_untracked and category_ids.numel() > 0:
        mask = category_ids[0] >= 0  # [N]
        bboxes = bboxes[:, mask]
        category_ids = category_ids[:, mask]
        if spectral_signatures is not None:
            spectral_signatures = spectral_signatures[:, mask]

    sigs = spectral_signatures if self.draw_sparklines else None
    return {
        "rgb_with_overlay": render_bboxes_overlay_torch(
            rgb_image=rgb_image,
            bboxes=bboxes,
            category_ids=category_ids,
            frame_id=frame_id,
            line_thickness=self.line_thickness,
            draw_labels=self.draw_labels,
            spectral_signatures=sigs,
            sparkline_height=self.sparkline_height,
        )
    }
```

`ChannelSelectorFalseRGBViz`cuvis_ai.node.anomaly_visualizationvisualizerhsirgbVisualize false RGB output from channel selectors with optional mask overlay.

#### ChannelSelectorFalseRGBViz

```python
ChannelSelectorFalseRGBViz(
    mask_overlay_alpha=0.4,
    max_samples=4,
    log_every_n_batches=1,
    **kwargs,
)
```

Bases: `ImageArtifactVizBase`

Visualize false RGB output from channel selectors with optional mask overlay.

Produces per-sample image artifacts:

- `false_rgb_sample_{b}`: Normalized false RGB image [H, W, 3]
- `mask_overlay_sample_{b}`: False RGB with red alpha-blend on foreground pixels (if mask provided)

Parameters:

| Name                  | Type    | Description                                                           | Default |
| --------------------- | ------- | --------------------------------------------------------------------- | ------- |
| `mask_overlay_alpha`  | `float` | Alpha value for red mask overlay on foreground pixels (default: 0.4). | `0.4`   |
| `max_samples`         | `int`   | Maximum number of batch elements to visualize (default: 4).           | `4`     |
| `log_every_n_batches` | `int`   | Log every N-th batch (default: 1).                                    | `1`     |

Source code in `cuvis_ai/node/anomaly_visualization.py`

```python
def __init__(
    self,
    mask_overlay_alpha: float = 0.4,
    max_samples: int = 4,
    log_every_n_batches: int = 1,
    **kwargs,
) -> None:
    self.mask_overlay_alpha = mask_overlay_alpha
    super().__init__(
        max_samples=max_samples,
        log_every_n_batches=log_every_n_batches,
        mask_overlay_alpha=mask_overlay_alpha,
        **kwargs,
    )
```

##### forward

```python
forward(rgb_output, context, mask=None, mesu_index=None)
```

Generate false RGB and mask overlay artifacts.

Parameters:

| Name         | Type      | Description                                     | Default                                                                |
| ------------ | --------- | ----------------------------------------------- | ---------------------------------------------------------------------- |
| `rgb_output` | `Tensor`  | False RGB tensor [B, H, W, 3].                  | *required*                                                             |
| `context`    | `Context` | Execution context with stage, epoch, batch_idx. | *required*                                                             |
| `mask`       | \`Tensor  | None\`                                          | Optional segmentation mask [B, H, W].                                  |
| `mesu_index` | \`Tensor  | None\`                                          | Optional measurement indices [B] for frame-identified artifact naming. |

Returns:

| Type                        | Description                                                 |
| --------------------------- | ----------------------------------------------------------- |
| `dict[str, list[Artifact]]` | Dictionary with "artifacts" key containing image artifacts. |

Source code in `cuvis_ai/node/anomaly_visualization.py`

```python
def forward(
    self,
    rgb_output: torch.Tensor,
    context: Context,
    mask: torch.Tensor | None = None,
    mesu_index: torch.Tensor | None = None,
) -> dict[str, list[Artifact]]:
    """Generate false RGB and mask overlay artifacts.

    Parameters
    ----------
    rgb_output : torch.Tensor
        False RGB tensor [B, H, W, 3].
    context : Context
        Execution context with stage, epoch, batch_idx.
    mask : torch.Tensor | None
        Optional segmentation mask [B, H, W].
    mesu_index : torch.Tensor | None
        Optional measurement indices [B] for frame-identified artifact naming.

    Returns
    -------
    dict[str, list[Artifact]]
        Dictionary with "artifacts" key containing image artifacts.
    """
    if not self._should_log():
        return {"artifacts": []}

    batch_size = min(rgb_output.shape[0], self.max_samples)
    artifacts = []

    for b in range(batch_size):
        # Use mesu_index for naming if available, otherwise fall back to batch index
        frame_id = f"mesu_{mesu_index[b].item()}" if mesu_index is not None else f"sample_{b}"

        # Normalized false RGB
        rgb_np = self._normalize_image(rgb_output[b].detach().cpu().numpy())

        artifact_rgb = Artifact(
            name=f"false_rgb_{frame_id}",
            value=rgb_np,
            el_id=b,
            desc=f"False RGB for {frame_id}",
            type=ArtifactType.IMAGE,
            stage=context.stage,
            epoch=context.epoch,
            batch_idx=context.batch_idx,
        )
        artifacts.append(artifact_rgb)

        # Mask overlay (if mask provided and has foreground)
        if mask is not None and mask[b].any():
            overlay = create_mask_overlay(
                torch.from_numpy(rgb_np), mask[b].cpu(), alpha=self.mask_overlay_alpha
            ).numpy()
            artifact_overlay = Artifact(
                name=f"mask_overlay_{frame_id}",
                value=overlay,
                el_id=b,
                desc=f"False RGB with mask overlay for {frame_id}",
                type=ArtifactType.IMAGE,
                stage=context.stage,
                epoch=context.epoch,
                batch_idx=context.batch_idx,
            )
            artifacts.append(artifact_overlay)

    return {"artifacts": artifacts}
```

`ChannelWeightsViz`cuvis_ai.node.anomaly_visualizationvisualizerhsirgbVisualize channel mixer weights as a heatmap.

#### ChannelWeightsViz

```python
ChannelWeightsViz(
    max_samples=1,
    log_every_n_batches=1,
    cell_height=60,
    cell_width=12,
    **kwargs,
)
```

Bases: `ImageArtifactVizBase`

Visualize channel mixer weights as a heatmap.

Produces a `[K, C]` mixing matrix heatmap with output channels on the y-axis and input channels on the x-axis. Uses a diverging blue-white-red colormap centred at zero so positive/negative contributions are immediately visible.

Implemented in pure PyTorch (no matplotlib) so it adds negligible overhead to the training loop.

Parameters:

| Name                  | Type  | Description                                                                                     | Default |
| --------------------- | ----- | ----------------------------------------------------------------------------------------------- | ------- |
| `max_samples`         | `int` | Ignored (weights are per-model, not per-sample). Kept for base class compatibility. Default: 1. | `1`     |
| `log_every_n_batches` | `int` | Log every N-th batch (default: 1).                                                              | `1`     |
| `cell_height`         | `int` | Pixel height per matrix row (default: 40).                                                      | `60`    |
| `cell_width`          | `int` | Pixel width per matrix column (default: 6).                                                     | `12`    |

Source code in `cuvis_ai/node/anomaly_visualization.py`

```python
def __init__(
    self,
    max_samples: int = 1,
    log_every_n_batches: int = 1,
    cell_height: int = 60,
    cell_width: int = 12,
    **kwargs,
) -> None:
    super().__init__(
        max_samples=max_samples,
        log_every_n_batches=log_every_n_batches,
        **kwargs,
    )
    self.cell_height = cell_height
    self.cell_width = cell_width
```

##### forward

```python
forward(weights, context, wavelengths=None)
```

Generate mixing matrix heatmap artifact.

Pure-torch rendering with R/G/B indicator bars, grid lines, and a diverging colorbar — no matplotlib for training-loop speed.

Parameters:

| Name          | Type      | Description                                      | Default    |
| ------------- | --------- | ------------------------------------------------ | ---------- |
| `weights`     | `Tensor`  | Mixing matrix [K, C].                            | *required* |
| `context`     | `Context` | Execution context with stage, epoch, batch_idx.  | *required* |
| `wavelengths` | `ndarray` | Wavelengths [C] in nm (reserved for future use). | `None`     |

Returns:

| Type                        | Description                      |
| --------------------------- | -------------------------------- |
| `dict[str, list[Artifact]]` | Dictionary with "artifacts" key. |

Source code in `cuvis_ai/node/anomaly_visualization.py`

```python
def forward(
    self,
    weights: torch.Tensor,
    context: Context,
    wavelengths: np.ndarray | None = None,
) -> dict[str, list[Artifact]]:
    """Generate mixing matrix heatmap artifact.

    Pure-torch rendering with R/G/B indicator bars, grid lines, and a
    diverging colorbar — no matplotlib for training-loop speed.

    Parameters
    ----------
    weights : Tensor
        Mixing matrix ``[K, C]``.
    context : Context
        Execution context with stage, epoch, batch_idx.
    wavelengths : ndarray, optional
        Wavelengths ``[C]`` in nm (reserved for future use).

    Returns
    -------
    dict[str, list[Artifact]]
        Dictionary with ``"artifacts"`` key.
    """
    if not self._should_log():
        return {"artifacts": []}

    w = weights.detach().float()
    if w.ndim == 1:
        w = w.unsqueeze(0)  # [1, C]

    K, C = w.shape
    vmax = w.abs().max().clamp_min(1e-8)
    t = (w / vmax + 1.0) * 0.5  # [K, C] in [0, 1], 0.5 = zero

    # Colormap the heatmap → [K, C, 3]
    heatmap = _diverging_colormap(t)

    # Build upscaled heatmap canvas with 1px black grid lines
    ch, cw = self.cell_height, self.cell_width
    grid_h = K * ch + (K + 1)
    grid_w = C * cw + (C + 1)
    canvas = torch.zeros(grid_h, grid_w, 3)  # black grid lines
    for r in range(K):
        y0 = 1 + r * (ch + 1)
        for c in range(C):
            x0 = 1 + c * (cw + 1)
            canvas[y0 : y0 + ch, x0 : x0 + cw] = heatmap[r, c]

    # Left margin: colored R/G/B indicator bars
    indicator_w = max(ch // 3, 8)
    _channel_colors = torch.tensor(
        [
            [1.0, 0.0, 0.0],  # R
            [0.0, 0.7, 0.0],  # G
            [0.0, 0.0, 1.0],  # B
        ]
    )
    indicator = torch.ones(grid_h, indicator_w, 3)  # white background
    for r in range(min(K, len(_channel_colors))):
        y0 = 1 + r * (ch + 1)
        indicator[y0 : y0 + ch, :] = _channel_colors[r]

    # Right margin: colorbar gradient (top=positive/red, bottom=negative/blue)
    cbar_w = max(ch // 3, 8)
    cbar_t = torch.linspace(1.0, 0.0, grid_h).unsqueeze(1)  # [grid_h, 1]
    cbar = _diverging_colormap(cbar_t).expand(grid_h, cbar_w, 3).contiguous()

    # Assemble: [indicator | gap | heatmap | gap | colorbar]
    gap = torch.ones(grid_h, 2, 3)  # 2px white gap
    full = torch.cat([indicator, gap, canvas, gap, cbar], dim=1)

    # Convert to uint8 numpy [H, W, 3]
    heatmap_np = (full * 255).clamp(0, 255).byte().cpu().numpy()

    # --- Text annotations via PIL (fast, no matplotlib) ---
    vmax_val = vmax.item()
    heat_h, heat_w = heatmap_np.shape[:2]
    left_margin, top_margin = 30, 5
    bottom_margin, right_margin = 45, 50

    canvas_img = Image.new(
        "RGB",
        (left_margin + heat_w + right_margin, top_margin + heat_h + bottom_margin),
        (255, 255, 255),
    )
    canvas_img.paste(Image.fromarray(heatmap_np), (left_margin, top_margin))
    draw = ImageDraw.Draw(canvas_img)
    font = ImageFont.load_default()

    # Y-axis: R / G / B labels (centered on each row)
    row_labels = ["R", "G", "B"] if K == 3 else [str(i) for i in range(K)]
    for r in range(K):
        y_center = top_margin + 1 + r * (ch + 1) + ch // 2
        draw.text((6, y_center - 5), row_labels[r], fill=(0, 0, 0), font=font)

    # X-axis: wavelength tick labels (every Nth to avoid overlap)
    # heatmap grid starts after: indicator_w + 2px gap + 1px border
    heatmap_x0 = left_margin + indicator_w + 2 + 1
    if wavelengths is not None and len(wavelengths) == C:
        step = max(1, C // 15)
        for i in range(0, C, step):
            x = heatmap_x0 + i * (cw + 1) + cw // 2
            y = top_margin + heat_h + 2
            label = str(int(wavelengths[i]))
            draw.text((x - len(label) * 3, y), label, fill=(0, 0, 0), font=font)

    # Colorbar scale: +vmax at top, 0 at middle, -vmax at bottom
    cbar_x = left_margin + heat_w + 4
    draw.text((cbar_x, top_margin), f"+{vmax_val:.2f}", fill=(0, 0, 0), font=font)
    draw.text((cbar_x, top_margin + heat_h // 2 - 5), "0", fill=(0, 0, 0), font=font)
    draw.text((cbar_x, top_margin + heat_h - 10), f"-{vmax_val:.2f}", fill=(0, 0, 0), font=font)

    img_np = np.array(canvas_img)

    stage = context.stage.name.lower() if context.stage else "unknown"
    return {
        "artifacts": [
            Artifact(
                name="mixing_matrix",
                value=img_np,
                el_id=0,
                desc=f"Mixing matrix — {stage} epoch {context.epoch}",
                type=ArtifactType.IMAGE,
                stage=context.stage,
                epoch=context.epoch,
                batch_idx=context.batch_idx,
            )
        ],
    }
```

`ClassMapToRGB`cuvis_ai.node.colormapvisualizermaskrgbColourise an integer class-index map `[B, H, W]` into an RGB image `[B, H, W, 3]`.

#### ClassMapToRGB

```python
ClassMapToRGB(palette=None, background_value=-1, **kwargs)
```

Bases: `Node`

Colourise an integer class-index map `[B, H, W]` into an RGB image `[B, H, W, 3]`.

Each class id indexes a palette colour; pixels equal to `background_value` (and, when a `mask` is connected, pixels where `mask == 0`) render black. The explicit `palette` lets it colourise arbitrary integer id-maps (compartment ids, cluster ids, class indices).

Parameters:

| Name               | Type                           | Description                                                                       | Default                                                                                                                                                                  |
| ------------------ | ------------------------------ | --------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `palette`          | \`list\[tuple[int, int, int]\] | None\`                                                                            | Per-id RGB colours in 0-255, indexed by class id. When None, a Tableau-20 palette is cycled. The lookup wraps modulo the palette length, so ids beyond it reuse colours. |
| `background_value` | `int`                          | Class id rendered black (default -1, so id 0 stays a valid class for clustering). | `-1`                                                                                                                                                                     |

Source code in `cuvis_ai/node/colormap.py`

```python
def __init__(
    self,
    palette: list[tuple[int, int, int]] | None = None,
    background_value: int = -1,
    **kwargs: Any,
) -> None:
    colors = list(palette) if palette is not None else list(_TAB20)
    if not colors:
        raise ValueError("palette must be a non-empty list of (r, g, b)")
    self.background_value = int(background_value)
    super().__init__(
        palette=[[int(c) for c in rgb] for rgb in colors],
        background_value=self.background_value,
        **kwargs,
    )
    self._lut = torch.tensor(
        [[c / 255.0 for c in rgb] for rgb in colors], dtype=torch.float32
    )  # [P, 3] in [0, 1]
```

##### forward

```python
forward(class_map, mask=None, **_)
```

Look the palette up per pixel; background and masked-out pixels stay black.

Source code in `cuvis_ai/node/colormap.py`

```python
@torch.no_grad()
def forward(self, class_map: Tensor, mask: Tensor | None = None, **_: Any) -> dict[str, Tensor]:
    """Look the palette up per pixel; background and masked-out pixels stay black."""
    lut = self._lut.to(class_map.device)
    idx = class_map.clamp(min=0) % lut.shape[0]
    rgb = lut[idx]  # [B, H, W, 3]
    valid = class_map != self.background_value
    if mask is not None:
        valid = valid & (mask.to(class_map.device) != 0)
    return {"label_rgb": torch.where(valid.unsqueeze(-1), rgb, torch.zeros_like(rgb))}
```

`CubeRGBVisualizer`cuvis_ai.node.pipeline_visualizationvisualizerhsirgbCreates false-color RGB images from hyperspectral cube using channel weights.

#### CubeRGBVisualizer

```python
CubeRGBVisualizer(name=None, up_to=5)
```

Bases: `Node`

Creates false-color RGB images from hyperspectral cube using channel weights.

Selects 3 channels with highest weights for R, G, B channels and creates a false-color visualization with wavelength annotations.

Source code in `cuvis_ai/node/pipeline_visualization.py`

```python
def __init__(self, name: str | None = None, up_to: int = 5) -> None:
    super().__init__(name=name, execution_stages={ExecutionStage.INFERENCE, ExecutionStage.VAL})
    self.up_to = up_to
```

##### forward

```python
forward(cube, weights, wavelengths, context)
```

Generate false-color RGB visualizations from hyperspectral cube.

Selects the 3 channels with highest weights and creates RGB images with wavelength annotations. Also generates a bar chart showing channel weights with the selected channels highlighted.

Parameters:

| Name          | Type      | Description                                                          | Default    |
| ------------- | --------- | -------------------------------------------------------------------- | ---------- |
| `cube`        | `Tensor`  | Hyperspectral cube [B, H, W, C].                                     | *required* |
| `weights`     | `Tensor`  | Channel selection weights [C] indicating importance of each channel. | *required* |
| `wavelengths` | `Tensor`  | Wavelengths for each channel [C] in nanometers.                      | *required* |
| `context`     | `Context` | Execution context with stage, epoch, batch_idx information.          | *required* |

Returns:

| Type                        | Description                                                                 |
| --------------------------- | --------------------------------------------------------------------------- |
| `dict[str, list[Artifact]]` | Dictionary with "artifacts" key containing list of visualization artifacts. |

Source code in `cuvis_ai/node/pipeline_visualization.py`

```python
def forward(self, cube, weights, wavelengths, context) -> dict[str, list[Artifact]]:
    """Generate false-color RGB visualizations from hyperspectral cube.

    Selects the 3 channels with highest weights and creates RGB images
    with wavelength annotations. Also generates a bar chart showing
    channel weights with the selected channels highlighted.

    Parameters
    ----------
    cube : Tensor
        Hyperspectral cube [B, H, W, C].
    weights : Tensor
        Channel selection weights [C] indicating importance of each channel.
    wavelengths : Tensor
        Wavelengths for each channel [C] in nanometers.
    context : Context
        Execution context with stage, epoch, batch_idx information.

    Returns
    -------
    dict[str, list[Artifact]]
        Dictionary with "artifacts" key containing list of visualization artifacts.
    """
    top3_indices = torch.topk(weights, k=3).indices.cpu().numpy()
    top3_wavelengths = wavelengths[top3_indices]

    batch_size = min(cube.shape[0], self.up_to)
    artifacts = []

    for b in range(batch_size):
        rgb_channels = cube[b, :, :, top3_indices].cpu().numpy()

        rgb_img = (rgb_channels - rgb_channels.min()) / (
            rgb_channels.max() - rgb_channels.min() + 1e-8
        )

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        ax1.imshow(rgb_img)
        ax1.set_title(
            f"False RGB: R={top3_wavelengths[0]:.1f}nm, "
            f"G={top3_wavelengths[1]:.1f}nm, B={top3_wavelengths[2]:.1f}nm"
        )
        ax1.axis("off")

        ax2.bar(range(len(wavelengths)), weights.detach().cpu().numpy())
        ax2.scatter(
            top3_indices,
            weights[top3_indices].detach().cpu().numpy(),
            c="red",
            s=100,
            zorder=3,
        )
        ax2.set_xlabel("Channel Index")
        ax2.set_ylabel("Weight")
        ax2.set_title("Channel Selection Weights")
        ax2.grid(True, alpha=0.3)

        for idx in top3_indices:
            ax2.annotate(
                f"{wavelengths[idx]:.0f}nm",
                xy=(idx, weights[idx].item()),
                xytext=(0, 10),
                textcoords="offset points",
                ha="center",
                fontsize=8,
            )

        plt.tight_layout()

        img_array = fig_to_array(fig, dpi=150)

        artifact = Artifact(
            name=f"viz_rgb_sample_{b}",
            value=img_array,
            el_id=b,
            desc=f"False RGB visualization for sample {b}",
            type=ArtifactType.IMAGE,
        )
        artifacts.append(artifact)
        plt.close(fig)

    return {"artifacts": artifacts}
```

`ImageArtifactVizBase`cuvis_ai.node.anomaly_visualizationvisualizeranomimgBase class for visualization nodes that produce image artifacts.

#### ImageArtifactVizBase

```python
ImageArtifactVizBase(
    max_samples=4,
    log_every_n_batches=1,
    execution_stages=None,
    **kwargs,
)
```

Bases: `Node`

Base class for visualization nodes that produce image artifacts.

Source code in `cuvis_ai/node/anomaly_visualization.py`

```python
def __init__(
    self,
    max_samples: int = 4,
    log_every_n_batches: int = 1,
    execution_stages: set[ExecutionStage] | None = None,
    **kwargs,
) -> None:
    self.max_samples = max_samples
    self.log_every_n_batches = log_every_n_batches
    self._batch_counter = 0
    if execution_stages is None:
        execution_stages = {ExecutionStage.TRAIN, ExecutionStage.VAL, ExecutionStage.TEST}
    super().__init__(
        execution_stages=execution_stages,
        **kwargs,
    )
```

`MaskOverlayNode`cuvis_ai.node.anomaly_visualizationvisualizermaskrgbAlpha-blend a coloured mask overlay onto RGB frames.

#### MaskOverlayNode

```python
MaskOverlayNode(
    alpha=0.4, overlay_color=(1.0, 0.0, 0.0), **kwargs
)
```

Bases: `Node`

Alpha-blend a coloured mask overlay onto RGB frames.

Pure PyTorch processing node (no matplotlib, no gradients). When *mask* is `None` or entirely zero the input RGB is passed through unchanged.

Parameters:

| Name            | Type                         | Description                                            | Default           |
| --------------- | ---------------------------- | ------------------------------------------------------ | ----------------- |
| `alpha`         | `float`                      | Blend factor for the overlay colour (default: 0.4).    | `0.4`             |
| `overlay_color` | `tuple[float, float, float]` | RGB overlay colour in [0, 1] (default: red (1, 0, 0)). | `(1.0, 0.0, 0.0)` |

Source code in `cuvis_ai/node/anomaly_visualization.py`

```python
def __init__(
    self,
    alpha: float = 0.4,
    overlay_color: Sequence[float] = (1.0, 0.0, 0.0),
    **kwargs,
) -> None:
    if len(overlay_color) != 3:
        raise ValueError(
            f"overlay_color must contain exactly 3 channels (R, G, B), got {overlay_color}"
        )

    parsed_overlay_color = tuple(float(channel) for channel in overlay_color)
    if any(channel < 0.0 or channel > 1.0 for channel in parsed_overlay_color):
        raise ValueError(
            f"overlay_color channels must be within [0, 1], got {parsed_overlay_color}"
        )

    self.overlay_color = parsed_overlay_color
    self.alpha = alpha
    super().__init__(alpha=alpha, overlay_color=self.overlay_color, **kwargs)
```

##### forward

```python
forward(rgb_image, mask=None, **_)
```

Apply mask overlay to RGB frames.

Source code in `cuvis_ai/node/anomaly_visualization.py`

```python
@torch.no_grad()
def forward(
    self,
    rgb_image: torch.Tensor,
    mask: torch.Tensor | None = None,
    **_,
) -> dict[str, torch.Tensor]:
    """Apply mask overlay to RGB frames."""
    if mask is None or not mask.any():
        return {"rgb_with_overlay": rgb_image}
    return {
        "rgb_with_overlay": create_mask_overlay(
            rgb_image, mask, alpha=self.alpha, color=self.overlay_color
        )
    }
```

`PCAVisualization`cuvis_ai.node.pipeline_visualizationvisualizerhsirgbVisualize PCA-projected data with scatter and image plots.

#### PCAVisualization

```python
PCAVisualization(up_to=None, **kwargs)
```

Bases: `Node`

Visualize PCA-projected data with scatter and image plots.

Creates visualizations for each batch element showing:

1. Scatter plot of H\*W points in 2D PC space (using first 2 PCs)
1. Image representation of the 2D projection reshaped to [H, W, 2]

Points in scatter plot are colored by spatial position. Returns artifacts for monitoring systems.

Executes only during validation stage.

Parameters:

| Name    | Type  | Description                                                                            | Default |
| ------- | ----- | -------------------------------------------------------------------------------------- | ------- |
| `up_to` | `int` | Maximum number of batch elements to visualize. If None, visualizes all (default: None) | `None`  |

Examples:

```pycon
>>> pca_viz = PCAVisualization(up_to=10)
>>> tensorboard_node = TensorBoardMonitorNode(output_dir="./runs")
>>> graph.connect(
...     (pca.projected, pca_viz.data),
...     (pca_viz.artifacts, tensorboard_node.artifacts),
... )
```

Source code in `cuvis_ai/node/pipeline_visualization.py`

```python
def __init__(self, up_to: int | None = None, **kwargs) -> None:
    self.up_to = up_to

    super().__init__(execution_stages={ExecutionStage.VAL}, up_to=up_to, **kwargs)
```

##### forward

```python
forward(data, context)
```

Create PCA projection visualizations as Artifact objects.

Parameters:

| Name      | Type      | Description                                                      | Default    |
| --------- | --------- | ---------------------------------------------------------------- | ---------- |
| `data`    | `Tensor`  | PCA-projected data tensor [B, H, W, C] (uses first 2 components) | *required* |
| `context` | `Context` | Execution context with stage, epoch, batch_idx                   | *required* |

Returns:

| Type   | Description                                                         |
| ------ | ------------------------------------------------------------------- |
| `dict` | Dictionary with "artifacts" key containing list of Artifact objects |

Source code in `cuvis_ai/node/pipeline_visualization.py`

```python
def forward(self, data: torch.Tensor, context: Context) -> dict:
    """Create PCA projection visualizations as Artifact objects.

    Parameters
    ----------
    data : torch.Tensor
        PCA-projected data tensor [B, H, W, C] (uses first 2 components)
    context : Context
        Execution context with stage, epoch, batch_idx

    Returns
    -------
    dict
        Dictionary with "artifacts" key containing list of Artifact objects
    """
    # Convert to numpy
    data_np = data.detach().cpu().numpy()

    # Handle input shape: [B, H, W, C]
    if data_np.ndim != 4:
        raise ValueError(f"Expected 4D input [B, H, W, C], got shape: {data_np.shape}")

    B, H, W, C = data_np.shape

    if C < 2:
        raise ValueError(f"Expected at least 2 components, got {C}")

    # Extract context information
    stage = context.stage.value
    epoch = context.epoch
    batch_idx = context.batch_idx

    # Determine how many images to visualize from this batch
    up_to_batch = B if self.up_to is None else min(B, self.up_to)

    # List to collect artifacts
    artifacts = []

    # Loop through each batch element
    for i in range(up_to_batch):
        # Get projection for this batch element: [H, W, C]
        projection = data_np[i]

        # Use only first 2 components
        projection_2d = projection[:, :, :2]  # [H, W, 2]

        # Flatten spatial dimensions for scatter plot
        projection_flat = projection_2d.reshape(-1, 2)  # [H*W, 2]

        # Create spatial position colors using 2D HSV encoding
        # x-coordinate maps to Hue (0-1)
        # y-coordinate maps to Saturation (0-1)
        # Value is constant at 1.0 for brightness
        y_coords, x_coords = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")

        # Normalize coordinates to [0, 1]
        x_norm = x_coords / (W - 1) if W > 1 else np.zeros_like(x_coords)
        y_norm = y_coords / (H - 1) if H > 1 else np.zeros_like(y_coords)

        # Create HSV colors: H from x, S from y, V constant
        hsv_colors = np.stack(
            [
                x_norm.flatten(),  # Hue from x-coordinate
                y_norm.flatten(),  # Saturation from y-coordinate
                np.ones(H * W),  # Value constant at 1.0
            ],
            axis=-1,
        )

        # Convert HSV to RGB for matplotlib
        from matplotlib.colors import hsv_to_rgb

        rgb_colors = hsv_to_rgb(hsv_colors)

        # Create figure with 3 subplots
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))

        # Subplot 1: Scatter plot colored by 2D spatial position
        axes[0].scatter(
            projection_flat[:, 0],
            projection_flat[:, 1],
            c=rgb_colors,
            alpha=0.6,
            s=20,
        )
        axes[0].set_xlabel("PC1 (1st component)")
        axes[0].set_ylabel("PC2 (2nd component)")
        axes[0].set_title(f"PCA Scatter - {stage} E{epoch} B{batch_idx} Img{i}")
        axes[0].grid(True, alpha=0.3)

        # Subplot 2: Spatial reference image
        # Create reference image showing the spatial color coding
        spatial_reference = hsv_to_rgb(
            np.stack([x_norm, y_norm, np.ones_like(x_norm)], axis=-1)
        )
        axes[1].imshow(spatial_reference, aspect="auto")
        axes[1].set_xlabel("Width (→ Hue)")
        axes[1].set_ylabel("Height (→ Saturation)")
        axes[1].set_title("Spatial Color Reference")

        # Subplot 3: Image representation
        # Normalize each channel to [0, 1] for visualization
        pc1_norm = (projection_2d[:, :, 0] - projection_2d[:, :, 0].min()) / (
            projection_2d[:, :, 0].max() - projection_2d[:, :, 0].min() + 1e-8
        )
        pc2_norm = (projection_2d[:, :, 1] - projection_2d[:, :, 1].min()) / (
            projection_2d[:, :, 1].max() - projection_2d[:, :, 1].min() + 1e-8
        )

        # Create RGB image: PC1 in red channel, PC2 in green channel, zeros in blue
        img_rgb = np.stack([pc1_norm, pc2_norm, np.zeros_like(pc1_norm)], axis=-1)

        axes[2].imshow(img_rgb, aspect="auto")
        axes[2].set_xlabel("Width")
        axes[2].set_ylabel("Height")
        axes[2].set_title("PCA Image (R=PC1, G=PC2)")

        # Add statistics text
        pc1_min = projection_2d[:, :, 0].min()
        pc1_max = projection_2d[:, :, 0].max()
        pc2_min = projection_2d[:, :, 1].min()
        pc2_max = projection_2d[:, :, 1].max()
        stats_text = (
            f"Shape: [{H}, {W}]\n"
            f"Points: {H * W}\n"
            f"PC1 range: [{pc1_min:.3f}, {pc1_max:.3f}]\n"
            f"PC2 range: [{pc2_min:.3f}, {pc2_max:.3f}]"
        )
        fig.text(
            0.98,
            0.5,
            stats_text,
            ha="left",
            va="center",
            bbox={
                "boxstyle": "round",
                "facecolor": "wheat",
                "alpha": 0.5,
            },
        )

        plt.tight_layout()

        # Convert figure to numpy array (RGB format)
        img_array = fig_to_array(fig, dpi=150)

        # Create Artifact object
        artifact = Artifact(
            name=f"pca_projection_img{i:02d}",
            stage=context.stage,
            epoch=context.epoch,
            batch_idx=context.batch_idx,
            value=img_array,
            el_id=i,
            desc=f"PCA projection for {stage} epoch {epoch}, batch {batch_idx}, image {i}",
            type=ArtifactType.IMAGE,
        )
        artifacts.append(artifact)

        progress_total = self.up_to if self.up_to else B
        description = (
            f"Created PCA projection artifact ({i + 1}/{progress_total}): {artifact.name}"
        )
        logger.info(description)

        plt.close(fig)

    # Return artifacts
    return {"artifacts": artifacts}
```

`PipelineComparisonVisualizer`cuvis_ai.node.pipeline_visualizationvisualizerhsirgbTensorBoard visualization node for comparing pipeline stages.

#### PipelineComparisonVisualizer

```python
PipelineComparisonVisualizer(
    hsi_channels=None,
    max_samples=4,
    log_every_n_batches=1,
    **kwargs,
)
```

Bases: `Node`

TensorBoard visualization node for comparing pipeline stages.

Creates image artifacts for logging to TensorBoard:

- Input HSI cube visualization (false-color RGB from selected channels)
- Mixer output (3-channel RGB-like image that downstream model sees)
- Ground truth anomaly mask
- Anomaly scores (as heatmap)

Parameters:

| Name                  | Type        | Description                                                                                                                          | Default |
| --------------------- | ----------- | ------------------------------------------------------------------------------------------------------------------------------------ | ------- |
| `hsi_channels`        | `list[int]` | Channel indices to use for false-color RGB visualization of HSI input (default: [0, 20, 40] for a simple false-color representation) | `None`  |
| `max_samples`         | `int`       | Maximum number of samples to log per batch (default: 4)                                                                              | `4`     |
| `log_every_n_batches` | `int`       | Log images every N batches to reduce TensorBoard size (default: 1, log every batch)                                                  | `1`     |

Source code in `cuvis_ai/node/pipeline_visualization.py`

```python
def __init__(
    self,
    hsi_channels: list[int] | None = None,
    max_samples: int = 4,
    log_every_n_batches: int = 1,
    **kwargs,
) -> None:
    if hsi_channels is None:
        hsi_channels = [0, 20, 40]  # Default: use channels 0, 20, 40 for false-color RGB
    self.hsi_channels = hsi_channels
    self.max_samples = max_samples
    self.log_every_n_batches = log_every_n_batches
    self._batch_counter = 0

    super().__init__(
        execution_stages={ExecutionStage.TRAIN, ExecutionStage.VAL, ExecutionStage.TEST},
        hsi_channels=hsi_channels,
        max_samples=max_samples,
        log_every_n_batches=log_every_n_batches,
        **kwargs,
    )
```

##### forward

```python
forward(
    hsi_cube,
    mixer_output,
    ground_truth_mask,
    anomaly_scores,
    context=None,
    **_,
)
```

Create image artifacts for TensorBoard logging.

Parameters:

| Name                | Type      | Description                                         | Default    |
| ------------------- | --------- | --------------------------------------------------- | ---------- |
| `hsi_cube`          | `Tensor`  | Input HSI cube [B, H, W, C]                         | *required* |
| `mixer_output`      | `Tensor`  | Mixer output (RGB-like) [B, H, W, 3]                | *required* |
| `ground_truth_mask` | `Tensor`  | Ground truth anomaly mask [B, H, W, 1]              | *required* |
| `anomaly_scores`    | `Tensor`  | Anomaly scores [B, H, W, 1]                         | *required* |
| `context`           | `Context` | Execution context with stage, epoch, batch_idx info | `None`     |

Returns:

| Type                        | Description                                                         |
| --------------------------- | ------------------------------------------------------------------- |
| `dict[str, list[Artifact]]` | Dictionary with "artifacts" key containing list of Artifact objects |

Source code in `cuvis_ai/node/pipeline_visualization.py`

```python
def forward(
    self,
    hsi_cube: Tensor,
    mixer_output: Tensor,
    ground_truth_mask: Tensor,
    anomaly_scores: Tensor,
    context: Context | None = None,
    **_: Any,
) -> dict[str, list[Artifact]]:
    """Create image artifacts for TensorBoard logging.

    Parameters
    ----------
    hsi_cube : Tensor
        Input HSI cube [B, H, W, C]
    mixer_output : Tensor
        Mixer output (RGB-like) [B, H, W, 3]
    ground_truth_mask : Tensor
        Ground truth anomaly mask [B, H, W, 1]
    anomaly_scores : Tensor
        Anomaly scores [B, H, W, 1]
    context : Context, optional
        Execution context with stage, epoch, batch_idx info

    Returns
    -------
    dict[str, list[Artifact]]
        Dictionary with "artifacts" key containing list of Artifact objects
    """
    if context is None:
        context = Context()

    # Skip logging if not the right batch interval
    self._batch_counter += 1
    if (self._batch_counter - 1) % self.log_every_n_batches != 0:
        return {"artifacts": []}

    artifacts = []
    B = hsi_cube.shape[0]
    num_samples = min(B, self.max_samples)

    # Convert tensors to numpy for visualization
    hsi_np = hsi_cube.detach().cpu().numpy()
    mixer_np = mixer_output.detach().cpu().numpy()
    mask_np = ground_truth_mask.detach().cpu().numpy()
    scores_np = anomaly_scores.detach().cpu().numpy()

    for b in range(num_samples):
        # 1. HSI Input Visualization (false-color RGB)
        hsi_img = self._create_hsi_visualization(hsi_np[b])
        artifact = Artifact(
            name=f"hsi_input_sample_{b}",
            value=hsi_img,
            el_id=b,
            desc=f"HSI input (false-color RGB) for sample {b}",
            type=ArtifactType.IMAGE,
            stage=context.stage,
            epoch=context.epoch,
            batch_idx=context.batch_idx,
        )
        artifacts.append(artifact)

        # 2. Mixer Output (what downstream model sees as input)
        mixer_img = self._normalize_image(mixer_np[b])  # Already [H, W, 3]
        artifact = Artifact(
            name=f"mixer_output_adaclip_input_sample_{b}",
            value=mixer_img,
            el_id=b,
            desc=f"Mixer output (model input) for sample {b}",
            type=ArtifactType.IMAGE,
            stage=context.stage,
            epoch=context.epoch,
            batch_idx=context.batch_idx,
        )
        artifacts.append(artifact)

        # 3. Ground Truth Mask
        mask_img = self._create_mask_visualization(mask_np[b])  # [H, W, 1] -> [H, W, 3]
        artifact = Artifact(
            name=f"ground_truth_mask_sample_{b}",
            value=mask_img,
            el_id=b,
            desc=f"Ground truth anomaly mask for sample {b}",
            type=ArtifactType.IMAGE,
            stage=context.stage,
            epoch=context.epoch,
            batch_idx=context.batch_idx,
        )
        artifacts.append(artifact)

        # 4. Anomaly Scores (as heatmap)
        scores_img = self._create_scores_heatmap(scores_np[b])  # [H, W, 1] -> [H, W, 3]
        artifact = Artifact(
            name=f"anomaly_scores_heatmap_sample_{b}",
            value=scores_img,
            el_id=b,
            desc=f"Anomaly scores (heatmap) for sample {b}",
            type=ArtifactType.IMAGE,
            stage=context.stage,
            epoch=context.epoch,
            batch_idx=context.batch_idx,
        )
        artifacts.append(artifact)

    return {"artifacts": artifacts}
```

`RGBAnomalyMask`cuvis_ai.node.anomaly_visualizationvisualizeranommaskrgbVisualize anomaly detection with GT and predicted masks on RGB images.

#### RGBAnomalyMask

```python
RGBAnomalyMask(up_to=None, **kwargs)
```

Bases: `Node`

Visualize anomaly detection with GT and predicted masks on RGB images.

Similar to AnomalyMask but designed for RGB images (e.g., from band selectors). Creates side-by-side visualizations showing ground truth masks, predicted masks, and overlay comparisons on RGB images. The overlay shows:

- Green: True Positives (correct anomaly detection)
- Red: False Positives (false alarms)
- Yellow: False Negatives (missed anomalies)

Also displays IoU and other metrics. Returns a list of Artifact objects for logging to monitoring systems.

Executes during validation and inference stages.

Parameters:

| Name    | Type  | Description                                                                    | Default |
| ------- | ----- | ------------------------------------------------------------------------------ | ------- |
| `up_to` | `int` | Maximum number of images to visualize. If None, visualizes all (default: None) | `None`  |

Examples:

```pycon
>>> decider = BinaryDecider(threshold=0.2)
>>> viz_mask = RGBAnomalyMask(up_to=5)
>>> tensorboard_node = TensorBoardMonitorNode(output_dir="./runs")
>>> graph.connect(
...     (decider.decisions, viz_mask.decisions),
...     (data_node.mask, viz_mask.mask),
...     (band_selector.rgb_image, viz_mask.rgb_image),
...     (viz_mask.artifacts, tensorboard_node.artifacts),
... )
```

Initialize RGBAnomalyMask visualizer.

Parameters:

| Name    | Type  | Description | Default                                                                        |
| ------- | ----- | ----------- | ------------------------------------------------------------------------------ |
| `up_to` | \`int | None\`      | Maximum number of images to visualize. If None, visualizes all (default: None) |

Source code in `cuvis_ai/node/anomaly_visualization.py`

```python
def __init__(self, up_to: int | None = None, **kwargs) -> None:
    """Initialize RGBAnomalyMask visualizer.

    Parameters
    ----------
    up_to : int | None, optional
        Maximum number of images to visualize. If None, visualizes all (default: None)
    """
    self.up_to = up_to
    super().__init__(
        execution_stages={ExecutionStage.VAL, ExecutionStage.TEST, ExecutionStage.INFERENCE},
        up_to=up_to,
        **kwargs,
    )
```

##### forward

```python
forward(
    decisions,
    rgb_image,
    mask=None,
    context=None,
    scores=None,
)
```

Create anomaly mask visualizations with GT/pred comparison on RGB images.

Parameters:

| Name        | Type      | Description                              | Default                                           |
| ----------- | --------- | ---------------------------------------- | ------------------------------------------------- |
| `decisions` | `Tensor`  | Binary anomaly decisions [B, H, W, 1]    | *required*                                        |
| `rgb_image` | `Tensor`  | RGB image [B, H, W, 3] for visualization | *required*                                        |
| `mask`      | \`Tensor  | None\`                                   | Ground truth anomaly mask [B, H, W, 1] (optional) |
| `context`   | \`Context | None\`                                   | Execution context with stage, epoch, batch_idx    |
| `scores`    | \`Tensor  | None\`                                   | Optional anomaly logits/scores [B, H, W, 1]       |

Returns:

| Type   | Description                                                         |
| ------ | ------------------------------------------------------------------- |
| `dict` | Dictionary with "artifacts" key containing list of Artifact objects |

Source code in `cuvis_ai/node/anomaly_visualization.py`

```python
def forward(
    self,
    decisions: torch.Tensor,
    rgb_image: torch.Tensor,
    mask: torch.Tensor | None = None,
    context: Context | None = None,
    scores: torch.Tensor | None = None,
) -> dict:
    """Create anomaly mask visualizations with GT/pred comparison on RGB images.

    Parameters
    ----------
    decisions : torch.Tensor
        Binary anomaly decisions [B, H, W, 1]
    rgb_image : torch.Tensor
        RGB image [B, H, W, 3] for visualization
    mask : torch.Tensor | None
        Ground truth anomaly mask [B, H, W, 1] (optional)
    context : Context | None
        Execution context with stage, epoch, batch_idx
    scores : torch.Tensor | None
        Optional anomaly logits/scores [B, H, W, 1]

    Returns
    -------
    dict
        Dictionary with "artifacts" key containing list of Artifact objects
    """
    if context is None:
        raise ValueError("RGBAnomalyMask.forward() requires a Context object")

    # Convert to numpy only at this point (keep on device until last moment)
    pred_mask_np: np.ndarray = tensor_to_numpy(decisions.float().squeeze(-1))  # [B, H, W]
    rgb_np: np.ndarray = tensor_to_numpy(rgb_image)  # [B, H, W, 3]

    # Normalize RGB to [0, 1]
    if rgb_np.max() > 1.0:
        rgb_np = rgb_np / 255.0
    rgb_np = np.clip(rgb_np, 0.0, 1.0)

    # Check if GT available and valid
    use_gt = (
        mask is not None and context.stage != ExecutionStage.INFERENCE and mask.any().item()
    )

    # Validate and convert GT if available
    gt_mask_np: np.ndarray | None = None
    batch_iou: float | None = None
    if use_gt:
        assert mask is not None
        gt_mask_np = tensor_to_numpy(mask.squeeze(-1))  # [B, H, W]
        unique_values = np.unique(gt_mask_np)
        if not np.all(np.isin(unique_values, [0, 1, True, False])):
            raise ValueError(f"RGBAnomalyMask expects binary masks, found: {unique_values}")
        # Compute batch IoU
        batch_pred = pred_mask_np > 0.5
        batch_gt = gt_mask_np > 0.5
        tp = np.logical_and(batch_pred, batch_gt).sum()
        fp = np.logical_and(batch_pred, ~batch_gt).sum()
        fn = np.logical_and(~batch_pred, batch_gt).sum()
        batch_iou = float(tp / (tp + fp + fn + 1e-8))

    batch_size = pred_mask_np.shape[0]
    up_to_batch = min(batch_size, self.up_to or batch_size)
    artifacts = []

    # Loop through images and visualize
    for i in range(up_to_batch):
        pred = pred_mask_np[i] > 0.5
        rgb_img = rgb_np[i]
        gt = gt_mask_np[i] > 0.5 if gt_mask_np is not None else None

        # Compute metrics and AP if GT available
        metrics: dict | None = None
        per_image_ap: float | None = None
        if gt is not None:
            metrics = self._compute_metrics(pred, gt)
            if scores is not None and mask is not None:
                raw_scores = scores[i, ..., 0]
                probs = torch.sigmoid(raw_scores).flatten()
                target_tensor = mask[i, ..., 0].flatten().to(dtype=torch.long)
                if probs.numel() == target_tensor.numel():
                    per_image_ap = binary_average_precision(probs, target_tensor).item()

        # Create figure and plot
        ncols = 3 if gt is not None else 2
        fig, axes = plt.subplots(1, ncols, figsize=(6 * ncols, 6))
        if ncols == 1:
            axes = [axes]

        if gt is not None and metrics is not None and batch_iou is not None:
            self._plot_with_gt(
                axes, rgb_img, pred, gt, metrics, batch_iou, batch_size, per_image_ap
            )
            log_msg = (
                f"Created RGB anomaly mask ({i + 1}/{up_to_batch}): IoU={metrics['iou']:.3f}"
            )
        else:
            self._plot_no_gt(axes, rgb_img, pred)
            log_msg = f"Created RGB anomaly mask ({i + 1}/{up_to_batch}) (no GT)"

        plt.tight_layout()
        img_array = fig_to_array(fig, dpi=150)
        plt.close(fig)

        artifact = Artifact(
            name=f"rgb_anomaly_mask_img{i:02d}",
            value=img_array,
            el_id=i,
            desc=log_msg,
            type=ArtifactType.IMAGE,
            stage=context.stage,
            epoch=context.epoch,
            batch_idx=context.batch_idx,
        )
        artifacts.append(artifact)

    return {"artifacts": artifacts}
```

`ScalarHSVColormapNode`cuvis_ai.node.colormapvisualizerrgbMap a scalar BHWC image to RGB using an HSV colormap.

#### ScalarHSVColormapNode

```python
ScalarHSVColormapNode(
    value_min=0.0, value_max=1.0, **kwargs
)
```

Bases: `Node`

Map a scalar BHWC image to RGB using an HSV colormap.

Source code in `cuvis_ai/node/colormap.py`

```python
def __init__(self, value_min: float = 0.0, value_max: float = 1.0, **kwargs: Any) -> None:
    if value_max <= value_min:
        raise ValueError("value_max must be greater than value_min")
    super().__init__(value_min=float(value_min), value_max=float(value_max), **kwargs)
    self.value_min = float(value_min)
    self.value_max = float(value_max)
    self._value_range = self.value_max - self.value_min
```

##### forward

```python
forward(data, **_)
```

Colorize a scalar image in BHWC format.

Source code in `cuvis_ai/node/colormap.py`

```python
def forward(self, data: Tensor, **_: Any) -> dict[str, Tensor]:
    """Colorize a scalar image in BHWC format."""
    if data.ndim != 4 or data.shape[-1] != 1:
        raise ValueError(
            f"Expected scalar data with shape [B, H, W, 1], got {tuple(data.shape)}"
        )
    normalized = ((data - self.value_min) / self._value_range).clamp(0.0, 1.0)
    return {"rgb_image": render_scalar_hsv_colormap(normalized)}
```

`ScoreHeatmapVisualizer`cuvis_ai.node.anomaly_visualizationvisualizeranomrgbLog LAD/RX score heatmaps as TensorBoard artifacts.

#### ScoreHeatmapVisualizer

```python
ScoreHeatmapVisualizer(
    normalize_scores=True, cmap="inferno", up_to=5, **kwargs
)
```

Bases: `Node`

Log LAD/RX score heatmaps as TensorBoard artifacts.

Source code in `cuvis_ai/node/anomaly_visualization.py`

```python
def __init__(
    self,
    normalize_scores: bool = True,
    cmap: str = "inferno",
    up_to: int | None = 5,
    **kwargs,
) -> None:
    self.normalize_scores = normalize_scores
    self.cmap = cmap
    self.up_to = up_to
    super().__init__(
        execution_stages={ExecutionStage.VAL, ExecutionStage.TEST, ExecutionStage.INFERENCE},
        normalize_scores=normalize_scores,
        cmap=cmap,
        up_to=up_to,
        **kwargs,
    )
```

##### forward

```python
forward(scores, context)
```

Generate heatmap visualizations of anomaly scores.

Creates color-mapped heatmaps of anomaly scores for visualization in TensorBoard. Optionally normalizes scores to [0, 1] range for consistent visualization across batches.

Parameters:

| Name      | Type      | Description                                                       | Default    |
| --------- | --------- | ----------------------------------------------------------------- | ---------- |
| `scores`  | `Tensor`  | Anomaly scores [B, H, W, 1] from detection nodes (e.g., RX, LAD). | *required* |
| `context` | `Context` | Execution context with stage, epoch, batch_idx information.       | *required* |

Returns:

| Type                        | Description                                                           |
| --------------------------- | --------------------------------------------------------------------- |
| `dict[str, list[Artifact]]` | Dictionary with "artifacts" key containing list of heatmap artifacts. |

Source code in `cuvis_ai/node/anomaly_visualization.py`

```python
def forward(self, scores: torch.Tensor, context: Context) -> dict[str, list[Artifact]]:
    """Generate heatmap visualizations of anomaly scores.

    Creates color-mapped heatmaps of anomaly scores for visualization
    in TensorBoard. Optionally normalizes scores to [0, 1] range for
    consistent visualization across batches.

    Parameters
    ----------
    scores : Tensor
        Anomaly scores [B, H, W, 1] from detection nodes (e.g., RX, LAD).
    context : Context
        Execution context with stage, epoch, batch_idx information.

    Returns
    -------
    dict[str, list[Artifact]]
        Dictionary with "artifacts" key containing list of heatmap artifacts.
    """
    artifacts: list[Artifact] = []
    batch_limit = scores.shape[0] if self.up_to is None else min(scores.shape[0], self.up_to)

    for idx in range(batch_limit):
        score_map = scores[idx, ..., 0].detach().cpu().numpy()

        if self.normalize_scores:
            min_v = float(score_map.min())
            max_v = float(score_map.max())
            if max_v - min_v > 1e-9:
                score_map = (score_map - min_v) / (max_v - min_v)
            else:
                score_map = np.zeros_like(score_map)

        fig, ax = plt.subplots(1, 1, figsize=(4, 4))
        im = ax.imshow(score_map, cmap=self.cmap)
        ax.set_title(f"Score Heatmap #{idx}")
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        img_array = fig_to_array(fig, dpi=150)
        plt.close(fig)

        artifact = Artifact(
            name=f"score_heatmap_img{idx:02d}",
            value=img_array,
            el_id=idx,
            desc="Anomaly score heatmap",
            type=ArtifactType.IMAGE,
            stage=context.stage,
            epoch=context.epoch,
            batch_idx=context.batch_idx,
        )
        artifacts.append(artifact)

    return {"artifacts": artifacts}
```

`SpectraPlot`cuvis_ai.node.spectrum_plotvisualizerhsirgbRender a multi-series line plot of per-class mean spectra into an RGB frame.

#### SpectraPlot

```python
SpectraPlot(
    palette=None,
    title="",
    xlabel="wavelength in [nm]",
    ylabel="value",
    plot_width=720,
    plot_height=540,
    dpi=150,
    linewidth=1.0,
    bg_color="white",
    fg_color="black",
    **kwargs,
)
```

Bases: `Node`

Render a multi-series line plot of per-class mean spectra into an RGB frame.

A multi-series sibling of :class:`SpectrumPlotNode`: instead of a tracked-vs-reference pair, it draws one line per row of `signatures` (e.g. the per-object mean spectra from :class:`~cuvis_ai.node.spectral_extractor.SpectralSignatureExtractor`), coloured by `palette`. The plot is rendered to a fixed-size RGB frame so it drops into the graph like any other image output (stitch several with `ImageConcatenator`).

Parameters:

| Name          | Type                           | Description                                                 | Default                                                                                                                                                                                   |
| ------------- | ------------------------------ | ----------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `palette`     | \`list\[tuple[int, int, int]\] | None\`                                                      | Per-row line colours in 0-255, indexed by row id and wrapped modulo its length. Share the same palette as a LegendStrip / ClassMapToRGB so colours agree. None uses a Tableau-20 palette. |
| `title`       | `str`                          | Axis title and labels.                                      | `''`                                                                                                                                                                                      |
| `xlabel`      | `str`                          | Axis title and labels.                                      | `''`                                                                                                                                                                                      |
| `ylabel`      | `str`                          | Axis title and labels.                                      | `''`                                                                                                                                                                                      |
| `plot_width`  | `int`                          | Pixel dimensions of the rendered frame (default 720 x 540). | `720`                                                                                                                                                                                     |
| `plot_height` | `int`                          | Pixel dimensions of the rendered frame (default 720 x 540). | `720`                                                                                                                                                                                     |
| `dpi`         | `int`                          | Figure dpi (default 150).                                   | `150`                                                                                                                                                                                     |
| `linewidth`   | `float`                        | Line width for each spectrum (default 1.0).                 | `1.0`                                                                                                                                                                                     |
| `bg_color`    | `str`                          | Background / foreground (axis, text) colours.               | `'white'`                                                                                                                                                                                 |
| `fg_color`    | `str`                          | Background / foreground (axis, text) colours.               | `'white'`                                                                                                                                                                                 |

Source code in `cuvis_ai/node/spectrum_plot.py`

```python
def __init__(
    self,
    palette: list[tuple[int, int, int]] | None = None,
    title: str = "",
    xlabel: str = "wavelength in [nm]",
    ylabel: str = "value",
    plot_width: int = 720,
    plot_height: int = 540,
    dpi: int = 150,
    linewidth: float = 1.0,
    bg_color: str = "white",
    fg_color: str = "black",
    **kwargs: Any,
) -> None:
    if plot_width < 32 or plot_height < 32:
        raise ValueError("plot dimensions must be >= 32 px")
    if dpi <= 0:
        raise ValueError("dpi must be > 0")
    from cuvis_ai.node.colormap import _TAB20

    colors = list(palette) if palette is not None else list(_TAB20)
    if not colors:
        raise ValueError("palette must be a non-empty list of (r, g, b)")
    self._palette = [tuple(int(c) for c in rgb) for rgb in colors]
    self.title = str(title)
    self.xlabel = str(xlabel)
    self.ylabel = str(ylabel)
    self.plot_width = int(plot_width)
    self.plot_height = int(plot_height)
    self.dpi = int(dpi)
    self.linewidth = float(linewidth)
    self.bg_color = str(bg_color)
    self.fg_color = str(fg_color)
    super().__init__(
        palette=[list(rgb) for rgb in self._palette],
        title=self.title,
        xlabel=self.xlabel,
        ylabel=self.ylabel,
        plot_width=self.plot_width,
        plot_height=self.plot_height,
        dpi=self.dpi,
        linewidth=self.linewidth,
        bg_color=self.bg_color,
        fg_color=self.fg_color,
        **kwargs,
    )
```

##### forward

```python
forward(signatures, wavelengths, valid=None, **_)
```

Render one multi-series plot frame per batch element.

Source code in `cuvis_ai/node/spectrum_plot.py`

```python
@torch.no_grad()
def forward(
    self,
    signatures: torch.Tensor,
    wavelengths: torch.Tensor,
    valid: torch.Tensor | None = None,
    **_: Any,
) -> dict[str, torch.Tensor]:
    """Render one multi-series plot frame per batch element."""
    sig = signatures.detach().cpu().numpy()  # [B, N, C]
    # wavelengths arrives as a numpy int32 array (the catalog's wavelength port dtype).
    wl = np.asarray(wavelengths, dtype=np.float32).ravel()
    b, n = sig.shape[0], sig.shape[1]
    valid_np = (
        valid.detach().cpu().numpy() if valid is not None else np.ones((b, n), dtype=bool)
    )
    frames = np.empty((b, self.plot_height, self.plot_width, 3), dtype=np.float32)
    for i in range(b):
        frames[i] = self._render(sig[i], wl, valid_np[i]).astype(np.float32) / 255.0
    return {
        "rgb_image": torch.from_numpy(frames).to(device=signatures.device, dtype=torch.float32)
    }
```

`SpectrumPlotNode`cuvis_ai.node.spectrum_plotvisualizerhsiRender a per-frame line plot of tracked vs reference spectrum.

#### SpectrumPlotNode

```python
SpectrumPlotNode(
    wavelengths,
    reference_wavelengths,
    plot_width=960,
    plot_height=720,
    dpi=150,
    xlabel="wavelength in [nm]",
    ylabel="spectral radiance in [W/m²/sr/µm]",
    tracked_label="",
    reference_label="",
    tracked_color="red",
    reference_color="lime",
    bg_color="black",
    fg_color="white",
    y_fixed_range=(0.0, 12.0),
    y_num_ticks=12,
    tracked_hold_frames=15,
    **kwargs,
)
```

Bases: `Node`

Render a per-frame line plot of tracked vs reference spectrum.

The tracked line is plotted against `wavelengths` (typically the full cube grid), while the reference line is plotted against `reference_wavelengths` (typically the narrower bandpass subset where the SAM reference is defined). Both are drawn on a single axes with a fixed x-range so the axis does not re-scale from frame to frame.

Output shape is fixed to `(plot_height, plot_width)` so downstream `ToVideoNode` gets consistent frame dimensions.

Parameters:

| Name                    | Type                  | Description                                                                                                                                                                                            | Default                                                                                                                          |
| ----------------------- | --------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------- |
| `wavelengths`           | `ndarray`             | Full wavelength grid in nm, shape [C_full]. Used as the x-axis for the tracked line.                                                                                                                   | *required*                                                                                                                       |
| `reference_wavelengths` | `ndarray`             | Wavelength grid for the reference line in nm, shape [C_ref].                                                                                                                                           | *required*                                                                                                                       |
| `plot_width`            | `int`                 | Pixel dimensions of the rendered frame. Default 960 x 720.                                                                                                                                             | `960`                                                                                                                            |
| `plot_height`           | `int`                 | Pixel dimensions of the rendered frame. Default 960 x 720.                                                                                                                                             | `960`                                                                                                                            |
| `dpi`                   | `int`                 | Figure dpi. Default 150.                                                                                                                                                                               | `150`                                                                                                                            |
| `xlabel`                | `str`                 | Axis labels.                                                                                                                                                                                           | `'wavelength in [nm]'`                                                                                                           |
| `ylabel`                | `str`                 | Axis labels.                                                                                                                                                                                           | `'wavelength in [nm]'`                                                                                                           |
| `tracked_label`         | `str`                 | Legend labels.                                                                                                                                                                                         | `''`                                                                                                                             |
| `reference_label`       | `str`                 | Legend labels.                                                                                                                                                                                         | `''`                                                                                                                             |
| `tracked_color`         | `str`                 | Matplotlib colour specs for the two lines.                                                                                                                                                             | `'red'`                                                                                                                          |
| `reference_color`       | `str`                 | Matplotlib colour specs for the two lines.                                                                                                                                                             | `'red'`                                                                                                                          |
| `bg_color`              | `str`                 | Background / foreground colours.                                                                                                                                                                       | `'black'`                                                                                                                        |
| `fg_color`              | `str`                 | Background / foreground colours.                                                                                                                                                                       | `'black'`                                                                                                                        |
| `y_fixed_range`         | \`tuple[float, float] | None\`                                                                                                                                                                                                 | If set, fixes the y-axis range. Otherwise auto-scales per-frame on the union of tracked+reference values, with a small headroom. |
| `y_num_ticks`           | `int`                 | When y_fixed_range is set, draw exactly this many evenly-spaced y-axis ticks spanning (y_min, y_max\]. Default 12.                                                                                     | `12`                                                                                                                             |
| `tracked_hold_frames`   | `int`                 | When valid=0 for a frame, keep drawing the most recent tracked spectrum for up to this many frames before falling back to reference-only. Smooths out brief dropouts. 0 disables the hold. Default 15. | `15`                                                                                                                             |

Source code in `cuvis_ai/node/spectrum_plot.py`

```python
def __init__(
    self,
    wavelengths: Sequence[float] | np.ndarray,
    reference_wavelengths: Sequence[float] | np.ndarray,
    plot_width: int = 960,
    plot_height: int = 720,
    dpi: int = 150,
    xlabel: str = "wavelength in [nm]",
    ylabel: str = "spectral radiance in [W/m²/sr/µm]",
    tracked_label: str = "",
    reference_label: str = "",
    tracked_color: str = "red",
    reference_color: str = "lime",
    bg_color: str = "black",
    fg_color: str = "white",
    y_fixed_range: tuple[float, float] | None = (0.0, 12.0),
    y_num_ticks: int = 12,
    tracked_hold_frames: int = 15,
    **kwargs: Any,
) -> None:
    if plot_width < 32 or plot_height < 32:
        raise ValueError("plot dimensions must be >= 32 px")
    if dpi <= 0:
        raise ValueError("dpi must be > 0")
    if y_num_ticks < 2:
        raise ValueError("y_num_ticks must be >= 2")
    if tracked_hold_frames < 0:
        raise ValueError("tracked_hold_frames must be >= 0")

    self._wavelengths = np.asarray(wavelengths, dtype=np.float32).ravel()
    self._ref_wavelengths = np.asarray(reference_wavelengths, dtype=np.float32).ravel()
    if self._wavelengths.size == 0:
        raise ValueError("wavelengths must be non-empty")
    if self._ref_wavelengths.size == 0:
        raise ValueError("reference_wavelengths must be non-empty")

    self.plot_width = int(plot_width)
    self.plot_height = int(plot_height)
    self.dpi = int(dpi)
    self.xlabel = str(xlabel)
    self.ylabel = str(ylabel)
    self.tracked_label = str(tracked_label)
    self.reference_label = str(reference_label)
    self.tracked_color = str(tracked_color)
    self.reference_color = str(reference_color)
    self.bg_color = str(bg_color)
    self.fg_color = str(fg_color)
    self.y_fixed_range = (
        None if y_fixed_range is None else (float(y_fixed_range[0]), float(y_fixed_range[1]))
    )
    self.y_num_ticks = int(y_num_ticks)
    self.tracked_hold_frames = int(tracked_hold_frames)

    super().__init__(
        wavelengths=self._wavelengths.tolist(),
        reference_wavelengths=self._ref_wavelengths.tolist(),
        plot_width=self.plot_width,
        plot_height=self.plot_height,
        dpi=self.dpi,
        xlabel=self.xlabel,
        ylabel=self.ylabel,
        tracked_label=self.tracked_label,
        reference_label=self.reference_label,
        tracked_color=self.tracked_color,
        reference_color=self.reference_color,
        bg_color=self.bg_color,
        fg_color=self.fg_color,
        y_fixed_range=self.y_fixed_range,
        y_num_ticks=self.y_num_ticks,
        tracked_hold_frames=self.tracked_hold_frames,
        **kwargs,
    )

    # Stateful hold across frames.
    self._last_tracked: np.ndarray | None = None
    self._hold_counter: int = 0
```

`TrackingOverlayNode`cuvis_ai.node.anomaly_visualizationvisualizerbboxrgbtrackAlpha-blend per-object coloured masks onto RGB frames.

#### TrackingOverlayNode

```python
TrackingOverlayNode(
    alpha=0.4,
    draw_contours=True,
    draw_ids=True,
    text_scale=2,
    **kwargs,
)
```

Bases: `Node`

Alpha-blend per-object coloured masks onto RGB frames.

Converts a SAM3-style label map (`mask`) into per-object binary masks and renders a coloured overlay with optional contour lines and object-ID labels using :func:`cuvis_ai.utils.torch_draw.overlay_instances`.

Parameters:

| Name            | Type    | Description                                                                                                     | Default |
| --------------- | ------- | --------------------------------------------------------------------------------------------------------------- | ------- |
| `alpha`         | `float` | Blend factor for the overlay colour (default 0.4).                                                              | `0.4`   |
| `draw_contours` | `bool`  | Draw contour outlines on mask edges (default True).                                                             | `True`  |
| `draw_ids`      | `bool`  | Render numeric object-ID labels above each mask (default True).                                                 | `True`  |
| `text_scale`    | `int`   | Integer scale of the object-ID label font (default 2). Use 1 for smaller ids on small frames with many objects. | `2`     |

Source code in `cuvis_ai/node/anomaly_visualization.py`

```python
def __init__(
    self,
    alpha: float = 0.4,
    draw_contours: bool = True,
    draw_ids: bool = True,
    text_scale: int = 2,
    **kwargs,
) -> None:
    self.alpha = float(alpha)
    self.draw_contours = bool(draw_contours)
    self.draw_ids = bool(draw_ids)
    self.text_scale = max(1, int(text_scale))
    super().__init__(
        alpha=alpha,
        draw_contours=draw_contours,
        draw_ids=draw_ids,
        text_scale=self.text_scale,
        **kwargs,
    )
```

##### forward

```python
forward(
    rgb_image, mask, object_ids=None, frame_id=None, **_
)
```

Render coloured per-object mask overlays onto *rgb_image*.

Parameters:

| Name         | Type             | Description                                                                                                                                                         | Default    |
| ------------ | ---------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------- |
| `rgb_image`  | `Tensor`         | Single RGB frame [1, H, W, 3] float32 in [0, 1].                                                                                                                    | *required* |
| `mask`       | `Tensor`         | SAM3 label map [1, H, W] int32.                                                                                                                                     | *required* |
| `object_ids` | `Tensor or None` | Active object IDs [1, N] int64. When provided, only these IDs are rendered and the ordering is preserved. When absent, all non-zero unique values in mask are used. | `None`     |
| `frame_id`   | `Tensor or None` | Frame / measurement index [1] int64. When provided, the frame number is rendered in the top-left corner.                                                            | `None`     |

Returns:

| Type   | Description                                                       |
| ------ | ----------------------------------------------------------------- |
| `dict` | {"rgb_with_overlay": torch.Tensor [1, H, W, 3] float32 in [0, 1]} |

Source code in `cuvis_ai/node/anomaly_visualization.py`

```python
@torch.no_grad()
def forward(
    self,
    rgb_image: torch.Tensor,
    mask: torch.Tensor,
    object_ids: torch.Tensor | None = None,
    frame_id: torch.Tensor | None = None,
    **_,
) -> dict[str, torch.Tensor]:
    """Render coloured per-object mask overlays onto *rgb_image*.

    Parameters
    ----------
    rgb_image : torch.Tensor
        Single RGB frame ``[1, H, W, 3]`` float32 in ``[0, 1]``.
    mask : torch.Tensor
        SAM3 label map ``[1, H, W]`` int32.
    object_ids : torch.Tensor or None
        Active object IDs ``[1, N]`` int64.  When provided, only these IDs
        are rendered and the ordering is preserved.  When absent, all
        non-zero unique values in *mask* are used.
    frame_id : torch.Tensor or None
        Frame / measurement index ``[1]`` int64.  When provided, the frame
        number is rendered in the top-left corner.

    Returns
    -------
    dict
        ``{"rgb_with_overlay": torch.Tensor [1, H, W, 3] float32 in [0, 1]}``
    """
    frame_u8 = (rgb_image[0].clamp(0.0, 1.0) * 255.0).to(torch.uint8)  # [H, W, 3]
    # Align mask/object-id tensors to the image device. Some upstream nodes can
    # emit CPU tensors even when the visualization path runs on CUDA.
    mask_t = mask[0].to(frame_u8.device)  # [H, W] int32
    present_ids_t = torch.unique(mask_t)
    present_ids_t = present_ids_t[present_ids_t > 0]

    if object_ids is not None:
        # Some trackers include background label 0 in object_ids; never render it.
        filtered_ids = object_ids[0].to(mask_t.device)
        filtered_ids = filtered_ids[filtered_ids > 0]
        if present_ids_t.numel() > 0:
            filtered_ids = filtered_ids[torch.isin(filtered_ids, present_ids_t)]
        else:
            filtered_ids = filtered_ids[:0]
        ids = []
        seen: set[int] = set()
        for raw_id in filtered_ids.tolist():
            obj_id = int(raw_id)
            if obj_id in seen:
                continue
            seen.add(obj_id)
            ids.append(obj_id)
    else:
        ids = [int(v) for v in present_ids_t.tolist()]

    per_obj_masks: list[tuple[int, torch.Tensor]] = [(oid, mask_t == oid) for oid in ids]

    rendered = overlay_instances(
        frame_u8,
        per_obj_masks,
        alpha=self.alpha,
        draw_edges=self.draw_contours,
        draw_ids=self.draw_ids,
        text_scale=self.text_scale,
    )

    if frame_id is not None:
        fid = int(frame_id.reshape(-1)[0].item())
        draw_text(rendered, 8, 8, f"frame {fid}", (255, 255, 255), scale=2, bg=True)

    out = rendered.to(torch.float32) / 255.0  # [H, W, 3]
    return {"rgb_with_overlay": out.unsqueeze(0)}  # [1, H, W, 3]
```

`TrackingPointerOverlayNode`cuvis_ai.node.anomaly_visualizationvisualizerbboxrgbtrackDraw downward triangle pointers for all tracked objects.

#### TrackingPointerOverlayNode

```python
TrackingPointerOverlayNode(
    alpha=0.4, draw_contours=True, draw_ids=True, **kwargs
)
```

Bases: `Node`

Draw downward triangle pointers for all tracked objects.

The node is composable by design: it renders only the pointer markers on top of an incoming RGB frame and does not perform any mask tinting itself. Colours are derived from object IDs using the same palette as :class:`TrackingOverlayNode`.

Parameters:

| Name            | Type    | Description                                                              | Default |
| --------------- | ------- | ------------------------------------------------------------------------ | ------- |
| `alpha`         | `float` | Reserved for API compatibility with :class:TrackingOverlayNode (unused). | `0.4`   |
| `draw_contours` | `bool`  | Reserved for API compatibility with :class:TrackingOverlayNode (unused). | `True`  |
| `draw_ids`      | `bool`  | Reserved for API compatibility with :class:TrackingOverlayNode (unused). | `True`  |

Source code in `cuvis_ai/node/anomaly_visualization.py`

```python
def __init__(
    self,
    alpha: float = 0.4,
    draw_contours: bool = True,
    draw_ids: bool = True,
    **kwargs,
) -> None:
    self.alpha = float(alpha)
    self.draw_contours = bool(draw_contours)
    self.draw_ids = bool(draw_ids)
    super().__init__(
        alpha=alpha,
        draw_contours=draw_contours,
        draw_ids=draw_ids,
        **kwargs,
    )
```

##### forward

```python
forward(
    rgb_image, mask, object_ids=None, frame_id=None, **_
)
```

Render pointer overlays for all objects onto *rgb_image*.

Source code in `cuvis_ai/node/anomaly_visualization.py`

```python
@torch.no_grad()
def forward(
    self,
    rgb_image: torch.Tensor,
    mask: torch.Tensor,
    object_ids: torch.Tensor | None = None,
    frame_id: torch.Tensor | None = None,
    **_,
) -> dict[str, torch.Tensor]:
    """Render pointer overlays for all objects onto *rgb_image*."""
    frame_u8 = (rgb_image[0].clamp(0.0, 1.0) * 255.0).to(torch.uint8)
    mask_t = mask[0].to(frame_u8.device)
    if tuple(mask_t.shape) != tuple(frame_u8.shape[:2]):
        raise ValueError(
            f"Mask shape {tuple(mask_t.shape)} does not match image shape {tuple(frame_u8.shape[:2])}."
        )

    present_ids_t = torch.unique(mask_t)
    present_ids_t = present_ids_t[present_ids_t > 0]

    if object_ids is not None:
        filtered_ids = object_ids[0].to(mask_t.device)
        filtered_ids = filtered_ids[filtered_ids > 0]
        if present_ids_t.numel() > 0:
            filtered_ids = filtered_ids[torch.isin(filtered_ids, present_ids_t)]
        else:
            filtered_ids = filtered_ids[:0]
        ids: list[int] = []
        seen: set[int] = set()
        for raw_id in filtered_ids.tolist():
            oid = int(raw_id)
            if oid in seen:
                continue
            seen.add(oid)
            ids.append(oid)
    else:
        ids = [int(v) for v in present_ids_t.tolist()]

    rendered = frame_u8.clone()

    for oid in ids:
        fg = mask_t == oid
        if not torch.any(fg):
            continue

        ys, xs = torch.where(fg)
        x_min = int(xs.min().item())
        x_max = int(xs.max().item())
        y_min = int(ys.min().item())

        bbox_width = x_max - x_min + 1
        tri_width = max(12, min(48, int(round(bbox_width * 0.45))))
        tri_width = min(tri_width, max(1, int(rendered.shape[1]) - 1))
        tri_height = max(10, min(36, int(round(tri_width * 0.8))))
        tri_height = min(tri_height, max(1, int(rendered.shape[0]) - 1))
        gap = max(4, min(10, int(round(tri_height * 0.3))))

        centroid_x = int(torch.round(xs.to(torch.float32).mean()).item())
        half_width = max(1, (tri_width + 1) // 2)
        max_tip_x = max(half_width, int(rendered.shape[1]) - 1 - half_width)
        tip_x = max(half_width, min(centroid_x, max_tip_x))
        desired_tip_y = y_min - gap
        tip_y = max(tri_height, min(desired_tip_y, int(rendered.shape[0]) - 1))

        color_t = id_to_color(torch.tensor([oid], device=rendered.device, dtype=torch.int64))[0]
        color = tuple(int(channel) for channel in color_t.tolist())

        outline_thickness = 2 if tri_width >= 20 and tri_height >= 16 else 1
        draw_downward_triangle(
            rendered,
            tip_x=tip_x,
            tip_y=tip_y,
            width=tri_width,
            height=tri_height,
            color=color,
            outline_color=(0, 0, 0),
            outline_thickness=outline_thickness,
        )

    if frame_id is not None:
        fid = int(frame_id.reshape(-1)[0].item())
        draw_text(rendered, 8, 8, f"frame {fid}", (255, 255, 255), scale=2, bg=True)

    out = rendered.to(torch.float32) / 255.0
    return {"rgb_with_overlay": out.unsqueeze(0)}
```

No items match your search and filters.

Clear search and filters
