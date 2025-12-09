"""High-level AdaCLIP model wrapper.

This module provides a simplified interface for using AdaCLIP for anomaly detection.
"""

from __future__ import annotations

from pathlib import Path

import torch
from loguru import logger
from torch import nn
from torchvision.transforms import Compose

from cuvis_ai.anomaly.adaclip.core.adaclip import AdaCLIP
from cuvis_ai.anomaly.adaclip.core.custom_clip import (
    OPENAI_DATASET_MEAN,
    OPENAI_DATASET_STD,
    create_model_and_transforms,
)
from cuvis_ai.anomaly.adaclip.weights import download_weights


class AdaCLIPModel(nn.Module):
    """High-level AdaCLIP model for zero-shot anomaly detection.

    This class wraps the AdaCLIP model and provides a simplified interface
    for inference. It handles model loading, preprocessing, and inference.

    Parameters
    ----------
    backbone : str
        CLIP backbone model name (e.g., 'ViT-L-14-336').
    image_size : int
        Input image size.
    prompting_depth : int
        Number of transformer layers to apply prompting.
    prompting_length : int
        Number of prompt tokens.
    prompting_branch : str
        Which branches to prompt ('V', 'L', or 'VL').
    prompting_type : str
        Type of prompting ('S' for static, 'D' for dynamic, 'SD' for both).
    use_hsf : bool
        Whether to use Hybrid Semantic Fusion.
    k_clusters : int
        Number of clusters for HSF.
    output_layers : list
        Which transformer layers to use for patch tokens.
    device : str, optional
        Device to run on.

    Example
    -------
    >>> from cuvis_ai.anomaly.adaclip import AdaCLIPModel, download_weights
    >>> weight_path = download_weights("pretrained_all")
    >>> model = AdaCLIPModel(backbone="ViT-L-14-336")
    >>> model.load_weights(weight_path)
    >>> model.eval()
    >>> anomaly_map, score = model.predict(image_tensor, prompt="candle")
    """

    def __init__(
        self,
        backbone: str = "ViT-L-14-336",
        image_size: int = 518,
        prompting_depth: int = 4,
        prompting_length: int = 5,
        prompting_branch: str = "VL",
        prompting_type: str = "SD",
        use_hsf: bool = True,
        k_clusters: int = 20,
        output_layers: list[int] | None = None,
        device: str | None = None,
    ) -> None:
        super().__init__()

        self.backbone = backbone
        self.image_size = image_size
        self.prompting_depth = prompting_depth
        self.prompting_length = prompting_length
        self.prompting_branch = prompting_branch
        self.prompting_type = prompting_type
        self.use_hsf = use_hsf
        self.k_clusters = k_clusters
        self.output_layers = output_layers or [6, 12, 18, 24]

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        self._clip_model: AdaCLIP | None = None
        self._model: AdaCLIP | None = None
        self._preprocess: Compose | None = None
        self._initialized = False

    def _init_model(self) -> None:
        """Initialize the CLIP backbone and AdaCLIP model."""
        if self._initialized:
            return

        logger.info(f"Initializing AdaCLIP with {self.backbone} backbone...")

        # Create CLIP model and transforms
        clip_model, preprocess_train, preprocess_val = create_model_and_transforms(
            model_name=self.backbone,
            img_size=self.image_size,
            pretrained="openai",
            device=self.device,
        )

        # Modify transforms to match legacy AdaCLIP trainer behavior:
        # The original trainer explicitly sets Resize and CenterCrop to use
        # tuple size (image_size, image_size) instead of just image_size.
        # This ensures exact resize instead of aspect-preserving resize.
        from torchvision import transforms as T

        preprocess_val.transforms[0] = T.Resize(
            size=(self.image_size, self.image_size),
            interpolation=T.InterpolationMode.BICUBIC,
        )
        preprocess_val.transforms[1] = T.CenterCrop(size=(self.image_size, self.image_size))

        # Get channel dimensions from config
        from cuvis_ai.anomaly.adaclip.core.custom_clip import get_model_config

        model_cfg = get_model_config(self.backbone)
        if model_cfg is None:
            raise ValueError(f"Model config for {self.backbone} not found")

        text_channel = model_cfg["embed_dim"]
        visual_channel = model_cfg["vision_cfg"]["width"]

        # Create AdaCLIP model
        self._clip_model = AdaCLIP(
            freeze_clip=clip_model,
            text_channel=text_channel,
            visual_channel=visual_channel,
            prompting_length=self.prompting_length,
            prompting_depth=self.prompting_depth,
            prompting_branch=self.prompting_branch,
            prompting_type=self.prompting_type,
            use_hsf=self.use_hsf,
            k_clusters=self.k_clusters,
            output_layers=self.output_layers,
            device=self.device,
            image_size=self.image_size,
        )
        self._clip_model.to(self.device)
        self._model = self._clip_model

        self._preprocess = preprocess_val
        self._initialized = True

        logger.info(f"AdaCLIP initialized on {self.device}")

    def load_weights(self, weight_path: str | Path) -> None:
        """Load pretrained AdaCLIP weights.

        Parameters
        ----------
        weight_path : str or Path
            Path to the pretrained weights file.
        """
        self._init_model()

        weight_path = Path(weight_path)
        if not weight_path.exists():
            raise FileNotFoundError(f"Weight file not found: {weight_path}")

        logger.info(f"Loading AdaCLIP weights from {weight_path}")

        # PyTorch 2.6+ defaults weights_only=True, which breaks older checkpoints.
        # Explicitly request full checkpoint loading, with backward-compatible fallback.
        try:
            checkpoint = torch.load(weight_path, map_location=self.device, weights_only=False)
        except TypeError:  # Older torch versions without weights_only argument
            checkpoint = torch.load(weight_path, map_location=self.device)
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint

        # Remove 'module.' prefix if present
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

        # The checkpoint keys have 'clip_model.' prefix (from AdaCLIP_Trainer.clip_model)
        # but our model expects keys without that prefix
        state_dict = {k.replace("clip_model.", ""): v for k, v in state_dict.items()}

        # Load ALL weights
        missing, unexpected = self._clip_model.load_state_dict(state_dict, strict=False)
        if missing:
            logger.debug(f"Missing keys: {missing[:5]}...")
        if unexpected:
            logger.debug(f"Unexpected keys: {unexpected[:5]}...")

        logger.info("AdaCLIP weights loaded successfully")

    def get_preprocess(self) -> Compose:
        """Get the preprocessing transform.

        Returns
        -------
        Compose
            Preprocessing transform for input images.
        """
        self._init_model()
        return self._preprocess

    def predict(
        self,
        image: torch.Tensor,
        prompt: str = "",
        sigma: float = 4.0,
        aggregation: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run anomaly detection on an image.

        Parameters
        ----------
        image : torch.Tensor
            Preprocessed image tensor [B, C, H, W].
        prompt : str
            Text prompt describing the object class.
        sigma : float
            Gaussian smoothing sigma for the anomaly map.
        aggregation : bool
            Whether to aggregate multi-scale features.

        Returns
        -------
        anomaly_map : torch.Tensor
            Pixel-level anomaly scores [B, H, W].
        anomaly_score : torch.Tensor
            Image-level anomaly scores [B].
        """
        self._init_model()

        if not self._initialized:
            raise RuntimeError("Model not initialized. Call load_weights() first.")

        image = image.to(self.device)

        # Use empty string as prompt if not provided
        cls_name = [prompt] if prompt else [""]

        with torch.no_grad():
            anomaly_map, anomaly_score = self._clip_model(image, cls_name, aggregation=aggregation)

        # Apply Gaussian smoothing using a pure PyTorch implementation on the
        # model device to avoid unnecessary NumPy/CPU round-trips.
        if sigma > 0:
            # anomaly_map: [B, H, W]
            b, h, w = anomaly_map.shape
            # Choose kernel size as 2*ceil(3*sigma)+1 (covers ~99% of Gaussian)
            kernel_size = int(2 * (int(3 * sigma) + 1) + 1)
            if kernel_size % 2 == 0:
                kernel_size += 1

            # Create 2D separable Gaussian kernel
            coords = torch.arange(kernel_size, device=anomaly_map.device, dtype=anomaly_map.dtype)
            coords = coords - (kernel_size - 1) / 2.0
            gauss_1d = torch.exp(-(coords**2) / (2 * float(sigma) ** 2))
            gauss_1d = gauss_1d / gauss_1d.sum()
            gauss_2d = gauss_1d[:, None] * gauss_1d[None, :]
            gauss_2d = gauss_2d / gauss_2d.sum()

            weight = gauss_2d.view(1, 1, kernel_size, kernel_size)

            # Apply depthwise conv2d
            anomaly_map = anomaly_map.unsqueeze(1)  # [B, 1, H, W]
            anomaly_map = torch.nn.functional.conv2d(
                anomaly_map,
                weight,
                padding=kernel_size // 2,
            )
            anomaly_map = anomaly_map.squeeze(1)  # [B, H, W]

        return anomaly_map, anomaly_score

    def forward(
        self,
        image: torch.Tensor,
        prompt: str = "",
        sigma: float = 4.0,
        aggregation: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for anomaly detection.

        This is an alias for predict().
        """
        return self.predict(image, prompt, sigma, aggregation)


def create_adaclip_model(
    weight_name: str = "pretrained_all",
    backbone: str = "ViT-L-14-336",
    image_size: int = 518,
    prompting_depth: int = 4,
    prompting_length: int = 5,
    device: str | None = None,
) -> AdaCLIPModel:
    """Create and load an AdaCLIP model.

    This is a convenience function that creates an AdaCLIPModel instance
    and loads pretrained weights.

    Parameters
    ----------
    weight_name : str
        Name of pretrained weights to download.
    backbone : str
        CLIP backbone model name.
    image_size : int
        Input image size.
    prompting_depth : int
        Number of transformer layers to apply prompting.
    prompting_length : int
        Number of prompt tokens.
    device : str, optional
        Device to run on.

    Returns
    -------
    AdaCLIPModel
        Loaded AdaCLIP model ready for inference.
    """
    model = AdaCLIPModel(
        backbone=backbone,
        image_size=image_size,
        prompting_depth=prompting_depth,
        prompting_length=prompting_length,
        device=device,
    )

    weight_path = download_weights(weight_name)
    model.load_weights(weight_path)
    model.eval()

    return model


__all__ = ["AdaCLIPModel", "create_adaclip_model", "OPENAI_DATASET_MEAN", "OPENAI_DATASET_STD"]
