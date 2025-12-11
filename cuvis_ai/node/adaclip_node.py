"""AdaCLIP detector node for zero-shot anomaly detection.

This module provides a port-based Node implementation for AdaCLIP,
enabling integration with the cuvis.ai pipeline system.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import Compose

from cuvis_ai.anomaly.adaclip import AdaCLIPModel, download_weights
from cuvis_ai.node.node import Node
from cuvis_ai.pipeline.ports import PortSpec
from cuvis_ai.utils.types import Context


class AdaCLIPDetector(Node):
    """AdaCLIP zero-shot anomaly detector node.

    This node applies AdaCLIP for anomaly detection on RGB images.
    It takes RGB images (either uint8 or float32) and outputs pixel-level
    anomaly scores and image-level anomaly scores.

    The node uses lazy loading to avoid initializing the model until
    it's actually needed (first forward pass). The underlying AdaCLIP model
    is registered as a submodule so that ``state_dict()`` captures its weights.

    Parameters
    ----------
    weight_name : str
        Name of pretrained weights to use. Options:
        - "pretrained_all": All datasets combined
        - "pretrained_mvtec_colondb": MVTec + ColonDB
        - "pretrained_visa_clinicdb": VisA + ClinicDB
    backbone : str
        CLIP backbone model name (e.g., 'ViT-L-14-336').
    prompt_text : str
        Text prompt describing the object/anomaly.
    image_size : int
        Model input image size.
    prompting_depth : int
        Number of transformer layers with prompting.
    prompting_length : int
        Number of prompt tokens.
    gaussian_sigma : float
        Sigma for Gaussian smoothing of anomaly map.

    Ports
    -----
    INPUT_SPECS
        ``rgb_image`` : float32, shape (-1, -1, -1, 3)
            RGB image in BHWC format (0-1 range or 0-255 range).
    OUTPUT_SPECS
        ``scores`` : float32, shape (-1, -1, -1, 1)
            Pixel-level anomaly scores in BHW1 format.
        ``anomaly_score`` : float32, shape (-1,)
            Image-level anomaly score per batch item.

    Example
    -------
    >>> from cuvis_ai.node.adaclip_node import AdaCLIPDetector
    >>> detector = AdaCLIPDetector(
    ...     weight_name="pretrained_all",
    ...     prompt_text="normal: lentils, anomaly: stones"
    ... )
    >>> # RGB image in BHWC format, float32, 0-1 range
    >>> result = detector(rgb_image=rgb_tensor)
    >>> anomaly_scores = result["scores"]  # [B, H, W, 1]
    >>> image_scores = result["anomaly_score"]  # [B]
    """

    INPUT_SPECS = {
        "rgb_image": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, 3),
            description="RGB image [B, H, W, 3] in float32 (0-1 or 0-255 range)",
        ),
    }

    OUTPUT_SPECS = {
        "scores": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, 1),
            description="Pixel-level anomaly scores [B, H, W, 1]",
        ),
        "anomaly_score": PortSpec(
            dtype=torch.float32,
            shape=(-1,),
            description="Image-level anomaly score [B]",
        ),
    }

    def __init__(
        self,
        weight_name: str = "pretrained_all",
        backbone: str = "ViT-L-14-336",
        prompt_text: str = "",
        image_size: int = 518,
        prompting_depth: int = 4,
        prompting_length: int = 5,
        gaussian_sigma: float = 4.0,
        **kwargs,
    ) -> None:
        super().__init__(
            weight_name=weight_name,
            backbone=backbone,
            prompt_text=prompt_text,
            image_size=image_size,
            prompting_depth=prompting_depth,
            prompting_length=prompting_length,
            gaussian_sigma=gaussian_sigma,
            **kwargs,
        )

        self.weight_name = weight_name
        self.backbone = backbone
        self.prompt_text = prompt_text
        self.image_size = image_size
        self.prompting_depth = prompting_depth
        self.prompting_length = prompting_length
        self.gaussian_sigma = gaussian_sigma

        # Lazy initialization - will be registered as submodule when loaded
        # Using a placeholder nn.Module so state_dict() works before loading
        self._adaclip_model: AdaCLIPModel | None = None
        self._preprocess: Compose | None = None

        # Use standard instance attribute for initialization tracking
        self._statistically_initialized = False

    @property
    def _model(self) -> AdaCLIPModel | None:
        """Backward-compatible alias for _adaclip_model."""
        return self._adaclip_model

    def _ensure_model_loaded(self) -> None:
        """Lazy load model on first forward pass."""
        if self._adaclip_model is not None:
            return

        # Download weights if not cached
        weight_path = download_weights(self.weight_name)

        # Determine device from current module parameters/buffers or default
        device = None
        for param in self.parameters():
            device = param.device
            break
        if device is None:
            for buf in self.buffers():
                device = buf.device
                break
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        # Create and register the model as a submodule
        model = AdaCLIPModel(
            backbone=self.backbone,
            image_size=self.image_size,
            prompting_depth=self.prompting_depth,
            prompting_length=self.prompting_length,
            device=str(device),
        )
        model.load_weights(weight_path)
        model.eval()

        # Assigning to an attribute of nn.Module automatically registers it
        # as a submodule, so we do not need (and must not) call add_module
        # with a name that already exists.
        self._adaclip_model = model
        self._preprocess = model.get_preprocess()
        self._statistically_initialized = True

    def _preprocess_rgb(self, rgb_bhwc: torch.Tensor) -> torch.Tensor:
        """Preprocess RGB tensor for model input.

        Converts BHWC tensor to preprocessed BCHW tensor suitable for the model.
        The output tensor is created on the same device as the AdaCLIP model
        (which handles its own device placement via pipeline.to()).

        Parameters
        ----------
        rgb_bhwc : torch.Tensor
            RGB tensor in BHWC format. Can be:
            - float32 in [0, 1] range
            - float32 in [0, 255] range
            - uint8 in [0, 255] range

        Returns
        -------
        torch.Tensor
            Preprocessed tensor in BCHW format, on the model's device.
        """
        B = rgb_bhwc.shape[0]

        # Convert to uint8 numpy for PIL processing
        rgb_np = rgb_bhwc.detach().cpu().numpy()

        # Handle different input ranges
        if rgb_np.max() <= 1.0:
            rgb_np = (rgb_np * 255).astype(np.uint8)
        else:
            rgb_np = rgb_np.astype(np.uint8)

        # Process each image through CLIP preprocessing
        preprocessed = []
        for i in range(B):
            pil_img = Image.fromarray(rgb_np[i], mode="RGB")
            img_tensor = self._preprocess(pil_img)
            preprocessed.append(img_tensor)

        batch_tensor = torch.stack(preprocessed, dim=0)

        # Place on model device - the AdaCLIP model manages its own device
        # via pipeline.to(), so we follow its placement for inference
        # if self._adaclip_model is not None:
        #    batch_tensor = batch_tensor.to(self._adaclip_model.device)

        return batch_tensor

    def forward(
        self,
        rgb_image: torch.Tensor,
        context: Context | None = None,  # noqa: ARG002
        **_: Any,
    ) -> dict[str, torch.Tensor]:
        """Run AdaCLIP inference on RGB images.

        Parameters
        ----------
        rgb_image : torch.Tensor
            RGB image tensor in BHWC format with 3 channels.

        Returns
        -------
        dict[str, torch.Tensor]
            Dictionary with:
            - "scores": Pixel-level anomaly scores [B, H, W, 1]
            - "anomaly_score": Image-level scores [B]

        Note
        ----
        Shape validation (C == 3) is handled by PortSpec; no explicit assert needed.
        Device placement is handled by pipeline.to(); outputs remain on model device.
        """
        self._ensure_model_loaded()

        B, H, W, _ = rgb_image.shape

        # Preprocess images (placed on model device internally)
        img_tensor = self._preprocess_rgb(rgb_image)

        # Run inference
        with torch.no_grad():
            anomaly_map, anomaly_score = self._adaclip_model.predict(
                img_tensor,
                prompt=self.prompt_text,
                sigma=self.gaussian_sigma,
            )

        # Resize anomaly map back to original size if needed
        if anomaly_map.shape[1] != H or anomaly_map.shape[2] != W:
            anomaly_map = torch.nn.functional.interpolate(
                anomaly_map.unsqueeze(1),  # [B, 1, h, w]
                size=(H, W),
                mode="bilinear",
                align_corners=False,
            ).squeeze(1)  # [B, H, W]

        # Format outputs - device is managed by pipeline.to(), not explicit .to() calls
        scores = anomaly_map.unsqueeze(-1)  # [B, H, W, 1]

        return {
            "scores": scores,
            "anomaly_score": anomaly_score,
        }


__all__ = ["AdaCLIPDetector"]
