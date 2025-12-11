"""HuggingFace model integration nodes for cuvis.ai pipeline system.

This module provides nodes for interfacing with HuggingFace models via:
1. API backend - gradio_client for HF Spaces (Phase 1)
2. Local backend - transformers for local models (Phase 2)

Example
-------
>>> # API inference (Phase 1)
>>> adaclip_api = AdaCLIPAPINode(space_url="Caoyunkang/AdaCLIP")
>>> result = adaclip_api.forward(image=rgb_tensor)
"""

from __future__ import annotations

import os
import tempfile
from typing import Any

import numpy as np
import torch
from dotenv import load_dotenv
from gradio_client import Client, handle_file
from loguru import logger
from PIL import Image
from torch import Tensor
from transformers import AutoModel, CLIPVisionModel

from cuvis_ai.node.node import Node
from cuvis_ai.pipeline.ports import PortSpec

load_dotenv(override=True)


class HuggingFaceAPINode(Node):
    """Base class for HuggingFace Spaces API integration.

    This node enables calling HuggingFace Spaces via their API using gradio_client.
    API calls are NOT differentiable and should only be used for inference/testing.

    Parameters
    ----------
    space_url : str
        HuggingFace Space URL (e.g., "username/space-name")
    use_hf_token : bool, optional
        Whether to use HF_TOKEN from environment (default: True)
    api_timeout : float, optional
        Timeout for API calls in seconds (default: 60.0)
    **kwargs
        Additional arguments passed to Node base class

    Attributes
    ----------
    space_url : str
        The HuggingFace Space URL
    client : gradio_client.Client or None
        The Gradio client instance (lazy loaded)

    Notes
    -----
    - API calls are not differentiable (no gradient flow)
    - Requires internet connection
    - Subject to HF Space rate limits
    - Use local backend (Phase 2) for gradient training
    """

    def __init__(
        self,
        space_url: str,
        api_timeout: float = 60.0,
        **kwargs,
    ) -> None:
        self.space_url = space_url
        self.api_timeout = api_timeout

        super().__init__(
            space_url=space_url,
            api_timeout=api_timeout,
            **kwargs,
        )

        # Lazy load client (only when needed)
        self._client = None

    @property
    def client(self) -> Client:
        """Lazy load Gradio client."""
        if self._client is None:
            self._client = self._initialize_client()
        return self._client

    def _initialize_client(self) -> Client:
        """Initialize Gradio client with optional HF token.

        Returns
        -------
        gradio_client.Client
            Initialized Gradio client

        Raises
        ------
        ImportError
            If gradio_client is not installed
        RuntimeError
            If Space connection fails
        """

        # Get HF token from environment if requested
        hf_token = os.getenv("HF_TOKEN")

        # Initialize client
        try:
            logger.info(f"Connecting to HuggingFace Space: {self.space_url}")
            client = Client(self.space_url, hf_token=hf_token)

            logger.success(f"Successfully connected to {self.space_url}")
            return client
        except TypeError as e:
            raise RuntimeError(
                f"Failed to connect to HuggingFace Space '{self.space_url}': {e}"
            ) from e

    def forward(self, **inputs: Any) -> dict[str, Any]:
        """Execute API call to HuggingFace Space.

        This method should be overridden by subclasses to implement
        model-specific API calls.

        Parameters
        ----------
        **inputs
            Input tensors/data for the model

        Returns
        -------
        dict[str, Any]
            Model outputs

        Raises
        ------
        NotImplementedError
            If not overridden by subclass
        """
        raise NotImplementedError(f"{self.__class__.__name__} must implement forward() method")


class AdaCLIPAPINode(HuggingFaceAPINode):
    """AdaCLIP anomaly detection via HuggingFace Spaces API.

    This node calls the AdaCLIP Space for zero-shot anomaly detection.
    API backend is non-differentiable and suitable for inference only.

    Parameters
    ----------
    space_url : str, optional
        AdaCLIP Space URL (default: "Caoyunkang/AdaCLIP")
    use_hf_token : bool, optional
        Whether to use HF_TOKEN from environment (default: True)
    **kwargs
        Additional arguments passed to HuggingFaceAPINode

    Examples
    --------
    >>> # Create node
    >>> adaclip = AdaCLIPAPINode()
    >>>
    >>> # Run inference
    >>> rgb_image = torch.rand(1, 224, 224, 3)  # BHWC format
    >>> result = adaclip.forward(image=rgb_image)
    >>> anomaly_mask = result["anomaly_mask"]  # [B, H, W, 1]
    """

    INPUT_SPECS = {
        "image": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, 3),
            description="RGB image [B, H, W, 3] in range [0, 1] or [0, 255]",
        ),
        "text_prompt": PortSpec(
            dtype=str,
            shape=(),
            description="Text description for anomaly detection (e.g., 'defect', 'crack')",
            optional=True,
        ),
        "dataset_option": PortSpec(
            dtype=str,
            shape=(),
            description="Dataset preset for AdaCLIP Space (MVTec AD+Colondb, VisA+Clinicdb, All)",
            optional=True,
        ),
    }

    OUTPUT_SPECS = {
        "anomaly_mask": PortSpec(
            dtype=torch.bool,
            shape=(-1, -1, -1, 1),
            description="Binary anomaly mask [B, H, W, 1]",
        ),
        "anomaly_scores": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, 1),
            description="Anomaly scores [B, H, W, 1] (if available)",
            optional=True,
        ),
    }

    SUPPORTED_OPTIONS = (
        "MVTec AD+Colondb",
        "VisA+Clinicdb",
        "All",
    )

    def __init__(
        self,
        space_url: str = "Caoyunkang/AdaCLIP",
        dataset_option: str = "All",
        text_prompt: str = "normal: lentils, anomaly: stones",
        **kwargs,
    ) -> None:
        self.dataset_option = dataset_option
        self.text_prompt = text_prompt

        super().__init__(
            space_url=space_url,
            dataset_option=dataset_option,
            text_prompt=text_prompt,
            **kwargs,
        )

    def forward(
        self,
        image: Tensor,
        text_prompt: str | None = None,
        dataset_option: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Tensor]:
        """Run AdaCLIP anomaly detection via API.

        Parameters
        ----------
        image : Tensor
            RGB image [B, H, W, 3] in BHWC format
        text_prompt : str, optional
            Text description of anomaly to detect. If None, uses self.text_prompt.
        **kwargs
            Additional arguments (unused)

        Returns
        -------
        dict[str, Tensor]
            Dictionary with "anomaly_mask" and optionally "anomaly_scores"

        Raises
        ------
        RuntimeError
            If API call fails
        ValueError
            If image format is invalid
        """

        # Use instance variable if text_prompt not provided
        if text_prompt is None:
            text_prompt = self.text_prompt

        # Process each image in batch
        batch_size = image.shape[0]
        masks = []

        for i in range(batch_size):
            img = image[i]  # [H, W, 3]

            img_np = img.detach().cpu().numpy()

            # Normalize to [0, 255] if in [0, 1]
            if img_np.max() <= 1.0:
                img_np = (img_np * 255).astype(np.uint8)
            else:
                img_np = img_np.astype(np.uint8)

            # Convert to PIL Image
            pil_img = Image.fromarray(img_np)

            try:
                # Call API
                logger.debug(f"Calling AdaCLIP API for image {i + 1}/{batch_size}")
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_img:
                    pil_img.save(tmp_img.name)
                    tmp_path = tmp_img.name

                try:
                    result = self.client.predict(
                        handle_file(tmp_path),
                        text_prompt,
                        dataset_option,
                        api_name="/predict",
                    )
                finally:
                    try:
                        os.remove(tmp_path)
                    except OSError:
                        logger.warning(f"Failed to remove temp image: {tmp_path}")

                # Parse result
                # Note: Actual return format depends on AdaCLIP Space implementation
                # The Space currently returns (output_image_path, anomaly_score_str)
                if isinstance(result, np.ndarray):
                    mask_np = result
                elif isinstance(result, (list, tuple)):
                    first = result[0]
                    if isinstance(first, np.ndarray):
                        mask_np = first
                    elif isinstance(first, str):
                        # Gradio returns a temporary file path – load and convert to array
                        from PIL import Image as PILImage

                        mask_np = np.array(PILImage.open(first))
                    else:
                        raise ValueError(
                            f"Unexpected first element type in result tuple: {type(first)}"
                        )
                else:
                    raise ValueError(f"Unexpected API result type: {type(result)}")

                mask = torch.from_numpy(mask_np)

                # Ensure correct shape [H, W, 1]
                if mask.dim() == 2:
                    mask = mask.unsqueeze(-1)
                elif mask.dim() == 3 and mask.shape[-1] == 3:
                    # Convert RGB mask to single channel
                    mask = mask.float().mean(dim=-1, keepdim=True)

                # Resize to original spatial resolution if needed
                orig_h, orig_w = img.shape[0], img.shape[1]
                if mask.shape[0] != orig_h or mask.shape[1] != orig_w:
                    # interpolate expects NCHW
                    mask = mask.permute(2, 0, 1).unsqueeze(0).float()
                    mask = torch.nn.functional.interpolate(
                        mask,
                        size=(orig_h, orig_w),
                        mode="bilinear",
                        align_corners=False,
                    )
                    mask = mask.squeeze(0).permute(1, 2, 0)

                # Convert to binary mask
                if mask.dtype != torch.bool:
                    mask = mask > 0

                masks.append(mask)

            except Exception as e:
                logger.error(f"API call failed for image {i + 1}/{batch_size}: {e}")
                raise RuntimeError(
                    f"AdaCLIP API call failed: {e}\n"
                    f"Space: {self.space_url}\n"
                    f"Text prompt: {text_prompt}"
                ) from e

        # Stack batch
        anomaly_mask = torch.stack(masks, dim=0)  # [B, H, W, 1]

        return {
            "anomaly_mask": anomaly_mask,
        }


# ============================================================================
# Phase 2: Local Model Loading
# ============================================================================


class HuggingFaceLocalNode(Node):
    """Base class for local HuggingFace model integration."""

    def __init__(
        self,
        model_name: str,
        cache_dir: str | None = None,
        **kwargs,
    ) -> None:
        self.model_name = model_name
        self.cache_dir = cache_dir

        super().__init__(
            model_name=model_name,
            cache_dir=cache_dir,
            **kwargs,
        )

        self._model: torch.nn.Module | None = None

    @property
    def model(self) -> torch.nn.Module:
        """Lazy load and cache the HF model."""
        if self._model is None:
            self._model = self._load_model()
        return self._model

    def _load_model(self) -> torch.nn.Module:
        """Load model from HuggingFace Hub (frozen by default)."""
        hf_token = os.getenv("HF_TOKEN")

        try:
            logger.info(f"Loading HF model locally: {self.model_name}")
            logger.info(f"Cache dir: {self.cache_dir or 'default'}")

            model = AutoModel.from_pretrained(
                self.model_name,
                token=hf_token,
                cache_dir=self.cache_dir,
            )

            model.eval()
            for param in model.parameters():
                param.requires_grad = False

            logger.success(f"Model loaded and frozen: {self.model_name}")
            return model
        except Exception as exc:  # pragma: no cover - defensive logging
            raise RuntimeError(
                f"Failed to load model '{self.model_name}': {exc}\n"
                "Check model name, network access, or HF_TOKEN for private models."
            ) from exc

    def freeze(self) -> None:
        if self._model is not None:
            logger.info(f"Freezing {self.__class__.__name__}")
            self._model.eval()
            for param in self._model.parameters():
                param.requires_grad = False
        super().freeze()

    def unfreeze(self) -> None:
        if self._model is not None:
            logger.info(f"Unfreezing {self.__class__.__name__}")
            self._model.train()
            for param in self._model.parameters():
                param.requires_grad = True
        super().unfreeze()

    def forward(self, **inputs: Any) -> dict[str, Any]:
        raise NotImplementedError(f"{self.__class__.__name__} must implement forward()")


class AdaCLIPLocalNode(HuggingFaceLocalNode):
    """AdaCLIP anomaly detection with local HF loading."""

    INPUT_SPECS = {
        "image": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, 3),
            description="RGB image [B, H, W, 3] in range [0, 1] or [0, 255]",
        ),
        "text_prompt": PortSpec(
            dtype=str,
            shape=(),
            description="Text description for anomaly detection (e.g., 'defect', 'crack')",
            optional=True,
        ),
    }

    OUTPUT_SPECS = {
        "anomaly_mask": PortSpec(
            dtype=torch.bool,
            shape=(-1, -1, -1, 1),
            description="Binary anomaly mask [B, H, W, 1]",
        ),
        "anomaly_scores": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, 1),
            description="Anomaly scores [B, H, W, 1]",
            optional=True,
        ),
    }

    def __init__(
        self,
        model_name: str = "AdaCLIP",
        cache_dir: str | None = None,
        text_prompt: str = "normal: lentils, anomaly: stones",
        **kwargs,
    ) -> None:
        self.text_prompt = text_prompt

        super().__init__(
            model_name=model_name,
            cache_dir=cache_dir,
            text_prompt=text_prompt,
            **kwargs,
        )

    def _load_model(self) -> torch.nn.Module:
        """Load CLIP vision model (image-only, no text encoder)."""
        hf_token = os.getenv("HF_TOKEN")

        try:
            logger.info(f"Loading CLIP vision model locally: {self.model_name}")
            logger.info(f"Cache dir: {self.cache_dir or 'default'}")

            # Use CLIPVisionModel to avoid needing text inputs
            model = CLIPVisionModel.from_pretrained(
                self.model_name,
                token=hf_token,
                cache_dir=self.cache_dir,
            )

            model.eval()
            for param in model.parameters():
                param.requires_grad = False

            logger.success(f"CLIP vision model loaded and frozen: {self.model_name}")
            return model
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load CLIP vision model '{self.model_name}': {exc}\n"
                "Check model name, network access, or HF_TOKEN for private models."
            ) from exc

    def _preprocess_image(self, image: Tensor) -> Tensor:
        """Convert BHWC image to model-ready BCHW, resize to 224x224, and normalize."""
        # Convert BHWC to BCHW
        image = image.permute(0, 3, 1, 2)  # [B, 3, H, W]

        # Normalize to [0, 1] if needed
        if image.max() > 1.0:
            image = image / 255.0

        # Resize to CLIP's expected input size (224x224)
        # This is required for clip-vit-base-patch32 and similar models
        if image.shape[2] != 224 or image.shape[3] != 224:
            image = torch.nn.functional.interpolate(
                image, size=(224, 224), mode="bilinear", align_corners=False
            )

        return image

    def forward(
        self,
        image: Tensor,
        text_prompt: str | None = None,
        context: Any | None = None,  # context captured for pipeline compatibility
        **kwargs: Any,
    ) -> dict[str, Tensor]:
        # Use instance variable if text_prompt not provided
        if text_prompt is None:
            text_prompt = self.text_prompt

        image_processed = self._preprocess_image(image)

        # Ensure model is on the same device as input
        # Get model's current device
        model_device = next(self.model.parameters()).device
        if image_processed.device != model_device:
            # Move model to input device (more efficient than moving data back)
            self.model.to(image_processed.device)

        try:
            # CLIP models expect pixel_values as keyword argument
            # Keep gradients enabled for gradient passthrough
            outputs = self.model(pixel_values=image_processed)

            # CLIPVisionModel returns BaseModelOutputWithPooling with:
            # - pooler_output: [B, 768] global image features
            # - last_hidden_state: [B, 50, 768] patch-level features
            scores = outputs.pooler_output  # [B, 768]

            # Reshape global features to spatial format [B, 1, 1, 1]
            # Use feature norm as anomaly score
            batch_size = scores.shape[0]
            scores = scores.norm(dim=-1, keepdim=True)  # [B, 1]
            scores = scores.view(batch_size, 1, 1, 1)  # [B, 1, 1, 1]

            # Normalize scores to [0, 1] range for interpretability
            # Use min-max normalization to preserve gradients better
            scores_min = scores.min()
            scores_max = scores.max()
            if scores_max > scores_min:
                scores = (scores - scores_min) / (scores_max - scores_min + 1e-8)
            else:
                # All scores are the same - normalize to 0.5
                scores = torch.ones_like(scores) * 0.5

            # Create binary anomaly mask (threshold at 0.5)
            anomaly_mask = scores > 0.5

            return {
                "anomaly_mask": anomaly_mask,
                "anomaly_scores": scores,
            }
        except Exception as exc:  # pragma: no cover - defensive path
            logger.error(f"Local AdaCLIP inference failed: {exc}")
            raise RuntimeError(
                f"AdaCLIP local inference failed: {exc}\n"
                f"Model: {self.model_name}\n"
                f"Input shape: {image.shape}\n"
                f"Text prompt: {text_prompt}"
            ) from exc

    def serialize(self, serial_dir: str) -> dict:
        """Serialize node configuration."""
        return {**self.hparams}

    def load(self, params: dict, serial_dir: str) -> None:
        """Load node configuration."""
        self.model_name = params.get("model_name", self.model_name)
        self.cache_dir = params.get("cache_dir", self.cache_dir)
        self.text_prompt = params.get("text_prompt", self.text_prompt)


__all__ = [
    "HuggingFaceAPINode",
    "AdaCLIPAPINode",
    "HuggingFaceLocalNode",
    "AdaCLIPLocalNode",
]
