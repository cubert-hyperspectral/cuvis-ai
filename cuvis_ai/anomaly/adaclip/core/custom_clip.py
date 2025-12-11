"""Custom CLIP model creation and transforms for AdaCLIP.

Adapted from https://github.com/caoyunkang/AdaCLIP and open_clip.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import urllib
import warnings
from collections.abc import Callable
from copy import deepcopy
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision.transforms import (
    CenterCrop,
    Compose,
    InterpolationMode,
    Normalize,
    Resize,
    ToTensor,
)

from cuvis_ai.anomaly.adaclip.core.clip_model import (
    CLIP,
    build_model_from_openai_state_dict,
    convert_to_custom_text_state_dict,
    convert_weights_to_lp,
    get_cast_dtype,
    resize_pos_embed,
)

# OpenAI CLIP normalization constants
OPENAI_DATASET_MEAN = (0.48145466, 0.4578275, 0.40821073)
OPENAI_DATASET_STD = (0.26862954, 0.26130258, 0.27577711)


def _pcfg(
    url: str = "", hf_hub: str = "", mean: tuple | None = None, std: tuple | None = None
) -> dict:
    """Create a pretrained config dict."""
    return {"url": url, "hf_hub": hf_hub, "mean": mean, "std": std}


_PRETRAINED = {
    "ViT-B-32": {
        "openai": _pcfg(
            "https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt"
        ),
    },
    "ViT-B-16": {
        "openai": _pcfg(
            "https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt"
        ),
    },
    "ViT-L-14": {
        "openai": _pcfg(
            "https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt"
        ),
    },
    "ViT-L-14-336": {
        "openai": _pcfg(
            "https://openaipublic.azureedge.net/clip/models/3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02/ViT-L-14-336px.pt"
        ),
    },
}

# Model configs directory
_MODEL_CONFIG_PATHS = [Path(__file__).parent / "configs"]
_MODEL_CONFIGS: dict[str, dict] = {}


def _get_pretrained_cfg(model: str, tag: str) -> dict:
    """Get pretrained config for a model."""
    if model not in _PRETRAINED:
        return {}
    model_pretrained = _PRETRAINED[model]
    if "openai" in model_pretrained.keys():
        tag = "openai"
    else:
        tag = list(model_pretrained.keys())[0]
    return model_pretrained.get(tag, {})


def _download_pretrained_from_url(url: str, cache_dir: str | None = None) -> str:
    """Download pretrained weights from URL."""
    if not cache_dir:
        cache_dir = os.path.expanduser("~/.cache/clip")
    os.makedirs(cache_dir, exist_ok=True)
    filename = os.path.basename(url)

    if "openaipublic" in url:
        expected_sha256 = url.split("/")[-2]
    elif "mlfoundations" in url:
        expected_sha256 = os.path.splitext(filename)[0].split("-")[-1]
    else:
        expected_sha256 = ""

    download_target = os.path.join(cache_dir, filename)

    if os.path.exists(download_target) and not os.path.isfile(download_target):
        raise RuntimeError(f"{download_target} exists and is not a regular file")

    if os.path.isfile(download_target):
        if expected_sha256:
            if (
                hashlib.sha256(open(download_target, "rb").read())
                .hexdigest()
                .startswith(expected_sha256)
            ):
                return download_target
            else:
                warnings.warn(
                    f"{download_target} exists, but the SHA256 checksum does not match; re-downloading",
                    stacklevel=2,
                )
        else:
            return download_target

    from tqdm import tqdm

    with urllib.request.urlopen(url) as source, open(download_target, "wb") as output:
        with tqdm(
            total=int(source.headers.get("Content-Length")), ncols=80, unit="iB", unit_scale=True
        ) as loop:
            while True:
                buffer = source.read(8192)
                if not buffer:
                    break
                output.write(buffer)
                loop.update(len(buffer))

    if expected_sha256 and not hashlib.sha256(
        open(download_target, "rb").read()
    ).hexdigest().startswith(expected_sha256):
        raise RuntimeError("Model has been downloaded but the SHA256 checksum does not match")

    return download_target


def _download_pretrained(cfg: dict, cache_dir: str | None = None) -> str:
    """Download pretrained weights."""
    target = ""
    if not cfg:
        return target

    download_url = cfg.get("url", "")
    if download_url:
        target = _download_pretrained_from_url(download_url, cache_dir=cache_dir)

    return target


def _convert_to_rgb(image) -> Image.Image:
    """Convert image to RGB."""
    return image.convert("RGB")


def image_transform(
    image_size: int,
    is_train: bool = False,
    mean: tuple[float, ...] | None = None,
    std: tuple[float, ...] | None = None,
) -> Callable:
    """Create image transform for CLIP.

    Args:
        image_size: Target image size.
        is_train: Whether for training (not used for inference).
        mean: Normalization mean.
        std: Normalization std.

    Returns:
        Transform function.
    """
    mean = mean or OPENAI_DATASET_MEAN
    if not isinstance(mean, (list, tuple)):
        mean = (mean,) * 3

    std = std or OPENAI_DATASET_STD
    if not isinstance(std, (list, tuple)):
        std = (std,) * 3

    if isinstance(image_size, (list, tuple)) and image_size[0] == image_size[1]:
        image_size = image_size[0]

    normalize = Normalize(mean=mean, std=std)
    transforms = [
        Resize(image_size, interpolation=InterpolationMode.BICUBIC),
        CenterCrop(image_size),
        _convert_to_rgb,
        ToTensor(),
        normalize,
    ]
    return Compose(transforms)


def _rescan_model_configs() -> None:
    """Scan for model configuration files."""
    global _MODEL_CONFIGS

    config_ext = (".json",)
    config_files = []
    for config_path in _MODEL_CONFIG_PATHS:
        if config_path.is_file() and config_path.suffix in config_ext:
            config_files.append(config_path)
        elif config_path.is_dir():
            for ext in config_ext:
                config_files.extend(config_path.glob(f"*{ext}"))

    for cf in config_files:
        with open(cf) as f:
            model_cfg = json.load(f)
            if all(a in model_cfg for a in ("embed_dim", "vision_cfg", "text_cfg")):
                _MODEL_CONFIGS[cf.stem] = model_cfg


# Initial populate of model config registry
_rescan_model_configs()


def get_model_config(model_name: str) -> dict | None:
    """Get model config by name."""
    if model_name in _MODEL_CONFIGS:
        return deepcopy(_MODEL_CONFIGS[model_name])
    return None


def load_openai_model(
    name: str,
    precision: str | None = None,
    device: str | torch.device | None = None,
    jit: bool = True,
    cache_dir: str | None = None,
) -> CLIP:
    """Load a CLIP model from OpenAI.

    Args:
        name: Model name.
        precision: Model precision ('fp32', 'fp16', 'bf16').
        device: Device to load model on.
        jit: Whether to load JIT model.
        cache_dir: Directory to cache downloaded weights.

    Returns:
        CLIP model.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if precision is None:
        precision = "fp32" if device == "cpu" else "fp16"

    cfg = _get_pretrained_cfg(name, "openai")
    if cfg:
        model_path = _download_pretrained(cfg, cache_dir=cache_dir)
    elif os.path.isfile(name):
        model_path = name
    else:
        raise RuntimeError(f"Model {name} not found")

    try:
        model = torch.jit.load(model_path, map_location=device if jit else "cpu").eval()
        state_dict = None
    except RuntimeError:
        if jit:
            warnings.warn(
                f"File {model_path} is not a JIT archive. Loading as a state dict instead",
                stacklevel=2,
            )
            jit = False
        state_dict = torch.load(model_path, map_location="cpu")

    if not jit:
        cast_dtype = get_cast_dtype(precision)
        try:
            model = build_model_from_openai_state_dict(
                state_dict or model.state_dict(), cast_dtype=cast_dtype
            )
        except KeyError:
            sd = {k[7:]: v for k, v in state_dict["state_dict"].items()}
            model = build_model_from_openai_state_dict(sd, cast_dtype=cast_dtype)

        model = model.to(device)
        if precision.startswith("amp") or precision == "fp32":
            model.float()
        elif precision == "bf16":
            convert_weights_to_lp(model, dtype=torch.bfloat16)

        return model

    model.visual.image_size = model.input_resolution.item()
    return model


def load_state_dict(checkpoint_path: str, map_location: str = "cpu") -> dict:
    """Load state dict from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint
    if next(iter(state_dict.items()))[0].startswith("module"):
        state_dict = {k[7:]: v for k, v in state_dict.items()}
    return state_dict


def load_checkpoint(model: nn.Module, checkpoint_path: str, strict: bool = True) -> None:
    """Load checkpoint into model."""
    state_dict = load_state_dict(checkpoint_path)
    if "positional_embedding" in state_dict and not hasattr(model, "positional_embedding"):
        state_dict = convert_to_custom_text_state_dict(state_dict)
    resize_pos_embed(state_dict, model)
    incompatible_keys = model.load_state_dict(state_dict, strict=strict)
    return incompatible_keys


def create_model(
    model_name: str,
    img_size: int,
    pretrained: str | None = None,
    precision: str = "fp32",
    device: str | torch.device = "cpu",
    jit: bool = False,
    cache_dir: str | None = None,
) -> CLIP:
    """Create a CLIP model.

    Args:
        model_name: Model name (e.g., 'ViT-L-14-336').
        img_size: Image size.
        pretrained: Pretrained source (e.g., 'openai').
        precision: Model precision.
        device: Device.
        jit: Whether to JIT compile.
        cache_dir: Cache directory.

    Returns:
        CLIP model.
    """
    if model_name.count("ViT") < 1:
        raise NotImplementedError("Only support ViT model")

    model_name = model_name.replace("/", "-")
    model_cfg = get_model_config(model_name)

    if isinstance(device, str):
        device = torch.device(device)

    if pretrained and pretrained.lower() == "openai":
        logging.info(f"Loading pretrained {model_name} from OpenAI.")
    else:
        raise ValueError("Only 'openai' pretrained weights are supported")

    if model_cfg is None:
        raise ValueError(f"Model config for {model_name} not found")

    model_cfg["vision_cfg"]["image_size"] = img_size
    cast_dtype = get_cast_dtype(precision)

    model_pre = load_openai_model(
        model_name,
        precision=precision,
        device=device,
        jit=jit,
        cache_dir=cache_dir,
    )
    state_dict = model_pre.state_dict()

    model = CLIP(**model_cfg, cast_dtype=cast_dtype)
    resize_pos_embed(state_dict, model)
    model.load_state_dict(state_dict, strict=True)
    model.to(device=device)

    if precision in ("fp16", "bf16"):
        convert_weights_to_lp(model, dtype=torch.bfloat16 if precision == "bf16" else torch.float16)

    model.visual.image_mean = OPENAI_DATASET_MEAN
    model.visual.image_std = OPENAI_DATASET_STD

    if jit:
        model = torch.jit.script(model)

    return model


def create_model_and_transforms(
    model_name: str,
    img_size: int,
    pretrained: str | None = None,
    precision: str = "fp32",
    device: str | torch.device = "cpu",
    jit: bool = False,
    image_mean: tuple[float, ...] | None = None,
    image_std: tuple[float, ...] | None = None,
    cache_dir: str | None = None,
) -> tuple[CLIP, Callable, Callable]:
    """Create CLIP model and transforms.

    Args:
        model_name: Model name.
        img_size: Image size.
        pretrained: Pretrained source.
        precision: Model precision.
        device: Device.
        jit: Whether to JIT compile.
        image_mean: Normalization mean.
        image_std: Normalization std.
        cache_dir: Cache directory.

    Returns:
        Tuple of (model, train_transform, val_transform).
    """
    model = create_model(
        model_name,
        img_size,
        pretrained,
        precision=precision,
        device=device,
        jit=jit,
        cache_dir=cache_dir,
    )

    image_mean = image_mean or getattr(model.visual, "image_mean", None)
    image_std = image_std or getattr(model.visual, "image_std", None)
    preprocess_train = image_transform(
        model.visual.image_size,
        is_train=True,
        mean=image_mean,
        std=image_std,
    )
    preprocess_val = image_transform(
        model.visual.image_size,
        is_train=False,
        mean=image_mean,
        std=image_std,
    )

    return model, preprocess_train, preprocess_val


__all__ = [
    "OPENAI_DATASET_MEAN",
    "OPENAI_DATASET_STD",
    "create_model",
    "create_model_and_transforms",
    "get_model_config",
    "image_transform",
    "load_checkpoint",
    "load_openai_model",
    "load_state_dict",
]
