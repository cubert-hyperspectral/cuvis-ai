"""CLIP Model for AdaCLIP.

Adapted from https://github.com/openai/CLIP and https://github.com/caoyunkang/AdaCLIP
Originally MIT License, Copyright (c) 2021 OpenAI.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from cuvis_ai.anomaly.adaclip.core.transformer import (
    Attention,
    LayerNorm,
    LayerNormFp32,
    QuickGELU,
    TextTransformer,
    VisionTransformer,
)
from cuvis_ai.anomaly.adaclip.core.utils import to_2tuple


@dataclass
class CLIPVisionCfg:
    """Configuration for CLIP vision encoder."""

    layers: tuple[int, int, int, int] | int = 12
    width: int = 768
    head_width: int = 64
    mlp_ratio: float = 4.0
    patch_size: int = 16
    image_size: tuple[int, int] | int = 224
    ls_init_value: float | None = None
    patch_dropout: float = 0.0
    input_patchnorm: bool = False
    global_average_pool: bool = False
    attentional_pool: bool = False
    n_queries: int = 256
    attn_pooler_heads: int = 8
    timm_model_name: str | None = None
    timm_model_pretrained: bool = False
    timm_pool: str = "avg"
    timm_proj: str = "linear"
    timm_proj_bias: bool = False
    timm_drop: float = 0.0
    timm_drop_path: float | None = None
    output_tokens: bool = False


@dataclass
class CLIPTextCfg:
    """Configuration for CLIP text encoder."""

    context_length: int = 77
    vocab_size: int = 49408
    width: int = 512
    heads: int = 8
    layers: int = 12
    ls_init_value: float | None = None
    hf_model_name: str | None = None
    hf_tokenizer_name: str | None = None
    hf_model_pretrained: bool = True
    proj: str = "mlp"
    pooler_type: str = "mean_pooler"
    embed_cls: bool = False
    pad_id: int = 0
    output_tokens: bool = False


def get_cast_dtype(precision: str) -> torch.dtype | None:
    """Get cast dtype from precision string."""
    cast_dtype = None
    if precision == "bf16":
        cast_dtype = torch.bfloat16
    elif precision == "fp16":
        cast_dtype = torch.float16
    return cast_dtype


def _build_vision_tower(
    embed_dim: int,
    vision_cfg: CLIPVisionCfg,
    quick_gelu: bool = False,
    cast_dtype: torch.dtype | None = None,
) -> VisionTransformer:
    """Build vision tower from config."""
    if isinstance(vision_cfg, dict):
        vision_cfg = CLIPVisionCfg(**vision_cfg)

    act_layer = QuickGELU if quick_gelu else nn.GELU

    vision_heads = vision_cfg.width // vision_cfg.head_width
    norm_layer = LayerNormFp32 if cast_dtype in (torch.float16, torch.bfloat16) else LayerNorm
    visual = VisionTransformer(
        image_size=vision_cfg.image_size,
        patch_size=vision_cfg.patch_size,
        width=vision_cfg.width,
        layers=vision_cfg.layers,
        heads=vision_heads,
        mlp_ratio=vision_cfg.mlp_ratio,
        ls_init_value=vision_cfg.ls_init_value,
        patch_dropout=vision_cfg.patch_dropout,
        input_patchnorm=vision_cfg.input_patchnorm,
        global_average_pool=vision_cfg.global_average_pool,
        attentional_pool=vision_cfg.attentional_pool,
        n_queries=vision_cfg.n_queries,
        attn_pooler_heads=vision_cfg.attn_pooler_heads,
        output_tokens=vision_cfg.output_tokens,
        output_dim=embed_dim,
        act_layer=act_layer,
        norm_layer=norm_layer,
    )

    return visual


def _build_text_tower(
    embed_dim: int,
    text_cfg: CLIPTextCfg,
    quick_gelu: bool = False,
    cast_dtype: torch.dtype | None = None,
) -> TextTransformer:
    """Build text tower from config."""
    if isinstance(text_cfg, dict):
        text_cfg = CLIPTextCfg(**text_cfg)

    act_layer = QuickGELU if quick_gelu else nn.GELU
    norm_layer = LayerNormFp32 if cast_dtype in (torch.float16, torch.bfloat16) else LayerNorm

    text = TextTransformer(
        context_length=text_cfg.context_length,
        vocab_size=text_cfg.vocab_size,
        width=text_cfg.width,
        heads=text_cfg.heads,
        layers=text_cfg.layers,
        ls_init_value=text_cfg.ls_init_value,
        output_dim=embed_dim,
        embed_cls=text_cfg.embed_cls,
        output_tokens=text_cfg.output_tokens,
        pad_id=text_cfg.pad_id,
        act_layer=act_layer,
        norm_layer=norm_layer,
    )

    return text


class CLIP(nn.Module):
    """CLIP model combining vision and text encoders."""

    output_dict: torch.jit.Final[bool]

    def __init__(
        self,
        embed_dim: int,
        vision_cfg: CLIPVisionCfg,
        text_cfg: CLIPTextCfg,
        quick_gelu: bool = False,
        cast_dtype: torch.dtype | None = None,
        output_dict: bool = False,
    ) -> None:
        super().__init__()
        self.output_dict = output_dict
        self.visual = _build_vision_tower(embed_dim, vision_cfg, quick_gelu, cast_dtype)
        text = _build_text_tower(embed_dim, text_cfg, quick_gelu, cast_dtype)
        self.transformer = text.transformer
        self.vocab_size = text.vocab_size
        self.token_embedding = text.token_embedding
        self.positional_embedding = text.positional_embedding
        self.ln_final = text.ln_final
        self.text_projection = text.text_projection
        self.register_buffer("attn_mask", text.attn_mask, persistent=False)

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def lock_image_tower(self, unlocked_groups: int = 0, freeze_bn_stats: bool = False) -> None:
        """Lock image tower for fine-tuning."""
        self.visual.lock(unlocked_groups=unlocked_groups, freeze_bn_stats=freeze_bn_stats)

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable: bool = True) -> None:
        """Enable gradient checkpointing."""
        self.visual.set_grad_checkpointing(enable)
        self.transformer.grad_checkpointing = enable

    def encode_image(self, image: torch.Tensor, out_layers: list) -> tuple[torch.Tensor, list]:
        """Encode image to features."""
        x = image

        if self.visual.input_patchnorm:
            x = x.reshape(
                x.shape[0],
                x.shape[1],
                self.visual.grid_size[0],
                self.visual.patch_size[0],
                self.visual.grid_size[1],
                self.visual.patch_size[1],
            )
            x = x.permute(0, 2, 4, 1, 3, 5)
            x = x.reshape(x.shape[0], self.visual.grid_size[0] * self.visual.grid_size[1], -1)
            x = self.visual.patchnorm_pre_ln(x)
            x = self.visual.conv1(x)
        else:
            x = self.visual.conv1(x)
            x = x.reshape(x.shape[0], x.shape[1], -1)
            x = x.permute(0, 2, 1)

        x = torch.cat(
            [
                self.visual.class_embedding.to(x.dtype)
                + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
                x,
            ],
            dim=1,
        )

        x = x + self.visual.positional_embedding.to(x.dtype)
        x = self.visual.patch_dropout(x)
        x = self.visual.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND

        patch_tokens = []
        idx = 0
        for r in self.visual.transformer.resblocks:
            idx += 1
            x, attn_tmp = r(x, attn_mask=None)
            if idx in out_layers:
                patch_tokens.append(x)

        x = x.permute(1, 0, 2)  # LND -> NLD
        patch_tokens = [patch_tokens[t].permute(1, 0, 2) for t in range(len(patch_tokens))]

        if self.visual.attn_pool is not None:
            x = self.visual.attn_pool(x)
            x = self.visual.ln_post(x)
            pooled, tokens = self.visual._global_pool(x)
        else:
            pooled, tokens = self.visual._global_pool(x)
            pooled = self.visual.ln_post(pooled)

        if self.visual.proj is not None:
            pooled = pooled @ self.visual.proj

        if self.visual.output_tokens:
            return pooled, patch_tokens

        return pooled, patch_tokens

    def encode_text(self, text: torch.Tensor) -> torch.Tensor:
        """Encode text to features."""
        cast_dtype = self.transformer.get_cast_dtype()

        x = self.token_embedding(text).to(cast_dtype)
        x = x + self.positional_embedding.to(cast_dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND

        for r in self.visual.transformer.resblocks:
            x, attn_tmp = r(x, attn_mask=self.attn_mask)

        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)

        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
        return x


def convert_weights_to_lp(model: nn.Module, dtype: torch.dtype = torch.float16) -> None:
    """Convert applicable model parameters to low-precision (bf16 or fp16)."""

    def _convert_weights(layer) -> None:
        if isinstance(layer, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            layer.weight.data = layer.weight.data.to(dtype)
            if layer.bias is not None:
                layer.bias.data = layer.bias.data.to(dtype)

        if isinstance(layer, (nn.MultiheadAttention, Attention)):
            for attr in [
                *[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]],
                "in_proj_bias",
                "bias_k",
                "bias_v",
            ]:
                tensor = getattr(layer, attr, None)
                if tensor is not None:
                    tensor.data = tensor.data.to(dtype)

        for name in ["text_projection", "proj"]:
            if hasattr(layer, name):
                attr = getattr(layer, name)
                if attr is not None:
                    attr.data = attr.data.to(dtype)

    model.apply(_convert_weights)


convert_weights_to_fp16 = convert_weights_to_lp


def convert_to_custom_text_state_dict(state_dict: dict) -> dict:
    """Convert old format state_dict to new format."""
    if "text_projection" in state_dict:
        new_state_dict = {}
        for k, v in state_dict.items():
            if any(
                k.startswith(p)
                for p in (
                    "text_projection",
                    "positional_embedding",
                    "token_embedding",
                    "transformer",
                    "ln_final",
                )
            ):
                k = "text." + k
            new_state_dict[k] = v
        return new_state_dict
    return state_dict


def build_model_from_openai_state_dict(
    state_dict: dict,
    quick_gelu: bool = True,
    cast_dtype: torch.dtype = torch.float16,
) -> CLIP:
    """Build CLIP model from OpenAI state dict."""
    vit = "visual.proj" in state_dict

    if vit:
        vision_width = state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len(
            [
                k
                for k in state_dict.keys()
                if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")
            ]
        )
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_size = vision_patch_size * grid_size
    else:
        counts: list = [
            len({k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}")})
            for b in [1, 2, 3, 4]
        ]
        vision_layers = tuple(counts)
        vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
        output_width = round(
            (state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5
        )
        vision_patch_size = None
        assert output_width**2 + 1 == state_dict["visual.attnpool.positional_embedding"].shape[0]
        image_size = output_width * 32

    embed_dim = state_dict["text_projection"].shape[1]
    context_length = state_dict["positional_embedding"].shape[0]
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(
        {k.split(".")[2] for k in state_dict if k.startswith("transformer.resblocks")}
    )

    vision_cfg = CLIPVisionCfg(
        layers=vision_layers,
        width=vision_width,
        patch_size=vision_patch_size,
        image_size=image_size,
    )
    text_cfg = CLIPTextCfg(
        context_length=context_length,
        vocab_size=vocab_size,
        width=transformer_width,
        heads=transformer_heads,
        layers=transformer_layers,
    )
    model = CLIP(
        embed_dim,
        vision_cfg=vision_cfg,
        text_cfg=text_cfg,
        quick_gelu=quick_gelu,
        cast_dtype=cast_dtype,
    )

    for key in ["input_resolution", "context_length", "vocab_size"]:
        state_dict.pop(key, None)

    convert_weights_to_fp16(model)
    model.load_state_dict(state_dict)
    return model.eval()


def resize_pos_embed(state_dict: dict, model: CLIP, interpolation: str = "bicubic") -> None:
    """Rescale position embeddings when loading from state_dict."""
    old_pos_embed = state_dict.get("visual.positional_embedding", None)
    if old_pos_embed is None or not hasattr(model.visual, "grid_size"):
        return
    grid_size = to_2tuple(model.visual.grid_size)
    extra_tokens = 1
    new_seq_len = grid_size[0] * grid_size[1] + extra_tokens
    if new_seq_len == old_pos_embed.shape[0]:
        return

    if extra_tokens:
        pos_emb_tok, pos_emb_img = old_pos_embed[:extra_tokens], old_pos_embed[extra_tokens:]
    else:
        pos_emb_tok, pos_emb_img = None, old_pos_embed
    old_grid_size = to_2tuple(int(math.sqrt(len(pos_emb_img))))

    logging.info("Resizing position embedding grid-size from %s to %s", old_grid_size, grid_size)
    pos_emb_img = pos_emb_img.reshape(1, old_grid_size[0], old_grid_size[1], -1).permute(0, 3, 1, 2)
    pos_emb_img = F.interpolate(
        pos_emb_img,
        size=grid_size,
        mode=interpolation,
        align_corners=False,
    )
    pos_emb_img = pos_emb_img.permute(0, 2, 3, 1).reshape(1, grid_size[0] * grid_size[1], -1)[0]
    if pos_emb_tok is not None:
        new_pos_embed = torch.cat([pos_emb_tok, pos_emb_img], dim=0)
    else:
        new_pos_embed = pos_emb_img
    state_dict["visual.positional_embedding"] = new_pos_embed


__all__ = [
    "CLIP",
    "CLIPTextCfg",
    "CLIPVisionCfg",
    "build_model_from_openai_state_dict",
    "convert_to_custom_text_state_dict",
    "convert_weights_to_fp16",
    "convert_weights_to_lp",
    "get_cast_dtype",
    "resize_pos_embed",
]
