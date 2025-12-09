"""AdaCLIP model for zero-shot anomaly detection.

Adapted from https://github.com/caoyunkang/AdaCLIP

Reference:
    Cao, Y., et al. "AdaCLIP: Adapting CLIP with Hybrid Learnable Prompts for
    Zero-Shot Anomaly Detection." ECCV 2024.
"""

from __future__ import annotations

import numpy as np
import torch
from packaging import version
from sklearn.cluster import KMeans
from torch import nn
from torch.nn import functional as F

from cuvis_ai.anomaly.adaclip.core.clip_model import CLIP
from cuvis_ai.anomaly.adaclip.core.simple_tokenizer import SimpleTokenizer as _Tokenizer


class ProjectLayer(nn.Module):
    """Projection layer for patch tokens."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_replicas: int,
        stack: bool = False,
        is_array: bool = True,
    ) -> None:
        super().__init__()
        self.head = nn.ModuleList([nn.Linear(input_dim, output_dim) for _ in range(num_replicas)])
        self.num_replicas = num_replicas
        self.stack = stack
        self.is_array = is_array

    def forward(
        self, tokens: list[torch.Tensor] | torch.Tensor
    ) -> list[torch.Tensor] | torch.Tensor:
        out_tokens = []
        for i in range(self.num_replicas):
            if self.is_array:
                temp = self.head[i](tokens[i][:, 1:, :])  # exclude class token
            else:
                temp = self.head[i](tokens)
            out_tokens.append(temp)

        if self.stack:
            out_tokens = torch.stack(out_tokens, dim=1)

        return out_tokens


class PromptLayer(nn.Module):
    """Learnable prompt layer for text/visual encoder."""

    def __init__(
        self,
        channel: int,
        length: int,
        depth: int,
        is_text: bool,
        prompting_type: str,
        enabled: bool = True,
    ) -> None:
        super().__init__()
        self.channel = channel
        self.length = length
        self.depth = depth
        self.is_text = is_text
        self.enabled = enabled
        self.prompting_type = prompting_type

        if self.enabled:
            if "S" in prompting_type:  # static prompts
                self.static_prompts = nn.ParameterList(
                    [
                        nn.Parameter(torch.empty(self.length, self.channel))
                        for _ in range(self.depth)
                    ]
                )
                for single_para in self.static_prompts:
                    nn.init.normal_(single_para, std=0.02)

            if "D" in prompting_type:  # dynamic prompts
                self.dynamic_prompts = [0.0]  # placeholder

    def set_dynamic_prompts(self, dynamic_prompts: torch.Tensor) -> None:
        self.dynamic_prompts = dynamic_prompts

    def forward_text(
        self,
        resblock: nn.Module,
        indx: int,
        x: torch.Tensor,
        k_x: torch.Tensor | None = None,
        v_x: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
    ) -> tuple:
        if self.enabled:
            length = self.length

            if indx < self.depth:
                if "S" in self.prompting_type and "D" in self.prompting_type:
                    static_prompts = (
                        self.static_prompts[indx].unsqueeze(0).expand(x.shape[1], -1, -1)
                    )
                    # Expand dynamic_prompts to match text batch size (x.shape[1])
                    # dynamic_prompts has shape (image_batch, prompting_length, channel)
                    # but text processing creates x.shape[1] variations (typically 28)
                    # Use the first image's dynamic prompts and expand to match text variations
                    # This is consistent with text_features being shared across the image batch
                    if isinstance(self.dynamic_prompts, torch.Tensor):
                        if self.dynamic_prompts.shape[0] != x.shape[1]:
                            dynamic_prompts_expanded = self.dynamic_prompts[0:1].expand(
                                x.shape[1], -1, -1
                            )
                        else:
                            dynamic_prompts_expanded = self.dynamic_prompts
                    else:
                        dynamic_prompts_expanded = self.dynamic_prompts
                    textual_context = dynamic_prompts_expanded + static_prompts
                elif "S" in self.prompting_type:
                    static_prompts = (
                        self.static_prompts[indx].unsqueeze(0).expand(x.shape[1], -1, -1)
                    )
                    textual_context = static_prompts
                elif "D" in self.prompting_type:
                    # Expand dynamic_prompts to match text batch size
                    if isinstance(self.dynamic_prompts, torch.Tensor):
                        if self.dynamic_prompts.shape[0] != x.shape[1]:
                            dynamic_prompts_expanded = self.dynamic_prompts[0:1].expand(
                                x.shape[1], -1, -1
                            )
                        else:
                            dynamic_prompts_expanded = self.dynamic_prompts
                    else:
                        dynamic_prompts_expanded = self.dynamic_prompts
                    textual_context = dynamic_prompts_expanded
                else:
                    raise NotImplementedError("Choose at least one type of prompts")

            if indx == 0:
                x = x
            else:
                if indx < self.depth:
                    prefix = x[:1, :, :]
                    suffix = x[1 + length :, :, :]
                    textual_context = textual_context.permute(1, 0, 2).half()
                    x = torch.cat([prefix, textual_context, suffix], dim=0)
                else:
                    x = x
        else:
            x = x

        x, attn_tmp = resblock(q_x=x, k_x=k_x, v_x=v_x, attn_mask=attn_mask)
        return x, attn_tmp

    def forward_visual(
        self,
        resblock: nn.Module,
        indx: int,
        x: torch.Tensor,
        k_x: torch.Tensor | None = None,
        v_x: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
    ) -> tuple:
        if self.enabled:
            length = self.length

            if indx < self.depth:
                if "S" in self.prompting_type and "D" in self.prompting_type:
                    static_prompts = (
                        self.static_prompts[indx].unsqueeze(0).expand(x.shape[1], -1, -1)
                    )
                    # For visual prompts, x.shape[1] is the image batch size, which should match
                    # dynamic_prompts shape, but handle edge cases for consistency
                    if isinstance(self.dynamic_prompts, torch.Tensor):
                        if self.dynamic_prompts.shape[0] != x.shape[1]:
                            dynamic_prompts_expanded = self.dynamic_prompts[0:1].expand(
                                x.shape[1], -1, -1
                            )
                        else:
                            dynamic_prompts_expanded = self.dynamic_prompts
                    else:
                        dynamic_prompts_expanded = self.dynamic_prompts
                    visual_context = dynamic_prompts_expanded + static_prompts
                elif "S" in self.prompting_type:
                    static_prompts = (
                        self.static_prompts[indx].unsqueeze(0).expand(x.shape[1], -1, -1)
                    )
                    visual_context = static_prompts
                elif "D" in self.prompting_type:
                    # For visual prompts, expand if needed for consistency
                    if isinstance(self.dynamic_prompts, torch.Tensor):
                        if self.dynamic_prompts.shape[0] != x.shape[1]:
                            dynamic_prompts_expanded = self.dynamic_prompts[0:1].expand(
                                x.shape[1], -1, -1
                            )
                        else:
                            dynamic_prompts_expanded = self.dynamic_prompts
                    else:
                        dynamic_prompts_expanded = self.dynamic_prompts
                    visual_context = dynamic_prompts_expanded
                else:
                    raise NotImplementedError("Choose at least one type of prompts")

            if indx == 0:
                visual_context = visual_context.permute(1, 0, 2).half()
                x = torch.cat([x, visual_context], dim=0)
            else:
                if indx < self.depth:
                    prefix = x[0 : x.shape[0] - length, :, :]
                    visual_context = visual_context.permute(1, 0, 2).half()
                    x = torch.cat([prefix, visual_context], dim=0)
                else:
                    x = x
        else:
            x = x

        x, attn_tmp = resblock(q_x=x, k_x=k_x, v_x=v_x, attn_mask=attn_mask)

        if self.enabled:
            tokens = x[: x.shape[0] - length, :, :]
        else:
            tokens = x

        return x, tokens, attn_tmp

    def forward(
        self,
        resblock: nn.Module,
        indx: int,
        x: torch.Tensor,
        k_x: torch.Tensor | None = None,
        v_x: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        if self.is_text:
            return self.forward_text(resblock, indx, x, k_x, v_x, attn_mask)
        else:
            return self.forward_visual(resblock, indx, x, k_x, v_x, attn_mask)


class TextEmbeddingLayer(nn.Module):
    """Text embedding layer with prompt templates."""

    def __init__(self, fixed: bool) -> None:
        super().__init__()
        self.tokenizer = _Tokenizer()
        self.ensemble_text_features = {}
        self.prompt_normal = [
            "{}",
            "flawless {}",
            "perfect {}",
            "unblemished {}",
            "{} without flaw",
            "{} without defect",
            "{} without damage",
        ]
        self.prompt_abnormal = [
            "damaged {}",
            "broken {}",
            "{} with flaw",
            "{} with defect",
            "{} with damage",
        ]
        self.prompt_state = [self.prompt_normal, self.prompt_abnormal]
        self.prompt_templates = [
            "a bad photo of a {}.",
            "a low resolution photo of the {}.",
            "a bad photo of the {}.",
            "a cropped photo of the {}.",
        ]
        self.fixed = fixed

    def tokenize(
        self, texts: str | list[str], context_length: int = 77, truncate: bool = False
    ) -> torch.IntTensor | torch.LongTensor:
        if isinstance(texts, str):
            texts = [texts]

        sot_token = self.tokenizer.encoder["<|startoftext|>"]
        eot_token = self.tokenizer.encoder["<|endoftext|>"]
        all_tokens = [[sot_token] + self.tokenizer.encode(text) + [eot_token] for text in texts]

        if version.parse(torch.__version__) < version.parse("1.8.0"):
            result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)
        else:
            result = torch.zeros(len(all_tokens), context_length, dtype=torch.int)

        for i, tokens in enumerate(all_tokens):
            if len(tokens) > context_length:
                if truncate:
                    tokens = tokens[:context_length]
                    tokens[-1] = eot_token
                else:
                    raise RuntimeError(
                        f"Input {texts[i]} is too long for context length {context_length}"
                    )
            result[i, : len(tokens)] = torch.tensor(tokens)

        return result

    def forward(self, model: nn.Module, texts: list[str], device: torch.device) -> torch.Tensor:
        text_feature_list = []

        for _indx, text in enumerate(texts):
            if self.fixed:
                if self.ensemble_text_features.get(text) is None:
                    text_features = self.encode_text(model, text, device)
                    self.ensemble_text_features[text] = text_features
                else:
                    text_features = self.ensemble_text_features[text]
            else:
                text_features = self.encode_text(model, text, device)
                self.ensemble_text_features[text] = text_features

            text_feature_list.append(text_features)

        text_features = torch.stack(text_feature_list, dim=0)
        text_features = F.normalize(text_features, dim=1)

        return text_features

    def encode_text(self, model: nn.Module, text: str, device: torch.device) -> torch.Tensor:
        text_features = []
        for i in range(len(self.prompt_state)):
            text = text.replace("-", " ")
            prompted_state = [state.format(text) for state in self.prompt_state[i]]
            prompted_sentence = []
            for s in prompted_state:
                for template in self.prompt_templates:
                    prompted_sentence.append(template.format(s))
            prompted_sentence = self.tokenize(prompted_sentence, context_length=77).to(device)

            class_embeddings = model.encode_text(prompted_sentence)
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            text_features.append(class_embedding)

        text_features = torch.stack(text_features, dim=1)
        return text_features


class HybridSemanticFusion(nn.Module):
    """Hybrid Semantic Fusion module for aggregating patch features."""

    def __init__(self, k_clusters: int) -> None:
        super().__init__()
        self.k_clusters = k_clusters
        self.n_aggregate_patch_tokens = k_clusters * 5
        self.cluster_performer = KMeans(n_clusters=self.k_clusters, n_init="auto")

    def forward(self, patch_tokens: list, anomaly_maps: list) -> torch.Tensor:
        anomaly_map = torch.mean(torch.stack(anomaly_maps, dim=1), dim=1)
        anomaly_map = torch.softmax(anomaly_map, dim=2)[:, :, 1]  # B, L

        # extract most abnormal features
        selected_abnormal_tokens = []
        k = min(anomaly_map.shape[1], self.n_aggregate_patch_tokens)
        top_k_indices = torch.topk(anomaly_map, k=k, dim=1).indices
        for layer in range(len(patch_tokens)):
            selected_tokens = patch_tokens[layer].gather(
                dim=1,
                index=top_k_indices.unsqueeze(-1).expand(-1, -1, patch_tokens[layer].shape[-1]),
            )
            selected_abnormal_tokens.append(selected_tokens)

        # use kmeans to extract centroids
        stacked_data = torch.cat(selected_abnormal_tokens, dim=2)

        batch_cluster_centers = []
        for b in range(stacked_data.shape[0]):
            cluster_labels = self.cluster_performer.fit_predict(
                stacked_data[b, :, :].detach().cpu().numpy()
            )

            cluster_centers = []
            for cluster_id in range(self.k_clusters):
                collected_cluster_data = []
                for abnormal_tokens in selected_abnormal_tokens:
                    cluster_data = abnormal_tokens[b, :, :][cluster_labels == cluster_id]
                    collected_cluster_data.append(cluster_data)
                collected_cluster_data = torch.cat(collected_cluster_data, dim=0)
                cluster_center = torch.mean(collected_cluster_data, dim=0, keepdim=True)
                cluster_centers.append(cluster_center)

            cluster_centers = torch.cat(cluster_centers, dim=0)
            cluster_centers = torch.mean(cluster_centers, dim=0)
            batch_cluster_centers.append(cluster_centers)

        batch_cluster_centers = torch.stack(batch_cluster_centers, dim=0)
        batch_cluster_centers = F.normalize(batch_cluster_centers, dim=1)

        return batch_cluster_centers


class AdaCLIP(nn.Module):
    """AdaCLIP model for zero-shot anomaly detection.

    This model adapts CLIP with hybrid learnable prompts for anomaly detection.
    """

    def __init__(
        self,
        freeze_clip: CLIP,
        text_channel: int,
        visual_channel: int,
        prompting_length: int,
        prompting_depth: int,
        prompting_branch: str,
        prompting_type: str,
        use_hsf: bool,
        k_clusters: int,
        output_layers: list,
        device: str,
        image_size: int,
    ) -> None:
        super().__init__()
        self.freeze_clip = freeze_clip

        self.visual = self.freeze_clip.visual
        self.transformer = self.freeze_clip.transformer
        self.token_embedding = self.freeze_clip.token_embedding
        self.positional_embedding = self.freeze_clip.positional_embedding
        self.ln_final = self.freeze_clip.ln_final
        self.text_projection = self.freeze_clip.text_projection
        self.attn_mask = self.freeze_clip.attn_mask

        self.output_layers = output_layers
        self.prompting_branch = prompting_branch
        self.prompting_type = prompting_type
        self.prompting_depth = prompting_depth
        self.prompting_length = prompting_length
        self.use_hsf = use_hsf
        self.k_clusters = k_clusters

        self.enable_text_prompt = "L" in self.prompting_branch
        self.enable_visual_prompt = "V" in self.prompting_branch

        self.text_embedding_layer = TextEmbeddingLayer(fixed=(not self.enable_text_prompt))
        self.text_prompter = PromptLayer(
            text_channel,
            prompting_length,
            prompting_depth,
            is_text=True,
            prompting_type=prompting_type,
            enabled=self.enable_text_prompt,
        )
        self.visual_prompter = PromptLayer(
            visual_channel,
            prompting_length,
            prompting_depth,
            is_text=False,
            prompting_type=prompting_type,
            enabled=self.enable_visual_prompt,
        )

        self.patch_token_layer = ProjectLayer(
            visual_channel, text_channel, len(output_layers), stack=False, is_array=True
        )
        self.cls_token_layer = ProjectLayer(
            text_channel, text_channel, 1, stack=False, is_array=False
        )

        if "D" in self.prompting_type:  # dynamic prompts
            self.dynamic_visual_prompt_generator = ProjectLayer(
                text_channel, visual_channel, prompting_length, stack=True, is_array=False
            )
            self.dynamic_text_prompt_generator = ProjectLayer(
                text_channel, text_channel, prompting_length, stack=True, is_array=False
            )

        if self.use_hsf:
            self.HSF = HybridSemanticFusion(k_clusters)

        self.image_size = image_size
        self.device = device

    def generate_and_set_dynamic_prompts(self, image: torch.Tensor) -> None:
        with torch.no_grad():
            image_features, _ = self.visual.forward(image, self.output_layers)

        dynamic_visual_prompts = self.dynamic_visual_prompt_generator(image_features)
        dynamic_text_prompts = self.dynamic_text_prompt_generator(image_features)

        self.visual_prompter.set_dynamic_prompts(dynamic_visual_prompts)
        self.text_prompter.set_dynamic_prompts(dynamic_text_prompts)

    def encode_image(self, image: torch.Tensor) -> tuple:
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

        patch_embedding = x
        x = x.permute(1, 0, 2)  # NLD -> LND

        patch_tokens = []
        for indx, r in enumerate(self.visual.transformer.resblocks):
            x, tokens, attn_tmp = self.visual_prompter(
                r, indx, x, k_x=None, v_x=None, attn_mask=None
            )
            if (indx + 1) in self.output_layers:
                patch_tokens.append(tokens)

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

        return pooled, patch_tokens, patch_embedding

    def proj_visual_tokens(self, image_features: torch.Tensor, patch_tokens: list) -> tuple:
        proj_patch_tokens = self.patch_token_layer(patch_tokens)
        for layer in range(len(proj_patch_tokens)):
            proj_patch_tokens[layer] /= proj_patch_tokens[layer].norm(dim=-1, keepdim=True)

        proj_cls_tokens = self.cls_token_layer(image_features)[0]
        proj_cls_tokens /= proj_cls_tokens.norm(dim=-1, keepdim=True)

        return proj_cls_tokens, proj_patch_tokens

    def encode_text(self, text: torch.Tensor) -> torch.Tensor:
        cast_dtype = self.transformer.get_cast_dtype()

        x = self.token_embedding(text).to(cast_dtype)
        x = x + self.positional_embedding.to(cast_dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND

        for indx, r in enumerate(self.transformer.resblocks):
            x, attn_tmp = self.text_prompter(
                r, indx, x, k_x=None, v_x=None, attn_mask=self.attn_mask
            )

        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)

        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
        return x

    def visual_text_similarity(
        self,
        image_feature: torch.Tensor,
        patch_token: list,
        text_feature: torch.Tensor,
        aggregation: bool,
    ) -> tuple:
        anomaly_maps = []

        for layer in range(len(patch_token)):
            anomaly_map = 100.0 * patch_token[layer] @ text_feature
            anomaly_maps.append(anomaly_map)

        if self.use_hsf:
            alpha = 0.2
            clustered_feature = self.HSF.forward(patch_token, anomaly_maps)
            cur_image_feature = alpha * clustered_feature + (1 - alpha) * image_feature
            cur_image_feature = F.normalize(cur_image_feature, dim=1)
        else:
            cur_image_feature = image_feature

        anomaly_score = 100.0 * cur_image_feature.unsqueeze(1) @ text_feature
        anomaly_score = anomaly_score.squeeze(1)
        anomaly_score = torch.softmax(anomaly_score, dim=1)

        for i in range(len(anomaly_maps)):
            B, L, C = anomaly_maps[i].shape
            H = int(np.sqrt(L))
            anomaly_maps[i] = anomaly_maps[i].permute(0, 2, 1).view(B, 2, H, H)
            anomaly_maps[i] = F.interpolate(
                anomaly_maps[i], size=self.image_size, mode="bilinear", align_corners=True
            )

        if aggregation:
            anomaly_map = torch.mean(torch.stack(anomaly_maps, dim=1), dim=1)
            anomaly_map = torch.softmax(anomaly_map, dim=1)
            anomaly_map = (anomaly_map[:, 1:, :, :] + 1 - anomaly_map[:, 0:1, :, :]) / 2.0
            anomaly_score = anomaly_score[:, 1]
            return anomaly_map, anomaly_score
        else:
            for i in range(len(anomaly_maps)):
                anomaly_maps[i] = torch.softmax(anomaly_maps[i], dim=1)
            return anomaly_maps, anomaly_score

    def extract_feat(self, image: torch.Tensor, cls_name: list[str]) -> tuple:
        if "D" in self.prompting_type:
            self.generate_and_set_dynamic_prompts(image)

        if self.enable_visual_prompt:
            image_features, patch_tokens, _ = self.encode_image(image)
        else:
            with torch.no_grad():
                image_features, patch_tokens, _ = self.encode_image(image)

        if self.enable_text_prompt:
            text_features = self.text_embedding_layer(self, cls_name, self.device)
        else:
            with torch.no_grad():
                text_features = self.text_embedding_layer(self, cls_name, self.device)

        proj_cls_tokens, proj_patch_tokens = self.proj_visual_tokens(image_features, patch_tokens)

        return proj_cls_tokens, proj_patch_tokens, text_features

    @torch.cuda.amp.autocast()
    def forward(
        self, image: torch.Tensor, cls_name: list[str], aggregation: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for anomaly detection.

        Args:
            image: Input image tensor [B, C, H, W].
            cls_name: List of class names for text prompts.
            aggregation: Whether to aggregate multi-scale features.

        Returns:
            anomaly_map: Pixel-level anomaly scores.
            anomaly_score: Image-level anomaly scores.
        """
        image_features, patch_tokens, text_features = self.extract_feat(image, cls_name)
        anomaly_map, anomaly_score = self.visual_text_similarity(
            image_features, patch_tokens, text_features, aggregation
        )

        if aggregation:
            anomaly_map = anomaly_map.squeeze(1)
            return anomaly_map, anomaly_score
        else:
            return anomaly_map, anomaly_score


__all__ = ["AdaCLIP", "HybridSemanticFusion", "ProjectLayer", "PromptLayer", "TextEmbeddingLayer"]
