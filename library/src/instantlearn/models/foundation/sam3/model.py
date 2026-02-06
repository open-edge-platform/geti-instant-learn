# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# Copyright 2025 The Meta AI Authors and The HuggingFace Team. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SAM3 model and related components (GeometryEncoder, MaskDecoder, scoring, etc.)."""

import logging
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional
import torchvision
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from torch import nn
from transformers import CLIPTextConfig, CLIPTextModelWithProjection

from .common import (
    MLP,
    Attention,
    SinePositionEmbedding,
    box_cxcywh_to_xyxy,
    concat_padded_sequences,
    expand_attention_mask,
    inverse_sigmoid,
)
from .detr import DecoderMLP, DetrDecoder, DetrEncoder
from .vit import VisionModel


class GeometryEncoderLayer(nn.Module):
    """Transformer layer for geometry encoding with self and cross-attention.

    Applies layer normalization, self-attention on prompts, cross-attention to
    vision features, and feedforward processing with residual connections.
    """

    def __init__(
        self,
        hidden_size: int = 256,
        num_attention_heads: int = 8,
        intermediate_size: int = 2048,
        dropout: float = 0.1,
        hidden_act: str = "relu",
        hidden_dropout: float = 0.0,
    ) -> None:
        """Initialize geometry encoder layer.

        Args:
            hidden_size (int): Hidden dimension size. Default: 256.
            num_attention_heads (int): Number of attention heads. Default: 8.
            intermediate_size (int): Feedforward intermediate dimension. Default:
                2048.
            dropout (float): Dropout probability. Default: 0.1.
            hidden_act (str): Activation function type. Default: "relu".
            hidden_dropout (float): Hidden state dropout probability. Default: 0.0.
        """
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(hidden_size)
        self.self_attn = Attention(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
        )
        self.dropout = nn.Dropout(dropout)

        self.cross_attn = Attention(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
        )
        self.layer_norm2 = nn.LayerNorm(hidden_size)

        self.mlp = MLP(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            hidden_act=hidden_act,
            hidden_dropout=hidden_dropout,
        )
        self.layer_norm3 = nn.LayerNorm(hidden_size)

    def forward(
        self,
        prompt_feats: torch.Tensor,
        vision_feats: torch.Tensor,
        vision_pos_encoding: torch.Tensor,
        prompt_mask: torch.Tensor,
        **kwargs: dict,
    ) -> torch.Tensor:
        """Forward pass through the layer.

        Args:
            prompt_feats (torch.Tensor): Prompt features [batch_size, num_prompts,
                hidden_size].
            vision_feats (torch.Tensor): Vision features [batch_size, num_vision,
                hidden_size].
            vision_pos_encoding (torch.Tensor): Vision position encoding [batch_size,
                num_vision, hidden_size].
            prompt_mask (torch.Tensor): Attention mask for prompts [batch_size,
                num_prompts, num_prompts].
            **kwargs (dict): Additional keyword arguments for attention layers.

        Returns:
            torch.Tensor: Processed features [batch_size, num_prompts, hidden_size].
        """
        residual = prompt_feats
        hidden_states = self.layer_norm1(prompt_feats)
        hidden_states = self.self_attn(
            query=hidden_states,
            key=hidden_states,
            value=hidden_states,
            attention_mask=prompt_mask,
            **kwargs,
        )
        hidden_states = self.dropout(hidden_states) + residual
        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        key = vision_feats + vision_pos_encoding
        hidden_states = self.cross_attn(query=hidden_states, key=key, value=vision_feats, **kwargs)
        hidden_states = self.dropout(hidden_states) + residual
        residual = hidden_states
        hidden_states = self.layer_norm3(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return self.dropout(hidden_states) + residual


class GeometryEncoder(nn.Module):
    """Encoder for geometric prompts (boxes).

    Boxes are encoded using three approaches:
     - Direct projection: linear projection from coordinate space to hidden_size
     - Pooling: pool features from the backbone at the specified location (ROI align for boxes)
     - Position encoding: use position encoding of the box center

    These encodings are combined additively and further processed with transformer layers.

    Args:
        hidden_size: Dimensionality of the encoder layers. Default: 256.
        num_layers: Number of transformer encoder layers. Default: 3.
        num_attention_heads: Number of attention heads. Default: 8.
        intermediate_size: Dimensionality of the feedforward layers. Default: 2048.
        dropout: Dropout probability. Default: 0.1.
        hidden_act: Activation function in FFN. Default: "relu".
        hidden_dropout: Dropout probability for hidden states. Default: 0.0.
        roi_size: ROI size for box pooling operations. Default: 7.
    """

    def __init__(
        self,
        hidden_size: int = 256,
        num_layers: int = 3,
        num_attention_heads: int = 8,
        intermediate_size: int = 2048,
        dropout: float = 0.1,
        hidden_act: str = "relu",
        hidden_dropout: float = 0.0,
        roi_size: int = 7,
    ) -> None:
        """Initialize geometry encoder.

        Args:
            hidden_size (int): Hidden dimension size. Default: 256.
            num_layers (int): Number of transformer layers. Default: 3.
            num_attention_heads (int): Number of attention heads. Default: 8.
            intermediate_size (int): Feedforward intermediate dimension. Default:
                2048.
            dropout (float): Dropout probability. Default: 0.1.
            hidden_act (str): Activation function type. Default: "relu".
            hidden_dropout (float): Hidden state dropout probability. Default: 0.0.
            roi_size (int): ROI pool size for box features. Default: 7.
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.roi_size = roi_size

        self.position_encoding = SinePositionEmbedding(num_pos_feats=hidden_size // 2, normalize=True)
        self.label_embed = nn.Embedding(2, hidden_size)
        self.cls_embed = nn.Embedding(1, hidden_size)

        self.boxes_direct_project = nn.Linear(4, hidden_size)
        self.boxes_pool_project = nn.Conv2d(hidden_size, hidden_size, roi_size)
        self.boxes_pos_enc_project = nn.Linear(hidden_size + 2, hidden_size)

        self.vision_layer_norm = nn.LayerNorm(hidden_size)

        self.final_proj = nn.Linear(hidden_size, hidden_size)
        self.prompt_layer_norm = nn.LayerNorm(hidden_size)

        self.layers = nn.ModuleList([
            GeometryEncoderLayer(
                hidden_size=hidden_size,
                num_attention_heads=num_attention_heads,
                intermediate_size=intermediate_size,
                dropout=dropout,
                hidden_act=hidden_act,
                hidden_dropout=hidden_dropout,
            )
            for _ in range(num_layers)
        ])
        self.output_layer_norm = nn.LayerNorm(hidden_size)

    def _encode_box_coordinates(
        self,
        center_x: torch.Tensor,
        center_y: torch.Tensor,
        width: torch.Tensor,
        height: torch.Tensor,
    ) -> torch.Tensor:
        """Encode box coordinates with position encoding.

        Combines position-encoded centers with raw width/height for box
        representation.

        Args:
            center_x (torch.Tensor): Box center x coordinates [num_boxes].
            center_y (torch.Tensor): Box center y coordinates [num_boxes].
            width (torch.Tensor): Box widths [num_boxes].
            height (torch.Tensor): Box heights [num_boxes].

        Returns:
            torch.Tensor: Encoded box coordinates [num_boxes, embedding_dim].
        """
        pos_x, pos_y = self.position_encoding.encode_1d_positions(center_x, center_y)
        return torch.cat((pos_y, pos_x, height[:, None], width[:, None]), dim=1)

    def _encode_boxes(
        self,
        boxes: torch.Tensor,
        boxes_mask: torch.Tensor,
        boxes_labels: torch.Tensor,
        vision_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode box prompts with embeddings and labels.

        Mask convention: True=valid, False=padding.

        Args:
            boxes (torch.Tensor): Box coordinates [batch_size, num_boxes, 4].
            boxes_mask (torch.Tensor): Valid box mask [batch_size, num_boxes].
            boxes_labels (torch.Tensor): Box labels [batch_size, num_boxes].
            vision_features (torch.Tensor): Vision features [batch_size, hidden_size,
                height, width].

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Encoded boxes and mask.
        """
        batch_size, num_boxes = boxes.shape[:2]
        height, width = vision_features.shape[-2:]
        boxes_embed = self.boxes_direct_project(boxes)

        # Pool features using ROI align
        # Convert boxes from CxCyWH to xyxy format and denormalize
        boxes_xyxy = box_cxcywh_to_xyxy(boxes)
        scale = torch.tensor([width, height, width, height], dtype=boxes_xyxy.dtype, device=boxes_xyxy.device)
        scale = scale.view(1, 1, 4)
        boxes_xyxy *= scale
        # ROI align expects list of boxes per batch element,
        # convert from bfloat16 to float16 as roi_align only supports float16 and float32
        dtype = torch.float16 if vision_features.dtype == torch.bfloat16 else vision_features.dtype
        sampled_features = torchvision.ops.roi_align(
            vision_features.to(dtype),
            boxes_xyxy.to(dtype).unbind(0),
            self.roi_size,
        ).to(vision_features.dtype)

        pooled_projection = self.boxes_pool_project(sampled_features)
        pooled_projection = pooled_projection.view(batch_size, num_boxes, self.hidden_size)
        boxes_embed += pooled_projection

        # Add position encoding
        center_x, center_y, box_width, box_height = boxes.unbind(-1)
        pos_enc = self._encode_box_coordinates(
            center_x.flatten(),
            center_y.flatten(),
            box_width.flatten(),
            box_height.flatten(),
        )
        pos_enc = pos_enc.view(batch_size, num_boxes, pos_enc.shape[-1])
        pos_projection = self.boxes_pos_enc_project(pos_enc)
        boxes_embed += pos_projection

        # Add label embeddings (positive/negative)
        label_embed = self.label_embed(boxes_labels.long())
        return label_embed + boxes_embed, boxes_mask

    def forward(
        self,
        box_embeddings: torch.Tensor,
        box_mask: torch.Tensor,
        box_labels: torch.Tensor,
        img_feats: tuple[torch.Tensor, ...],
        img_pos_embeds: tuple[torch.Tensor, ...] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Encode geometric prompts with transformer layers.

        Args:
            box_embeddings (torch.Tensor): Box coordinates in CxCyWH format
                [batch_size, num_boxes, 4].
            box_mask (torch.Tensor): Attention mask for boxes [batch_size,
                num_boxes].
            box_labels (torch.Tensor): Labels for boxes (positive/negative)
                [batch_size, num_boxes].
            img_feats (tuple[torch.Tensor, ...]): Image features from vision
                encoder.
            img_pos_embeds (tuple[torch.Tensor, ...] | None): Optional position
                embeddings for image features. Default: None.

        Returns:
            dict[str, torch.Tensor]: Dictionary with 'last_hidden_state' containing
                encoded geometry features [batch_size, num_prompts, hidden_size]
                and 'attention_mask' for padding.
        """
        batch_size = box_embeddings.shape[0]

        # Prepare vision features for cross-attention: flatten spatial dimensions
        vision_feats = img_feats[-1]  # [B, C, H, W]
        vision_pos_embeds = img_pos_embeds[-1] if img_pos_embeds is not None else torch.zeros_like(vision_feats)
        vision_feats_flat = vision_feats.flatten(2).transpose(1, 2)  # [B, H*W, C]
        vision_pos_embeds_flat = vision_pos_embeds.flatten(2).transpose(1, 2)  # [B, H*W, C]

        # Normalize image features for pooling operations
        img_feats_last = img_feats[-1]  # [B, C, H, W]
        img_feats_last = img_feats_last.permute(0, 2, 3, 1)  # [B, H, W, C]
        normalized_img_feats = self.vision_layer_norm(img_feats_last)
        normalized_img_feats = normalized_img_feats.permute(0, 3, 1, 2)  # [B, C, H, W]

        prompt_embeds, prompt_mask = self._encode_boxes(box_embeddings, box_mask, box_labels, normalized_img_feats)

        # Add CLS token (always valid)
        cls_embed = self.cls_embed.weight.view(1, self.hidden_size).unsqueeze(0).expand(batch_size, -1, -1)
        cls_mask = torch.ones(batch_size, 1, dtype=prompt_mask.dtype, device=prompt_mask.device)
        prompt_embeds, prompt_mask = concat_padded_sequences(prompt_embeds, prompt_mask, cls_embed, cls_mask)

        prompt_embeds = self.prompt_layer_norm(self.final_proj(prompt_embeds))

        # Create bidirectional attention mask for transformer layers
        prompt_attention_mask = None
        if prompt_mask is not None:
            prompt_attention_mask = expand_attention_mask(prompt_mask)

        # Apply transformer layers with cross-attention to vision features
        for layer in self.layers:
            prompt_embeds = layer(
                prompt_feats=prompt_embeds,
                vision_feats=vision_feats_flat,
                vision_pos_encoding=vision_pos_embeds_flat,
                prompt_mask=prompt_attention_mask,
            )

        # Final output normalization
        prompt_embeds = self.output_layer_norm(prompt_embeds)

        return {
            "last_hidden_state": prompt_embeds,
            "attention_mask": prompt_mask,
        }


class DotProductScoring(nn.Module):
    """Compute classification scores via dot product between query and text.

    Computes scores by taking dot product between projected decoder queries
    and pooled text features to determine confidence/presence scores for each
    query.
    """

    def __init__(
        self,
        hidden_size: int = 256,
        intermediate_size: int = 2048,
        dropout: float = 0.1,
    ) -> None:
        """Initialize dot product scoring module.

        Args:
            hidden_size (int): Hidden dimension size. Default: 256.
            intermediate_size (int): Feedforward intermediate dimension. Default:
                2048.
            dropout (float): Dropout probability. Default: 0.1.
        """
        super().__init__()
        projection_dim = hidden_size

        self.text_mlp = DecoderMLP(
            input_dim=hidden_size,
            hidden_dim=intermediate_size,
            output_dim=hidden_size,
            num_layers=2,
        )
        self.text_mlp_dropout = nn.Dropout(dropout)
        self.text_mlp_out_norm = nn.LayerNorm(hidden_size)

        self.text_proj = nn.Linear(hidden_size, projection_dim)
        self.query_proj = nn.Linear(hidden_size, projection_dim)

        self.scale = float(1.0 / np.sqrt(projection_dim))

        # Clamping to avoid numerical issues
        self.clamp_logits = True
        self.clamp_max_val = 12.0

    @staticmethod
    def _pool_text_features(
        text_features: torch.Tensor,
        text_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """Mean pool text features accounting for padding.

        Args:
            text_features (torch.Tensor): Text features [batch_size, seq_len,
                hidden_size].
            text_mask (torch.Tensor | None): Valid token mask [batch_size, seq_len]
                where True=valid, False=padding.

        Returns:
            torch.Tensor: Pooled text features [batch_size, hidden_size].
        """
        if text_mask is None:
            # No padding, simple mean
            return text_features.mean(dim=1)

        is_valid = text_mask.to(text_features.dtype).unsqueeze(-1)  # [batch_size, seq_len, 1]

        # Count valid tokens per batch
        num_valid = is_valid.sum(dim=1).clamp(min=1.0)  # [batch_size, 1]

        # Mean pool only over valid tokens
        return (text_features * is_valid).sum(dim=1) / num_valid  # [batch_size, hidden_size]

    def forward(
        self,
        decoder_hidden_states: torch.Tensor,
        text_features: torch.Tensor,
        text_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute classification scores via dot product.

        Args:
            decoder_hidden_states (torch.Tensor): Decoder outputs
                [num_layers, batch_size, num_queries, hidden_size].
            text_features (torch.Tensor): Text embeddings [batch_size, seq_len,
                hidden_size].
            text_mask (torch.Tensor | None): Valid token mask [batch_size, seq_len]
                where True=valid, False=padding. Default: None.

        Returns:
            torch.Tensor: Classification scores [num_layers, batch_size,
                num_queries, 1].
        """
        orig_text_features = text_features
        text_features = self.text_mlp(text_features)
        text_features = self.text_mlp_dropout(text_features)
        text_features += orig_text_features
        text_features = self.text_mlp_out_norm(text_features)

        pooled_text = self._pool_text_features(text_features, text_mask)

        proj_text = self.text_proj(pooled_text)
        proj_queries = self.query_proj(decoder_hidden_states)

        proj_text = proj_text.unsqueeze(-1)
        scores = torch.matmul(proj_queries, proj_text.unsqueeze(0))
        scores *= self.scale
        if self.clamp_logits:
            scores = scores.clamp(min=-self.clamp_max_val, max=self.clamp_max_val)

        return scores


class MaskEmbedder(nn.Module):
    """MLP for embedding object queries for mask prediction.

    Similar to MaskFormer's mask embedder architecture with three linear layers.
    """

    def __init__(self, hidden_size: int = 256) -> None:
        """Initialize mask embedder.

        Args:
            hidden_size (int): Hidden dimension size. Default: 256.
        """
        super().__init__()
        self.layers = nn.ModuleList(
            [
                nn.Linear(hidden_size, hidden_size),
                nn.Linear(hidden_size, hidden_size),
                nn.Linear(hidden_size, hidden_size),
            ],
        )
        self.activation = nn.ReLU()

    def forward(self, queries: torch.Tensor) -> torch.Tensor:
        """Embed object queries for mask prediction.

        Args:
            queries (torch.Tensor): Query embeddings [batch_size, num_queries,
                hidden_size].

        Returns:
            torch.Tensor: Mask embeddings [batch_size, num_queries, hidden_size].
        """
        hidden_states = queries
        for i, layer in enumerate(self.layers):
            hidden_states = layer(hidden_states)
            if i < len(self.layers) - 1:
                hidden_states = self.activation(hidden_states)
        return hidden_states


class PixelDecoder(nn.Module):
    """Feature Pyramid Network (FPN) decoder for pixel-level features.

    Inspired by MaskFormer's pixel decoder, generates multi-scale features
    through upsampling and skip connections.
    """

    def __init__(
        self,
        hidden_size: int = 256,
        num_upsampling_stages: int = 3,
    ) -> None:
        """Initialize pixel decoder.

        Args:
            hidden_size (int): Hidden dimension size. Default: 256.
            num_upsampling_stages (int): Number of upsampling stages. Default: 3.
        """
        super().__init__()
        self.conv_layers = nn.ModuleList(
            [
                nn.Conv2d(hidden_size, hidden_size, kernel_size=3, stride=1, padding=1)
                for _ in range(num_upsampling_stages)
            ],
        )
        self.norms = nn.ModuleList([nn.GroupNorm(8, hidden_size) for _ in range(num_upsampling_stages)])

        self.out_channels = hidden_size

    def forward(self, backbone_features: list[torch.Tensor]) -> torch.Tensor:
        """Decode multi-scale backbone features to pixel embeddings.

        Args:
            backbone_features (list[torch.Tensor]): Backbone features
                [batch_size, hidden_size, H_i, W_i] from low to high resolution
                (assumes already projected to hidden_size).

        Returns:
            torch.Tensor: Pixel embeddings [batch_size, hidden_size, H, W] at the
                finest resolution.
        """
        # Start from the coarsest feature (last in list)
        prev_fpn = backbone_features[-1]
        # Iterate through features from coarse to fine (excluding the last which we started with)
        for layer_idx, backbone_feat in enumerate(reversed(backbone_features[:-1])):
            # Upsample previous FPN output to match current backbone feature size
            prev_fpn = torch.nn.functional.interpolate(prev_fpn, size=backbone_feat.shape[-2:], mode="nearest")

            # Add skip connection
            prev_fpn += backbone_feat

            # Apply conv and norm
            prev_fpn = self.conv_layers[layer_idx](prev_fpn)
            prev_fpn = self.norms[layer_idx](prev_fpn)
            prev_fpn = torch.nn.functional.relu(prev_fpn)

        return prev_fpn


class MaskDecoder(nn.Module):
    """Mask decoder for instance mask prediction from queries and features.

    Combines object queries with pixel-level features to predict instance masks
    and semantic segmentation. Supports cross-attention to prompts.

    Args:
        hidden_size (int): Dimensionality of the mask decoder. Default: 256.
        num_upsampling_stages (int): Number of upsampling stages in pixel
            decoder (FPN). Default: 3.
        num_attention_heads (int): Number of attention heads for prompt
            cross-attention. Default: 8.
        dropout (float): Dropout probability for prompt cross-attention.
            Default: 0.0.
    """

    def __init__(
        self,
        hidden_size: int = 256,
        num_upsampling_stages: int = 3,
        num_attention_heads: int = 8,
        dropout: float = 0.0,
    ) -> None:
        """Initialize mask decoder.

        Args:
            hidden_size (int): Dimensionality of the mask decoder. Default: 256.
            num_upsampling_stages (int): Number of upsampling stages in pixel
                decoder (FPN). Default: 3.
            num_attention_heads (int): Number of attention heads for prompt
                cross-attention. Default: 8.
            dropout (float): Dropout probability for prompt cross-attention.
                Default: 0.0.
        """
        super().__init__()
        self.pixel_decoder = PixelDecoder(
            hidden_size=hidden_size,
            num_upsampling_stages=num_upsampling_stages,
        )

        self.mask_embedder = MaskEmbedder(hidden_size=hidden_size)

        self.instance_projection = nn.Conv2d(self.pixel_decoder.out_channels, hidden_size, kernel_size=1)

        self.semantic_projection = nn.Conv2d(self.pixel_decoder.out_channels, 1, kernel_size=1)

        self.prompt_cross_attn = Attention(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
        )
        self.prompt_cross_attn_norm = nn.LayerNorm(hidden_size)
        self.prompt_cross_attn_dropout = nn.Dropout(dropout)

    def forward(
        self,
        decoder_queries: torch.Tensor,
        backbone_features: list[torch.Tensor],
        encoder_hidden_states: torch.Tensor,
        prompt_features: torch.Tensor | None = None,
        prompt_mask: torch.Tensor | None = None,
        **kwargs: dict,
    ) -> dict[str, torch.Tensor | None]:
        """Predict instance masks and semantic segmentation.

        Args:
            decoder_queries (torch.Tensor): Decoder output queries [batch_size,
                num_queries, hidden_size].
            backbone_features (list[torch.Tensor]): List of backbone features to
                process through FPN.
            encoder_hidden_states (torch.Tensor): Encoder outputs [batch_size,
                seq_len, hidden_size].
            prompt_features (torch.Tensor | None): Prompt features (text +
                geometry) for cross-attention [batch_size, prompt_len,
                hidden_size]. Default: None.
            prompt_mask (torch.Tensor | None): Padding mask [batch_size,
                prompt_len] where True=valid, False=padding. Default: None.
            **kwargs (dict): Additional keyword arguments for attention layers.

        Returns:
            dict[str, torch.Tensor | None]: Dictionary with 'pred_masks',
                'semantic_seg', and 'attentions'.
        """
        if prompt_features is not None:
            # Cross-attention: encoder features attend to prompt features
            residual = encoder_hidden_states
            normed_hidden_states = self.prompt_cross_attn_norm(encoder_hidden_states)

            cross_attn_mask = None
            if prompt_mask is not None:
                cross_attn_mask = expand_attention_mask(prompt_mask)

            attn_output = self.prompt_cross_attn(
                query=normed_hidden_states,
                key=prompt_features,
                value=prompt_features,
                attention_mask=cross_attn_mask,
                **kwargs,
            )
            encoder_hidden_states = residual + self.prompt_cross_attn_dropout(attn_output)

        # Process backbone features through FPN to get pixel embeddings
        pixel_embed = self._embed_pixels(
            backbone_features=backbone_features,
            encoder_hidden_states=encoder_hidden_states,
        )

        # Predict instance masks via dot product between query embeddings and pixel embeddings
        instance_embeds = self.instance_projection(pixel_embed)
        mask_embeddings = self.mask_embedder(decoder_queries)
        pred_masks = torch.einsum("bqc,bchw->bqhw", mask_embeddings, instance_embeds)

        # Generate semantic segmentation
        semantic_seg = self.semantic_projection(pixel_embed)

        return {
            "pred_masks": pred_masks,
            "semantic_seg": semantic_seg,
            "attentions": None,
        }

    def _embed_pixels(
        self,
        backbone_features: list[torch.Tensor],
        encoder_hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """Embed pixels from backbone FPN features and encoder vision features.

        The encoder vision features replace the finest-resolution backbone
        feature.

        Args:
            backbone_features (list[torch.Tensor]): List of backbone features
                [batch_size, C, H_i, W_i].
            encoder_hidden_states (torch.Tensor): Encoder outputs [batch_size,
                seq_len, hidden_size].

        Returns:
            torch.Tensor: Pixel embeddings [batch_size, hidden_size, H, W].
        """
        backbone_visual_feats = [feat.clone() for feat in backbone_features]

        # Extract vision features from encoder output and reshape to spatial format
        spatial_dim = backbone_features[-1].shape[-2] * backbone_features[-1].shape[-1]
        encoder_visual_embed = encoder_hidden_states[:, :spatial_dim, :]
        batch_size, _, hidden_size = encoder_visual_embed.shape
        height, width = backbone_features[-1].shape[-2:]
        encoder_visual_embed = encoder_visual_embed.transpose(1, 2).reshape(batch_size, hidden_size, height, width)

        # Replace finest backbone feature with encoder vision features
        backbone_visual_feats[-1] = encoder_visual_embed

        # Process through FPN decoder
        return self.pixel_decoder(backbone_visual_feats)


class Sam3Model(nn.Module):
    """SAM3 (Segment Anything Model 3) for open-vocabulary instance segmentation.

    Combines vision encoder (ViT with FPN), text encoder (CLIP), geometry
    encoder, DETR encoder/decoder for object detection, and mask decoder for
    instance segmentation.
    """

    def __init__(
        self,
        # Vision encoder args
        vision_hidden_size: int = 1024,
        vision_intermediate_size: int = 4736,
        vision_num_hidden_layers: int = 32,
        vision_num_attention_heads: int = 16,
        num_channels: int = 3,
        image_size: int = 1008,
        patch_size: int = 14,
        vision_hidden_act: str = "gelu",
        vision_layer_norm_eps: float = 1e-6,
        vision_attention_dropout: float = 0.0,
        rope_theta: float = 10000.0,
        window_size: int = 24,
        global_attn_indexes: list[int] | None = None,
        pretrain_image_size: int = 336,
        vision_hidden_dropout: float = 0.0,
        fpn_hidden_size: int = 256,
        scale_factors: list[float] | None = None,
        # Text encoder args
        text_vocab_size: int = 49408,
        text_hidden_size: int = 1024,
        text_intermediate_size: int = 4096,
        text_projection_dim: int = 512,
        text_num_hidden_layers: int = 24,
        text_num_attention_heads: int = 16,
        text_max_position_embeddings: int = 32,
        text_hidden_act: str = "gelu",
        # Geometry encoder args
        geometry_hidden_size: int = 256,
        geometry_num_layers: int = 3,
        geometry_num_attention_heads: int = 8,
        geometry_intermediate_size: int = 2048,
        geometry_dropout: float = 0.1,
        geometry_hidden_act: str = "relu",
        geometry_roi_size: int = 7,
        # DETR encoder args
        detr_encoder_hidden_size: int = 256,
        detr_encoder_num_layers: int = 6,
        detr_encoder_num_attention_heads: int = 8,
        detr_encoder_intermediate_size: int = 2048,
        detr_encoder_dropout: float = 0.1,
        detr_encoder_hidden_act: str = "relu",
        # DETR decoder args
        detr_decoder_hidden_size: int = 256,
        detr_decoder_num_layers: int = 6,
        detr_decoder_num_queries: int = 200,
        detr_decoder_num_attention_heads: int = 8,
        detr_decoder_intermediate_size: int = 2048,
        detr_decoder_dropout: float = 0.1,
        detr_decoder_hidden_act: str = "relu",
        # Mask decoder args
        mask_decoder_hidden_size: int = 256,
        mask_decoder_num_upsampling_stages: int = 3,
        mask_decoder_num_attention_heads: int = 8,
        mask_decoder_dropout: float = 0.0,
    ) -> None:
        """Initialize SAM3 model with all components.

        See class docstring for parameter descriptions. All size parameters are
        in dimensionality units, all dropout parameters are floats in [0, 1),
        and all count parameters are integers.
        """
        super().__init__()
        if global_attn_indexes is None:
            global_attn_indexes = [7, 15, 23, 31]
        if scale_factors is None:
            scale_factors = [4.0, 2.0, 1.0, 0.5]

        # Vision encoder
        self.vision_encoder = VisionModel(
            hidden_size=vision_hidden_size,
            intermediate_size=vision_intermediate_size,
            num_hidden_layers=vision_num_hidden_layers,
            num_attention_heads=vision_num_attention_heads,
            num_channels=num_channels,
            image_size=image_size,
            patch_size=patch_size,
            hidden_act=vision_hidden_act,
            layer_norm_eps=vision_layer_norm_eps,
            attention_dropout=vision_attention_dropout,
            rope_theta=rope_theta,
            window_size=window_size,
            global_attn_indexes=global_attn_indexes,
            pretrain_image_size=pretrain_image_size,
            hidden_dropout=vision_hidden_dropout,
            fpn_hidden_size=fpn_hidden_size,
            scale_factors=scale_factors,
        )

        # Text encoder (CLIP)
        text_config = CLIPTextConfig(
            vocab_size=text_vocab_size,
            hidden_size=text_hidden_size,
            intermediate_size=text_intermediate_size,
            projection_dim=text_projection_dim,
            num_hidden_layers=text_num_hidden_layers,
            num_attention_heads=text_num_attention_heads,
            max_position_embeddings=text_max_position_embeddings,
            hidden_act=text_hidden_act,
        )
        self.text_encoder = CLIPTextModelWithProjection(text_config)
        self.vocab_size = text_vocab_size

        self.text_projection = nn.Linear(text_hidden_size, detr_encoder_hidden_size)

        # Geometry encoder
        self.geometry_encoder = GeometryEncoder(
            hidden_size=geometry_hidden_size,
            num_layers=geometry_num_layers,
            num_attention_heads=geometry_num_attention_heads,
            intermediate_size=geometry_intermediate_size,
            dropout=geometry_dropout,
            hidden_act=geometry_hidden_act,
            roi_size=geometry_roi_size,
        )

        # DETR encoder
        self.detr_encoder = DetrEncoder(
            hidden_size=detr_encoder_hidden_size,
            num_layers=detr_encoder_num_layers,
            num_attention_heads=detr_encoder_num_attention_heads,
            intermediate_size=detr_encoder_intermediate_size,
            dropout=detr_encoder_dropout,
            hidden_act=detr_encoder_hidden_act,
        )

        # DETR decoder
        self.detr_decoder = DetrDecoder(
            hidden_size=detr_decoder_hidden_size,
            num_layers=detr_decoder_num_layers,
            num_queries=detr_decoder_num_queries,
            num_attention_heads=detr_decoder_num_attention_heads,
            intermediate_size=detr_decoder_intermediate_size,
            dropout=detr_decoder_dropout,
            hidden_act=detr_decoder_hidden_act,
        )

        # Mask decoder
        self.mask_decoder = MaskDecoder(
            hidden_size=mask_decoder_hidden_size,
            num_upsampling_stages=mask_decoder_num_upsampling_stages,
            num_attention_heads=mask_decoder_num_attention_heads,
            dropout=mask_decoder_dropout,
        )

        # Dot product scoring
        self.dot_product_scoring = DotProductScoring(
            hidden_size=detr_decoder_hidden_size,
            intermediate_size=detr_decoder_intermediate_size,
            dropout=detr_decoder_dropout,
        )

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        device: str | torch.device | None = None,
        dtype: torch.dtype | None = None,
        torch_dtype: torch.dtype | None = None,
        key_mapping: dict | None = None,
        **kwargs: dict,
    ) -> "Sam3Model":
        """Load a pretrained SAM3 model from HuggingFace Hub or local path.

        Args:
            pretrained_model_name_or_path: HuggingFace model ID or local path.
            device: Device to load the model on.
            dtype: Data type for the model weights (alias for torch_dtype).
            torch_dtype: Data type for the model weights.
            key_mapping: Optional regex mapping to transform state dict keys.
            attn_implementation: Attention implementation (e.g., "sdpa", "eager").
                Currently ignored - uses PyTorch default attention.
            **kwargs: Additional arguments passed to Sam3Model.__init__.

        Returns:
            Loaded Sam3Model instance.

        Example:
            >>> model = Sam3Model.from_pretrained("facebook/sam3-base-plus")
            >>> model = Sam3Model.from_pretrained("facebook/sam3-base-plus", device="cuda", dtype=torch.bfloat16)
        """
        # Handle dtype aliases
        if torch_dtype is not None and dtype is None:
            dtype = torch_dtype

        # Determine if local path or HuggingFace Hub
        path = Path(pretrained_model_name_or_path)
        if path.exists():
            # Local path
            model_path = path / "model.safetensors"
        else:
            model_path = hf_hub_download(
                repo_id=pretrained_model_name_or_path,
                filename="model.safetensors",
            )

        # Load state dict
        state_dict = load_file(model_path)

        # Handle sam3_video format: remove "detector_model." prefix
        new_state_dict = {}
        for key, value in state_dict.items():
            new_key = re.sub(r"^detector_model\.", "", key)
            new_state_dict[new_key] = value
        state_dict = new_state_dict

        # Apply any additional key mapping if provided
        if key_mapping:
            mapped_state_dict = {}
            for key, value in state_dict.items():
                new_key = key
                for pattern, replacement in key_mapping.items():
                    new_key = re.sub(pattern, replacement, new_key)
                mapped_state_dict[new_key] = value
            state_dict = mapped_state_dict

        # Create model with default args (can be overridden via kwargs)
        model = cls(**kwargs)

        # Load weights
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

        # Filter out expected missing/unexpected keys
        tracker_pattern = re.compile(r"^(tracker_model\.|tracker_neck\.)")
        unexpected_keys = [k for k in unexpected_keys if not tracker_pattern.match(k)]

        if missing_keys:
            logging.warning(f"Missing keys when loading SAM3 model: {missing_keys}")
        if unexpected_keys:
            logging.warning(f"Unexpected keys when loading SAM3 model: {unexpected_keys}")

        # Move to device/dtype if specified
        if device is not None:
            model = model.to(device)
        if dtype is not None:
            model = model.to(dtype)

        return model

    def get_text_features(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor | None = None,
        **kwargs: dict,
    ) -> torch.Tensor:
        """Extract text features from text encoder.

        Returns the CLIP text model output with pooler_output containing the
        projected text embeddings.

        Args:
            input_ids (torch.LongTensor): Token IDs [batch_size, seq_len].
            attention_mask (torch.Tensor | None): Attention mask [batch_size,
                seq_len]. Default: None.
            **kwargs (dict): Additional keyword arguments for text encoder.

        Returns:
            torch.Tensor: Text features with pooler_output projection.
        """
        text_outputs = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
            **kwargs,
        )
        last_hidden_state = text_outputs.last_hidden_state
        text_outputs.pooler_output = self.text_projection(last_hidden_state)

        return text_outputs

    def get_vision_features(
        self,
        pixel_values: torch.FloatTensor,
        **kwargs: dict,
    ) -> dict[str, torch.Tensor | None]:
        """Extract vision features from vision encoder.

        Can be used to pre-compute vision embeddings for reuse across multiple
        text prompts.

        Args:
            pixel_values (torch.FloatTensor): Input images [batch_size, channels,
                height, width].
            **kwargs (dict): Additional keyword arguments for vision encoder.

        Returns:
            dict[str, torch.Tensor | None]: Vision features and positional
                encodings.
        """
        return self.vision_encoder(pixel_values, **kwargs)

    def forward(
        self,
        pixel_values: torch.FloatTensor | None = None,
        vision_embeds: dict[str, torch.Tensor] | None = None,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        text_embeds: torch.FloatTensor | None = None,
        input_boxes: torch.FloatTensor | None = None,
        input_boxes_labels: torch.LongTensor | None = None,
        **kwargs: dict,
    ) -> dict[str, torch.Tensor | None]:
        """Predict instance masks, boxes, and logits for input images.

        Args:
            pixel_values (torch.FloatTensor | None): Input images [batch_size,
                channels, height, width]. Mutually exclusive with vision_embeds.
                Default: None.
            vision_embeds (dict[str, torch.Tensor] | None): Pre-computed vision
                embeddings. Mutually exclusive with pixel_values. Default: None.
            input_ids (torch.LongTensor | None): Text token IDs [batch_size,
                seq_len]. Mutually exclusive with text_embeds. Default: None.
            attention_mask (torch.Tensor | None): Text attention mask [batch_size,
                seq_len]. Default: None.
            text_embeds (torch.FloatTensor | None): Pre-computed text embeddings
                [batch_size, seq_len, hidden_size]. Mutually exclusive with
                input_ids. Default: None.
            input_boxes (torch.FloatTensor | None): Box prompts in CxCyWH format
                normalized to [0, 1] [batch_size, num_boxes, 4]. Default: None.
            input_boxes_labels (torch.LongTensor | None): Box labels (1=positive,
                0=negative) [batch_size, num_boxes]. Default: None.
            **kwargs (dict): Additional keyword arguments for model components.

        Returns:
            dict[str, torch.Tensor | None]: Dictionary with predicted masks
                [batch_size, num_queries, height, width], boxes [batch_size,
                num_queries, 4], logits [batch_size, num_queries, num_classes],
                and other outputs.

        Raises:
            ValueError: If pixel_values and vision_embeds are both or neither
                provided, or if input_ids and text_embeds are both or neither
                provided.
        """
        if (pixel_values is None) == (vision_embeds is None):
            msg = "You must specify exactly one of pixel_values or vision_embeds"
            raise ValueError

        if (input_ids is None) == (text_embeds is None):
            msg = "You must specify exactly one of input_ids or text_embeds"
            raise ValueError(msg)

        if pixel_values is not None:
            batch_size = pixel_values.shape[0]
            device = pixel_values.device
        else:
            batch_size = vision_embeds["fpn_hidden_states"][0].shape[0]
            device = vision_embeds["fpn_hidden_states"][0].device

        vision_outputs = self.vision_encoder(pixel_values, **kwargs) if vision_embeds is None else vision_embeds

        fpn_hidden_states = vision_outputs["fpn_hidden_states"][:-1]
        fpn_position_encoding = vision_outputs["fpn_position_encoding"][:-1]

        if text_embeds is None:
            text_features = self.get_text_features(
                input_ids=input_ids,
                attention_mask=attention_mask,
            ).pooler_output
        else:
            text_features = text_embeds

        text_mask = attention_mask.bool() if attention_mask is not None else None
        has_geometry_prompts = input_boxes is not None and input_boxes.numel() > 0

        geometry_prompt_features = None
        geometry_prompt_mask = None

        if has_geometry_prompts:
            if input_boxes is not None and input_boxes.numel() > 0:
                box_embeddings = input_boxes.to(dtype=text_features.dtype)  # [batch_size, num_boxes, 4]
                box_labels = (
                    input_boxes_labels
                    if input_boxes_labels is not None
                    else torch.ones_like(box_embeddings[..., 0], dtype=torch.long)
                )
                box_mask = (
                    (input_boxes_labels != -10)
                    if input_boxes_labels is not None
                    else torch.ones(batch_size, input_boxes.shape[1], dtype=torch.bool, device=device)
                )
                box_labels = torch.where(box_labels == -10, 0, box_labels)
            else:
                box_embeddings = torch.zeros(batch_size, 0, 4, dtype=text_features.dtype, device=device)
                box_labels = torch.zeros(batch_size, 0, dtype=torch.long, device=device)
                box_mask = torch.zeros(batch_size, 0, dtype=torch.bool, device=device)

            geometry_outputs = self.geometry_encoder(
                box_embeddings=box_embeddings,
                box_mask=box_mask,
                box_labels=box_labels,
                img_feats=fpn_hidden_states,
                img_pos_embeds=fpn_position_encoding,
            )

            geometry_prompt_features = geometry_outputs["last_hidden_state"]
            geometry_prompt_mask = geometry_outputs["attention_mask"]

        if geometry_prompt_features is not None:
            # Repeat text_features for all geometry prompts
            if text_features.shape[0] == 1 and geometry_prompt_features.shape[0] > 1:
                text_features = text_features.repeat(geometry_prompt_features.shape[0], 1, 1)
            combined_prompt_features = torch.cat([text_features, geometry_prompt_features], dim=1)
            if text_mask is not None and text_mask.shape[0] == 1 and geometry_prompt_mask.shape[0] > 1:
                text_mask = text_mask.repeat(geometry_prompt_mask.shape[0], 1)

            if text_mask is not None and geometry_prompt_mask is not None:
                combined_prompt_mask = torch.cat([text_mask, geometry_prompt_mask], dim=1)
            elif text_mask is not None:
                geo_valid_mask = torch.ones(
                    batch_size,
                    geometry_prompt_features.shape[1],
                    dtype=torch.bool,
                    device=device,
                )
                combined_prompt_mask = torch.cat([text_mask, geo_valid_mask], dim=1)
            elif geometry_prompt_mask is not None:
                text_valid_mask = torch.ones(batch_size, text_features.shape[1], dtype=torch.bool, device=device)
                combined_prompt_mask = torch.cat([text_valid_mask, geometry_prompt_mask], dim=1)
            else:
                combined_prompt_mask = None
        else:
            combined_prompt_features = text_features
            combined_prompt_mask = text_mask

        encoder_outputs = self.detr_encoder(
            vision_features=[fpn_hidden_states[-1]],
            text_features=combined_prompt_features,
            vision_pos_embeds=[fpn_position_encoding[-1]],
            text_mask=combined_prompt_mask,
            **kwargs,
        )

        decoder_outputs = self.detr_decoder(
            vision_features=encoder_outputs["last_hidden_state"],
            text_features=encoder_outputs["text_features"],
            vision_pos_encoding=encoder_outputs["pos_embeds_flattened"],
            text_mask=combined_prompt_mask,
            spatial_shapes=encoder_outputs["spatial_shapes"],
            **kwargs,
        )

        # Refine boxes from decoder
        all_box_offsets = self.detr_decoder.box_head(decoder_outputs["intermediate_hidden_states"])
        reference_boxes_inv_sig = inverse_sigmoid(decoder_outputs["reference_boxes"])
        all_pred_boxes_cxcywh = (reference_boxes_inv_sig + all_box_offsets).sigmoid()
        all_pred_boxes = box_cxcywh_to_xyxy(all_pred_boxes_cxcywh)

        all_pred_logits = self.dot_product_scoring(
            decoder_hidden_states=decoder_outputs["intermediate_hidden_states"],
            text_features=encoder_outputs["text_features"],
            text_mask=combined_prompt_mask,
        ).squeeze(-1)

        pred_logits = all_pred_logits[-1]
        pred_boxes = all_pred_boxes[-1]
        decoder_hidden_states = decoder_outputs["intermediate_hidden_states"][-1]
        presence_logits = decoder_outputs["presence_logits"][-1]

        mask_outputs = self.mask_decoder(
            decoder_queries=decoder_hidden_states,
            backbone_features=list(fpn_hidden_states),
            encoder_hidden_states=encoder_outputs["last_hidden_state"],
            prompt_features=combined_prompt_features,
            prompt_mask=combined_prompt_mask,
            **kwargs,
        )

        return {
            "pred_masks": mask_outputs["pred_masks"],
            "pred_boxes": pred_boxes,
            "pred_logits": pred_logits,
            "presence_logits": presence_logits,
            "semantic_seg": mask_outputs["semantic_seg"],
            "decoder_hidden_states": decoder_outputs.get("hidden_states"),
            "decoder_reference_boxes": decoder_outputs["reference_boxes"],
            "encoder_hidden_states": encoder_outputs.get("hidden_states"),
            "vision_hidden_states": vision_outputs.get("hidden_states"),
            "vision_attentions": vision_outputs.get("attentions"),
            "detr_encoder_attentions": encoder_outputs.get("attentions"),
            "detr_decoder_attentions": decoder_outputs.get("attentions"),
            "mask_decoder_attentions": mask_outputs.get("attentions"),
        }
