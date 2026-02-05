# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# Copyright 2025 The Meta AI Authors and The HuggingFace Team. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DETR (DEtection TRansformer) encoder and decoder components for SAM3."""

import math

import torch
from torch import nn
from torch.nn import functional
from transformers.pytorch_utils import compile_compatible_method_lru_cache

from .common import (
    MLP,
    Attention,
    SinePositionEmbedding,
    box_cxcywh_to_xyxy,
    expand_attention_mask,
    inverse_sigmoid,
)


class DecoderMLP(nn.Module):
    """Simple 2 or 3-layer MLP for decoder components.

    A flexible multi-layer perceptron that supports 2 or 3 linear layers with ReLU
    activation between layers.

    Args:
        input_dim (int): Input dimension.
        hidden_dim (int): Hidden dimension.
        output_dim (int): Output dimension.
        num_layers (int): Number of layers (2 or 3). Default: 2.
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int = 2) -> None:
        """Initialize the DecoderMLP module.

        Args:
            input_dim (int): Input dimension.
            hidden_dim (int): Hidden dimension.
            output_dim (int): Output dimension.
            num_layers (int): Number of layers (2 or 3). Default: 2.

        Raises:
            ValueError: If num_layers is not 2 or 3.
        """
        super().__init__()
        if num_layers == 2:
            self.layer1 = nn.Linear(input_dim, hidden_dim)
            self.layer2 = nn.Linear(hidden_dim, output_dim)
            self.layer3 = None
        elif num_layers == 3:
            self.layer1 = nn.Linear(input_dim, hidden_dim)
            self.layer2 = nn.Linear(hidden_dim, hidden_dim)
            self.layer3 = nn.Linear(hidden_dim, output_dim)
        else:
            msg = f"Only 2 or 3 layers supported, got {num_layers}"
            raise ValueError(msg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the MLP.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, ..., input_dim] with
                floating-point dtype.

        Returns:
            torch.Tensor: Output tensor of shape [batch_size, ..., output_dim] with
                floating-point dtype.
        """
        x = functional.relu(self.layer1(x))
        if self.layer3 is not None:
            x = functional.relu(self.layer2(x))
            x = self.layer3(x)
        else:
            x = self.layer2(x)
        return x


class DetrEncoderLayer(nn.Module):
    """DETR encoder layer with self-attention and cross-attention.

    Applies self-attention to vision features with position encoding, followed by
    cross-attention between vision and text features, and an MLP feedforward layer.

    Args:
        hidden_size (int): Dimensionality of hidden states. Default: 256.
        num_attention_heads (int): Number of attention heads. Default: 8.
        intermediate_size (int): Dimensionality of feedforward layer. Default: 2048.
        dropout (float): Dropout probability for attention and MLP. Default: 0.1.
        hidden_act (str): Activation function in FFN. Default: "relu".
        hidden_dropout (float): Dropout probability for hidden states. Default: 0.0.
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
        """Initialize the DetrEncoderLayer module.

        Args:
            hidden_size (int): Dimensionality of hidden states. Default: 256.
            num_attention_heads (int): Number of attention heads. Default: 8.
            intermediate_size (int): Dimensionality of feedforward layer. Default: 2048.
            dropout (float): Dropout probability for attention and MLP. Default: 0.1.
            hidden_act (str): Activation function in FFN. Default: "relu".
            hidden_dropout (float): Dropout probability for hidden states. Default: 0.0.
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
        vision_feats: torch.Tensor,
        prompt_feats: torch.Tensor,
        vision_pos_encoding: torch.Tensor,
        prompt_cross_attn_mask: torch.Tensor | None = None,
        **kwargs: dict,
    ) -> torch.Tensor:
        """Forward pass for DETR encoder layer.

        Applies self-attention to vision features with position encoding, followed by
        cross-attention between vision (query) and text (key/value) features, then an MLP.

        Args:
            vision_feats (torch.Tensor): Vision features [batch_size, vision_len, hidden_size]
                with floating-point dtype (main hidden states).
            prompt_feats (torch.Tensor): Text prompt features
                [batch_size, text_len, hidden_size] with floating-point dtype.
            vision_pos_encoding (torch.Tensor): Position encoding for vision
                [batch_size, vision_len, hidden_size] with floating-point dtype.
            prompt_cross_attn_mask (torch.Tensor | None): Cross-attention mask for prompt
                features. If provided, dtype is bool or floating-point. Default: None.
            **kwargs: Additional keyword arguments passed to attention modules.

        Returns:
            torch.Tensor: Updated vision features [batch_size, vision_len, hidden_size] with
                floating-point dtype.
        """
        # Self-attention on vision features with position encoding
        residual = vision_feats
        hidden_states = self.layer_norm1(vision_feats)
        hidden_states_with_pos = hidden_states + vision_pos_encoding
        hidden_states = self.self_attn(
            query=hidden_states_with_pos,
            key=hidden_states_with_pos,
            value=hidden_states,
            **kwargs,
        )
        hidden_states = self.dropout(hidden_states) + residual

        # Cross-attention: vision queries attend to text/prompt features
        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)

        hidden_states = self.cross_attn(
            query=hidden_states,
            key=prompt_feats,
            value=prompt_feats,
            attention_mask=prompt_cross_attn_mask,
            **kwargs,
        )
        hidden_states = self.dropout(hidden_states) + residual

        # MLP
        residual = hidden_states
        hidden_states = self.layer_norm3(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return self.dropout(hidden_states) + residual


class DetrEncoder(nn.Module):
    """DETR-style encoder that processes multi-level vision features with text fusion.

    This encoder processes vision features from multiple levels (e.g., FPN features at
    different resolutions) and fuses them with text prompts through a stack of
    transformer encoder layers.

    Args:
        hidden_size (int): Dimensionality of the encoder layers. Default: 256.
        num_layers (int): Number of encoder layers. Default: 6.
        num_attention_heads (int): Number of attention heads. Default: 8.
        intermediate_size (int): Dimensionality of the feedforward layers.
            Default: 2048.
        dropout (float): Dropout probability. Default: 0.1.
        hidden_act (str): Activation function in FFN. Default: "relu".
        hidden_dropout (float): Dropout probability for hidden states. Default: 0.0.
    """

    def __init__(
        self,
        hidden_size: int = 256,
        num_layers: int = 6,
        num_attention_heads: int = 8,
        intermediate_size: int = 2048,
        dropout: float = 0.1,
        hidden_act: str = "relu",
        hidden_dropout: float = 0.0,
    ) -> None:
        """Initialize the DetrEncoder module.

        Args:
            hidden_size (int): Dimensionality of the encoder layers. Default: 256.
            num_layers (int): Number of encoder layers. Default: 6.
            num_attention_heads (int): Number of attention heads. Default: 8.
            intermediate_size (int): Dimensionality of the feedforward layers.
                Default: 2048.
            dropout (float): Dropout probability. Default: 0.1.
            hidden_act (str): Activation function in FFN. Default: "relu".
            hidden_dropout (float): Dropout probability for hidden states. Default: 0.0.
        """
        super().__init__()
        self.hidden_size = hidden_size

        self.layers = nn.ModuleList(
            [
                DetrEncoderLayer(
                    hidden_size=hidden_size,
                    num_attention_heads=num_attention_heads,
                    intermediate_size=intermediate_size,
                    dropout=dropout,
                    hidden_act=hidden_act,
                    hidden_dropout=hidden_dropout,
                )
                for _ in range(num_layers)
            ],
        )

    @staticmethod
    def _prepare_multilevel_features(
        vision_features: list[torch.Tensor],
        vision_pos_embeds: list[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Prepare multi-level vision features by flattening spatial dimensions.

        Flattens spatial dimensions and adds level embeddings for multi-level vision
        features.

        Args:
            vision_features (list[torch.Tensor]): List of vision features at different
                levels [batch_size, channels, height, width] with floating-point dtype.
            vision_pos_embeds (list[torch.Tensor]): List of position embeddings for each
                level [batch_size, channels, height, width] with floating-point dtype.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Tuple containing:
                - Flattened features [batch_size, total_seq_len, channels] with
                  floating-point dtype
                - Flattened position embeddings [batch_size, total_seq_len, channels]
                  with floating-point dtype
                - Spatial metadata tensor [num_levels, 2] with (height, width)
        """
        features_flattened = []
        pos_embeds_flattened = []
        spatial_shapes = []

        for feature_map, pos_embed in zip(vision_features, vision_pos_embeds, strict=False):
            height, width = feature_map.shape[-2:]
            spatial_shapes.append((height, width))

            # Flatten spatial dimensions: [batch_size, channels, height, width] -> [batch_size, height*width, channels]
            flattened_features = feature_map.flatten(2).transpose(1, 2)
            flattened_pos_embed = pos_embed.flatten(2).transpose(1, 2)

            features_flattened.append(flattened_features)
            pos_embeds_flattened.append(flattened_pos_embed)

        # Concatenate all levels into single sequence
        features_flattened = torch.cat(features_flattened, dim=1)
        pos_embeds_flattened = torch.cat(pos_embeds_flattened, dim=1)

        spatial_shapes = torch.tensor(spatial_shapes, dtype=torch.long, device=features_flattened.device)

        return (
            features_flattened,
            pos_embeds_flattened,
            spatial_shapes,
        )

    def forward(
        self,
        vision_features: list[torch.Tensor],
        text_features: torch.Tensor,
        vision_pos_embeds: list[torch.Tensor] | None = None,
        text_mask: torch.Tensor | None = None,
        spatial_sizes: list[tuple[int, int]] | None = None,
        **kwargs: dict,
    ) -> dict[str, torch.Tensor | None]:
        """Forward pass for the DETR encoder.

        Processes multi-level vision features and fuses them with text features through
        a series of transformer encoder layers with self-attention and cross-attention.

        Args:
            vision_features (list[torch.Tensor]): List of vision features at different
                levels with floating-point dtype.
            text_features (torch.Tensor): Text prompt features
                [batch_size, seq_len, hidden_size] with floating-point dtype.
            vision_pos_embeds (list[torch.Tensor] | None): Optional list of position
                embeddings for each level. If None, zeros are used. Default: None.
            text_mask (torch.Tensor | None): Optional text padding mask
                [batch_size, seq_len] where True=valid, False=padding. Default: None.
            spatial_sizes (list[tuple[int, int]] | None): Optional list of (height,
                width) tuples for reshaping. Default: None.
            **kwargs: Additional keyword arguments passed to encoder layers.

        Returns:
            dict[str, torch.Tensor | None]: Dictionary containing:
                - 'last_hidden_state': Encoded features [batch_size, seq_len, hidden_size]
                - 'pos_embeds_flattened': Flattened position embeddings
                - 'text_features': Text features
                - 'spatial_shapes': Spatial shape metadata
                - 'hidden_states': None (reserved for future use)
                - 'attentions': None (reserved for future use)

        Raises:
            ValueError: If ``spatial_sizes`` is provided and its length does not match
                the number of vision feature levels.
        """
        batch_size = vision_features[0].shape[0] if vision_features[0].dim() == 4 else vision_features[0].shape[1]

        if vision_pos_embeds is None:
            vision_pos_embeds = [torch.zeros_like(feature) for feature in vision_features]

        if spatial_sizes is not None:
            if len(spatial_sizes) != len(vision_features):
                msg = "spatial_sizes must match the number of vision feature levels"
                raise ValueError(msg)
            reshaped_vision_features: list[torch.Tensor] = []
            reshaped_pos_embeds: list[torch.Tensor] = []
            for level_index, (height, width) in enumerate(spatial_sizes):
                # Reshape from [height*width, batch_size, channels] to [batch_size, channels, height, width]
                reshaped_feature = vision_features[level_index].reshape(
                    height,
                    width,
                    batch_size,
                    -1,
                )
                reshaped_vision_features.append(reshaped_feature.permute(2, 3, 0, 1))

                reshaped_pos_embed = vision_pos_embeds[level_index].reshape(
                    height,
                    width,
                    batch_size,
                    -1,
                )
                reshaped_pos_embeds.append(reshaped_pos_embed.permute(2, 3, 0, 1))
            vision_features = reshaped_vision_features
            vision_pos_embeds = reshaped_pos_embeds

        # Flatten multi-level features for encoder processing
        (
            features_flattened,
            pos_embeds_flattened,
            spatial_shapes,
        ) = self._prepare_multilevel_features(vision_features, vision_pos_embeds)

        prompt_cross_attn_mask = expand_attention_mask(text_mask) if text_mask is not None else None

        hidden_states = features_flattened
        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                prompt_feats=text_features,
                vision_pos_encoding=pos_embeds_flattened,
                prompt_cross_attn_mask=prompt_cross_attn_mask,
                **kwargs,
            )
        return {
            "last_hidden_state": hidden_states,
            "pos_embeds_flattened": pos_embeds_flattened,
            "text_features": text_features,
            "spatial_shapes": spatial_shapes,
            "hidden_states": None,
            "attentions": None,
        }


class DetrDecoderLayer(nn.Module):
    """DETR decoder layer with self-attention, text cross-attention, and vision cross-attention.

    Applies sequential attention mechanisms: self-attention on queries, cross-attention
    with text features, cross-attention with vision features, and an MLP feedforward layer.

    Args:
        hidden_size (int): Dimensionality of hidden states. Default: 256.
        num_attention_heads (int): Number of attention heads. Default: 8.
        intermediate_size (int): Dimensionality of feedforward layer. Default: 2048.
        dropout (float): Dropout probability for attention and MLP. Default: 0.1.
        hidden_act (str): Activation function in FFN. Default: "relu".
        hidden_dropout (float): Dropout probability for hidden states. Default: 0.0.
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
        """Initialize the DetrDecoderLayer module.

        Args:
            hidden_size (int): Dimensionality of hidden states. Default: 256.
            num_attention_heads (int): Number of attention heads. Default: 8.
            intermediate_size (int): Dimensionality of feedforward layer. Default: 2048.
            dropout (float): Dropout probability for attention and MLP. Default: 0.1.
            hidden_act (str): Activation function in FFN. Default: "relu".
            hidden_dropout (float): Dropout probability for hidden states. Default: 0.0.
        """
        super().__init__()
        self.self_attn = Attention(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
        )
        self.self_attn_dropout = nn.Dropout(dropout)
        self.self_attn_layer_norm = nn.LayerNorm(hidden_size)

        self.text_cross_attn = Attention(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
        )
        self.text_cross_attn_dropout = nn.Dropout(dropout)
        self.text_cross_attn_layer_norm = nn.LayerNorm(hidden_size)

        self.vision_cross_attn = Attention(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
        )
        self.vision_cross_attn_dropout = nn.Dropout(dropout)
        self.vision_cross_attn_layer_norm = nn.LayerNorm(hidden_size)

        self.mlp = MLP(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            hidden_act=hidden_act,
            hidden_dropout=hidden_dropout,
        )
        self.mlp_layer_norm = nn.LayerNorm(hidden_size)
        self.mlp_dropout = nn.Dropout(dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        query_pos: torch.Tensor,
        text_features: torch.Tensor,
        vision_features: torch.Tensor,
        vision_pos_encoding: torch.Tensor,
        text_cross_attn_mask: torch.Tensor | None = None,
        vision_cross_attn_mask: torch.Tensor | None = None,
        **kwargs: dict,
    ) -> torch.Tensor:
        """Forward pass for decoder layer.

        Applies self-attention on query features with position encoding, followed by
        cross-attention with text and vision features, and an MLP feedforward layer.

        Args:
            hidden_states (torch.Tensor): Query features
                [batch_size, num_queries + 1, hidden_size] (includes presence token at
                position 0) with floating-point dtype.
            query_pos (torch.Tensor): Query position embeddings
                [batch_size, num_queries, hidden_size] with floating-point dtype.
            text_features (torch.Tensor): Text features [batch_size, seq_len, hidden_size]
                with floating-point dtype.
            vision_features (torch.Tensor): Vision features
                [batch_size, height*width, hidden_size] with floating-point dtype.
            vision_pos_encoding (torch.Tensor): Vision position encoding
                [batch_size, height*width, hidden_size] with floating-point dtype.
            text_cross_attn_mask (torch.Tensor | None): Text cross-attention mask. If
                provided, dtype is bool or floating-point. Default: None.
            vision_cross_attn_mask (torch.Tensor | None): Vision cross-attention mask,
                already expanded for presence token. If provided, dtype is bool or
                floating-point. Default: None.
            **kwargs: Additional keyword arguments passed to attention modules.

        Returns:
            torch.Tensor: Updated hidden states (including presence token at
                position 0) [batch_size, num_queries + 1, hidden_size] with
                floating-point dtype.
        """
        # Prepend zeros to query_pos for presence token
        query_pos = functional.pad(query_pos, (0, 0, 1, 0), mode="constant", value=0)

        # Self-attention with query position encoding
        residual = hidden_states
        query_with_pos = hidden_states + query_pos
        attn_output = self.self_attn(
            query=query_with_pos,
            key=query_with_pos,
            value=hidden_states,
            attention_mask=None,
            **kwargs,
        )
        hidden_states = residual + self.self_attn_dropout(attn_output)
        hidden_states = self.self_attn_layer_norm(hidden_states)

        # Text cross-attention: queries attend to text features
        residual = hidden_states
        query_with_pos = hidden_states + query_pos

        attn_output = self.text_cross_attn(
            query=query_with_pos,
            key=text_features,
            value=text_features,
            attention_mask=text_cross_attn_mask,
            **kwargs,
        )
        hidden_states = residual + self.text_cross_attn_dropout(attn_output)
        hidden_states = self.text_cross_attn_layer_norm(hidden_states)

        # Vision cross-attention: queries attend to vision features (with RPB)
        residual = hidden_states
        query_with_pos = hidden_states + query_pos
        key_with_pos = vision_features + vision_pos_encoding
        attn_output = self.vision_cross_attn(
            query=query_with_pos,
            key=key_with_pos,
            value=vision_features,
            attention_mask=vision_cross_attn_mask,
            **kwargs,
        )
        hidden_states = residual + self.vision_cross_attn_dropout(attn_output)
        hidden_states = self.vision_cross_attn_layer_norm(hidden_states)

        # MLP
        residual = hidden_states
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + self.mlp_dropout(hidden_states)
        return self.mlp_layer_norm(hidden_states)


class DetrDecoder(nn.Module):
    """DETR-style decoder with box refinement and presence token.

    Simplified version that assumes:
    - Box refinement is always enabled
    - Intermediate outputs are always returned
    - BoxRPB (relative position bias) with log-scale encoding
    - Presence token is used

    Args:
        hidden_size (int): Dimensionality of the decoder layers. Default: 256.
        num_layers (int): Number of decoder layers. Default: 6.
        num_queries (int): Number of object queries. Default: 200.
        num_attention_heads (int): Number of attention heads. Default: 8.
        intermediate_size (int): Dimensionality of the feedforward layers.
            Default: 2048.
        dropout (float): Dropout probability. Default: 0.1.
        hidden_act (str): Activation function in FFN. Default: "relu".
        hidden_dropout (float): Dropout probability for hidden states. Default: 0.0.
    """

    def __init__(
        self,
        hidden_size: int = 256,
        num_layers: int = 6,
        num_queries: int = 200,
        num_attention_heads: int = 8,
        intermediate_size: int = 2048,
        dropout: float = 0.1,
        hidden_act: str = "relu",
        hidden_dropout: float = 0.0,
    ) -> None:
        """Initialize the DetrDecoder module.

        Args:
            hidden_size (int): Dimensionality of the decoder layers. Default: 256.
            num_layers (int): Number of decoder layers. Default: 6.
            num_queries (int): Number of object queries. Default: 200.
            num_attention_heads (int): Number of attention heads. Default: 8.
            intermediate_size (int): Dimensionality of the feedforward layers.
                Default: 2048.
            dropout (float): Dropout probability. Default: 0.1.
            hidden_act (str): Activation function in FFN. Default: "relu".
            hidden_dropout (float): Dropout probability for hidden states. Default: 0.0.
        """
        super().__init__()
        self.hidden_size = hidden_size

        self.layers = nn.ModuleList(
            [
                DetrDecoderLayer(
                    hidden_size=hidden_size,
                    num_attention_heads=num_attention_heads,
                    intermediate_size=intermediate_size,
                    dropout=dropout,
                    hidden_act=hidden_act,
                    hidden_dropout=hidden_dropout,
                )
                for _ in range(num_layers)
            ],
        )

        self.output_layer_norm = nn.LayerNorm(hidden_size)

        self.box_head = DecoderMLP(hidden_size, hidden_size, 4, 3)

        self.query_embed = nn.Embedding(num_queries, hidden_size)
        self.reference_points = nn.Embedding(num_queries, 4)

        self.presence_token = nn.Embedding(1, hidden_size)
        self.presence_head = DecoderMLP(hidden_size, hidden_size, 1, 3)
        self.presence_layer_norm = nn.LayerNorm(hidden_size)
        self.clamp_presence_logit_max_val = 10.0

        self.ref_point_head = DecoderMLP(2 * hidden_size, hidden_size, hidden_size, 2)

        self.box_rpb_embed_x = DecoderMLP(2, hidden_size, num_attention_heads, 2)
        self.box_rpb_embed_y = DecoderMLP(2, hidden_size, num_attention_heads, 2)

        self.position_encoding = SinePositionEmbedding(num_pos_feats=hidden_size // 2, normalize=False)

    @compile_compatible_method_lru_cache(maxsize=1)
    def _get_coords(
        self,
        height: int,
        width: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate normalized coordinate grids for relative position bias computation.

        Args:
            height (int): Height of the spatial grid.
            width (int): Width of the spatial grid.
            dtype (torch.dtype): Data type for the output tensors.
            device (torch.device): Device for the output tensors.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Tuple of normalized coordinate grids:
                - coords_h: Height coordinates [height] normalized by height
                - coords_w: Width coordinates [width] normalized by width
        """
        coords_h = torch.arange(0, height, device=device, dtype=dtype) / height
        coords_w = torch.arange(0, width, device=device, dtype=dtype) / width
        return coords_h, coords_w

    def _get_rpb_matrix(
        self,
        reference_boxes: torch.Tensor,
        spatial_shape: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        """Compute box relative position bias (RPB) matrix using log-scale encoding.

        Computes relative position biases based on predicted box positions and spatial
        locations. Uses log-scale encoding to handle varying spatial scales. RPB helps
        the decoder attend to relevant spatial locations based on predicted box positions.

        Args:
        reference_boxes (torch.Tensor): Reference boxes [batch_size, num_queries, 4]
            in sigmoid space with floating-point dtype.
        spatial_shape (tuple[torch.Tensor, torch.Tensor]): (height, width) of the
            vision features as tensors.

        Returns:
        torch.Tensor: RPB matrix [batch_size, num_heads, num_queries, height*width]
            containing attention biases for each query to each spatial location with
            floating-point dtype.
        """
        height, width = spatial_shape
        boxes_xyxy = box_cxcywh_to_xyxy(reference_boxes)
        batch_size, num_queries, _ = boxes_xyxy.shape

        # Generate coordinate grids
        height_int = int(height)
        width_int = int(width)
        coords_h, coords_w = self._get_coords(
            height=height_int,
            width=width_int,
            dtype=reference_boxes.dtype,
            device=reference_boxes.device,
        )

        # Compute deltas between coordinates and box boundaries
        deltas_y = coords_h.view(1, -1, 1) - boxes_xyxy.reshape(-1, 1, 4)[:, :, 1:4:2]
        deltas_y = deltas_y.view(batch_size, num_queries, -1, 2)
        deltas_x = coords_w.view(1, -1, 1) - boxes_xyxy.reshape(-1, 1, 4)[:, :, 0:3:2]
        deltas_x = deltas_x.view(batch_size, num_queries, -1, 2)

        # Apply log-scale encoding
        deltas_x_log = deltas_x * 8
        deltas_x_log = torch.sign(deltas_x_log) * torch.log2(torch.abs(deltas_x_log) + 1.0) / math.log2(8)
        deltas_y_log = deltas_y * 8
        deltas_y_log = torch.sign(deltas_y_log) * torch.log2(torch.abs(deltas_y_log) + 1.0) / math.log2(8)

        # Embed deltas
        deltas_x = self.box_rpb_embed_x(deltas_x_log)  # [batch_size, num_queries, width, num_heads]
        deltas_y = self.box_rpb_embed_y(deltas_y_log)  # [batch_size, num_queries, height, num_heads]

        # Combine into 2D bias matrix
        rpb_matrix = deltas_y.unsqueeze(3) + deltas_x.unsqueeze(
            2,
        )  # [batch_size, num_queries, height, width, num_heads]
        rpb_matrix = rpb_matrix.flatten(2, 3)  # [batch_size, num_queries, height*width, num_heads]
        return rpb_matrix.permute(0, 3, 1, 2).contiguous()  # [batch_size, num_heads, num_queries, height*width]

    def forward(
        self,
        vision_features: torch.Tensor,
        text_features: torch.Tensor,
        vision_pos_encoding: torch.Tensor,
        text_mask: torch.Tensor | None = None,
        spatial_shapes: torch.Tensor | None = None,
        **kwargs: dict,
    ) -> dict[str, torch.Tensor | None]:
        """Forward pass for the DETR decoder.

        Processes queries through stacked decoder layers with progressive box refinement
        and presence token prediction. Intermediate outputs from all layers are collected.

        Args:
            vision_features (torch.Tensor): Vision features [batch_size, height*width,
                hidden_size] with floating-point dtype.
            text_features (torch.Tensor): Text features [batch_size, seq_len, hidden_size]
                with floating-point dtype.
            vision_pos_encoding (torch.Tensor): Vision position encoding
                [batch_size, height*width, hidden_size] with floating-point dtype.
            text_mask (torch.Tensor | None): Text padding mask [batch_size, seq_len] where
                True=valid, False=padding (dtype=bool). Default: None.
            spatial_shapes (torch.Tensor | None): Spatial shapes [num_levels, 2].
                Default: None.
            **kwargs: Additional keyword arguments passed to decoder layers.

        Returns:
            dict[str, torch.Tensor | None]: Dictionary containing:
                - 'intermediate_hidden_states': Stacked outputs from all layers
                    [num_layers, batch_size, num_queries, hidden_size]
                - 'reference_boxes': Refined box predictions
                    [num_layers, batch_size, num_queries, 4]
                - 'presence_logits': Presence token logits from all layers
                    [num_layers, batch_size, 1]
                - 'hidden_states': None (reserved for future use)
                - 'attentions': None (reserved for future use)

        Raises:
            ValueError: If ``spatial_shapes`` is provided with more than one level.
        """
        batch_size = vision_features.shape[0]

        query_embeds = self.query_embed.weight.unsqueeze(0).expand(batch_size, -1, -1)
        reference_boxes = self.reference_points.weight.unsqueeze(0).expand(batch_size, -1, -1)
        reference_boxes = reference_boxes.sigmoid()
        presence_token = self.presence_token.weight.unsqueeze(0).expand(batch_size, -1, -1)

        # Concatenate presence token with query embeddings
        hidden_states = torch.cat([presence_token, query_embeds], dim=1)

        text_cross_attn_mask = expand_attention_mask(text_mask) if text_mask is not None else None

        intermediate_outputs = []
        intermediate_boxes = [reference_boxes]
        intermediate_presence_logits = []

        for layer in self.layers:
            # Generate sine embeddings for conditional queries
            reference_points_input = reference_boxes.unsqueeze(2)
            query_sine_embed = self.position_encoding.encode_boxes(reference_points_input[:, :, 0, :])
            query_pos = self.ref_point_head(query_sine_embed)

            # Compute box relative position bias (RPB) attention mask
            vision_cross_attn_mask = None
            if spatial_shapes is not None:
                if spatial_shapes.shape[0] != 1:
                    msg = "RPB mask expects a single spatial level"
                    raise ValueError(msg)
                spatial_shape = (spatial_shapes[0, 0], spatial_shapes[0, 1])
                rpb_matrix = self._get_rpb_matrix(reference_boxes, spatial_shape)
                # Prepend zeros row for presence token (it attends to all vision tokens equally)
                vision_cross_attn_mask = functional.pad(rpb_matrix, (0, 0, 1, 0), mode="constant", value=0)

            hidden_states = layer(
                hidden_states,
                query_pos=query_pos,
                text_features=text_features,
                vision_features=vision_features,
                vision_pos_encoding=vision_pos_encoding,
                text_cross_attn_mask=text_cross_attn_mask,
                vision_cross_attn_mask=vision_cross_attn_mask,
                **kwargs,
            )

            # Extract query hidden states (without presence token) for box refinement
            query_hidden_states = hidden_states[:, 1:]

            # Box refinement: predict delta and update reference boxes
            reference_boxes_before_sigmoid = inverse_sigmoid(reference_boxes)
            delta_boxes = self.box_head(self.output_layer_norm(query_hidden_states))
            new_reference_boxes = (delta_boxes + reference_boxes_before_sigmoid).sigmoid()
            reference_boxes = new_reference_boxes.detach()

            intermediate_outputs.append(self.output_layer_norm(query_hidden_states))
            intermediate_boxes.append(new_reference_boxes)

            # Process presence token
            presence_hidden = hidden_states[:, :1]
            presence_logits = self.presence_head(self.presence_layer_norm(presence_hidden)).squeeze(-1)
            presence_logits = presence_logits.clamp(
                min=-self.clamp_presence_logit_max_val,
                max=self.clamp_presence_logit_max_val,
            )
            intermediate_presence_logits.append(presence_logits)

        # Stack outputs from all layers
        intermediate_outputs = torch.stack(intermediate_outputs)
        intermediate_boxes = torch.stack(intermediate_boxes[:-1])
        intermediate_presence_logits = torch.stack(intermediate_presence_logits)

        return {
            "intermediate_hidden_states": intermediate_outputs,
            "reference_boxes": intermediate_boxes,
            "presence_logits": intermediate_presence_logits,
            "hidden_states": None,
            "attentions": None,
        }
