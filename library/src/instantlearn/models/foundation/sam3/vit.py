# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# Copyright 2025 The Meta AI Authors and The HuggingFace Team. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Vision Transformer (ViT) components for SAM3."""

from collections.abc import Iterable

import torch
from torch import nn
from torch.nn import functional

from .common import MLP, SinePositionEmbedding


class ViTRotaryEmbedding(nn.Module):
    """Vision Rotary Position Embedding for SAM3.

    Supports 2D (axial) rotary embeddings for spatial dimensions, following
    transformers library standards.

    Attributes:
        end_x (int): X dimension size for rotary embeddings.
        end_y (int): Y dimension size for rotary embeddings.
        dim (int): Dimension size for each position (head_dim).
        rope_theta (float): Base frequency for rotary embeddings.
        scale (float): Scale factor for rotary embeddings.
        rope_embeddings_cos (torch.Tensor): Cosine position embeddings.
        rope_embeddings_sin (torch.Tensor): Sine position embeddings.
    """

    def __init__(
        self,
        end_x: int,
        end_y: int,
        hidden_size: int = 1024,
        num_attention_heads: int = 16,
        rope_theta: float = 10000.0,
        scale: float = 1.0,
    ) -> None:
        """Initialize ViTRotaryEmbedding.

        Args:
            end_x (int): X dimension size for rotary embeddings.
            end_y (int): Y dimension size for rotary embeddings.
            hidden_size (int): Hidden size of the model. Default: 1024.
            num_attention_heads (int): Number of attention heads. Default: 16.
            rope_theta (float): Base frequency for rotary embeddings. Default: 10000.0.
            scale (float): Scale factor for position indices. Default: 1.0.

        Raises:
            ValueError: If dimension is not divisible by 4 for axial RoPE.
        """
        super().__init__()
        dim = hidden_size // num_attention_heads
        if dim % 4 != 0:
            msg = "Dimension must be divisible by 4 for axial RoPE"
            raise ValueError(msg)
        self.end_x, self.end_y = end_x, end_y
        self.dim = dim
        self.rope_theta = rope_theta
        self.scale = scale

        # Register buffers (will be initialized by _init_weights)
        self.register_buffer("rope_embeddings_cos", torch.empty(end_x * end_y, dim), persistent=False)
        self.register_buffer("rope_embeddings_sin", torch.empty(end_x * end_y, dim), persistent=False)
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize rotary position embeddings."""
        freqs = 1.0 / (self.rope_theta ** (torch.arange(0, self.dim, 4)[: (self.dim // 4)].float() / self.dim))
        flattened_indices = torch.arange(self.end_x * self.end_y, dtype=torch.long)
        x_positions = (flattened_indices % self.end_x) * self.scale
        y_positions = torch.div(flattened_indices, self.end_x, rounding_mode="floor") * self.scale
        freqs_x = torch.outer(x_positions, freqs).float()
        freqs_y = torch.outer(y_positions, freqs).float()
        inv_freq = torch.cat([freqs_x, freqs_y], dim=-1)
        inv_freq = inv_freq.repeat_interleave(2, dim=-1)
        self.rope_embeddings_cos.data.copy_(inv_freq.cos())
        self.rope_embeddings_sin.data.copy_(inv_freq.sin())

    @torch.no_grad()
    def forward(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return pre-computed rotary position embeddings.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Cosine and sine embeddings tensors.
        """
        # As the feature map size is fixed for each stage, we can just return the pre-computed embeddings.
        return self.rope_embeddings_cos, self.rope_embeddings_sin


class ViTRoPEAttention(nn.Module):
    """Self-attention with rotary position encoding.

    Implements multi-head self-attention with rotary position embeddings
    for Vision Transformer.

    Attributes:
        hidden_size (int): Hidden dimension size.
        num_attention_heads (int): Number of attention heads.
        head_dim (int): Dimension per attention head.
        scaling (float): Scaling factor for attention scores.
        attention_dropout (float): Dropout rate for attention.
        q_proj (Linear): Query projection layer.
        k_proj (Linear): Key projection layer.
        v_proj (Linear): Value projection layer.
        o_proj (Linear): Output projection layer.
    """

    def __init__(
        self,
        hidden_size: int = 1024,
        num_attention_heads: int = 16,
        attention_dropout: float = 0.0,
    ) -> None:
        """Initialize ViTRoPEAttention.

        Args:
            hidden_size (int): Hidden dimension size. Default: 1024.
            num_attention_heads (int): Number of attention heads. Default: 16.
            attention_dropout (float): Dropout rate for attention. Default: 0.0.
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.head_dim = hidden_size // num_attention_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = attention_dropout

        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.o_proj = nn.Linear(hidden_size, hidden_size)

    @staticmethod
    def _rotate_pairwise(x: torch.Tensor) -> torch.Tensor:
        """Pairwise rotation of the hidden dims of the input.

        Different from Llama Half-Tensor Rotation. This is an optimized version of
        the following more explicit implementation:

        ```python
        x_rotated = torch.zeros_like(x, dtype=x.dtype, device=x.device)
        x_rotated[..., ::2] = -x[..., 1::2]
        x_rotated[..., 1::2] = x[..., ::2]
        return x_rotated
        ```

        Args:
            x (Tensor): Input tensor of shape (..., hidden_dim).

        Returns:
            Tensor: Rotated tensor of shape (..., hidden_dim).
        """
        x = x.view(*x.shape[:-1], -1, 2)
        x1, x2 = x.unbind(dim=-1)
        x = torch.stack((-x2, x1), dim=-1)
        return x.flatten(start_dim=-2)

    @staticmethod
    def _apply_rotary_pos_emb_2d(
        q: torch.Tensor,
        k: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply rotary position embedding to query and key tensors for self-attention.

        Args:
            q (torch.Tensor): Query tensor of shape (batch_size, num_windows, seq_len,
                num_heads, head_dim).
            k (torch.Tensor): Key tensor of shape (batch_size, num_windows, seq_len,
                num_heads, head_dim).
            cos (torch.Tensor): Cosine position embedding of shape (seq_len, head_dim).
            sin (torch.Tensor): Sine position embedding of shape (seq_len, head_dim).

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Rotated (q, k) tensors with same shapes as input.
        """
        rotate = ViTRoPEAttention._rotate_pairwise
        q_embed = q.float()
        q_embed = (q_embed * cos) + (rotate(q_embed) * sin)

        k_embed = k.float()
        k_embed = (k_embed * cos) + (rotate(k_embed) * sin)

        return q_embed.type_as(q), k_embed.type_as(k)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        """Apply self-attention with rotary position embeddings.

        Args:
            hidden_states (torch.Tensor): Input hidden states of shape (batch_size,
                height, width, hidden_size).
            position_embeddings (tuple[torch.Tensor, torch.Tensor]): Cosine and sine
                position embeddings of shape (seq_len, head_dim).

        Returns:
            torch.Tensor: Attention output of shape (batch_size, height, width, hidden_size).
        """
        batch_size, height, width, _ = hidden_states.shape
        seq_len = height * width
        new_shape = (batch_size, seq_len, self.num_attention_heads, self.head_dim)
        query = self.q_proj(hidden_states).view(*new_shape).transpose(1, 2)
        key = self.k_proj(hidden_states).view(*new_shape).transpose(1, 2)
        value = self.v_proj(hidden_states).view(*new_shape).transpose(1, 2)
        cos, sin = position_embeddings
        query, key = self._apply_rotary_pos_emb_2d(query, key, cos=cos, sin=sin)

        attn_output = functional.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=None,
            dropout_p=0.0 if not self.training else self.attention_dropout,
            scale=self.scaling,
        )

        attn_output = attn_output.transpose(1, 2).reshape(batch_size, height, width, -1).contiguous()
        return self.o_proj(attn_output)


class ViTPatchEmbeddings(nn.Module):
    """Convert pixel values to patch embeddings for ViT.

    This class converts image pixels of shape (batch_size, num_channels, height,
    width) into patch embeddings (batch_size, seq_length, hidden_size) to be
    consumed by a Transformer.

    Attributes:
        image_size (tuple[int, int]): Expected input image size (height, width).
        patch_size (tuple[int, int]): Size of each patch (height, width).
        num_channels (int): Number of input image channels.
        num_patches (int): Total number of patches.
        projection (Conv2d): Convolutional layer for patch projection.
    """

    def __init__(
        self,
        pretrain_image_size: int = 336,
        patch_size: int = 14,
        num_channels: int = 3,
        hidden_size: int = 1024,
    ) -> None:
        """Initialize ViTPatchEmbeddings.

        Args:
            pretrain_image_size (int): Pretraining image size. Default: 336.
            patch_size (int): Patch size. Default: 14.
            num_channels (int): Number of input channels. Default: 3.
            hidden_size (int): Hidden dimension size. Default: 1024.
        """
        super().__init__()
        image_size = (
            pretrain_image_size
            if isinstance(pretrain_image_size, Iterable)
            else (pretrain_image_size, pretrain_image_size)
        )
        patch_size = patch_size if isinstance(patch_size, Iterable) else (patch_size, patch_size)
        num_patches = (image_size[1] // patch_size[1]) * (image_size[0] // patch_size[0])
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.num_patches = num_patches

        self.projection = nn.Conv2d(num_channels, hidden_size, kernel_size=patch_size, stride=patch_size, bias=False)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Convert pixel values to patch embeddings.

        Args:
            pixel_values (torch.Tensor): Input images of shape (batch_size, num_channels, height, width).

        Returns:
            torch.Tensor: Patch embeddings of shape (batch_size, seq_length, hidden_size).
        """
        return self.projection(pixel_values.to(self.projection.weight.dtype)).flatten(2).transpose(1, 2)


class ViTEmbeddings(nn.Module):
    """Patch embeddings and position embeddings for SAM3 ViT.

    Combines patch embeddings with learnable position embeddings. Position
    embeddings are tiled (not interpolated) when resizing to match different
    input sizes.

    Attributes:
        patch_embeddings (ViTPatchEmbeddings): Patch embedding layer.
        position_embeddings (Parameter): Learnable position embedding parameters.
        dropout (Dropout): Dropout layer.
        patch_size (int): Patch size.
    """

    def __init__(
        self,
        pretrain_image_size: int = 336,
        patch_size: int = 14,
        num_channels: int = 3,
        hidden_size: int = 1024,
        hidden_dropout: float = 0.0,
        initializer_range: float = 0.02,
    ) -> None:
        """Initialize ViTEmbeddings.

        Args:
            pretrain_image_size (int): Pretraining image size. Default: 336.
            patch_size (int): Patch size. Default: 14.
            num_channels (int): Number of input channels. Default: 3.
            hidden_size (int): Hidden dimension size. Default: 1024.
            hidden_dropout (float): Dropout rate for hidden states. Default: 0.0.
            initializer_range (float): Std deviation for weight initialization.
                Default: 0.02.
        """
        super().__init__()
        self.initializer_range = initializer_range

        self.patch_embeddings = ViTPatchEmbeddings(
            pretrain_image_size=pretrain_image_size,
            patch_size=patch_size,
            num_channels=num_channels,
            hidden_size=hidden_size,
        )
        num_patches = self.patch_embeddings.num_patches
        self.position_embeddings = nn.Parameter(torch.empty(1, num_patches, hidden_size))

        self.dropout = nn.Dropout(hidden_dropout)
        self.patch_size = patch_size
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize position embeddings with normal distribution."""
        nn.init.normal_(self.position_embeddings, mean=0.0, std=self.initializer_range)

    @staticmethod
    def _tile_position_embeddings(
        position_embeddings: torch.Tensor,
        height: int,
        width: int,
    ) -> torch.Tensor:
        """Tile position embeddings to match target spatial dimensions.

        Args:
            position_embeddings (torch.Tensor): Shape [1, num_pretrain_patches,
                hidden_size].
            height (int): Target height in patches.
            width (int): Target width in patches.

        Returns:
            torch.Tensor: Tiled position embeddings of shape [1, height * width,
                hidden_size].
        """
        pretrain_size = int(position_embeddings.shape[1] ** 0.5)

        # Skip tiling if sizes match (but always tile during tracing for consistent graph)
        if not torch.jit.is_tracing() and pretrain_size == height and pretrain_size == width:
            return position_embeddings.reshape(1, height * width, -1)

        # Tile position embeddings to match target spatial dimensions
        hidden_size = position_embeddings.shape[-1]
        pos_embed = position_embeddings.reshape(1, pretrain_size, pretrain_size, hidden_size).permute(0, 3, 1, 2)
        repeat_h = height // pretrain_size + 1
        repeat_w = width // pretrain_size + 1
        pos_embed = pos_embed.tile([1, 1, repeat_h, repeat_w])[:, :, :height, :width]
        return pos_embed.permute(0, 2, 3, 1).reshape(1, height * width, hidden_size)

    def forward(
        self,
        pixel_values: torch.Tensor,
    ) -> torch.Tensor:
        """Add patch and position embeddings.

        Args:
            pixel_values (torch.Tensor): Input images of shape (batch_size, num_channels,
                height, width).

        Returns:
            torch.Tensor: Combined embeddings of shape (batch_size, seq_length,
                hidden_size).
        """
        height, width = pixel_values.shape[-2:]
        embeddings = self.patch_embeddings(pixel_values)

        # Calculate spatial dimensions in patches
        height_patches = height // self.patch_size
        width_patches = width // self.patch_size

        position_embeddings = self._tile_position_embeddings(
            self.position_embeddings,
            height_patches,
            width_patches,
        )
        embeddings += position_embeddings
        return self.dropout(embeddings)


class ViTLayerScale(nn.Module):
    """Layer scaling for Vision Transformer residual connections.

    Applies learnable scaling to layer outputs for improved training stability.

    Attributes:
        lambda1 (Parameter): Learnable scaling parameter.
    """

    def __init__(
        self,
        hidden_size: int = 1024,
        layer_scale_init_value: float = 1.0,
    ) -> None:
        """Initialize ViTLayerScale.

        Args:
            hidden_size (int): Hidden dimension size. Default: 1024.
            layer_scale_init_value (float): Initial value for scaling parameter.
                Default: 1.0.
        """
        super().__init__()
        self.lambda1 = nn.Parameter(layer_scale_init_value * torch.ones(hidden_size))

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        """Scale hidden state.

        Args:
            hidden_state (torch.Tensor): Input hidden state tensor.

        Returns:
            torch.Tensor: Scaled hidden state.
        """
        return hidden_state * self.lambda1


class ViTLayer(nn.Module):
    """Vision Transformer layer with rotary position embeddings and windowed attention.

    Implements a single transformer block with layer normalization, multi-head
    self-attention, and feed-forward network. Supports optional windowed
    attention for efficiency.

    Attributes:
        layer_norm1 (LayerNorm): Layer normalization for attention.
        rotary_emb (ViTRotaryEmbedding): Rotary position embeddings.
        attention (ViTRoPEAttention): Multi-head attention with RoPE.
        layer_norm2 (LayerNorm): Layer normalization for MLP.
        mlp (MLP): Feed-forward network.
        dropout (Dropout): Dropout layer.
        window_size (int): Window size for windowed attention (0 = global).
    """

    def __init__(
        self,
        hidden_size: int = 1024,
        intermediate_size: int = 4736,
        num_attention_heads: int = 16,
        image_size: int = 1008,
        patch_size: int = 14,
        layer_norm_eps: float = 1e-6,
        hidden_act: str = "gelu",
        hidden_dropout: float = 0.0,
        attention_dropout: float = 0.0,
        rope_theta: float = 10000.0,
        config_window_size: int = 24,
        window_size: int = 0,
    ) -> None:
        """Initialize ViTLayer.

        Args:
            hidden_size (int): Hidden dimension size. Default: 1024.
            intermediate_size (int): MLP intermediate dimension. Default: 4736.
            num_attention_heads (int): Number of attention heads. Default: 16.
            image_size (int): Input image size in pixels. Default: 1008.
            patch_size (int): Patch size in pixels. Default: 14.
            layer_norm_eps (float): Layer norm epsilon. Default: 1e-6.
            hidden_act (str): Activation function name. Default: "gelu".
            hidden_dropout (float): Hidden state dropout rate. Default: 0.0.
            attention_dropout (float): Attention dropout rate. Default: 0.0.
            rope_theta (float): RoPE base frequency. Default: 10000.0.
            config_window_size (int): Configured window size. Default: 24.
            window_size (int): Actual window size for attention (0=global).
                Default: 0.
        """
        super().__init__()

        img_size = image_size if isinstance(image_size, (list, tuple)) else (image_size, image_size)
        p_size = patch_size if isinstance(patch_size, (list, tuple)) else (patch_size, patch_size)

        input_size = (img_size[0] // p_size[0], img_size[1] // p_size[1])
        self.layer_norm1 = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        rotary_input_size = input_size if window_size == 0 else (window_size, window_size)
        rotary_scale = config_window_size / rotary_input_size[0]
        self.rotary_emb = ViTRotaryEmbedding(
            end_x=rotary_input_size[0],
            end_y=rotary_input_size[1],
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            rope_theta=rope_theta,
            scale=rotary_scale,
        )
        self.attention = ViTRoPEAttention(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            attention_dropout=attention_dropout,
        )
        self.layer_norm2 = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.mlp = MLP(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            hidden_act=hidden_act,
            hidden_dropout=hidden_dropout,
        )
        self.dropout = nn.Dropout(hidden_dropout)

        self.window_size = window_size

    @staticmethod
    def _window_partition(
        hidden_state: torch.Tensor,
        window_size: int,
    ) -> tuple[torch.Tensor, tuple[int, int]]:
        """Partition into non-overlapping windows with padding if needed.

        Args:
            hidden_state (torch.Tensor): Input tokens with shape [batch_size, height, width,
                num_channels].
            window_size (int): Window size for partitioning.

        Returns:
            tuple[torch.Tensor, tuple[int, int]]: A tuple containing:
                - windows (torch.Tensor): Windows after partition with shape [batch_size *
                  num_windows, window_size, window_size, num_channels].
                - (padded_height, padded_width) (tuple[int, int]): Padded height
                  and width before partition.
        """
        batch_size, height, width, num_channels = hidden_state.shape
        pad_height = (window_size - height % window_size) % window_size
        pad_width = (window_size - width % window_size) % window_size

        # Noop in case pad_width == 0 and pad_height == 0.
        hidden_state = nn.functional.pad(hidden_state, (0, 0, 0, pad_width, 0, pad_height))

        padded_height, padded_width = height + pad_height, width + pad_width

        hidden_state = hidden_state.view(
            batch_size,
            padded_height // window_size,
            window_size,
            padded_width // window_size,
            window_size,
            num_channels,
        )
        windows = hidden_state.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, num_channels)
        return windows, (padded_height, padded_width)

    @staticmethod
    def _window_unpartition(
        windows: torch.Tensor,
        window_size: int,
        pad_height_width: tuple[int, int],
        height_width: tuple[int, int],
    ) -> torch.Tensor:
        """Window unpartition into original sequences and removing padding.

        Args:
            windows (torch.Tensor): Input tokens with shape [batch_size * num_windows,
                window_size, window_size, num_channels].
            window_size (int): Window size used for partitioning.
            pad_height_width (tuple[int, int]): Padded (height, width) dimensions
                from window_partition.
            height_width (tuple[int, int]): Original (height, width) dimensions
                before padding.

        Returns:
            torch.Tensor: Unpartitioned sequences with shape [batch_size, height, width,
                num_channels].
        """
        padded_height, padded_width = pad_height_width
        height, width = height_width
        batch_size = windows.shape[0] // (padded_height * padded_width // window_size // window_size)
        hidden_state = windows.view(
            batch_size,
            padded_height // window_size,
            padded_width // window_size,
            window_size,
            window_size,
            -1,
        )
        hidden_state = hidden_state.permute(0, 1, 3, 2, 4, 5).contiguous()
        hidden_state = hidden_state.view(batch_size, padded_height, padded_width, -1)

        # We always have height <= padded_height and width <= padded_width
        return hidden_state[:, :height, :width, :].contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """Apply transformer layer with attention and MLP.

        Args:
            hidden_states (torch.Tensor): Input hidden states of shape (batch_size,
                height, width, hidden_size).

        Returns:
            torch.Tensor: Output hidden states of shape (batch_size, height, width,
                hidden_size).
        """
        residual = hidden_states

        hidden_states = self.layer_norm1(hidden_states)

        if self.window_size > 0:
            height, width = hidden_states.shape[1], hidden_states.shape[2]
            # Partition into non-overlapping windows for efficient attention
            hidden_states, pad_height_width = self._window_partition(hidden_states, self.window_size)

        position_embeddings = self.rotary_emb()
        hidden_states = self.attention(hidden_states, position_embeddings)

        if self.window_size > 0:
            # Reverse window partition to restore original spatial layout
            hidden_states = self._window_unpartition(hidden_states, self.window_size, pad_height_width, (height, width))

        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return residual + self.dropout(hidden_states)


class ViTModel(nn.Module):
    """Vision Transformer backbone for SAM3.

    Implements a complete Vision Transformer with patch embeddings, learnable
    position embeddings, and stacked transformer layers with optional windowed
    attention.

    Args:
        hidden_size (int): Dimensionality of the encoder layers. Default: 1024.
        intermediate_size (int): Dimensionality of the feedforward (MLP) layers.
            Default: 4736.
        num_hidden_layers (int): Number of hidden layers in the Transformer
            encoder. Default: 32.
        num_attention_heads (int): Number of attention heads. Default: 16.
        num_channels (int): Number of input image channels. Default: 3.
        image_size (int): Expected input image size. Default: 1008.
        patch_size (int): Size of image patches. Default: 14.
        hidden_act (str): The non-linear activation function. Default: "gelu".
        layer_norm_eps (float): Epsilon for layer normalization. Default: 1e-6.
        attention_dropout (float): Dropout ratio for attention probabilities.
            Default: 0.0.
        rope_theta (float): Base frequency for RoPE. Default: 10000.0.
        window_size (int): Window size for windowed attention. Default: 24.
        global_attn_indexes (list[int] | None): Indexes of layers with global
            attention. Default: [7, 15, 23, 31].
        pretrain_image_size (int): Pretrained model image size for position
            embedding init. Default: 336.
        hidden_dropout (float): Dropout probability for hidden states.
            Default: 0.0.
        initializer_range (float): Std deviation for weight initialization.
            Default: 0.02.
    """

    def __init__(
        self,
        hidden_size: int = 1024,
        intermediate_size: int = 4736,
        num_hidden_layers: int = 32,
        num_attention_heads: int = 16,
        num_channels: int = 3,
        image_size: int = 1008,
        patch_size: int = 14,
        hidden_act: str = "gelu",
        layer_norm_eps: float = 1e-6,
        attention_dropout: float = 0.0,
        rope_theta: float = 10000.0,
        window_size: int = 24,
        global_attn_indexes: list[int] | None = None,
        pretrain_image_size: int = 336,
        hidden_dropout: float = 0.0,
        initializer_range: float = 0.02,
    ) -> None:
        """Initialize ViTModel.

        Args:
            hidden_size (int): Dimensionality of encoder layers. Default: 1024.
            intermediate_size (int): Dimensionality of feedforward layers.
                Default: 4736.
            num_hidden_layers (int): Number of transformer layers. Default: 32.
            num_attention_heads (int): Number of attention heads. Default: 16.
            num_channels (int): Number of input image channels. Default: 3.
            image_size (int): Input image size in pixels. Default: 1008.
            patch_size (int): Patch size in pixels. Default: 14.
            hidden_act (str): Activation function name. Default: "gelu".
            layer_norm_eps (float): Layer norm epsilon. Default: 1e-6.
            attention_dropout (float): Attention dropout rate. Default: 0.0.
            rope_theta (float): RoPE base frequency. Default: 10000.0.
            window_size (int): Window size for windowed attention. Default: 24.
            global_attn_indexes (list[int] | None): Layer indices with global
                attention. Default: [7, 15, 23, 31].
            pretrain_image_size (int): Pretrained image size. Default: 336.
            hidden_dropout (float): Hidden state dropout rate. Default: 0.0.
            initializer_range (float): Weight initialization std deviation.
                Default: 0.02.
        """
        super().__init__()
        if global_attn_indexes is None:
            global_attn_indexes = [7, 15, 23, 31]

        # Store for forward pass
        self.patch_size = patch_size

        self.embeddings = ViTEmbeddings(
            pretrain_image_size=pretrain_image_size,
            patch_size=patch_size,
            num_channels=num_channels,
            hidden_size=hidden_size,
            hidden_dropout=hidden_dropout,
            initializer_range=initializer_range,
        )
        self.layer_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.layers = nn.ModuleList(
            [
                ViTLayer(
                    hidden_size=hidden_size,
                    intermediate_size=intermediate_size,
                    num_attention_heads=num_attention_heads,
                    image_size=image_size,
                    patch_size=patch_size,
                    layer_norm_eps=layer_norm_eps,
                    hidden_act=hidden_act,
                    hidden_dropout=hidden_dropout,
                    attention_dropout=attention_dropout,
                    rope_theta=rope_theta,
                    config_window_size=window_size,
                    window_size=window_size if i not in global_attn_indexes else 0,
                )
                for i in range(num_hidden_layers)
            ],
        )

    def get_input_embeddings(self) -> ViTPatchEmbeddings:
        """Get the patch embedding layer.

        Returns:
            ViTPatchEmbeddings: The patch embeddings module.
        """
        return self.embeddings.patch_embeddings

    def forward(
        self,
        pixel_values: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Apply Vision Transformer backbone.

        Args:
            pixel_values (torch.Tensor): Input images of shape (batch_size, num_channels,
                height, width).

        Returns:
            dict[str, torch.Tensor]: Dictionary with key "last_hidden_state" containing
                output hidden states of shape (batch_size, seq_len, hidden_size).
        """
        hidden_states = self.embeddings(pixel_values)  # [batch_size, seq_len, hidden_size]

        batch_size = hidden_states.shape[0]
        height = pixel_values.shape[-2] // self.patch_size
        width = pixel_values.shape[-1] // self.patch_size
        hidden_size = hidden_states.shape[-1]

        # Reshape to spatial format for windowed attention: [batch_size, height, width, hidden_size]
        hidden_states = hidden_states.view(batch_size, height, width, hidden_size)

        hidden_states = self.layer_norm(hidden_states)
        for layer in self.layers:
            hidden_states = layer(hidden_states)

        # Reshape back to sequence format: [batch_size, height*width, hidden_size]
        hidden_states = hidden_states.view(batch_size, height * width, hidden_size)

        return {"last_hidden_state": hidden_states}


class FPNLayer(nn.Module):
    """Feature Pyramid Network layer for multi-scale feature extraction.

    Applies upsampling/downsampling and convolutions to adjust feature maps
    to a target scale.

    Attributes:
        scale_factor (float): Scaling factor for this FPN layer.
        scale_layers (ModuleList): Scaling operation layers.
        proj1 (Conv2d): First projection convolution layer.
        proj2 (Conv2d): Second projection convolution layer.
    """

    def __init__(
        self,
        in_channels: int,
        fpn_dim: int,
        scale_factor: float,
    ) -> None:
        """Initialize FPNLayer.

        Args:
            in_channels (int): Number of input channels.
            fpn_dim (int): Output feature dimension.
            scale_factor (float): Scaling factor for upsampling/downsampling.

        Raises:
            NotImplementedError: If scale_factor is not in [4.0, 2.0, 1.0, 0.5].
        """
        super().__init__()
        self.scale_factor = scale_factor

        # Build the upsampling/downsampling layers based on scale factor
        self.scale_layers = nn.ModuleList()

        if scale_factor == 4.0:
            self.scale_layers.append(nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2))
            self.scale_layers.append(nn.GELU())
            self.scale_layers.append(nn.ConvTranspose2d(in_channels // 2, in_channels // 4, kernel_size=2, stride=2))
            intermediate_channels = in_channels // 4
        elif scale_factor == 2.0:
            self.scale_layers.append(nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2))
            intermediate_channels = in_channels // 2
        elif scale_factor == 1.0:
            intermediate_channels = in_channels
        elif scale_factor == 0.5:
            self.scale_layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            intermediate_channels = in_channels
        else:
            msg = f"scale_factor={scale_factor} is not supported yet."
            raise NotImplementedError(msg)

        self.proj1 = nn.Conv2d(in_channels=intermediate_channels, out_channels=fpn_dim, kernel_size=1)
        self.proj2 = nn.Conv2d(in_channels=fpn_dim, out_channels=fpn_dim, kernel_size=3, padding=1)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Apply FPN scaling and projections.

        Args:
            hidden_states (torch.Tensor): Input feature map.

        Returns:
            torch.Tensor: Output feature map of shape (..., fpn_dim, *, *).
        """
        hidden_states = hidden_states.to(self.proj1.weight.dtype)
        for layer in self.scale_layers:
            hidden_states = layer(hidden_states)

        hidden_states = self.proj1(hidden_states)
        return self.proj2(hidden_states)


class VisionNeck(nn.Module):
    """Vision Transformer feature pyramid neck for multi-scale features.

    Applies FPN layers to generate multi-scale feature representations from
    ViT backbone outputs.

    Attributes:
        position_encoding (SinePositionEmbedding): Sinusoidal position embedding.
        fpn_layers (ModuleList): List of FPN layers for different scales.
    """

    def __init__(
        self,
        backbone_hidden_size: int = 1024,
        fpn_hidden_size: int = 256,
        scale_factors: list[float] | None = None,
    ) -> None:
        """Initialize VisionNeck.

        Args:
            backbone_hidden_size (int): Backbone hidden dimension. Default: 1024.
            fpn_hidden_size (int): FPN output dimension. Default: 256.
            scale_factors (list[float] | None): Scale factors for FPN layers.
                Default: [4.0, 2.0, 1.0, 0.5].
        """
        super().__init__()
        if scale_factors is None:
            scale_factors = [4.0, 2.0, 1.0, 0.5]

        self.position_encoding = SinePositionEmbedding(num_pos_feats=fpn_hidden_size // 2, normalize=True)

        self.fpn_layers = nn.ModuleList(
            [
                FPNLayer(in_channels=backbone_hidden_size, fpn_dim=fpn_hidden_size, scale_factor=scale)
                for scale in scale_factors
            ],
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
        """Generate multi-scale features and position encodings.

        Args:
            hidden_states (torch.Tensor): Backbone features of shape (batch_size,
                hidden_size, height, width).

        Returns:
            tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]: A tuple containing:
                - fpn_hidden_states: Multi-scale feature maps.
                - fpn_position_encoding: Position encodings for each scale.
        """
        fpn_hidden_states: tuple[torch.Tensor, ...] = ()
        fpn_position_encoding: tuple[torch.Tensor, ...] = ()

        for fpn_layer in self.fpn_layers:
            fpn_output = fpn_layer(hidden_states)
            fpn_hidden_states += (fpn_output,)
            pos_enc = self.position_encoding(fpn_output.shape, fpn_output.device, fpn_output.dtype)
            fpn_position_encoding += (pos_enc,)

        return fpn_hidden_states, fpn_position_encoding


class VisionModel(nn.Module):
    """The vision model from SAM3 without any head or projection on top.

    Combines ViT backbone with FPN neck for multi-scale feature extraction.

    Args:
        hidden_size (int): Dimensionality of the ViT encoder layers. Default: 1024.
        intermediate_size (int): Dimensionality of the ViT feedforward (MLP)
            layers. Default: 4736.
        num_hidden_layers (int): Number of hidden layers in the ViT encoder.
            Default: 32.
        num_attention_heads (int): Number of attention heads. Default: 16.
        num_channels (int): Number of input image channels. Default: 3.
        image_size (int): Expected input image size. Default: 1008.
        patch_size (int): Size of image patches. Default: 14.
        hidden_act (str): The non-linear activation function. Default: "gelu".
        layer_norm_eps (float): Epsilon for layer normalization. Default: 1e-6.
        attention_dropout (float): Dropout ratio for attention probabilities.
            Default: 0.0.
        rope_theta (float): Base frequency for RoPE. Default: 10000.0.
        window_size (int): Window size for windowed attention. Default: 24.
        global_attn_indexes (list[int] | None): Indexes of layers with global
            attention. Default: [7, 15, 23, 31].
        pretrain_image_size (int): Pretrained model image size for position
            embedding init. Default: 336.
        hidden_dropout (float): Dropout probability for hidden states.
            Default: 0.0.
        initializer_range (float): Std deviation for weight initialization.
            Default: 0.02.
        fpn_hidden_size (int): The hidden dimension of the FPN. Default: 256.
        scale_factors (list[float] | None): Scale factors for FPN multi-scale
            features. Default: [4.0, 2.0, 1.0, 0.5].
    """

    def __init__(
        self,
        hidden_size: int = 1024,
        intermediate_size: int = 4736,
        num_hidden_layers: int = 32,
        num_attention_heads: int = 16,
        num_channels: int = 3,
        image_size: int = 1008,
        patch_size: int = 14,
        hidden_act: str = "gelu",
        layer_norm_eps: float = 1e-6,
        attention_dropout: float = 0.0,
        rope_theta: float = 10000.0,
        window_size: int = 24,
        global_attn_indexes: list[int] | None = None,
        pretrain_image_size: int = 336,
        hidden_dropout: float = 0.0,
        initializer_range: float = 0.02,
        fpn_hidden_size: int = 256,
        scale_factors: list[float] | None = None,
    ) -> None:
        """Initialize VisionModel.

        Args:
            hidden_size (int): ViT encoder hidden dimension. Default: 1024.
            intermediate_size (int): ViT feedforward dimension. Default: 4736.
            num_hidden_layers (int): Number of ViT layers. Default: 32.
            num_attention_heads (int): Number of attention heads. Default: 16.
            num_channels (int): Input image channels. Default: 3.
            image_size (int): Input image size in pixels. Default: 1008.
            patch_size (int): Patch size in pixels. Default: 14.
            hidden_act (str): Activation function name. Default: "gelu".
            layer_norm_eps (float): Layer norm epsilon. Default: 1e-6.
            attention_dropout (float): Attention dropout rate. Default: 0.0.
            rope_theta (float): RoPE base frequency. Default: 10000.0.
            window_size (int): Window size for windowed attention. Default: 24.
            global_attn_indexes (list[int] | None): Layer indices with global
                attention. Default: [7, 15, 23, 31].
            pretrain_image_size (int): Pretrained image size. Default: 336.
            hidden_dropout (float): Hidden state dropout rate. Default: 0.0.
            initializer_range (float): Weight initialization std deviation.
                Default: 0.02.
            fpn_hidden_size (int): FPN output dimension. Default: 256.
            scale_factors (list[float] | None): FPN scale factors. Default:
                [4.0, 2.0, 1.0, 0.5].
        """
        super().__init__()
        if global_attn_indexes is None:
            global_attn_indexes = [7, 15, 23, 31]
        if scale_factors is None:
            scale_factors = [4.0, 2.0, 1.0, 0.5]

        # Store for forward pass
        self.patch_size = patch_size

        self.backbone = ViTModel(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            num_channels=num_channels,
            image_size=image_size,
            patch_size=patch_size,
            hidden_act=hidden_act,
            layer_norm_eps=layer_norm_eps,
            attention_dropout=attention_dropout,
            rope_theta=rope_theta,
            window_size=window_size,
            global_attn_indexes=global_attn_indexes,
            pretrain_image_size=pretrain_image_size,
            hidden_dropout=hidden_dropout,
            initializer_range=initializer_range,
        )
        self.neck = VisionNeck(
            backbone_hidden_size=hidden_size,
            fpn_hidden_size=fpn_hidden_size,
            scale_factors=scale_factors,
        )

    def get_input_embeddings(self) -> ViTPatchEmbeddings:
        """Get the patch embedding layer.

        Returns:
            ViTPatchEmbeddings: The patch embeddings module from backbone.
        """
        return self.backbone.get_input_embeddings()

    def forward(
        self,
        pixel_values: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | None]:
        """Apply complete vision model (ViT + FPN).

        Args:
            pixel_values (torch.Tensor | None): Input images of shape (batch_size,
                num_channels, height, width). Required.

        Returns:
            dict[str, torch.Tensor | None]: Dictionary containing:
                - "last_hidden_state": ViT output of shape (batch_size, seq_len,
                  hidden_size).
                - "fpn_hidden_states": Tuple of FPN feature maps at different scales.
                - "fpn_position_encoding": Tuple of position encodings for each FPN scale.
                - "hidden_states": None (for compatibility).
                - "attentions": None (for compatibility).

        Raises:
            ValueError: If pixel_values is None.
        """
        if pixel_values is None:
            msg = "You have to specify pixel_values"
            raise ValueError(msg)

        backbone_output = self.backbone(pixel_values)
        hidden_states = backbone_output["last_hidden_state"]  # [batch_size, seq_len, hidden_size]

        # Reshape for FPN neck: [batch_size, seq_len, hidden_size] -> [batch_size, hidden_size, height, width]
        batch_size = hidden_states.shape[0]
        height = pixel_values.shape[-2] // self.patch_size
        width = pixel_values.shape[-1] // self.patch_size
        hidden_states_spatial = hidden_states.view(batch_size, height, width, -1).permute(0, 3, 1, 2)
        fpn_hidden_states, fpn_position_encoding = self.neck(hidden_states_spatial)

        return {
            "last_hidden_state": hidden_states,
            "fpn_hidden_states": fpn_hidden_states,
            "fpn_position_encoding": fpn_position_encoding,
            "hidden_states": None,
            "attentions": None,
        }
