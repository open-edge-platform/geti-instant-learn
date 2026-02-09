# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# Copyright 2025 The Meta AI Authors and The HuggingFace Team. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Common utilities and base classes for SAM3 model components."""

import logging
import math

import torch
from torch import nn
from torch.nn import functional
from transformers.pytorch_utils import compile_compatible_method_lru_cache

# Activation functions used by SAM3
ACT2FN = {
    "gelu": functional.gelu,
    "relu": functional.relu,
}


logger = logging.getLogger(__name__)


def inverse_sigmoid(x: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    """Compute the inverse of the sigmoid activation.

    Args:
        x (torch.Tensor): Input probability tensor of shape (...,) with floating-point
            dtype (typically float32/float16).
        eps (float): Clamp value to avoid numerical instability. Defaults to 1e-3.

    Returns:
        torch.Tensor: Inverse-sigmoid values with the same dtype as ``x``.
    """
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)


def concat_padded_sequences(
    seq1: torch.Tensor,
    mask1: torch.Tensor,
    seq2: torch.Tensor,
    mask2: torch.Tensor,
    return_index: bool = False,
) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Concatenate two right-padded sequences into a single contiguous right-padded sequence.

    Concatenates two right-padded sequences such that the resulting sequence is contiguous
    and also right-padded. Tensors are batch-first with masks in batch-first format
    (True=valid, False=padding).

    Args:
        seq1 (torch.Tensor): Sequence tensor of shape (batch_size, seq1_length, hidden_size)
            with floating-point dtype.
        mask1 (torch.Tensor): Boolean mask of shape (batch_size, seq1_length) where
            True=valid and False=padding (dtype=bool).
        seq2 (torch.Tensor): Sequence tensor of shape (batch_size, seq2_length, hidden_size)
            with floating-point dtype.
        mask2 (torch.Tensor): Boolean mask of shape (batch_size, seq2_length) where
            True=valid and False=padding (dtype=bool).
        return_index (bool): If True, also return the indices of seq2 elements in the
            concatenated sequence. Defaults to False.

    Returns:
        If return_index is False: tuple of (concatenated_sequence, concatenated_mask)
            - concatenated_sequence (torch.Tensor): Concatenated sequence tensor with
              floating-point dtype.
            - concatenated_mask (torch.Tensor): Concatenated mask using True=valid,
              False=padding convention (dtype=bool).
        If return_index is True: tuple of (concatenated_sequence, concatenated_mask, index)
            - index (torch.Tensor): Indices of seq2 elements in concatenated sequence
              (dtype=long).

    Raises:
        ValueError: If batch sizes or sequence lengths do not match between inputs.
    """
    batch1_size, seq1_length, hidden1_size = seq1.shape
    batch2_size, seq2_length, hidden2_size = seq2.shape

    if not (batch1_size == batch2_size == mask1.size(0) == mask2.size(0)):
        msg = "Sequence batch sizes and mask batch sizes must match."
        raise ValueError(msg)
    if hidden1_size != hidden2_size:
        msg = "Sequence hidden sizes must match."
        raise ValueError(msg)
    if seq1_length != mask1.size(1):
        msg = "seq1_length must match mask1 length."
        raise ValueError(msg)
    if seq2_length != mask2.size(1):
        msg = "seq2_length must match mask2 length."
        raise ValueError(msg)

    actual_seq1_lengths = mask1.sum(dim=-1)
    actual_seq2_lengths = mask2.sum(dim=-1)

    final_lengths = actual_seq1_lengths + actual_seq2_lengths
    max_length = seq1_length + seq2_length

    concatenated_mask = torch.arange(max_length, device=seq2.device)[None].repeat(batch1_size, 1)
    concatenated_mask = concatenated_mask < final_lengths[:, None]

    concatenated_sequence = torch.zeros((batch1_size, max_length, hidden1_size), device=seq2.device, dtype=seq2.dtype)
    concatenated_sequence[:, :seq1_length, :] = seq1

    # Shift seq2 elements to start at the end of valid seq1
    index = torch.arange(seq2_length, device=seq2.device)[None].repeat(batch1_size, 1)
    index += actual_seq1_lengths[:, None]

    # Scatter seq2 into the right positions
    concatenated_sequence = concatenated_sequence.scatter(
        1,
        index[:, :, None].expand(-1, -1, hidden1_size),
        seq2,
    )

    if return_index:
        return concatenated_sequence, concatenated_mask, index

    return concatenated_sequence, concatenated_mask


def box_cxcywh_to_xyxy(x: torch.Tensor) -> torch.Tensor:
    """Convert bounding boxes from (cx, cy, w, h) format to (x1, y1, x2, y2) format.

    Args:
        x (torch.Tensor): Box coordinates in center-width-height format of shape
            (..., 4) where last dimension is [cx, cy, w, h] with floating-point dtype.

    Returns:
        torch.Tensor: Box coordinates in corner format of shape (..., 4) where last
            dimension is [x1, y1, x2, y2] with floating-point dtype.
    """
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


def expand_attention_mask(
    attention_mask: torch.Tensor,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Expand a 2D padding mask to 4D for use with scaled_dot_product_attention.

    For bidirectional (full) attention, queries can attend to all valid (non-padding)
    keys. The mask is expanded from [batch_size, key_len] to [batch_size, 1, 1, key_len].

    Args:
        attention_mask (torch.Tensor): 2D boolean mask of shape [batch_size, key_len]
            where True=valid and False=padding (dtype=bool).
        dtype (torch.dtype | None): Optional dtype for float conversion. If None, returns
            boolean mask. Defaults to None.

    Returns:
        torch.Tensor: 4D mask of shape [batch_size, 1, 1, key_len] compatible with SDPA
            attn_mask parameter. SDPA convention: True (or 0.0) = attend, False (or -inf)
            = mask out. Dtype is bool when ``dtype`` is None, otherwise floating-point.

    Raises:
        ValueError: If attention_mask is not 2-dimensional.
    """
    if attention_mask.ndim != 2:
        msg = f"Expected 2D attention_mask, got {attention_mask.ndim}D"
        raise ValueError(msg)

    # Expand: [B, S] -> [B, 1, 1, S]
    expanded_mask = attention_mask[:, None, None, :]

    if dtype is not None:
        # Convert to float: True -> 0.0, False -> -inf
        expanded_mask = torch.where(
            expanded_mask,
            torch.tensor(0.0, dtype=dtype, device=attention_mask.device),
            torch.finfo(dtype).min,
        )

    return expanded_mask


class MLP(nn.Module):
    """Multi-layer perceptron with configurable activation and dropout.

    A simple feed-forward network with two linear layers and an activation function.
    """

    def __init__(
        self,
        hidden_size: int = 1024,
        intermediate_size: int = 4736,
        hidden_act: str = "gelu",
        hidden_dropout: float = 0.0,
    ) -> None:
        """Initialize MLP module.

        Args:
            hidden_size (int): Dimension of input and output features. Defaults to 1024.
            intermediate_size (int): Dimension of the intermediate hidden layer.
                Defaults to 4736.
            hidden_act (str): Name of the activation function. Must be a key in ACT2FN.
                Defaults to "gelu".
            hidden_dropout (float): Dropout probability applied after the first linear
                layer. Defaults to 0.0.
        """
        super().__init__()
        self.activation_fn = ACT2FN[hidden_act]
        self.fc1 = nn.Linear(hidden_size, intermediate_size)
        self.fc2 = nn.Linear(intermediate_size, hidden_size)
        self.dropout = nn.Dropout(hidden_dropout)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Apply MLP transformation to hidden states.

        Args:
        hidden_states (torch.Tensor): Input tensor of shape (..., hidden_size) with
            floating-point dtype.

        Returns:
        torch.Tensor: Output tensor of same shape as input (..., hidden_size) with
            floating-point dtype.
        """
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        return self.fc2(hidden_states)


class Attention(nn.Module):
    """Multi-head attention module.

    Implements scaled dot-product attention with support for optional attention masks.
    Handles standard [batch_size, seq_len, hidden_size] tensors.
    """

    def __init__(
        self,
        hidden_size: int = 256,
        num_attention_heads: int = 8,
    ) -> None:
        """Initialize attention module.

        Args:
            hidden_size (int): Total dimension of the model. Defaults to 256.
            num_attention_heads (int): Number of attention heads. Must divide hidden_size
                evenly. Defaults to 8.
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.head_dim = hidden_size // num_attention_heads
        self.scaling = self.head_dim**-0.5

        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.o_proj = nn.Linear(hidden_size, hidden_size)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        **kwargs: dict,  # noqa: ARG002
    ) -> torch.Tensor:
        """Apply multi-head attention.

        Computes scaled dot-product attention over queries, keys, and values. Supports
        optional attention masks for masking invalid positions.

        Args:
        query (torch.Tensor): Query tensor of shape [batch_size, query_len, hidden_size]
            with floating-point dtype.
        key (torch.Tensor): Key tensor of shape [batch_size, key_len, hidden_size]
            with floating-point dtype.
        value (torch.Tensor): Value tensor of shape [batch_size, value_len, hidden_size]
            with floating-point dtype.
        attention_mask (torch.Tensor | None): Optional attention mask. Can be shape
            [batch_size, num_heads, query_len, key_len] or broadcastable. If provided,
            dtype is bool or floating-point. Defaults to None.

        **kwargs (dict): Additional keyword arguments (unused, for API compatibility).

        Returns:
            torch.Tensor: Output tensor of shape [batch_size, query_len, hidden_size] with floating-point dtype.
                Attention weights are not returned by SDPA.
        """
        batch_size = query.shape[0]
        query_len = query.shape[1]
        key_len = key.shape[1]

        query = self.q_proj(query)
        query = query.view(batch_size, query_len, self.num_attention_heads, self.head_dim).transpose(1, 2)
        key = self.k_proj(key)
        key = key.view(batch_size, key_len, self.num_attention_heads, self.head_dim).transpose(1, 2)
        value = self.v_proj(value)
        value = value.view(batch_size, key_len, self.num_attention_heads, self.head_dim).transpose(1, 2)

        attn_output = functional.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=0.0,
            scale=self.scaling,
        )

        attn_output = attn_output.transpose(1, 2).reshape(batch_size, query_len, self.hidden_size).contiguous()
        return self.o_proj(attn_output)


class SinePositionEmbedding(nn.Module):
    """Sine/cosine positional embeddings for sequences and bounding boxes.

    A standard version of position embedding similar to "Attention is All You Need",
    generalized to work on images. Supports both 1D coordinate encoding and 4D box
    coordinate encoding.
    """

    def __init__(
        self,
        num_pos_feats: int = 64,
        temperature: int = 10000,
        normalize: bool = False,
        scale: float | None = None,
    ) -> None:
        """Initialize sine position embedding.

        Args:
            num_pos_feats (int): Number of positional features (half of output dimension
                for 1D, quarter for 4D boxes). Defaults to 64.
            temperature (int): Temperature parameter for positional embeddings. Controls
                the frequency range of sine/cosine waves. Defaults to 10000.
            normalize (bool): Whether to normalize position embeddings. Defaults to False.
            scale (float | None): Scale factor for normalized embeddings. If provided,
                normalize must be True. Defaults to None.

        Raises:
            ValueError: If scale is provided but normalize is False.
        """
        super().__init__()
        if scale is not None and normalize is False:
            msg = "normalize should be True if scale is passed"
            raise ValueError(msg)
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        self.scale = 2 * math.pi if scale is None else scale

    def encode_1d_positions(self, x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode 1D coordinate pairs using sine/cosine positional embeddings.

        Args:
        x (torch.Tensor): 1D tensor of x coordinates (flattened) of shape [n_coords]
            with floating-point dtype.
        y (torch.Tensor): 1D tensor of y coordinates (flattened) of shape [n_coords]
            with floating-point dtype.

        Returns:
        tuple[torch.Tensor, torch.Tensor]: Tuple of (pos_x, pos_y) where each
            element has shape [n_coords, num_pos_feats*2] with floating-point dtype.
        """
        x_embed = x * self.scale
        y_embed = y * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.int64, device=x.device).to(x.dtype)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, None] / dim_t
        pos_y = y_embed[:, None] / dim_t
        pos_x = torch.stack((pos_x[:, 0::2].sin(), pos_x[:, 1::2].cos()), dim=2).flatten(1)
        pos_y = torch.stack((pos_y[:, 0::2].sin(), pos_y[:, 1::2].cos()), dim=2).flatten(1)
        return pos_x, pos_y

    def encode_boxes(self, boxes: torch.Tensor) -> torch.Tensor:
        """Encode 4D box coordinates for decoder conditioning using sine/cosine embeddings.

        Args:
        boxes (torch.Tensor): Box coordinates of shape [batch_size, num_queries, 4]
            in (x, y, w, h) format with floating-point dtype.

        Returns:
        torch.Tensor: Position embeddings of shape
            [batch_size, num_queries, num_pos_feats*4] with floating-point dtype.

        Raises:
        ValueError: If the last dimension of ``boxes`` is not 4.
        """
        if boxes.size(-1) != 4:
            msg = f"Expected 4D box coordinates (x, y, w, h), got shape {boxes.shape}"
            raise ValueError(msg)
        dim_t = torch.arange(self.num_pos_feats, dtype=torch.int64, device=boxes.device).to(boxes.dtype)
        dim_t = self.temperature ** (2 * torch.div(dim_t, 2, rounding_mode="floor") / self.num_pos_feats)

        x_embed = boxes[:, :, 0] * self.scale
        y_embed = boxes[:, :, 1] * self.scale
        w_embed = boxes[:, :, 2] * self.scale
        h_embed = boxes[:, :, 3] * self.scale

        pos_x = x_embed[:, :, None] / dim_t
        pos_y = y_embed[:, :, None] / dim_t
        pos_w = w_embed[:, :, None] / dim_t
        pos_h = h_embed[:, :, None] / dim_t

        pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)
        pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3).flatten(2)
        pos_w = torch.stack((pos_w[:, :, 0::2].sin(), pos_w[:, :, 1::2].cos()), dim=3).flatten(2)
        pos_h = torch.stack((pos_h[:, :, 0::2].sin(), pos_h[:, :, 1::2].cos()), dim=3).flatten(2)

        return torch.cat((pos_y, pos_x, pos_w, pos_h), dim=2)

    @compile_compatible_method_lru_cache(maxsize=4)
    def forward(
        self,
        shape: torch.Size,
        device: torch.device | str,
        dtype: torch.dtype,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Generate sine/cosine positional embeddings for image features.

        Args:
        shape (torch.Size): Shape of the feature map [batch_size, channels, height, width].
        device (torch.device | str): Device for the embeddings.
        dtype (torch.dtype): Floating-point dtype for the embeddings.
        mask (torch.Tensor | None): Optional binary mask of shape [batch_size, height, width]
            where True=valid and False=padding (dtype=bool). If None, creates an
            all-valid mask. Defaults to None.

        Returns:
        torch.Tensor: Positional embeddings of shape [batch_size, num_pos_feats*2, height, width]
            with dtype matching ``dtype``.
        """
        if mask is None:
            mask = torch.zeros((shape[0], shape[2], shape[3]), device=device, dtype=torch.bool)
        not_mask = (~mask).to(dtype)
        y_embed = not_mask.cumsum(1)
        x_embed = not_mask.cumsum(2)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.int64, device=device).to(dtype)
        dim_t = self.temperature ** (2 * torch.div(dim_t, 2, rounding_mode="floor") / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        return torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
