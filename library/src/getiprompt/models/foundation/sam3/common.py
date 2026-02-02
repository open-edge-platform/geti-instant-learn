# Copyright 2025 The Meta AI Authors and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Common utilities and base classes for SAM3 model components."""

import logging
import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from transformers.pytorch_utils import compile_compatible_method_lru_cache

# Activation functions used by SAM3
ACT2FN = {
    "gelu": F.gelu,
    "relu": F.relu,
}


logger = logging.getLogger(__name__)


def inverse_sigmoid(x: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    """The inverse function for sigmoid activation function."""
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)


def concat_padded_sequences(seq1, mask1, seq2, mask2, return_index: bool = False):
    """Concatenates two right-padded sequences, such that the resulting sequence
    is contiguous and also right-padded.

    Tensors are batch-first, masks are batch-first with True=valid, False=padding.

    Args:
        seq1: A tensor of shape (batch_size, seq1_length, hidden_size).
        mask1: A tensor of shape (batch_size, seq1_length) with True=valid, False=padding.
        seq2: A tensor of shape (batch_size, seq2_length, hidden_size).
        mask2: A tensor of shape (batch_size, seq2_length) with True=valid, False=padding.
        return_index: If True, also returns the index of the ids of the element of seq2
            in the concatenated sequence. This can be used to retrieve the elements of seq2.

    Returns:
        A tuple (concatenated_sequence, concatenated_mask) if return_index is False,
        otherwise (concatenated_sequence, concatenated_mask, index).
        The concatenated_mask uses True=valid, False=padding convention.
    """
    batch_size, seq1_length, hidden_size = seq1.shape
    batch_size2, seq2_length, hidden_size2 = seq2.shape

    assert batch_size == batch_size2 == mask1.size(0) == mask2.size(0)
    assert hidden_size == hidden_size2
    assert seq1_length == mask1.size(1)
    assert seq2_length == mask2.size(1)

    actual_seq1_lengths = mask1.sum(dim=-1)
    actual_seq2_lengths = mask2.sum(dim=-1)

    final_lengths = actual_seq1_lengths + actual_seq2_lengths
    max_length = seq1_length + seq2_length

    concatenated_mask = (
        torch.arange(max_length, device=seq2.device)[None].repeat(batch_size, 1) < final_lengths[:, None]
    )

    concatenated_sequence = torch.zeros((batch_size, max_length, hidden_size), device=seq2.device, dtype=seq2.dtype)
    concatenated_sequence[:, :seq1_length, :] = seq1

    # Shift seq2 elements to start at the end of valid seq1
    index = torch.arange(seq2_length, device=seq2.device)[None].repeat(batch_size, 1)
    index = index + actual_seq1_lengths[:, None]

    # Scatter seq2 into the right positions
    concatenated_sequence = concatenated_sequence.scatter(1, index[:, :, None].expand(-1, -1, hidden_size), seq2)

    if return_index:
        return concatenated_sequence, concatenated_mask, index

    return concatenated_sequence, concatenated_mask


def box_cxcywh_to_xyxy(x):
    """Convert boxes from (cx, cy, w, h) format to (x1, y1, x2, y2) format."""
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


def expand_attention_mask(attention_mask: torch.Tensor, dtype: torch.dtype | None = None) -> torch.Tensor:
    """Expand a 2D padding mask to 4D for use with scaled_dot_product_attention.

    For bidirectional (full) attention, queries can attend to all valid (non-padding) keys.
    The mask is expanded from [batch_size, key_len] to [batch_size, 1, 1, key_len].

    Args:
        attention_mask: 2D boolean mask [batch_size, key_len] where True=valid, False=padding
        dtype: Optional dtype for float conversion. If None, returns boolean mask.

    Returns:
        4D mask [batch_size, 1, 1, key_len] compatible with SDPA attn_mask parameter.
        SDPA convention: True (or 0.0) = attend, False (or -inf) = mask out.
    """
    if attention_mask.ndim != 2:
        raise ValueError(f"Expected 2D attention_mask, got {attention_mask.ndim}D")

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
    def __init__(
        self,
        hidden_size: int = 1024,
        intermediate_size: int = 4736,
        hidden_act: str = "gelu",
        hidden_dropout: float = 0.0,
    ):
        super().__init__()
        self.activation_fn = ACT2FN[hidden_act]
        self.fc1 = nn.Linear(hidden_size, intermediate_size)
        self.fc2 = nn.Linear(intermediate_size, hidden_size)
        self.dropout = nn.Dropout(hidden_dropout)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


class Attention(nn.Module):
    """Multi-head attention.
    Handles standard [batch_size, seq_len, hidden_size] tensors.
    """

    def __init__(
        self,
        hidden_size: int = 256,
        num_attention_heads: int = 8,
    ):
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
        **kwargs,
    ) -> tuple[torch.Tensor, None]:
        """Args:
            query: [batch_size, query_len, hidden_size]
            key: [batch_size, key_len, hidden_size]
            value: [batch_size, value_len, hidden_size]
            attention_mask: [batch_size, num_heads, query_len, key_len] or broadcastable

        Returns:
            Tuple of (output, None) - attention weights not returned by SDPA
                output: [batch_size, query_len, hidden_size]
        """
        batch_size = query.shape[0]
        query_len = query.shape[1]
        key_len = key.shape[1]

        query = self.q_proj(query).view(batch_size, query_len, self.num_attention_heads, self.head_dim).transpose(1, 2)
        key = self.k_proj(key).view(batch_size, key_len, self.num_attention_heads, self.head_dim).transpose(1, 2)
        value = self.v_proj(value).view(batch_size, key_len, self.num_attention_heads, self.head_dim).transpose(1, 2)

        attn_output = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=0.0,
            scale=self.scaling,
        )

        attn_output = attn_output.transpose(1, 2).reshape(batch_size, query_len, self.hidden_size).contiguous()
        attn_output = self.o_proj(attn_output)

        return attn_output, None


class SinePositionEmbedding(nn.Module):
    """This is a more standard version of the position embedding, very similar to the one used by the Attention is all you
    need paper, generalized to work on images.
    """

    def __init__(
        self,
        num_pos_feats: int = 64,
        temperature: int = 10000,
        normalize: bool = False,
        scale: float | None = None,
    ):
        super().__init__()
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        self.scale = 2 * math.pi if scale is None else scale

    def encode_1d_positions(self, x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode 1D coordinate pairs using sine/cosine positional embeddings.

        Args:
            x: 1D tensor of x coordinates (flattened)
            y: 1D tensor of y coordinates (flattened)

        Returns:
            Tuple of (pos_x, pos_y) positional embeddings
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
        """Encode 4D box coordinates (x, y, w, h) for decoder conditioning using sine/cosine embeddings.

        Args:
            boxes: Box coordinates [batch_size, num_queries, 4] in (x, y, w, h) format

        Returns:
            Position embeddings [batch_size, num_queries, num_pos_feats*4]
        """
        assert boxes.size(-1) == 4, f"Expected 4D box coordinates (x, y, w, h), got shape {boxes.shape}"
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

        pos = torch.cat((pos_y, pos_x, pos_w, pos_h), dim=2)

        return pos

    @compile_compatible_method_lru_cache(maxsize=4)
    def forward(
        self,
        shape: torch.Size,
        device: torch.device | str,
        dtype: torch.dtype,
        mask: Tensor | None = None,
    ) -> Tensor:
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
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos
