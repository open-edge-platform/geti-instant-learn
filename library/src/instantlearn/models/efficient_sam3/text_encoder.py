# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# Portions adapted from Apple MobileCLIP (MIT License)
# https://github.com/apple/ml-mobileclip

"""MobileCLIP-S1 text encoder for EfficientSAM3.

A standard pre-LayerNorm transformer encoder with learnable positional embeddings
and a final linear projection, matching the MobileCLIP-S1 architecture used for
text encoding in EfficientSAM3.
"""

import math

import torch
from torch import nn
from torch.nn import functional


class LayerNormFP32(nn.LayerNorm):
    """LayerNorm that casts to float32 internally for numerical stability."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply LayerNorm in float32 precision.

        Args:
            x: Input tensor of any dtype.

        Returns:
            Normalized tensor cast back to input dtype.
        """
        inp_dtype = x.dtype
        return super().forward(x.to(torch.float32)).to(inp_dtype)


class LearnablePositionalEmbedding(nn.Module):
    """Learnable positional embedding with bilinear interpolation for variable lengths."""

    def __init__(self, num_embeddings: int, embedding_dim: int) -> None:
        """Initialize learnable positional embedding.

        Args:
            num_embeddings: Maximum sequence length.
            embedding_dim: Embedding dimension.
        """
        super().__init__()
        self.pos_embed = nn.Parameter(torch.empty(1, 1, num_embeddings, embedding_dim))
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        nn.init.trunc_normal_(self.pos_embed, mean=0, std=embedding_dim**-0.5)

    def forward(self, seq_len: int) -> torch.Tensor:
        """Get positional embeddings, interpolating if needed.

        Args:
            seq_len: Target sequence length.

        Returns:
            Positional embeddings [1, seq_len, embedding_dim].
        """
        pos_embed = self.pos_embed
        if seq_len != self.num_embeddings:
            pos_embed = functional.interpolate(
                pos_embed,
                size=(seq_len, self.embedding_dim),
                mode="bilinear",
            )
        return pos_embed.reshape(1, seq_len, self.embedding_dim)


class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention with fused QKV projection."""

    def __init__(self, embed_dim: int, num_heads: int) -> None:
        """Initialize multi-head self-attention.

        Args:
            embed_dim: Total embedding dimension.
            num_heads: Number of attention heads.
        """
        super().__init__()
        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim, bias=True)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.head_dim = embed_dim // num_heads
        self.scaling = self.head_dim**-0.5
        self.num_heads = num_heads

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply multi-head self-attention.

        Args:
            x: Input [B, S, D].
            attn_mask: Optional causal mask [B, S, S] or [S, S].
            key_padding_mask: Optional padding mask [B, S] (True=masked).

        Returns:
            Attention output [B, S, D].
        """
        b, s, _ = x.shape
        qkv = self.qkv_proj(x).reshape(b, s, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        query, key, value = qkv[0], qkv[1], qkv[2]

        query *= self.scaling
        attn = torch.matmul(query, key.transpose(-1, -2))

        if attn_mask is not None:
            if attn_mask.ndim == 3:
                attn_mask = attn_mask.unsqueeze(1)
            attn += attn_mask

        if key_padding_mask is not None:
            attn = attn.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2).to(torch.bool),
                float("-inf"),
            )

        attn = attn.softmax(dim=-1).to(query.dtype)
        out = torch.matmul(attn, value)
        out = out.transpose(1, 2).reshape(b, s, -1)
        return self.out_proj(out)


class TransformerEncoderLayer(nn.Module):
    """Pre-norm transformer encoder layer: LN -> MHA -> residual -> LN -> FFN -> residual.

    Attribute names match the original MobileCLIP TransformerEncoder for checkpoint
    compatibility (pre_norm_mha, self_attn, pre_norm_ffn, ffn).
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ffn_dim: int,
    ) -> None:
        """Initialize transformer encoder layer.

        Args:
            embed_dim: Embedding dimension.
            num_heads: Number of attention heads.
            ffn_dim: Feed-forward network intermediate dimension.
        """
        super().__init__()
        self.pre_norm_mha = LayerNormFP32(embed_dim)
        self.self_attn = MultiHeadSelfAttention(embed_dim, num_heads)
        self.pre_norm_ffn = LayerNormFP32(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim, bias=True),
            nn.GELU(),
            nn.Linear(ffn_dim, embed_dim, bias=True),
        )

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass through the encoder layer.

        Args:
            x: Input [B, S, D].
            attn_mask: Optional attention mask.
            key_padding_mask: Optional padding mask.

        Returns:
            Output [B, S, D].
        """
        residual = x
        x = self.pre_norm_mha(x)
        x = self.self_attn(x, attn_mask=attn_mask, key_padding_mask=key_padding_mask)
        x += residual

        residual = x
        x = self.pre_norm_ffn(x)
        x = self.ffn(x)
        return x + residual


class MobileCLIPTextEncoder(nn.Module):
    """MobileCLIP-S1 text encoder: 12-layer transformer with 512-dim embeddings.

    Architecture matches the EfficientSAM3 student text encoder (MobileCLIP-S1 "base"
    variant) for exact weight compatibility.
    """

    def __init__(
        self,
        vocab_size: int = 49408,
        dim: int = 512,
        n_layers: int = 12,
        n_heads: int = 8,
        ffn_multiplier: float = 4.0,
        context_length: int = 77,
    ) -> None:
        """Initialize MobileCLIP text encoder.

        Args:
            vocab_size: Vocabulary size. Default: 49408.
            dim: Model dimension. Default: 512.
            n_layers: Number of transformer layers. Default: 12.
            n_heads: Number of attention heads. Default: 8.
            ffn_multiplier: FFN dimension multiplier. Default: 4.0.
            context_length: Maximum context length for positional embeddings. Default: 77.
        """
        super().__init__()
        self.dim = dim
        self.context_length = context_length

        # Token embedding (no scaling — matches the original EfficientSAM3 repo's
        # MobileCLIPTextTransformer.forward_embedding which defines embed_scale
        # but never applies it)
        self.embedding_layer = nn.Embedding(vocab_size, dim)

        # Learnable positional embedding
        self.positional_embedding = LearnablePositionalEmbedding(context_length, dim)

        # Transformer layers
        ffn_dim = int(math.ceil(dim * ffn_multiplier / 16.0) * 16.0)
        self.transformer = nn.ModuleList([
            TransformerEncoderLayer(embed_dim=dim, num_heads=n_heads, ffn_dim=ffn_dim) for _ in range(n_layers)
        ])

        self.final_layer_norm = LayerNormFP32(dim)

        # Projection layer (MobileCLIP uses nn.Parameter matrix multiplication)
        self.projection_layer = nn.Parameter(torch.empty(dim, dim))
        nn.init.normal_(self.projection_layer, std=dim**-0.5)

    def forward_embedding(self, tokens: torch.Tensor) -> torch.Tensor:
        """Convert tokens to embeddings with positional encoding.

        Note:
            No embedding scaling is applied here. Although the original MobileCLIP
            code defines ``embed_scale = dim**-0.5``, the ``forward_embedding``
            method in the upstream EfficientSAM3 repo never uses it. The distilled
            weights were trained without scaling, so applying it would suppress
            text features and break detection confidence.

        Args:
            tokens: Token IDs [B, S].

        Returns:
            Token embeddings [B, S, dim].
        """
        token_emb = self.embedding_layer(tokens)
        seq_len = token_emb.shape[1]
        pos_emb = self.positional_embedding(seq_len).to(token_emb.dtype)
        return token_emb + pos_emb

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,  # noqa: ARG002
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode text tokens.

        Args:
            input_ids: Token IDs [B, S].
            attention_mask: HuggingFace-style mask [B, S] where 1=valid, 0=padding.

        Returns:
            Tuple of:
                - all_tokens: All token features [B, S, dim] after final layer norm.
                - input_embeds: Raw input embeddings [B, S, dim] before transformer.
        """
        input_embeds = self.forward_embedding(input_ids)

        # NOTE: No key_padding_mask is passed to the transformer layers.
        # The original MobileCLIP-S1 text encoder used in EfficientSAM3
        # distillation runs without any masking (causal_masking=False and
        # key_padding_mask=None). Passing a padding mask produces different
        # text features and degrades detection quality — in particular,
        # presence_logits become strongly negative for certain text prompts
        # (e.g. "person") because the cross-attention in the DETR decoder
        # sees altered text representations.
        hidden_states = input_embeds
        for layer in self.transformer:
            hidden_states = layer(hidden_states)

        hidden_states = self.final_layer_norm(hidden_states)
        return hidden_states, input_embeds
