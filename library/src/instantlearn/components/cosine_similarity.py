# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Cosine similarity matcher."""

import torch
import torch.nn.functional
from torch import nn


class CosineSimilarity(nn.Module):
    """Computes cosine similarity between reference and target embeddings.

    All outputs are tensors for full traceability (ONNX/TorchScript compatible).

    Args:
        feature_size: Feature grid size for output similarity maps. Default: 64.

    Examples:
        >>> from instantlearn.components import CosineSimilarity
        >>> import torch
        >>>
        >>> similarity_matcher = CosineSimilarity(feature_size=64)
        >>> # masked_ref_embeddings: [C, 1, embed_dim]
        >>> masked_ref_embeddings = torch.randn(2, 1, 256)
        >>> # target_embeddings: [T, num_patches, embed_dim]
        >>> target_embeddings = torch.randn(1, 4096, 256)
        >>> category_ids = [1, 2]
        >>>
        >>> similarities = similarity_matcher(
        ...     masked_ref_embeddings,
        ...     target_embeddings,
        ...     category_ids,
        ... )
        >>> similarities.shape  # [T, C, feature_size, feature_size]
        torch.Size([1, 2, 64, 64])
    """

    def __init__(self, feature_size: int = 64) -> None:
        """Initialize the CosineSimilarity module."""
        super().__init__()
        self.feature_size = feature_size

    @torch.inference_mode()
    def forward(
        self,
        reference_embeddings: torch.Tensor,
        target_embeddings: torch.Tensor,
        category_ids: list[int],
    ) -> torch.Tensor:
        """Compute cosine similarity between reference and target features.

        Args:
            masked_ref_embeddings: Reference embeddings [C, 1, embed_dim]
            target_embeddings: Target embeddings [T, num_patches, embed_dim]
            category_ids: List of category IDs [C]

        Returns:
            similarities: Similarity maps [T, C, feature_size, feature_size]
        """
        num_targets = target_embeddings.shape[0]
        num_categories = len(category_ids)
        device = target_embeddings.device
        dtype = target_embeddings.dtype
        feat_size = self.feature_size

        similarities = torch.zeros(num_targets, num_categories, feat_size, feat_size, device=device, dtype=dtype)

        for t_idx in range(num_targets):
            target_embed = target_embeddings[t_idx]
            target_embed = target_embed / target_embed.norm(dim=-1, keepdim=True)

            # Reshape if needed
            if target_embed.dim() == 3:
                target_embed = target_embed.reshape(-1, target_embed.shape[-1])

            num_patches = target_embed.shape[0]
            grid_size = int(num_patches**0.5)

            for c_idx in range(num_categories):
                ref_embed = reference_embeddings[c_idx]  # [1, embed_dim]

                # Compute similarity
                sim_map = ref_embed @ target_embed.T  # [1, num_patches]
                sim_map = sim_map.reshape(1, 1, grid_size, grid_size)

                # Resize to feature_size
                sim_resized = (
                    torch.nn.functional.interpolate(
                        sim_map,
                        size=(feat_size, feat_size),
                        mode="bilinear",
                        align_corners=False,
                    )
                    .squeeze(0)
                    .squeeze(0)
                )

                similarities[t_idx, c_idx] = sim_resized

        return similarities
