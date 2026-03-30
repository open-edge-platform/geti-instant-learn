# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Mixin for negative embedding extraction and similarity adjustment."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch.nn import functional

from instantlearn.data.base.sample import BACKGROUND_CATEGORY_ID

if TYPE_CHECKING:
    from instantlearn.data.base.batch import Batch


class NegativeEmbeddingMixin:
    """Mixin providing negative embedding extraction and similarity adjustment.

    Requires the host class to have:
        - ``masked_feature_extractor``: with a ``.transform()`` method
        - ``_negative_embedding``: ``torch.Tensor | None``
    """

    _negative_embedding: torch.Tensor | None

    def _extract_negative_embedding(
        self,
        embeddings: torch.Tensor,
        batch: Batch,
    ) -> torch.Tensor | None:
        """Extract averaged feature embedding from background mask regions.

        Pools background masks to the patch grid and averages the corresponding
        encoder features. The result represents "what NOT to match" in feature space.

        Args:
            embeddings: Encoder features [B, num_patches, embed_dim].
            batch: Reference batch potentially containing background masks.

        Returns:
            Normalized negative embedding (1, embed_dim), or None.
        """
        neg_embeds: list[torch.Tensor] = []
        for idx, sample in enumerate(batch.samples):
            if sample.masks is None or sample.category_ids is None:
                continue
            embed = embeddings[idx]  # [num_patches, embed_dim]
            for cid, mask in zip(sample.category_ids, sample.masks, strict=True):
                val = int(cid.item()) if isinstance(cid, torch.Tensor) else int(cid)
                if val != BACKGROUND_CATEGORY_ID:
                    continue
                pooled = self.masked_feature_extractor.transform(mask).to(embed.device)
                keep = pooled.flatten().bool()
                if keep.any():
                    neg_embeds.append(embed[keep])
        if not neg_embeds:
            return None
        combined = torch.cat(neg_embeds, dim=0)  # [N, embed_dim]
        avg = combined.mean(dim=0, keepdim=True)  # [1, embed_dim]
        return functional.normalize(avg, p=2, dim=-1)

    def _adjust_similarities(
        self,
        similarities: torch.Tensor,
        target_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """Suppress target regions that resemble the negative reference.

        Computes a per-target "negative similarity map" and subtracts its
        mean-centered version from each category's similarity map.  Mean-centering
        removes the constant baseline similarity that every patch shares with the
        (often very general) negative embedding, so only patches that are *more*
        background-like than average get penalized.

        Args:
            similarities: [T, C, feat_size, feat_size] per-category similarity maps.
            target_embeddings: [T, num_patches, embed_dim] target features.

        Returns:
            Adjusted similarities with above-average negative contribution subtracted.
        """
        neg_embed = self._negative_embedding  # (1, embed_dim)
        if neg_embed is None:
            return similarities

        feat_size = similarities.shape[-1]
        neg_embed = neg_embed.to(device=similarities.device, dtype=similarities.dtype)

        for t in range(similarities.shape[0]):
            target_embed = target_embeddings[t]  # [num_patches, embed_dim]
            neg_sim = (neg_embed @ target_embed.T).reshape(feat_size, feat_size)
            # Only penalize patches with above-average negative similarity;
            # this keeps absolute scores stable for true positive regions.
            penalty = torch.relu(neg_sim - neg_sim.mean())
            for c in range(similarities.shape[1]):
                similarities[t, c] -= penalty

        return similarities
