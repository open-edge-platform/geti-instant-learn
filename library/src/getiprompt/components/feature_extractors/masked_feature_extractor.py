# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Masked Feature Extractor module."""

from collections import defaultdict

import torch
from torch import nn
from torch.nn import functional as F

from getiprompt.components.feature_extractors.reference_features import ReferenceFeatures


class MaskedFeatureExtractor(nn.Module):
    """Extracts localized patch features inside binary masks.

    Given batched patch embeddings and region masks, pools the masks to the patch
    grid and selects features corresponding to masked regions. The resulting local
    features are aggregated by category and returned as a ReferenceFeatures dataclass.

    Args:
        input_size: The input image size.
        patch_size: The patch size of the encoder.
        device: The device to use.

    Example:
        >>> extractor = MaskedFeatureExtractor(input_size=1024, patch_size=16, device="cuda")
        >>> ref_features = extractor(embeddings, masks, category_ids)
        >>> ref_features.ref_embeddings.shape  # [C, num_patches_total, embed_dim]
    """

    def __init__(self, input_size: int, patch_size: int, device: str) -> None:
        """Initialize the masked feature extractor."""
        super().__init__()
        self.input_size = input_size
        self.patch_size = patch_size
        self.device = device
        self.num_patches = (input_size // patch_size) ** 2

        # Traceable mask pooling (replaces transforms.Compose with lambdas)
        self.mask_pool = nn.MaxPool2d(kernel_size=patch_size)

    def _pool_mask(self, mask: torch.Tensor, target_device: torch.device) -> torch.Tensor:
        """Pool mask to patch grid.

        Uses max pooling with inversion to preserve foreground regions.
        This is a traceable alternative to the previous transform pipeline.

        Args:
            mask: Binary mask of shape [H, W] or [1, H, W]
            target_device: Device to place the result on

        Returns:
            Pooled mask of shape [num_patches] as float tensor
        """
        # Ensure 4D: [1, 1, H, W]
        if mask.ndim == 2:
            mask = mask.unsqueeze(0).unsqueeze(0)
        elif mask.ndim == 3:
            mask = mask.unsqueeze(0)

        # Resize to input_size
        mask = F.interpolate(
            mask.float(),
            size=(self.input_size, self.input_size),
            mode="nearest",
        )

        # Invert -> maxpool -> invert back (preserves foreground under pooling)
        mask = 1 - mask
        mask = self.mask_pool(mask)
        mask = 1 - mask

        return mask.flatten().to(target_device)

    def forward(
        self,
        embeddings: torch.Tensor,
        masks: torch.Tensor,
        category_ids: torch.Tensor,
    ) -> ReferenceFeatures:
        """Extract masked features from batched inputs, aggregated by category.

        This method:
        1. Pools binary masks to the patch grid
        2. Extracts features corresponding to masked regions
        3. Aggregates results by category ID into stacked tensors

        Args:
            embeddings: Feature tensor of shape (batch_size, num_patches, embedding_dim)
            masks: Binary masks of shape (batch_size, num_masks, height, width)
            category_ids: Category IDs for each mask of shape (batch_size, num_masks)

        Returns:
            ReferenceFeatures containing:
                - ref_embeddings: [C, num_patches_total, embed_dim]
                - masked_ref_embeddings: [C, embed_dim]
                - flatten_ref_masks: [C, num_patches_total]
                - category_ids: [C]
        """
        device = embeddings.device
        embed_dim = embeddings.shape[-1]

        # Collect features per category
        ref_embeddings_per_cat: dict[int, list[torch.Tensor]] = defaultdict(list)
        masked_embeddings_per_cat: dict[int, list[torch.Tensor]] = defaultdict(list)
        masks_per_cat: dict[int, list[torch.Tensor]] = defaultdict(list)

        # Process each sample in the batch
        for embedding, masks_tensor, category_ids_tensor in zip(
            embeddings,
            masks,
            category_ids,
            strict=True,
        ):
            for category_id, mask in zip(category_ids_tensor, masks_tensor, strict=True):
                cat_id = category_id.item()
                pooled_mask = self._pool_mask(mask, device)
                masks_per_cat[cat_id].append(pooled_mask)

                # Extract masked embeddings
                keep = pooled_mask.bool()
                masked_embedding = embedding[keep]
                masked_embeddings_per_cat[cat_id].append(masked_embedding)

                # Store full embedding for this reference
                ref_embeddings_per_cat[cat_id].append(embedding)

        # Get unique categories in sorted order for deterministic output
        unique_cats = sorted(ref_embeddings_per_cat.keys())

        # Aggregate by category
        ref_embeddings_list: list[torch.Tensor] = []
        masked_ref_embeddings_list: list[torch.Tensor] = []
        flatten_ref_masks_list: list[torch.Tensor] = []

        for cat_id in unique_cats:
            # Stack reference embeddings for this category: [num_refs, num_patches, embed_dim]
            # Then reshape to [num_refs * num_patches, embed_dim]
            cat_ref_embeds = torch.stack(ref_embeddings_per_cat[cat_id], dim=0)
            cat_ref_embeds = cat_ref_embeds.reshape(-1, embed_dim)
            ref_embeddings_list.append(cat_ref_embeds)

            # Average masked embeddings for this category
            cat_masked_embeds = torch.cat(masked_embeddings_per_cat[cat_id], dim=0)
            if cat_masked_embeds.numel() > 0:
                averaged = cat_masked_embeds.mean(dim=0)
                averaged = averaged / (averaged.norm(dim=-1, keepdim=True) + 1e-8)
            else:
                averaged = torch.zeros(embed_dim, device=device, dtype=embeddings.dtype)
            masked_ref_embeddings_list.append(averaged)

            # Concatenate masks for this category
            cat_masks = torch.cat(masks_per_cat[cat_id], dim=0)
            flatten_ref_masks_list.append(cat_masks)

        # Stack into final tensors
        # Note: Different categories may have different num_patches_total if different num_refs
        # For now, we assume same number of references per category (padding would be needed otherwise)
        ref_embeddings_tensor = torch.stack(ref_embeddings_list, dim=0)  # [C, P, D]
        masked_ref_embeddings_tensor = torch.stack(masked_ref_embeddings_list, dim=0)  # [C, D]
        flatten_ref_masks_tensor = torch.stack(flatten_ref_masks_list, dim=0)  # [C, P]
        category_ids_tensor = torch.tensor(unique_cats, device=device, dtype=torch.int64)  # [C]

        return ReferenceFeatures(
            ref_embeddings=ref_embeddings_tensor,
            masked_ref_embeddings=masked_ref_embeddings_tensor,
            flatten_ref_masks=flatten_ref_masks_tensor,
            category_ids=category_ids_tensor,
        )
