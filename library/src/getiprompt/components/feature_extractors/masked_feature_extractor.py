# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Masked Feature Extractor module."""

from collections import defaultdict

import torch
from torch import nn
from torchvision import transforms

from getiprompt.components.feature_extractors.reference_features import ReferenceFeatures
from getiprompt.data.transforms import ToTensor


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

        self.transform = transforms.Compose([
            ToTensor(),
            transforms.Lambda(lambda x: x.unsqueeze(0) if x.ndim == 2 else x),
            transforms.Lambda(lambda x: x.float()),
            transforms.Resize([input_size, input_size]),
            transforms.Lambda(lambda x: (x * -1) + 1),
            torch.nn.MaxPool2d(kernel_size=(patch_size, patch_size)),
            transforms.Lambda(lambda x: (x * -1) + 1),
        ])

    def forward(
        self,
        embeddings: torch.Tensor,
        masks: torch.Tensor,
        category_ids: torch.Tensor,
    ) -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor], dict[int, torch.Tensor]]:
        """Extract masked, mask-conditioned features from batched inputs.

        This method:
        1. Pools binary masks to the patch grid
        2. Extracts features corresponding to masked regions
        3. Aggregates results by category ID into stacked tensors

        Args:
            embeddings: Feature tensor of shape (batch_size, num_patches, embedding_dim)
            masks: Binary masks of shape (batch_size, num_masks, height, width)
            category_ids: Category IDs for each mask of shape (batch_size, num_masks)

        Returns:
            tuple[dict[int, torch.Tensor], dict[int, torch.Tensor], dict[int, torch.Tensor]]:
                - masked_ref_embeddings: Dictionary of masked reference features grouped by category.
                - flatten_ref_masks: Dictionary of flattened masks grouped by category.
                - ref_embeddings: Dictionary of all reference features grouped by category.
        """
        masked_embeddings_per_cat = defaultdict(list)
        masks_per_cat = defaultdict(list)
        ref_embeddings_per_cat = defaultdict(list)

        for embedding, masks_tensor, category_ids_tensor in zip(
            embeddings,
            masks,
            category_ids,
            strict=True,
        ):
            for category_id, mask in zip(category_ids_tensor, masks_tensor, strict=True):
                if isinstance(category_id, torch.Tensor):
                    category_id = int(category_id.item())
                pooled_mask = self.transform(mask).to(embedding.device)
                masks_per_cat[category_id].append(pooled_mask)

                # Extract masked embeddings
                keep = pooled_mask.flatten().bool()
                masked_embeddings_per_cat[category_id].append(embedding[keep])

                # Store full embedding for this reference
                ref_embeddings_per_cat[category_id].append(embedding)

        # Get unique categories in sorted order for deterministic output
        unique_cats = sorted(masked_embeddings_per_cat.keys())

        # Find max number of reference patches across all categories (for padding)
        max_ref_patches = max(torch.cat(ref_embeddings_per_cat[cat_id], dim=0).shape[0] for cat_id in unique_cats)

        # Aggregate by category
        ref_embeddings_list: list[torch.Tensor] = []
        masked_ref_embeddings_list: list[torch.Tensor] = []
        flatten_ref_masks_list: list[torch.Tensor] = []

        for cat_id in unique_cats:
            # Average masked embeddings for this category
            cat_masked_embeds = torch.cat(masked_embeddings_per_cat[cat_id], dim=0)
            if cat_masked_embeds.numel():  # num of elements > 0
                averaged_embed = cat_masked_embeds.mean(dim=0, keepdim=True)
                averaged_embed /= averaged_embed.norm(dim=-1, keepdim=True)
            else:
                averaged_embed = cat_masked_embeds  # Empty tensor
            masked_ref_embeddings_list.append(averaged_embed)

            # Get ref embeddings and masks for this category
            ref_embed = torch.cat(ref_embeddings_per_cat[cat_id], dim=0)
            flatten_mask = torch.cat(masks_per_cat[cat_id], dim=0).reshape(-1)

            # Pad to max_ref_patches for consistent stacking across categories
            num_patches = ref_embed.shape[0]
            if num_patches < max_ref_patches:
                pad_size = max_ref_patches - num_patches
                embed_padding = torch.zeros(
                    pad_size,
                    ref_embed.shape[1],
                    device=ref_embed.device,
                    dtype=ref_embed.dtype,
                )
                mask_padding = torch.zeros(pad_size, device=flatten_mask.device, dtype=flatten_mask.dtype)
                ref_embed = torch.cat([ref_embed, embed_padding], dim=0)
                flatten_mask = torch.cat([flatten_mask, mask_padding], dim=0)

            ref_embeddings_list.append(ref_embed)
            flatten_ref_masks_list.append(flatten_mask)

        return ReferenceFeatures(
            ref_embeddings=torch.stack(ref_embeddings_list, dim=0),
            masked_ref_embeddings=torch.stack(masked_ref_embeddings_list, dim=0),
            flatten_ref_masks=torch.stack(flatten_ref_masks_list, dim=0),
            category_ids=unique_cats,
        )
