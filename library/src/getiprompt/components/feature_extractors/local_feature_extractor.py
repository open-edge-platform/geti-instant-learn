# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Local feature extractor."""

from __future__ import annotations

from collections import defaultdict

import torch
from torch import nn
from torchvision import transforms

from getiprompt.types import Masks
from getiprompt.utils import MaybeToTensor


class LocalFeatureExtractor(nn.Module):
    """Extracts local features inside pooled masks and records Masks.

    Given batched feature tensors and samples with masks, pools pixel-space masks
    to the patch grid and extracts local features inside each mask.

    Examples:
        >>> import torch
        >>> from getiprompt.components.feature_extractors import LocalFeatureExtractor
        >>> from getiprompt.types import Features, Masks

        >>> # Initialize the extractor
        >>> extractor = LocalFeatureExtractor(
        ...     input_size=224,
        ...     patch_size=14,
        ...     device="cpu",
        ... )

        >>> # Create mock batched inputs
        >>> batch_size = 2
        >>> patches_per_dim = 16  # 224 // 14 = 16
        >>> total_patches = patches_per_dim * patches_per_dim  # 16 * 16 = 256
        >>> embedding_dim = 768
        >>> num_masks = 2
        >>> mask_height, mask_width = 224, 224

        >>> # Batched features: (batch_size, total_patches, embedding_dim)
        >>> batched_features = torch.randn(batch_size, total_patches, embedding_dim)

        >>> # Batched masks: (batch_size, num_masks, height, width)
        >>> # Create masks with some foreground regions
        >>> batched_masks = torch.zeros(batch_size, num_masks, mask_height, mask_width, dtype=torch.bool)
        >>> batched_masks[0, 0, 50:100, 50:100] = True  # First sample, first mask
        >>> batched_masks[0, 1, 150:200, 150:200] = True  # First sample, second mask
        >>> batched_masks[1, 0, 30:80, 30:80] = True  # Second sample, first mask

        >>> # Batched category IDs: (batch_size, num_masks)
        >>> batched_category_ids = torch.tensor([[1, 2], [1, 0]], dtype=torch.long)

        >>> # Extract local features
        >>> features_list, masks_list = extractor(
        ...     batched_features,
        ...     batched_masks,
        ...     batched_category_ids,
        ... )

        >>> # Check outputs
        >>> len(features_list) == batch_size
        True
        >>> len(masks_list) == batch_size
        True

        >>> # Each Features object has global features and local features added
        >>> isinstance(features_list[0], Features)
        True
        >>> features_list[0].global_features.shape == (total_patches, embedding_dim)
        True
        >>> 1 in features_list[0].local_features
        True
        >>> len(features_list[0].local_features[1]) > 0
        True

        >>> # Each Masks object contains pooled masks
        >>> isinstance(masks_list[0], Masks)
        True
        >>> 1 in masks_list[0].data
        True
        >>> masks_list[0].data[1].shape[0] == 1  # One mask per class
        True
    """

    def __init__(self, input_size: int, patch_size: int, device: str) -> None:
        """Initialize the local feature extractor.

        Args:
            input_size: The input image size.
            patch_size: The patch size of the encoder.
            device: The device to use.
        """
        super().__init__()
        self.input_size = input_size
        self.patch_size = patch_size
        self.device = device
        self.transform = transforms.Compose([
            MaybeToTensor(),
            transforms.Lambda(lambda x: x.unsqueeze(0) if x.ndim == 2 else x),
            transforms.Lambda(lambda x: x.float()),
            transforms.Resize([input_size, input_size]),
            transforms.Lambda(lambda x: (x * -1) + 1),
            torch.nn.MaxPool2d(kernel_size=(patch_size, patch_size)),
            transforms.Lambda(lambda x: (x * -1) + 1),
        ])

    def forward(
        self,
        batch_ref_embeds: torch.Tensor,
        batch_ref_masks: torch.Tensor,
        batch_ref_category_ids: torch.Tensor,
    ) -> tuple[dict[int, list[torch.Tensor]], list[Masks]]:
        """Extract local features from batched features and samples.

        Args:
            batched_features: Batched feature tensor of shape
                (batch_size, num_patches, embedding_dim).
            batched_masks: Batched mask tensor of shape
                (batch_size, num_masks, height, width).
            batched_category_ids: Batched category ID tensor of shape
                (batch_size, num_masks).

        Returns:
            tuple[dict[int, list[torch.Tensor]], list[Masks]]:
                masked_embeds: Dictionary of masked embeddings.
                resized_masks_per_image: List of resized masks.
        """
        # Split batched tensor into individual tensors and wrap in Features objects

        resized_masks_per_image = []
        masked_embeds = defaultdict(list)
        for ref_embed, ref_masks, ref_cat_ids in zip(
            batch_ref_embeds,
            batch_ref_masks,
            batch_ref_category_ids,
            strict=True,
        ):
            resized_ref_masks = Masks()
            for ref_cat_id, ref_mask in zip(ref_cat_ids, ref_masks, strict=True):
                ref_cat_id = ref_cat_id.item()
                pooled_mask = self.transform(ref_mask)
                resized_ref_masks.add(mask=pooled_mask, class_id=ref_cat_id)
                keep = pooled_mask.flatten().bool()
                local_features = ref_embed[keep]
                masked_embeds[ref_cat_id].append(local_features)
            resized_masks_per_image.append(resized_ref_masks)

        masked_embeds = {
            ref_cat_id: torch.cat(masked_embed_list, dim=0) for ref_cat_id, masked_embed_list in masked_embeds.items()
        }

        return masked_embeds, resized_masks_per_image
