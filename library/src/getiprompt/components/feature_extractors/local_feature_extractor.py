# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Local feature extractor."""

from __future__ import annotations

import torch
from torch import nn
from torchvision import transforms

from getiprompt.types import Features, Masks
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
        batched_features: torch.Tensor,
        batched_masks: torch.Tensor,
        batched_category_ids: torch.Tensor,
    ) -> tuple[list[Features], list[Masks]]:
        """Extract local features from batched features and samples.

        Args:
            batched_features: Batched feature tensor of shape
                (batch_size, num_patches, embedding_dim).
            batched_masks: Batched mask tensor of shape
                (batch_size, num_masks, height, width).
            batched_category_ids: Batched category ID tensor of shape
                (batch_size, num_masks).

        Returns:
            tuple[list[Features], list[Masks]]: Tuple of Features objects (with local features added)
                and Masks objects, one per sample.
        """
        # Split batched tensor into individual tensors and wrap in Features objects
        features_list = [Features(global_features=emb) for emb in batched_features.unbind(0)]

        resized_masks_per_image = []
        for embedding, masks_tensor, category_ids in zip(
            features_list,
            batched_masks,
            batched_category_ids,
            strict=True,
        ):
            resized_masks = Masks()
            for category_id, mask in zip(category_ids, masks_tensor, strict=True):
                category_id = category_id.item()
                pooled_mask = self.transform(mask)
                resized_masks.add(mask=pooled_mask, class_id=category_id)
                keep = pooled_mask.flatten().bool()
                local_features = embedding.global_features[keep]
                embedding.add_local_features(local_features=local_features, class_id=category_id)
            resized_masks_per_image.append(resized_masks)

        return features_list, resized_masks_per_image
