# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Masked Feature Extractor module."""

import torch
from torch import nn
from torchvision import transforms

from getiprompt.types import Features, Masks
from getiprompt.utils import MaybeToTensor


class MaskedFeatureExtractor(nn.Module):
    """Extracts localized patch features inside binary masks.

    Given batched patch embeddings and region masks, pools the masks to the patch
    grid and selects features corresponding to masked regions. The resulting local
    features are grouped by category ID and recorded in :class:`Features` and
    :class:`Masks` objects.

    Args:
        input_size: The input image size.
        patch_size: The patch size of the encoder.
        device: The device to use.

    Example:
        >>> extractor = MaskedFeatureExtractor(input_size=224, patch_size=14, device="cpu")
        >>> features_list, masks_list = extractor(batched_features, batched_masks, batched_category_ids)
    """

    def __init__(self, input_size: int, patch_size: int, device: str) -> None:
        """Initialize the masked feature extractor."""
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
        """Extract local, mask-conditioned features from batched inputs.

        This method aligns binary masks to the patch grid, selects feature embeddings
        corresponding to masked regions, and associates them with their category IDs.
        Each sample in the batch produces:
            - A :class:`Features` object containing both global and per-category local features.
            - A :class:`Masks` object containing the pooled, patch-aligned masks.

        Args:
            batched_features (torch.Tensor): Feature tensor of shape
                ``(batch_size, num_patches, embedding_dim)``, typically the patch embeddings
                from an encoder (e.g., ViT).
            batched_masks (torch.Tensor): Binary masks of shape
                ``(batch_size, num_masks, height, width)``, defining spatial regions
                to extract local features from.
            batched_category_ids (torch.Tensor): Category IDs for each mask of shape
                ``(batch_size, num_masks)``.

        Returns:
            tuple[list[Features], list[Masks]]: A pair of lists, each of length ``batch_size``:
                - The first list contains :class:`Features` objects with local features grouped by category.
                - The second list contains :class:`Masks` objects with corresponding pooled masks.

        Example:
            >>> features_list, masks_list = extractor(
            ...     batched_features, batched_masks, batched_category_ids
            ... )
            >>> len(features_list) == len(masks_list)
            True
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
