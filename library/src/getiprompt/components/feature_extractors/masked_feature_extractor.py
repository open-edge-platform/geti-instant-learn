# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Masked Feature Extractor module."""

from collections import defaultdict

import torch
from torch import nn
from torchvision import transforms

from getiprompt.types import Masks
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
    ) -> tuple[dict[int, torch.Tensor], list[Masks]]:
        """Extract masked, mask-conditioned features from batched inputs.

        This method aligns binary masks to the patch grid, selects feature embeddings
        corresponding to masked regions, and associates them with their category IDs.

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
            tuple[dict[int, torch.Tensor], list[Masks]]:
                - masked_ref_embeds: Dictionary of masked reference features grouped by category.
                - resized_masks_per_image: List of resized masks.
        """
        resized_masks_per_image = []
        masked_ref_embeds = defaultdict(list)
        for embedding, masks_tensor, category_ids in zip(
            batched_features,
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
                local_features = embedding[keep]
                masked_ref_embeds[category_id].append(local_features)
            resized_masks_per_image.append(resized_masks)

        for category_id, masked_embed_list in masked_ref_embeds.items():
            _embed = torch.cat(masked_embed_list, dim=0)
            _embed = _embed.mean(dim=0, keepdim=True)
            _embed /= _embed.norm(dim=-1, keepdim=True)
            masked_ref_embeds[category_id] = _embed

        return masked_ref_embeds, resized_masks_per_image
