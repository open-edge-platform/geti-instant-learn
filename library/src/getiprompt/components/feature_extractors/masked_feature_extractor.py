# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Masked Feature Extractor module."""

from collections import defaultdict

import torch
from torch import nn
from torchvision import transforms

from getiprompt.data.transforms import ToTensor
from getiprompt.types import Masks


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
    ) -> tuple[dict[int, torch.Tensor], list[Masks]]:
        """Extract masked, mask-conditioned features from batched inputs.

        This method aligns binary masks to the patch grid, selects feature embeddings
        corresponding to masked regions, and associates them with their category IDs.

        Args:
            embeddings(torch.Tensor): Feature tensor of shape
                ``(batch_size, num_patches, embedding_dim)``, typically the patch embeddings
                from an encoder (e.g., ViT).
            masks (torch.Tensor): Binary masks of shape
                ``(batch_size, num_masks, height, width)``, defining spatial regions
                to extract local features from.
            category_ids (torch.Tensor): Category IDs for each mask of shape ``(batch_size, num_masks)``.

        Returns:
            tuple[dict[int, torch.Tensor], list[Masks]]:
                - masked_ref_embeds: Dictionary of masked reference features grouped by category.
                - resized_masks_per_image: List of resized masks.
        """
        resized_masks_per_image = []
        masked_ref_embeddings = defaultdict(list)
        for embedding, masks_tensor, category_ids_tensor in zip(
            embeddings,
            masks,
            category_ids,
            strict=True,
        ):
            resized_masks = Masks()
            for category_id, mask in zip(category_ids_tensor, masks_tensor, strict=True):
                category_id = category_id.item()
                pooled_mask = self.transform(mask)
                resized_masks.add(mask=pooled_mask, class_id=category_id)
                keep = pooled_mask.flatten().bool()
                masked_embedding = embedding[keep]
                masked_ref_embeddings[category_id].append(masked_embedding)
            resized_masks_per_image.append(resized_masks)

        for category_id, masked_embed_list in masked_ref_embeddings.items():
            _embedding = torch.cat(masked_embed_list, dim=0)
            _embedding = _embedding.mean(dim=0, keepdim=True)
            _embedding /= _embedding.norm(dim=-1, keepdim=True)
            masked_ref_embeddings[category_id] = _embedding

        return masked_ref_embeddings, resized_masks_per_image
