# Copyright (C) 2025 Intel Corporation
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
        masked_ref_embeddings = defaultdict(list)
        flatten_ref_masks = defaultdict(list)
        ref_embeddings = defaultdict(list)

        for embedding, masks_tensor, category_ids_tensor in zip(
            embeddings,
            masks,
            category_ids,
            strict=True,
        ):
            for category_id, mask in zip(category_ids_tensor, masks_tensor, strict=True):
                pooled_mask = self.transform(mask).to(embedding.device)
                flatten_ref_masks[category_id].append(pooled_mask)
                keep = pooled_mask.flatten().bool()
                masked_embedding = embedding[keep]
                masked_ref_embeddings[category_id].append(masked_embedding)
                ref_embeddings[category_id].append(embedding)

        for category_id, masked_embedding in masked_ref_embeddings.items():
            masked_embedding = torch.cat(masked_embedding, dim=0)
            if masked_embedding.numel():  # num of elements > 0
                masked_embedding = masked_embedding.mean(dim=0, keepdim=True)
                masked_embedding /= masked_embedding.norm(dim=-1, keepdim=True)
            masked_ref_embeddings[category_id] = masked_embedding

        for category_id, flatten_ref_mask_list in flatten_ref_masks.items():
            flatten_ref_mask_list = torch.cat(flatten_ref_mask_list, dim=0)
            flatten_ref_masks[category_id] = flatten_ref_mask_list.reshape(-1)

        for category_id, ref_embedding_list in ref_embeddings.items():
            ref_embedding_list = torch.cat(ref_embedding_list, dim=0)
            ref_embeddings[category_id] = ref_embedding_list

        return masked_ref_embeddings, flatten_ref_masks, ref_embeddings