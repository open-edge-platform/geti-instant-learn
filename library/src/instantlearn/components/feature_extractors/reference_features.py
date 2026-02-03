# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Reference features dataclass for traceable inference."""

from dataclasses import dataclass

import torch


@dataclass
class ReferenceFeatures:
    """Container for reference features - all tensors for traceability.

    This dataclass holds the extracted reference features organized by category.
    The first dimension of all tensors corresponds to the number of unique categories.

    Attributes:
        ref_embeddings: Reference patch embeddings per category.
            Shape: [C, num_patches_total, embed_dim] where num_patches_total = num_refs * num_patches
        masked_ref_embeddings: Averaged masked reference embeddings per category.
            Shape: [C, embed_dim]
        flatten_ref_masks: Flattened reference masks per category.
            Shape: [C, num_patches_total]
        category_ids: Mapping from index to actual class ID.
            Shape: [C]

    Example:
        For 2-shot learning with 2 categories (cat=5, dog=3):
        - ref_embeddings.shape = [2, 2048, 1024]  # 2 categories, 2*1024 patches
        - masked_ref_embeddings.shape = [2, 1024]
        - flatten_ref_masks.shape = [2, 2048]
        - category_ids = [5, 3]  # index 0 -> class 5, index 1 -> class 3
    """

    ref_embeddings: torch.Tensor
    masked_ref_embeddings: torch.Tensor
    flatten_ref_masks: torch.Tensor
    category_ids: list[int]

    @property
    def num_categories(self) -> int:
        """Return the number of unique categories."""
        return len(self.category_ids)

    @property
    def device(self) -> torch.device:
        """Return the device of the tensors."""
        return self.ref_embeddings.device

    def to(self, device: torch.device | str) -> "ReferenceFeatures":
        """Move all tensors to the specified device.

        Args:
            device: Target device

        Returns:
            New ReferenceFeatures with tensors on the target device
        """
        return ReferenceFeatures(
            ref_embeddings=self.ref_embeddings.to(device),
            masked_ref_embeddings=self.masked_ref_embeddings.to(device),
            flatten_ref_masks=self.flatten_ref_masks.to(device),
            category_ids=self.category_ids,
        )
