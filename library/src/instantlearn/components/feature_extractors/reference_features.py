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
            Shape: [C, 1, embed_dim]. Categories whose annotation mask covers zero
            encoder patch cells (polygon too small relative to patch grid) receive a
            zero-filled [1, embed_dim] row and will produce no detections at inference.
        flatten_ref_masks: Flattened reference masks per category.
            Shape: [C, num_patches_total]
        category_ids: Mapping from index to actual class ID.
            Shape: [C]

    Example:
        For 2-shot learning with 2 categories (cat=5, dog=3):
        - ref_embeddings.shape = [2, 2048, 1024]  # 2 categories, 2*1024 patches
        - masked_ref_embeddings.shape = [2, 1, 1024]  # 2 categories, 1 averaged embedding each
        - flatten_ref_masks.shape = [2, 2048]
        - category_ids = [5, 3]  # index 0 -> class 5, index 1 -> class 3
    """

    ref_embeddings: torch.Tensor
    masked_ref_embeddings: torch.Tensor
    flatten_ref_masks: torch.Tensor
    category_ids: list[int]

    def __post_init__(self) -> None:
        """Validate that ``flatten_ref_masks`` is strictly binary.

        The export prompt paths threshold the reference masks (``> 0``) to pick
        foreground patches. If the masks are not strictly ``{0, 1}`` — e.g. a
        numpy ``Sample.mask`` scaled by torchvision ``ToTensor`` (``/255``) — the
        threshold degenerates and prompts collapse to a raster grid (the "sky"
        bug). Binarization happens upstream in ``MaskedFeatureExtractor``; this
        guard makes a regression fail loudly at the source instead of silently
        producing garbage masks.

        Raises:
            ValueError: If ``flatten_ref_masks`` holds values other than 0/1.
        """
        masks = self.flatten_ref_masks
        # Cheap strict-binary check on the fast path: avoid torch.unique (sort +
        # host sync) on every construction. Only the single boolean reduction syncs.
        if not bool(((masks == 0) | (masks == 1)).all()):
            unique_vals = torch.unique(masks)
            msg = (
                "flatten_ref_masks must be strictly binary (values in {0, 1}), but got "
                f"unique values {unique_vals.tolist()}. Reference masks are expected to be "
                "binarized in MaskedFeatureExtractor; a non-binary mask usually means a "
                "numpy Sample.mask was rescaled by torchvision ToTensor (/255)."
            )
            raise ValueError(msg)

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
