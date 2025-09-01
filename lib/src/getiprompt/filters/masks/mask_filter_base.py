# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import torch

from getiprompt.filters import Filter
from getiprompt.types.masks import Masks


class MaskFilter(Filter):
    """This is the base class for all mask filters."""

    def __call__(self, masks_per_image: list[Masks]) -> list[Masks]:
        """Filter the masks."""

    @staticmethod
    def _calculate_mask_iou(
        mask1: torch.Tensor,
        mask2: torch.Tensor,
    ) -> tuple[float, torch.Tensor | None]:
        """Calculate the IoU between two masks.

        Args:
            mask1: First mask
            mask2: Second mask

        Returns:
            IoU between the two masks and the intersection
        """
        assert mask1.dim() == 2
        assert mask2.dim() == 2
        # Avoid division by zero
        union = (mask1 | mask2).sum().item()
        if union == 0:
            return 0.0, None
        intersection = mask1 & mask2
        return intersection.sum().item() / union, intersection
