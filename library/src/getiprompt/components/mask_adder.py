# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Add masks to priors from points if masks are missing."""

from torch import nn
from torchvision import tv_tensors

from getiprompt.components.mask_decoder import SamDecoder
from getiprompt.types import Priors


class MaskAdder(nn.Module):
    """Add masks from points in the priors if they are missing.

    This is used when only points are supplied for the reference images and the masks are not available.
    """

    def __init__(self, segmenter: SamDecoder) -> None:
        """Initialize the prior mask from points filter.

        Args:
            segmenter: SamDecoder to use to create masks from points
        """
        super().__init__()
        self.segmenter = segmenter

    def forward(self, images: list[tv_tensors.Image], priors: list[Priors]) -> list[Priors]:
        """Create masks from points in the priors.

        Args:
            images (list[tv_tensors.Image]): List of images to segment
            priors (list[Priors]): List of Priors objects to segment

        Returns:
            The same Priors list with masks
        """
        if all(p.masks.is_empty for p in priors) and not any(p.points.is_empty for p in priors):
            masks, _ = self.segmenter(images, priors)
            for p, m in zip(priors, masks, strict=True):
                p.masks = m
        return priors
