# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""This module provides a filter that choces the top K foreground points."""

from getiprompt.filters.priors.prior_filter_base import PriorFilter
from getiprompt.processes.segmenters import Segmenter
from getiprompt.types import Image, Priors


class PriorMaskFromPoints(PriorFilter):
    """Filter that creates masks from points in priors.

    This is used when only points are supplied for the reference images and the masks are not available.
    """

    def __init__(self, segmenter: Segmenter) -> None:
        """Initialize the prior mask from points filter.

        Args:
            segmenter: Segmenter to use to create masks from points
        """
        self.segmenter = segmenter

    def __call__(self, images: list[Image], priors: list[Priors]) -> list[Priors]:
        """Create masks from points in the priors.

        Args:
            images: List of images to segment
            priors: List of Priors objects to segment

        Returns:
            The same Priors list with masks
        """
        if all(p.masks.is_empty for p in priors) and not any(p.points.is_empty for p in priors):
            masks, _ = self.segmenter(images, priors)
            for p, m in zip(priors, masks, strict=True):
                p.masks = m
        return priors
