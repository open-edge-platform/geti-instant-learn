# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Base class for similarity matchers."""

import math

import torch
import torch.nn.functional as F
from torch import nn


class SimilarityMatcher(nn.Module):
    """This class calculates the (cosine) similarities between reference features and target features.

    Examples:
        >>> from getiprompt.processes.similarity_matchers import SimilarityMatcher
        >>> from getiprompt.types import Features, Similarities
        >>>
        >>> class MySimilarityMatcher(SimilarityMatcher):
        ...     def __call__(self, reference_features: list[Features],
        ...         target_features: list[Features],
        ...         **kwargs,
        ...     ) -> list[Similarities]:
        ...         return []
        >>>
        >>> my_matcher = MySimilarityMatcher()
        >>> similarities = my_matcher([Features()], [Features()])
    """

    @staticmethod
    def _resize_similarities_to_target_size(
        similarities: torch.Tensor,
        target_size: tuple[int, int] | int,
        unpadded_image_size: tuple[int, int] | None = None,
    ) -> torch.Tensor:
        """This function resizes the similarities to the target image size while removing padding.

        Args:
            similarities: torch.Tensor The similarities to resize
            target_size: tuple[int, int] | int | None The size of the target image,
            unpadded_image_size: tuple[int, int] | None The size of the unpadded image

        Returns:
            torch.Tensor The resized similarities
        """
        square_size = int(math.sqrt(similarities.shape[-1]))
        # put in batched square shape
        similarities = similarities.reshape(
            similarities.shape[0],
            1,
            square_size,
            square_size,
        )
        # SAM models can in some cases add padding to the image, we need to remove it
        if unpadded_image_size is not None:
            similarities = F.interpolate(
                similarities,
                size=max(unpadded_image_size),
                mode="bilinear",
                align_corners=False,
            )
            similarities = similarities[
                ...,
                : unpadded_image_size[0],
                : unpadded_image_size[1],
            ]

        # resize to (original) target size
        similarities = F.interpolate(
            similarities,
            size=target_size,
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)
        if similarities.ndim == 4:
            similarities = similarities.squeeze(0)

        return similarities
