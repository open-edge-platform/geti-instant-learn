# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Cosine similarity matcher."""

import torch
import torch.nn.functional as F
from torch import nn

from getiprompt.types import Features, Similarities
from getiprompt.utils.similarity_resize import resize_similarity_map


class CosineSimilarity(nn.Module):
    """Compute cosine similarity between reference and target features.
    
    This class calculates cosine similarity scores between reference features
    and target features, then resizes the similarity maps to match the target image size.
    """

    def __init__(self) -> None:
        """Initialize the cosine similarity matcher."""
        super().__init__()

    def __call__(
        self,
        reference_features: list[Features],
        target_features: list[Features],
        target_size: tuple[int, int],
        unpadded_image_size: tuple[int, int] | None = None,
    ) -> list[Similarities]:
        """Compute cosine similarities between reference and target features.

        Args:
            reference_features: List of reference features.
            target_features: List of target features.
            target_size: Target image size as (height, width).
            unpadded_image_size: Original image size before padding as (height, width).

        Returns:
            List of similarity maps for each target image.
        """
        all_similarities = []
        
        for target_feat in target_features:
            similarities_per_target = []
            
            for ref_feat in reference_features:
                # Compute cosine similarity
                similarities = F.cosine_similarity(
                    ref_feat.global_features,
                    target_feat.global_features,
                    dim=-1,
                )
                
                # Resize similarity map
                similarities = resize_similarity_map(
                    similarities,
                    target_size=target_size,
                    unpadded_image_size=unpadded_image_size,
                )
                
                similarities_per_target.append(similarities)
            
            all_similarities.append(Similarities(similarities=similarities_per_target))
        
        return all_similarities
