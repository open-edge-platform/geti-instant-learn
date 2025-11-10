# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Cosine similarity matcher."""

from collections import defaultdict

import torch
from torch import nn

from getiprompt.utils.similarity_resize import resize_similarity_maps


class CosineSimilarity(nn.Module):
    """This class computes the cosine similarity.

    Examples:
        >>> from getiprompt.processes.similarity_matchers import CosineSimilarity
        >>> import torch
        >>> import numpy as np

        >>> similarity_matcher = CosineSimilarity()
        >>> masked_ref_embeddings = {1: torch.randn(1, 256)}
        >>> target_embeddings = torch.randn(64, 64, 256)
        >>> original_sizes = [(1024, 1024)]
        >>>
        >>> similarities = similarity_matcher(
        ...     masked_ref_embeddings=masked_ref_embeddings,
        ...     target_embeddings=target_embeddings,
        ...     original_sizes=original_sizes,
        ... )

        >>> isinstance(similarities, dict) and similarities[1].shape == (1, 1024, 1024)
        True
    """

    @torch.inference_mode()
    def forward(
        self,
        masked_ref_embeddings: dict[int, torch.Tensor],
        target_embeddings: torch.Tensor,
        original_sizes: list[tuple[int, int]],
    ) -> list[dict[int, torch.Tensor]]:
        """This function computes the cosine similarity between the reference features and the target features.

        Args:
            masked_ref_embeddings (dict[int, torch.Tensor]): Dictionary of masked reference embeddings
            target_embeddings (torch.Tensor): Target embeddings
            original_sizes (list[tuple[int, int]]): List of original sizes of the target images

        Returns:
            list[dict[int, torch.Tensor]]: List of similarities dictionaries, one per target image instance
              which are resized to the original image size
        """
        per_image_similarities: list[dict[int, torch.Tensor]] = []
        for target_embedding, original_size in zip(target_embeddings, original_sizes, strict=True):
            target_embedding /= target_embedding.norm(dim=-1, keepdim=True)
            # reshape from (encoder_shape, encoder_shape, embed_dim)
            # to (encoder_shape*encoder_shape, embed_dim) if necessary
            if target_embedding.dim() == 3:
                target_embedding = target_embedding.reshape(
                    target_embedding.shape[0] * target_embedding.shape[1],
                    target_embedding.shape[2],
                )
            # compute cosine similarity of (1,1,embed_dim) and (encoder_shape*encoder_shape, embed_dim)
            all_similarities: dict[int, list[torch.Tensor]] = defaultdict(list)
            for class_id, masked_embedding in masked_ref_embeddings.items():
                # Need to loop since number of reference features can differ per input mask.
                similarity_map = masked_embedding @ target_embedding.T
                similarity_map = resize_similarity_maps(similarity_maps=similarity_map, target_size=original_size)
                all_similarities[class_id].append(similarity_map)

            # Concatenate all tensors once per class
            concatenated_similarity_maps = {
                class_id: torch.cat(tensor_list, dim=0) for class_id, tensor_list in all_similarities.items()
            }
            per_image_similarities.append(concatenated_similarity_maps)
        return per_image_similarities
