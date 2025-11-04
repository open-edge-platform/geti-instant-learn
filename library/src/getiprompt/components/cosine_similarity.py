# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Cosine similarity matcher."""

import torch
from torch import nn
from torchvision import tv_tensors

from getiprompt.types import Features, Similarities
from getiprompt.utils.similarity_resize import resize_similarity_map


class CosineSimilarity(nn.Module):
    """This class computes the cosine similarity.

    Examples:
        >>> from getiprompt.processes.similarity_matchers import CosineSimilarity
        >>> from getiprompt.types import Features, Image, Similarities
        >>> import torch
        >>> import numpy as np
        >>>
        >>> similarity_matcher = CosineSimilarity()
        >>> ref_features = Features()
        >>> ref_features.local_features = {1: [torch.randn(1, 256)]}
        >>> target_embeddings = torch.randn(64, 64, 256)
        >>> target_image = torch.zeros((3, 1024, 1024))
        >>>
        >>> similarities = similarity_matcher(
        ...     reference_features=ref_features,
        ...     target_embeddings=target_embeddings,
        ...     target_images=target_image,
        ... )
        >>>
        >>> isinstance(similarities, Similarities) and similarities._data[1].shape == (1, 1024, 1024)
        True
    """

    @torch.inference_mode()
    def forward(
        self,
        reference_features: Features,
        target_embeddings: torch.Tensor,
        target_images: list[tv_tensors.Image],
    ) -> list[Similarities]:
        """This function computes the cosine similarity between the reference features and the target features.

        This similarity matcher expects the features of multiple reference images
        to be reduced (averaged/clustered) into a single Features object.

        Args:
            reference_features (list[Features]): List of reference features, one per prior image instance
            target_embeddings (torch.Tensor): Target embeddings
            target_images (list[tv_tensors.Image]): List of target images

        Returns:
            list[Similarities]: List of similarities, one per target image instance which are resized to
              the original image size
        """
        per_image_similarities: list[Similarities] = []
        for target_embedding, target_image in zip(target_embeddings, target_images, strict=True):
            target_embedding /= target_embedding.norm(dim=-1, keepdim=True)
            # reshape from (encoder_shape, encoder_shape, embed_dim)
            # to (encoder_shape*encoder_shape, embed_dim) if necessary
            if target_embedding.dim() == 3:
                target_embedding = target_embedding.reshape(
                    target_embedding.shape[0] * target_embedding.shape[1],
                    target_embedding.shape[2],
                )
            # compute cosine similarity of (1,1,embed_dim) and (encoder_shape*encoder_shape, embed_dim)
            all_similarities = Similarities()
            for class_id, local_reference_features_per_mask in reference_features.local_features.items():
                # Need to loop since number of reference features can differ per input mask.
                for local_reference_features in local_reference_features_per_mask:
                    similarities = local_reference_features @ target_embedding.T
                    similarities = resize_similarity_map(similarities=similarities, target_size=target_image.shape[-2:])
                    all_similarities.add(similarities=similarities, class_id=class_id)
            per_image_similarities.append(all_similarities)
        return per_image_similarities
