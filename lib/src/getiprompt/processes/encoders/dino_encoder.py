# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""DINOv2 encoder."""

from logging import getLogger

import torch
from torch import nn

from getiprompt.models.dinov2 import DinoV2
from getiprompt.processes.encoders.encoder_base import Encoder
from getiprompt.types import Features, Image, Masks, Priors

logger = getLogger("Geti Prompt")


class DinoEncoder(Encoder):
    """This encoder uses a HuggingFace model to encode the images.

    Examples:
        >>> from getiprompt.processes.encoders import DinoEncoder
        >>> from getiprompt.types import Image, Priors, Features
        >>> import torch
        >>> import numpy as np
        >>>
        >>> # Create a sample image
        >>> sample_image = np.zeros((224, 224, 3), dtype=np.uint8)
        >>> encoder = DinoEncoder()
        >>> features, masks = encoder([Image(sample_image)], priors_per_image=[Priors()])
        >>> len(features), len(masks)
        (1, 1)
        >>> # Each image gets a Features object with global features and a Masks object
        >>> isinstance(features[0], Features), isinstance(masks[0], Masks)
        (True, True)
        >>> # DINOv2-large outputs 1024-dimensional feature vectors
        >>> features[0].global_features.shape
        torch.Size([1369, 1024])

    """

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        logger.info("Loading DINOv2 encoder model")
        self.model = model

    def __call__(
        self,
        images: list[Image] | None = None,
        priors_per_image: list[Priors] | None = None,
    ) -> tuple[list[Features], list[Masks]]:
        """This method creates an embedding from the images for locations inside the mask.

        Args:
            images: A list of images.
            priors_per_image: A list of priors per image.

        Returns:
            A list of extracted features.
        """
        resized_masks_per_image: list[Masks] = []
        image_features: list[Features] = self._extract_global_features_batch(images)

        if priors_per_image is not None:
            for features, priors in zip(image_features, priors_per_image, strict=False):
                _, resized_masks = self._extract_local_features(features=features, masks_per_class=priors.masks)
                resized_masks_per_image.append(resized_masks)
        else:
            resized_masks_per_image = [Masks() for _ in image_features]

        return image_features, resized_masks_per_image

    def _extract_local_features(self, features: Features, masks_per_class: Masks) -> tuple[Features, Masks]:
        """This method extracts the local features from the image.

         This only keeps the features that are inside the masks.

        Args:
            features: The features to extract the local features from.
            masks_per_class: The masks to extract the local features from.

        Returns:
            The features with the local features extracted.
        """
        resized_masks = Masks()
        for class_id, masks in masks_per_class.data.items():
            for mask in masks:
                # preprocess mask, add batch dim, convert to float and resize
                pooled_mask = self.model.mask_transform(mask.data).to(self.model.device)
                resized_masks.add(mask=pooled_mask, class_id=class_id)
                # extract local features
                indices = pooled_mask.flatten().bool()
                local_features = features.global_features[indices]
                if local_features.shape[0] == 0:
                    e = f"The reference mask is too small to detect any features for class {class_id}"
                    raise ValueError(e)
                features.add_local_features(local_features=local_features, class_id=class_id)
        return features, resized_masks

    @torch.inference_mode()
    def _extract_global_features_batch(self, images: list[Image]) -> list[Features]:
        """Extract all global features from the images."""
        image_tensors = [image.data for image in images]
        features = self.model(image_tensors)
        image_features: list[Features] = []
        for idx, _image in enumerate(images):
            image_features.append(Features(global_features=features[idx]))
        return image_features


if __name__ == "__main__":
    import numpy as np

    image = Image(np.zeros((224, 224, 3), dtype=np.uint8))
    encoder = DinoEncoder(model=DinoV2(size=DinoV2.Size.SMALL, use_registers=True))
    features, masks = encoder([image], priors_per_image=[Priors()])
    print(features[0].global_features.shape)
