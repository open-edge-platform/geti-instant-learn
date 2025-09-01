# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""DINOv2 encoder."""

from logging import getLogger

import torch
from torch.nn import functional as F
from torchvision import transforms
from transformers import AutoImageProcessor, AutoModel

from getiprompt.models.model_optimizer import optimize_model
from getiprompt.processes.encoders.encoder_base import Encoder
from getiprompt.types import Features, Image, Masks, Priors
from getiprompt.utils import MaybeToTensor, precision_to_torch_dtype

logger = getLogger("Vision Prompt")


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
        >>> encoder = DinoEncoder(precision=torch.float32, compile_models=False, verbose=False)
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

    def __init__(
        self,
        precision: str,
        compile_models: bool,
        verbose: bool,
        model_id: str = "facebook/dinov2-large",
        device: str = "cuda",
    ) -> None:
        super().__init__()
        model = AutoModel.from_pretrained(model_id).to(device).eval()

        self.encoder_input_size = model.config.image_size
        self.patch_size = model.config.patch_size
        self.feature_size = self.encoder_input_size // self.patch_size

        self.model = optimize_model(
            model=model,
            precision=precision_to_torch_dtype(precision),
            compile_models=compile_models,
            verbose=verbose,
            device=device,
        ).eval()

        self.processor = AutoImageProcessor.from_pretrained(
            model_id,
            size={"height": self.encoder_input_size, "width": self.encoder_input_size},
            do_center_crop=False,
            use_fast=True,  # uses Rust based image processor
        )
        self.encoder_mask_transform = transforms.Compose([
            MaybeToTensor(),
            transforms.Lambda(lambda x: x.unsqueeze(0) if x.ndim == 2 else x),
            transforms.Lambda(lambda x: x.float()),
            transforms.Resize([self.encoder_input_size, self.encoder_input_size]),
            # MinPool to make sure we do not use background features
            transforms.Lambda(lambda x: (x * -1) + 1),
            torch.nn.MaxPool2d(
                kernel_size=(self.patch_size, self.patch_size),
            ),
            transforms.Lambda(lambda x: (x * -1) + 1),
        ])

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
                pooled_mask = self.encoder_mask_transform(mask.data).to(self.model.device)
                resized_masks.add(mask=pooled_mask, class_id=class_id)
                # extract local features
                indices = pooled_mask.flatten().bool()
                local_features = features.global_features[indices]
                if local_features.shape[0] == 0:
                    e = f"The reference mask is too small to detect any features for class {class_id}"
                    raise ValueError(e)
                features.add_local_features(local_features=local_features, class_id=class_id)
        return features, resized_masks

    @torch.no_grad()
    def _extract_global_features_batch(self, images: list[Image]) -> list[Features]:
        """Extract all global features from the images."""
        image_tensors = [image.data for image in images]
        inputs = self.processor(images=image_tensors, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        features = self.model(**inputs).last_hidden_state[:, 1:, :]  # Remove CLS token
        features = F.normalize(features, p=2, dim=-1)
        image_features: list[Features] = []
        for idx, _image in enumerate(images):
            image_features.append(Features(global_features=features[idx]))
        return image_features
