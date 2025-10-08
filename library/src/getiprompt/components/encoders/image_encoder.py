# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Wrapper class that uses a DINO model to encode images and process them into Features and Masks."""

from logging import getLogger

import torch
from torch import nn
from torch.nn import functional as F
from torchvision import transforms
from transformers import AutoImageProcessor, AutoModel

from getiprompt.foundation.model_optimizer import optimize_model
from getiprompt.types import Features, Image, Masks, Priors
from getiprompt.utils.utils import MaybeToTensor, precision_to_torch_dtype

logger = getLogger("Geti Prompt")

AVAILABLE_IMAGE_ENCODERS = {
    "dinov2_small": "facebook/dinov2-with-registers-small",
    "dinov2_base": "facebook/dinov2-with-registers-base",
    "dinov2_large": "facebook/dinov2-with-registers-large",
    "dinov2_giant": "facebook/dinov2-with-registers-giant",
    "dinov3_small": "facebook/dinov3-vits16-pretrain-lvd1689m",
    "dinov3_small_plus": "facebook/dinov3-vits16plus-pretrain-lvd1689m",
    "dinov3_base": "facebook/dinov3-vitb16-pretrain-lvd1689m",
    "dinov3_large": "facebook/dinov3-vitl16-pretrain-lvd1689m",
    "dinov3_huge": "facebook/dinov3-vith16plus-pretrain-lvd1689m",
}


class ImageEncoder(nn.Module):
    """This encoder uses a model from HuggingFace to encode the images.

    Examples:
        >>> from getiprompt.processes.encoders import Encoder
        >>> from getiprompt.types import Image, Priors, Features
        >>> import torch
        >>> import numpy as np
        >>>
        >>> # Create a sample image
        >>> sample_image = np.zeros((518, 518, 3), dtype=np.uint8)
        >>> encoder = Encoder(model_id="dinov2_large")
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
        model_id: str = "dinov3_large",
        device: str = "cuda",
        precision: str = "bf16",
        compile_models: bool = False,
        benchmark_inference_speed: bool = False,
        input_size: int = 518,
    ) -> None:
        """Initialize the encoder.

        Args:
            model_id: The model id to use.
            device: The device to use.
            precision: The precision to use.
            compile_models: Whether to compile the models.
            benchmark_inference_speed: Whether to benchmark the inference speed.
            input_size: The input size to use.
        """
        super().__init__()

        if model_id not in AVAILABLE_IMAGE_ENCODERS:
            msg = f"Invalid model ID: {model_id}. Valid model IDs: {list(AVAILABLE_IMAGE_ENCODERS.keys())}"
            raise ValueError(msg)

        self.model_id = model_id
        self.input_size = input_size
        self.device = device

        logger.info(f"Loading DINO model {model_id}")
        self.model, self.processor = self._load_hf_model(AVAILABLE_IMAGE_ENCODERS[model_id], input_size)
        self.model = self.model.to(device).eval()
        self.patch_size = self.model.config.patch_size
        self.feature_size = self.input_size // self.patch_size
        # Ignore CLS token and register tokens
        self.ignore_token_length = 1 + self.model.config.num_register_tokens

        self.precision = precision_to_torch_dtype(precision)

        self.model = optimize_model(
            model=self.model,
            precision=self.precision,
            device=device,
            compile_models=compile_models,
            benchmark_inference_speed=benchmark_inference_speed,
        ).eval()

        # Mask transform based on the model variant output size
        self.mask_transform = transforms.Compose([
            MaybeToTensor(),
            transforms.Lambda(lambda x: x.unsqueeze(0) if x.ndim == 2 else x),
            transforms.Lambda(lambda x: x.float()),
            transforms.Resize([self.input_size, self.input_size]),
            # MinPool to make sure we do not use background features
            transforms.Lambda(lambda x: (x * -1) + 1),
            torch.nn.MaxPool2d(
                kernel_size=(self.patch_size, self.patch_size),
            ),
            transforms.Lambda(lambda x: (x * -1) + 1),
        ])

    @torch.inference_mode()
    def _embed(self, x: list[torch.Tensor]) -> torch.Tensor:
        """Embed images.

        Args:
            x: The input images.

        Returns:
            The normalized features.
        """
        inputs = self.processor(images=x, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        last_hidden_state = self.model(**inputs).last_hidden_state
        features = last_hidden_state[:, self.ignore_token_length :, :]  # Remove CLS token (and register tokens if used)
        return F.normalize(features, p=2, dim=-1)

    def forward(
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
                pooled_mask = self.mask_transform(mask.data).to(self.model.device)
                resized_masks.add(mask=pooled_mask, class_id=class_id)
                # extract local features
                indices = pooled_mask.flatten().bool()
                local_features = features.global_features[indices]
                if local_features.shape[0] == 0:
                    e = f"The reference mask is too small to detect any features for class {class_id}"
                    raise ValueError(e)
                features.add_local_features(local_features=local_features, class_id=class_id)
        return features, resized_masks

    def _extract_global_features_batch(self, images: list[Image]) -> list[Features]:
        """Extract all global features from the images.

        Args:
            images: A list of images.

        Returns:
            A list of features.
        """
        image_tensors = [image.data for image in images]
        features = self._embed(image_tensors)
        image_features: list[Features] = []
        for idx, _image in enumerate(images):
            image_features.append(Features(global_features=features[idx]))
        return image_features

    @staticmethod
    def _load_hf_model(model_id: str, input_size: int) -> tuple[nn.Module, AutoImageProcessor]:
        """Load DINO model from HuggingFace with error handling.

        Meta requires huggingface users to access weights by first requesting access on the HuggingFace website.
        This function will raise an error if the user does not have access to the weights.

        Args:
            model_id: The model id of the model.
            input_size: The size of the input image.

        Returns:
            The model and processor.
        """
        err_msg = (
            "User does not have access to the weights of the DinoV3 model.\n"
            "Please follow these steps:\n"
            f"1. Request access on the HuggingFace website: https://huggingface.co/{model_id}\n"
            "2. Set your HuggingFace credentials using one of these methods:\n"
            "   - Run: hf auth login\n"
            "   - Set environment variable: export HUGGINGFACE_HUB_TOKEN=your_token\n"
        )
        try:
            model = AutoModel.from_pretrained(model_id)
            processor = AutoImageProcessor.from_pretrained(
                model_id,
                size={"height": input_size, "width": input_size},
                do_center_crop=False,
                use_fast=True,  # uses Rust based image processor
            )
        except OSError as e:
            # Check if this is specifically a HuggingFace gated repo access error
            if "gated repo" in str(e).lower():
                raise ValueError(err_msg) from None
            raise
        return model, processor
