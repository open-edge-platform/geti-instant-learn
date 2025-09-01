# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""SAM MAPI encoder."""

import cv2
import numpy as np
import torch
from model_api.models import Prompt
from model_api.models.visual_prompting import SAMLearnableVisualPrompter
from torch.nn import functional as F

from getiprompt.processes.encoders.encoder_base import Encoder
from getiprompt.types.features import Features
from getiprompt.types.image import Image
from getiprompt.types.masks import Masks
from getiprompt.types.priors import Priors


class SamMAPIEncoder(Encoder):
    """This is a wrapper around the ModelAPI SAM encoder.

    This encoder extracts features from images using a SAM model. It can be used to extract reference/local features.
    The ModelAPI implementation only returns a single feature vector per image.

    Examples:
        >>> from model_api.models.visual_prompting import SAMLearnableVisualPrompter
        >>> from model_api.models.model import Model
        >>> from getiprompt.processes.encoders import SamMAPIEncoder
        >>> from getiprompt.types import Image, Priors, Features, Masks
        >>> from getiprompt.utils.constants import MAPI_ENCODER_PATH, MAPI_DECODER_PATH
        >>> import numpy as np
        >>> import torch
        >>>
        >>> encoder = Model.create_model(MAPI_ENCODER_PATH)
        >>> decoder = Model.create_model(MAPI_DECODER_PATH)
        >>> model = SAMLearnableVisualPrompter(encoder, decoder)
        >>> encoder = SamMAPIEncoder(model)
        >>> sample_image = np.zeros((1024, 1024, 3), dtype=np.uint8)
        >>> sample_mask = np.zeros((1, 1024, 1024), dtype=np.uint8)
        >>> sample_mask[0, 50:100, 50:100] = 1  # A single square
        >>> sample_prior = Priors()
        >>> sample_prior.masks.add(sample_mask)
        >>> features, masks = encoder([Image(sample_image)], priors_per_image=[sample_prior])
        >>> len(features), len(masks)
        (1, 1)
        >>> isinstance(features[0], Features) and isinstance(masks[0], Masks)
        True
        >>> features[0].global_features.shape
        torch.Size([1, 1, 256])
    """

    def __init__(self, model: SAMLearnableVisualPrompter) -> None:
        super().__init__()
        self._model = model

    @staticmethod
    def _mask_to_polygons(mask: np.ndarray) -> list[np.ndarray]:
        """Converts a binary mask to a list of polygons in format XY."""
        # Find contours
        contours, _hierarchy = cv2.findContours(
            mask.astype(np.uint8) * 255,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        # Loop through contours
        polygons = []
        for cnt in contours:
            # Approximate contour to polygon (optional, for simplification)
            epsilon = 0.01 * cv2.arcLength(cnt, closed=True)
            approx = cv2.approxPolyDP(cnt, epsilon, closed=True)
            polygon = approx.reshape(-1, 2)
            polygons.append(polygon)
        return polygons

    def __call__(
        self,
        images: list[Image],
        priors_per_image: list[Priors] | None = None,
    ) -> tuple[list[Features], list[Masks]]:
        """This method creates an embedding from the images.

        If masks are provided, it extracts local features from masked regions.
        If no masks are provided, it extracts global features.

        Args:
            images: A list of images, expected to be in HWC uint8 format, with pixel values in [0, 255].
            priors_per_image: Optional list of priors per image. If None, returns global features.

        Returns:
            A list of extracted features per image (local reference if masks provided, global if not).
            A list of resized masks per image.
        """
        # Convert input to MAPI format
        if len(images) != len(priors_per_image):
            msg = "Both images and priors need to be specified"
            raise ValueError(msg)

        features: list[Features] = []
        masks: list[Masks] = []

        for image, prior in zip(images, priors_per_image, strict=False):
            image_np = image.to_numpy()
            mask_np = np.moveaxis(prior.masks.to_numpy(), 0, 2) * 255
            polygons = self._mask_to_polygons(mask_np)
            prompt = [Prompt(data=polygon, label=0) for polygon in polygons]
            # Learn features
            sam_features, sam_masks = self._model.learn(image=image_np, polygons=prompt)
            features.append(
                Features(global_features=torch.from_numpy(sam_features.feature_vectors)),
            )
            m = Masks()
            m.add(np.moveaxis(sam_masks, 0, 2))
            masks.append(m)

        return features, masks

    def _extract_global_features(self, image: Image) -> torch.Tensor:
        """Extract image embedding from the image.

        Args:
            image: The image to extract the embedding from.

        Returns:
            The image embedding.
        """
        self.predictor.set_image(image.data)
        # save the size after preprocessing for later use
        image.sam_preprocessed_size = self.predictor.input_size
        embedding = self.predictor.get_image_embedding().squeeze().permute(1, 2, 0)
        return F.normalize(embedding, p=2, dim=-1)

    def _extract_local_features(
        self,
        features: Features,
        masks_per_class: Masks,
    ) -> tuple[Features, Masks]:
        """This method extracts the local features from the image.

         This only keeping the features that are inside the masks.

        Args:
            features: The features to extract the local features from.
            masks_per_class: The masks to extract the features from.

        Returns:
            Features object containing the local features per class and mask.
            The processed masks. These are resized to match the encoder embedding shape.
        """
        resized_masks = Masks()
        for class_id, masks in masks_per_class.data.items():
            # perform per mask as the current predictor does not support batches
            masks: torch.Tensor  # 3D tensor with n_masks x H x W
            for mask in masks:
                input_mask = self.predictor.transform.apply_image(
                    mask.numpy().astype(np.uint8) * 255,
                )
                input_mask_torch = torch.as_tensor(
                    input_mask,
                    device=self.predictor.device,
                )
                input_mask_torch = input_mask_torch.unsqueeze(0).unsqueeze(
                    0,
                )  # add color and batch dimension
                input_mask = self.predictor.model.preprocess(
                    input_mask_torch,
                )  # (normalize) and pad
                # transform mask to embedding shape
                input_mask = F.interpolate(
                    input_mask,
                    size=features.global_features_shape[:2],
                    mode="bilinear",
                )
                input_mask = input_mask.squeeze(0)[0]  # (emb_shape, emb_shape)
                local_features = features.global_features[input_mask > 0]
                if local_features.shape[0] == 0:
                    e = f"The reference mask is too small to detect any features for class {class_id}"
                    raise ValueError(e)
                features.add_local_features(
                    local_features=local_features,
                    class_id=class_id,
                )
                resized_masks.add(mask=input_mask, class_id=class_id)

        return features, resized_masks
