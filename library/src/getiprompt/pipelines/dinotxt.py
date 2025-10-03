# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""DINOv3 zero-shot classification pipeline."""

import torch

from getiprompt.models.dinotxt import IMAGENET_TEMPLATES, DinoTextEncoder
from getiprompt.pipelines.pipeline_base import Pipeline
from getiprompt.types import Image, Masks, Priors, Results
from getiprompt.utils import precision_to_torch_dtype
from getiprompt.utils.constants import DINOv3BackboneSize


class DinoTxtZeroShotClassification(Pipeline):
    """DinoTxt pipeline.

    Args:
        pretrained: Whether to use pretrained weights.
        prompt_templates: The prompt templates to use for the model.
        precision: The precision to use for the model.
        device: The device to use for the model.
        image_size: The size of the image to use.

    Examples:
        >>> from getiprompt.pipelines import DinoTxtZeroShotClassification
        >>> from getiprompt.types import Image, Priors
        >>> from getiprompt.utils.constants import DINOv3BackboneSize
        >>>
        >>> dinotxt = DinoTxtZeroShotClassification(
        >>>     prompt_templates=["a photo of a {}."],  # default is IMAGENET_TEMPLATES
        >>>     precision="bf16",
        >>>     device="cuda",
        >>>     image_size=(512, 512),
        >>>     backbone_size=DINOv3BackboneSize.LARGE,
        >>> )
        >>> ref_priors = Priors(text={0: "cat", 1: "dog"})
    """

    def __init__(
        self,
        prompt_templates: list[str] = IMAGENET_TEMPLATES,
        precision: str = "bf16",
        device: str = "cuda",
        image_size: tuple[int, int] | None = (512, 512),
        backbone_size: DINOv3BackboneSize = DINOv3BackboneSize.LARGE,
    ) -> None:
        super().__init__(image_size=image_size)
        self.precision = precision = precision_to_torch_dtype(precision)
        self.dino_encoder = DinoTextEncoder(
            device=device,
            image_size=image_size,
            precision=precision,
            backbone_size=backbone_size,
        )
        self.prompt_templates = prompt_templates

    def learn(
        self,
        reference_images: list[Image],  # noqa: ARG002
        reference_priors: list[Priors],
    ) -> None:
        """Perform learning step on the priors.

        DINOTxt does not need reference images, but we keep it for consistency.

        Args:
            reference_images: A list of reference images.
            reference_priors: A list of reference priors.

        Returns:
            None

        Examples:
            >>> import torch
            >>> import numpy as np
            >>> from getiprompt.pipelines import DINOTxt
            >>> from getiprompt.types import Image, Priors
            >>> dinotxt = DINOTxt()
            >>> ref_priors = Priors(text={0: "cat", 1: "dog"})
            >>> dinotxt.learn(reference_images=[], reference_priors=[ref_priors])
            >>> dinotxt.infer(target_images=[Image()])
        """
        if not reference_priors:
            msg = "reference_priors must be provided"
            raise ValueError(msg)

        reference_prior = reference_priors[0]
        self.class_maps = reference_prior.text.items()
        # reference features is zero shot weights from DinoTxtEncoder
        self.reference_features = self.dino_encoder.encode_text(reference_prior, self.prompt_templates)

    @torch.no_grad()
    def infer(self, target_images: list[Image]) -> Results:
        """Perform inference on the target images.

        Args:
            target_images: A list of target images.

        Returns:
            Result object containing the masks.

        Examples:
            >>> import torch
            >>> from getiprompt.pipelines import DinoTxtZeroShotClassification
            >>> from getiprompt.types import Image, Priors
            >>> dinotxt = DinoTxtZeroShotClassification()
            >>> ref_priors = Priors(text={0: "cat", 1: "dog"})
            >>> dinotxt.learn(reference_images=[], reference_priors=[ref_priors])
            >>> target_image = Image(data=torch.randn(512, 512, 3))
            >>> result = dinotxt.infer(target_images=[target_image])
            >>> result.masks  # doctest: +SKIP
            [Masks(num_masks=1)]
        """
        target_features = self.dino_encoder.encode_image(target_images)
        target_features /= target_features.norm(dim=-1, keepdim=True)
        logits = 100.0 * target_features @ self.reference_features
        scores = logits.softmax(dim=1)
        _, max_class_ids = scores.max(dim=1)

        masks = []
        for target_image, max_class_id in zip(target_images, max_class_ids, strict=False):
            m = torch.zeros(target_image.shape)
            # NOTE: Due to the current type contract, for zero-shot classification,
            # we need to create a mask for each target image
            # This part should be refactored when we have a Label type class
            mask_type = Masks()
            mask_type.add(mask=m, class_id=max_class_id)
            masks.append(mask_type)
        result = Results()
        result.masks = masks
        return result
