# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""DINOv3 zero-shot classification model."""

import torch

from getiprompt.data.base.batch import Batch
from getiprompt.models.base import Model
from getiprompt.models.foundation.dinotxt import IMAGENET_TEMPLATES, DinoTextEncoder
from getiprompt.types import Masks, Results
from getiprompt.utils import precision_to_torch_dtype
from getiprompt.utils.constants import DINOv3BackboneSize


class DinoTxtZeroShotClassification(Model):
    """DinoTxt model.

    Args:
        pretrained: Whether to use pretrained weights.
        prompt_templates: The prompt templates to use for the model.
        precision: The precision to use for the model.
        device: The device to use for the model.
        image_size: The size of the image to use.

    Examples:
        >>> from getiprompt.models import DinoTxtZeroShotClassification
        >>> from getiprompt.data.base import Batch
        >>> from getiprompt.data.base.sample import Sample
        >>> from getiprompt.types import Results
        >>> import torch
        >>> import numpy as np
        >>>
        >>> dinotxt = DinoTxtZeroShotClassification(
        ...     prompt_templates=["a photo of a {}."],  # default is IMAGENET_TEMPLATES
        ...     precision="bf16",
        ...     device="cpu",
        ...     image_size=(512, 512),
        ...     backbone_size=DINOv3BackboneSize.LARGE,
        ... )
        >>>
        >>> # Create reference sample with categories
        >>> ref_sample = Sample(
        ...     image=torch.zeros((3, 512, 512)),
        ...     categories=["cat", "dog"],
        ...     category_ids=np.array([0, 1]),
        ...     is_reference=[True, True],
        ... )
        >>> ref_batch = Batch.collate([ref_sample])
        >>>
        >>> # Create target sample
        >>> target_sample = Sample(
        ...     image=torch.zeros((3, 512, 512)),
        ...     is_reference=[False],
        ...     categories=["object"],
        ... )
        >>> target_batch = Batch.collate([target_sample])
        >>>
        >>> # Run learn and infer
        >>> dinotxt.learn(ref_batch)
        >>> infer_results = dinotxt.infer(target_batch)
        >>>
        >>> isinstance(infer_results, Results)
        True
    """

    def __init__(
        self,
        prompt_templates: list[str] = IMAGENET_TEMPLATES,
        precision: str = "bf16",
        device: str = "cuda",
        image_size: tuple[int, int] | None = (512, 512),
        backbone_size: DINOv3BackboneSize = DINOv3BackboneSize.LARGE,
    ) -> None:
        """Initialize the DinoTxtZeroShotClassification."""
        super().__init__()
        self.precision = precision = precision_to_torch_dtype(precision)
        self.dino_encoder = DinoTextEncoder(
            device=device,
            image_size=image_size,
            precision=precision,
            backbone_size=backbone_size,
        )
        self.prompt_templates = prompt_templates

    def learn(self, reference_batch: Batch) -> None:
        """Perform learning step on the reference batch.

        DINOTxt extracts categories from the reference batch to create text priors.

        Args:
            reference_batch: The reference batch containing samples with categories.

        Raises:
            ValueError: If no reference samples with categories are provided.

        Examples:
            >>> from getiprompt.models import DinoTxtZeroShotClassification
            >>> from getiprompt.data.base import Batch
            >>> from getiprompt.data.base.sample import Sample
            >>> import numpy as np
            >>> dinotxt = DinoTxtZeroShotClassification(device="cpu")
            >>> ref_sample = Sample(
            ...     image=torch.zeros((3, 512, 512)),
            ...     categories=["cat", "dog"],
            ...     category_ids=np.array([0, 1]),
            ...     is_reference=[True, True],
            ... )
            >>> ref_batch = Batch.collate([ref_sample])
            >>> dinotxt.learn(ref_batch)
        """
        if not reference_batch.samples:
            msg = "reference_batch must contain at least one sample"
            raise ValueError(msg)

        # Extract categories and category_ids from the batch to create category mapping
        category_mapping: dict[int, str] = {}

        for sample in reference_batch.samples:
            if sample.categories is not None and sample.category_ids is not None:
                for category_id, category in zip(sample.category_ids, sample.categories, strict=False):
                    category_id_int = int(category_id)
                    # Avoid duplicates - use first occurrence
                    if category_id_int not in category_mapping:
                        category_mapping[category_id_int] = category

        if not category_mapping:
            msg = "reference_batch must contain samples with categories"
            raise ValueError(msg)

        self.category_mapping = category_mapping
        # reference features is zero shot weights from DinoTxtEncoder
        self.reference_features = self.dino_encoder.encode_text(category_mapping, self.prompt_templates)

    @torch.no_grad()
    def infer(self, target_batch: Batch) -> Results:
        """Perform inference on the target batch.

        Args:
            target_batch: The target batch containing images to classify.

        Returns:
            Results object containing the masks with predicted class IDs.

        Examples:
            >>> from getiprompt.models import DinoTxtZeroShotClassification
            >>> from getiprompt.data.base import Batch
            >>> from getiprompt.data.base.sample import Sample
            >>> import torch
            >>> import numpy as np
            >>> dinotxt = DinoTxtZeroShotClassification(device="cpu")
            >>> ref_sample = Sample(
            ...     image=torch.zeros((3, 512, 512)),
            ...     categories=["cat", "dog"],
            ...     category_ids=np.array([0, 1]),
            ...     is_reference=[True, True],
            ... )
            >>> ref_batch = Batch.collate([ref_sample])
            >>> dinotxt.learn(ref_batch)
            >>> target_sample = Sample(
            ...     image=torch.zeros((3, 512, 512)),
            ...     is_reference=[False],
            ...     categories=["object"],
            ... )
            >>> target_batch = Batch.collate([target_sample])
            >>> result = dinotxt.infer(target_batch)
            >>> isinstance(result, Results)
            True
            >>> result.masks is not None
            True
        """
        target_images = target_batch.images
        target_features = self.dino_encoder.encode_image(target_images)
        target_features /= target_features.norm(dim=-1, keepdim=True)
        logits = 100.0 * target_features @ self.reference_features
        scores = logits.softmax(dim=1)
        _, max_class_ids = scores.max(dim=1)

        masks = []
        for target_image, max_class_id in zip(target_images, max_class_ids, strict=False):
            m = torch.zeros(target_image.shape[-2:], dtype=torch.bool)
            # NOTE: Due to the current type contract, for zero-shot classification,
            # we need to create a mask for each target image
            # This part should be refactored when we have a Label type class
            mask_type = Masks()
            mask_type.add(mask=m, class_id=max_class_id.item())
            masks.append(mask_type)
        result = Results()
        result.masks = masks
        return result
