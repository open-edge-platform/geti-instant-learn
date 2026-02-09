# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""DINOv3 zero-shot classification model."""

import torch

from instantlearn.data.base.batch import Batch, Collatable
from instantlearn.data.base.sample import Sample
from instantlearn.models.base import Model
from instantlearn.utils import precision_to_torch_dtype
from instantlearn.utils.constants import DINOv3BackboneSize

from .encoder import IMAGENET_TEMPLATES, DinoTextEncoder


class DinoTxtZeroShotClassification(Model):
    """DinoTxt model.

    Args:
        pretrained: Whether to use pretrained weights.
        prompt_templates: The prompt templates to use for the model.
        precision: The precision to use for the model.
        device: The device to use for the model.
        image_size: The size of the image to use.

    Examples:
        >>> from instantlearn.models import DinoTxtZeroShotClassification
        >>> from instantlearn.data.base import Batch
        >>> from instantlearn.data.base.sample import Sample
        >>> from instantlearn.types import Results
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

    def fit(self, reference: Sample | list[Sample] | Batch) -> None:
        """Perform learning step on the reference batch.

        DINOTxt extracts categories from the reference batch to create text priors.

        Args:
            reference: Reference data to learn from. Accepts:
                - Sample: A single reference sample
                - list[Sample]: A list of reference samples
                - Batch: A batch of reference samples

        Raises:
            ValueError: If no reference samples with categories are provided.

        Examples:
            >>> from instantlearn.models import DinoTxtZeroShotClassification
            >>> from instantlearn.data.base import Batch
            >>> from instantlearn.data.base.sample import Sample
            >>> import numpy as np
            >>> dinotxt = DinoTxtZeroShotClassification(device="cpu")
            >>> ref_sample = Sample(
            ...     image=torch.zeros((3, 512, 512)),
            ...     categories=["cat", "dog"],
            ...     category_ids=np.array([0, 1]),
            ...     is_reference=[True, True],
            ... )
            >>> dinotxt.fit(ref_sample)  # Can pass Sample directly
        """
        reference_batch = Batch.collate(reference)
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
    def predict(self, target: Collatable) -> list[dict[str, torch.Tensor]]:
        """Perform inference on the target batch.

        Args:
            target: Target data to infer. Accepts:
                - Sample: A single target sample
                - list[Sample]: A list of target samples
                - Batch: A batch of target samples
                - str | Path: A single image path
                - list[str] | list[Path]: Multiple image paths

        Returns:
            A list of predictions, one per sample.

        Examples:
            >>> from instantlearn.models import DinoTxtZeroShotClassification
            >>> from instantlearn.data.base.sample import Sample
            >>> import torch
            >>> import numpy as np
            >>> dinotxt = DinoTxtZeroShotClassification(device="cpu")
            >>> ref_sample = Sample(
            ...     image=torch.zeros((3, 512, 512)),
            ...     categories=["cat", "dog"],
            ...     category_ids=np.array([0, 1]),
            ...     is_reference=[True, True],
            ... )
            >>> dinotxt.fit(ref_sample)  # Can pass Sample directly
            >>> target_sample = Sample(
            ...     image=torch.zeros((3, 512, 512)),
            ...     is_reference=[False],
            ...     categories=["object"],
            ... )
            >>> result = dinotxt.predict(target_sample)  # Can pass Sample directly
        """
        target_batch = Batch.collate(target)
        target_images = target_batch.images
        target_features = self.dino_encoder.encode_image(target_images)
        target_features /= target_features.norm(dim=-1, keepdim=True)
        logits = 100.0 * target_features @ self.reference_features
        scores = logits.softmax(dim=1)
        _, max_class_ids = scores.max(dim=1)

        predictions = []
        for max_class_id, score in zip(max_class_ids, scores, strict=False):
            predictions.append({
                "pred_scores": score[max_class_id],
                "pred_labels": max_class_id,
            })
        return predictions
