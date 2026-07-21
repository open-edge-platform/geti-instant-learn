# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""DINOv3 zero-shot classification model."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
from torchvision import tv_tensors

from instantlearn.data.base.batch import Batch, Collatable
from instantlearn.models.torch_adapter import arrays_to_prediction, batch_to_tensors
from instantlearn.models.torch_base import ExportConfig, TorchModel
from instantlearn.utils import precision_to_torch_dtype
from instantlearn.utils.constants import DINOv3BackboneSize

from ._card import _DINOTXT_CARD
from .encoder import IMAGENET_TEMPLATES, DinoTextEncoder

if TYPE_CHECKING:
    from pathlib import Path

    from instantlearn.components.postprocessing import PostProcessor
    from instantlearn.data.base.prediction import Prediction
    from instantlearn.data.base.sample import Sample
    from instantlearn.models.model_card import ModelCard


class DinoTxtZeroShotClassification(TorchModel):
    """DINOv3 zero-shot image classifier.

    Examples:
        >>> from instantlearn.models import DinoTxtZeroShotClassification
        >>> from instantlearn.data.base.sample import Category, Sample
        >>> import numpy as np
        >>> model = DinoTxtZeroShotClassification(device="cpu")
        >>> sample = Sample(
        ...     image=np.zeros((512, 512, 3), dtype=np.uint8),
        ...     categories=[Category(0, "cat"), Category(1, "dog")],
        ... )
        >>> predictions = model.predict(sample)
        >>> len(predictions)
        1
    """

    def __init__(
        self,
        prompt_templates: list[str] = IMAGENET_TEMPLATES,
        precision: str = "bf16",
        device: str = "cuda",
        image_size: tuple[int, int] | None = (512, 512),
        backbone_size: DINOv3BackboneSize = DINOv3BackboneSize.LARGE,
        postprocessor: PostProcessor | None = None,
    ) -> None:
        """Initialize the DinoTxtZeroShotClassification."""
        super().__init__(
            device=device,
            precision=precision,
            postprocessor=postprocessor,
        )
        self.torch_precision = precision_to_torch_dtype(precision)
        self.dino_encoder = DinoTextEncoder(
            device=device,
            image_size=image_size,
            precision=self.torch_precision,
            backbone_size=backbone_size,
        )
        self.prompt_templates = prompt_templates
        self.category_mapping: dict[int, str] = {}
        self.reference_features: torch.Tensor | None = None

    @classmethod
    def card(cls) -> ModelCard:
        """Return the static capability descriptor for DinoTxt."""
        return _DINOTXT_CARD

    def fit(self, reference: Sample | list[Sample] | Batch) -> None:
        """Perform learning step on the reference batch.

        DinoTxt extracts categories from the reference batch to create text priors.

        Args:
            reference: Reference data to learn from. Accepts:
                - Sample: A single reference sample
                - list[Sample]: A list of reference samples
                - Batch: A batch of reference samples

        Raises:
            ValueError: If no reference samples with categories are provided.
        """
        reference_batch = Batch.collate(reference)
        if not reference_batch.samples:
            msg = "reference_batch must contain at least one sample"
            raise ValueError(msg)

        category_mapping = _category_mapping(reference_batch)

        if not category_mapping:
            msg = "reference_batch must contain samples with categories"
            raise ValueError(msg)

        self.category_mapping = category_mapping
        # reference features is zero shot weights from DinoTxtEncoder
        self.reference_features = self.dino_encoder.encode_text(category_mapping, self.prompt_templates)

    @torch.no_grad()
    def predict(self, target: Collatable) -> list[Prediction]:
        """Perform inference on the target batch.

        Args:
            target: Target samples or a compatibility ``Batch``.

        Returns:
            A classification-shaped ``Prediction`` for each target sample.

        Raises:
            ValueError: If categories or an image are missing from a target.
        """
        target_batch = Batch.collate(target)
        category_mapping = self.category_mapping or _category_mapping(target_batch)
        if not category_mapping:
            msg = "DinoTxt requires categories from fit() or target samples."
            raise ValueError(msg)

        reference_features = self.reference_features
        if reference_features is None or category_mapping != self.category_mapping:
            reference_features = self.dino_encoder.encode_text(category_mapping, self.prompt_templates)

        tensor_batch = batch_to_tensors(target_batch, device=self.device)
        if any(image is None for image in tensor_batch.images):
            msg = "DinoTxt.predict() requires each sample to contain an image."
            raise ValueError(msg)
        target_images = [tv_tensors.Image(image) for image in tensor_batch.images if image is not None]
        target_features = self.dino_encoder.encode_image(target_images)
        target_features /= target_features.norm(dim=-1, keepdim=True)
        logits = 100.0 * target_features @ reference_features
        scores = logits.softmax(dim=1)
        max_scores, max_class_indexes = scores.max(dim=1)
        category_ids = sorted(category_mapping)

        predictions: list[Prediction] = []
        for sample, max_score, max_class_index in zip(
            target_batch.samples,
            max_scores,
            max_class_indexes,
            strict=True,
        ):
            if sample.image is None:
                msg = "DinoTxt.predict() requires each sample to contain an image."
                raise ValueError(msg)
            height, width = sample.image.shape[:2]
            label_id = category_ids[int(max_class_index)]
            predictions.append(
                arrays_to_prediction(
                    masks=np.empty((0, height, width), dtype=np.uint8),
                    scores=np.array([max_score.detach().cpu().item()], dtype=np.float32),
                    label_ids=np.array([label_id], dtype=np.int32),
                    categories=category_mapping,
                ),
            )
        return predictions

    def to_openvino(  # noqa: PLR6301
        self,
        export_path: Path | None = None,
        config: ExportConfig | None = None,
    ) -> Path:
        """Raise until DinoTxt has a functional OpenVINO sibling."""
        del export_path, config
        msg = "DinoTxt does not support OpenVINO export because no DinoTxtOpenVINO implementation exists."
        raise NotImplementedError(msg)


def _category_mapping(batch: Batch) -> dict[int, str]:
    """Build a stable category-ID-to-label mapping from batch samples."""
    mapping: dict[int, str] = {}
    for sample in batch.samples:
        for category in sample.categories:
            mapping.setdefault(category.id, category.label)
    return mapping
