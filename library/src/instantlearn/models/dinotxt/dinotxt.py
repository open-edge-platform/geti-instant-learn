# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""DINOv3 zero-shot classification model."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
from torchvision import tv_tensors

from instantlearn.data.base.batch import Batch, Collatable
from instantlearn.models.torch_adapter import CategoryRegistry, arrays_to_prediction, batch_to_tensors
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
    from instantlearn.device import DeviceInfo
    from instantlearn.models.model_card import ModelCard


class DinoTxtZeroShotClassification(TorchModel):
    """DINOv3 zero-shot image classifier.

    Examples:
        >>> from instantlearn.models import DinoTxtZeroShotClassification
        >>> from instantlearn.data.base.sample import Category, Sample
        >>> import numpy as np
        >>> model = DinoTxtZeroShotClassification()
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
        device: DeviceInfo | None = None,
        image_size: tuple[int, int] | None = (512, 512),
        backbone_size: DINOv3BackboneSize = DINOv3BackboneSize.LARGE,
        weights_location: str | Path | None = None,
        postprocessor: PostProcessor | None = None,
    ) -> None:
        """Initialize the DinoTxtZeroShotClassification.

        Args:
            prompt_templates: Text templates for zero-shot classification.
            precision: Weight precision (``"fp32"``, ``"fp16"``, ``"bf16"``).
            device: Physical device, or ``None`` to select automatically.
            image_size: Input image size.
            backbone_size: DINOv3 backbone variant (only ``LARGE`` supported).
            weights_location: Path to pre-downloaded DINOv3 weights directory,
                or ``None`` to auto-download from Meta (requires prior access
                approval at https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/).
            postprocessor: Optional post-processor.
        """
        super().__init__(
            device=device,
            precision=precision,
            postprocessor=postprocessor,
        )
        self.torch_precision = precision_to_torch_dtype(precision)
        self.dino_encoder = DinoTextEncoder(
            device=self.device,
            image_size=image_size,
            precision=self.torch_precision,
            backbone_size=backbone_size,
            weights_location=weights_location,
        )
        self.prompt_templates = prompt_templates
        self.categories: CategoryRegistry = CategoryRegistry()
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

        categories = CategoryRegistry.from_samples(reference_batch)

        if not categories:
            msg = "reference_batch must contain samples with categories"
            raise ValueError(msg)

        self.categories = categories
        # reference features is zero shot weights from DinoTxtEncoder
        self.reference_features = self.dino_encoder.encode_text(categories.id_to_name, self.prompt_templates)

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
        categories = self.categories or CategoryRegistry.from_samples(target_batch)
        if not categories:
            msg = "DinoTxt requires categories from fit() or target samples."
            raise ValueError(msg)

        reference_features = self.reference_features
        if reference_features is None or categories != self.categories:
            reference_features = self.dino_encoder.encode_text(categories.id_to_name, self.prompt_templates)

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
        category_ids = sorted(categories)

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
                    categories=categories,
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

