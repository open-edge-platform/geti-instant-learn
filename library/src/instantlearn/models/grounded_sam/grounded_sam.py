# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Zero-shot object detection with text grounding followed by SAM masks."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torchvision import tv_tensors

from instantlearn.components import SamDecoder
from instantlearn.components.postprocessing import default_postprocessor
from instantlearn.components.postprocessing.base import apply_postprocessing
from instantlearn.components.sam import load_sam_model
from instantlearn.data.base.batch import Batch, Collatable
from instantlearn.models.torch_adapter import batch_to_tensors, dict_to_prediction, CategoryRegistry
from instantlearn.models.torch_base import ExportConfig, TorchModel
from instantlearn.utils.constants import SAMModelName

from ._card import _GROUNDED_SAM_CARD
from .grounded import GroundingModel, TextToBoxPromptGenerator
from .prompt_filter import BoxPromptFilter

if TYPE_CHECKING:
    from pathlib import Path

    from instantlearn.components.postprocessing import PostProcessor
    from instantlearn.data.base.prediction import Prediction
    from instantlearn.data.base.sample import Sample
    from instantlearn.models.model_card import ModelCard


class GroundedSAM(TorchModel):
    """This model uses a zero-shot object detector (from Huggingface) to generate boxes for SAM."""

    def __init__(
        self,
        sam: SAMModelName = SAMModelName.SAM_HQ_TINY,
        grounding_model: GroundingModel = GroundingModel.LLMDET_TINY,
        precision: str = "bf16",
        compile_models: bool = False,
        box_threshold: float = 0.4,
        text_threshold: float = 0.3,
        device: str = "cuda",
        postprocessor: PostProcessor | None = None,
    ) -> None:
        """Initialize the model.

        Args:
            sam: The SAM model name.
            grounding_model: The grounding model to use.
            precision: The precision to use for the model.
            compile_models: Whether to compile the models.
            box_threshold: The box threshold.
            text_threshold: The text threshold.
            device: The device to use.
            postprocessor: Post-processor applied after predict().
                Defaults to :func:`~instantlearn.components.postprocessing.default_postprocessor`
                (MaskIoMNMS + BoxIoMNMS).
        """
        if postprocessor is None:
            postprocessor = default_postprocessor()
        super().__init__(
            device=device,
            precision=precision,
            postprocessor=postprocessor,
        )
        self.sam_predictor = load_sam_model(
            sam,
            device=device,
            precision=precision,
            compile_models=compile_models,
        )
        self.prompt_generator: TextToBoxPromptGenerator = TextToBoxPromptGenerator(
            device=device,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            template=TextToBoxPromptGenerator.Template.specific_object,
            model_id=grounding_model,
            precision=precision,
            compile_models=compile_models,
        )
        self.segmenter: SamDecoder = SamDecoder(sam_predictor=self.sam_predictor)
        self.prompt_filter: BoxPromptFilter = BoxPromptFilter()
        # Category identity (populated by fit()).
        self.categories: CategoryRegistry = CategoryRegistry()

    @classmethod
    def card(cls) -> ModelCard:
        """Return the static capability descriptor for GroundedSAM."""
        return _GROUNDED_SAM_CARD

    def fit(self, reference: Sample | list[Sample] | Batch) -> None:
        """Optionally cache category names and IDs for later prediction.

        Args:
            reference: Reference data to learn from. Accepts:
                - Sample: A single reference sample
                - list[Sample]: A list of reference samples
                - Batch: A batch of reference samples
        """
        reference_batch = Batch.collate(reference)
        self.categories = CategoryRegistry.from_samples(reference_batch)

    @torch.no_grad()
    def predict(self, target: Collatable) -> list[Prediction]:
        """Run zero-shot grounded segmentation on one or more targets.

        Args:
            target: Target samples or a compatibility ``Batch``.

        Returns:
            A numpy-based ``Prediction`` for each target sample.

        Raises:
            ValueError: If neither ``fit()`` nor the target samples provide a
                category name.
        """
        target_batch = Batch.collate(target)
        # Prefer categories from fit(); fall back to per-target-sample categories.
        categories = self.categories if self.categories else CategoryRegistry.from_samples(target_batch)
        if not categories:
            msg = "GroundedSAM requires categories from fit() or target samples."
            raise ValueError(msg)

        # Build the name→id mapping expected by the prompt generator.
        category_mapping = categories.name_to_id

        tensor_batch = batch_to_tensors(target_batch, device=self.device)
        images = [tv_tensors.Image(image) for image in tensor_batch.images if image is not None]
        if len(images) != len(target_batch):
            msg = "GroundedSAM.predict() requires each sample to contain an image."
            raise ValueError(msg)

        # Generate box prompts (tensor format)
        box_prompts, category_ids = self.prompt_generator(
            images,
            category_mapping,
        )

        # Filter box prompts
        box_prompts = self.prompt_filter(box_prompts)

        # Decode masks
        predictions = self.segmenter(
            images,
            category_ids,
            box_prompts=box_prompts,
        )
        predictions = apply_postprocessing(predictions, self.postprocessor)
        return [dict_to_prediction(prediction, categories) for prediction in predictions]

    def to_openvino(  # noqa: PLR6301
        self,
        export_path: Path | None = None,
        config: ExportConfig | None = None,
    ) -> Path:
        """Raise until GroundedSAM has a functional OpenVINO sibling."""
        del export_path, config
        msg = "GroundedSAM does not support OpenVINO export because no GroundedSAMOpenVINO implementation exists."
        raise NotImplementedError(msg)

