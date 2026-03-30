# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""This model uses a zero-shot object detector (from Huggingface) to generate boxes for SAM."""

import logging

import torch

from instantlearn.components import SamDecoder
from instantlearn.components.postprocessing import PostProcessor, default_postprocessor
from instantlearn.components.sam import load_sam_model
from instantlearn.data.base.batch import Batch, Collatable
from instantlearn.data.base.sample import BACKGROUND_CATEGORY_ID, Sample
from instantlearn.models.base import Model
from instantlearn.utils.constants import SAMModelName

from .grounded import GroundingModel, TextToBoxPromptGenerator
from .prompt_filter import BoxPromptFilter

logger = logging.getLogger(__name__)


class GroundedSAM(Model):
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
        super().__init__(postprocessor=postprocessor)
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

    def fit(self, reference: Sample | list[Sample] | Batch) -> None:
        """Perform learning step on the reference images and priors.

        Args:
            reference: Reference data to learn from. Accepts:
                - Sample: A single reference sample
                - list[Sample]: A list of reference samples
                - Batch: A batch of reference samples
        """
        reference_batch = Batch.collate(reference)
        self.category_mapping = {}
        has_background = False
        for sample in reference_batch.samples:
            for category_id, category in zip(sample.category_ids, sample.categories, strict=False):
                if int(category_id) == BACKGROUND_CATEGORY_ID:
                    has_background = True
                    continue
                if category not in self.category_mapping:
                    self.category_mapping[category] = int(category_id)
        if has_background:
            logger.warning(
                "GroundedSAM does not support negative prompts. "
                "Background masks will be ignored. Use Matcher, PerDino, "
                "SoftMatcher, or SAM3 for negative prompt support.",
            )

    def predict(self, target: Collatable) -> list[dict[str, torch.Tensor]]:
        """Perform inference step on the target images.

        Args:
            target: Target data to infer. Accepts:
                - Sample: A single target sample
                - list[Sample]: A list of target samples
                - Batch: A batch of target samples
                - str | Path: A single image path
                - list[str] | list[Path]: Multiple image paths

        Returns:
            A list of predictions, one per sample. Each prediction contains:
                "pred_masks": torch.Tensor of shape [num_masks, H, W]
                "pred_scores": torch.Tensor of shape [num_masks]
                "pred_labels": torch.Tensor of shape [num_masks]
                "pred_boxes": torch.Tensor of shape [num_boxes, 5] with [x1, y1, x2, y2, score]
        """
        target_batch = Batch.collate(target)
        # Generate box prompts (tensor format)
        box_prompts, category_ids = self.prompt_generator(
            target_batch.images,
            self.category_mapping,
        )

        # Filter box prompts
        box_prompts = self.prompt_filter(box_prompts)

        # Decode masks
        predictions = self.segmenter(
            target_batch.images,
            category_ids,
            box_prompts=box_prompts,
        )
        return self.apply_postprocessing(predictions)
