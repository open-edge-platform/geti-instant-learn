# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Reference batch construction for the inference pipeline."""

import logging
from collections.abc import Sequence
from typing import Any
from uuid import UUID

import cv2
import numpy as np
from instantlearn.data.base.batch import Batch
from instantlearn.data.base.sample import Sample
from sqlalchemy.orm import Session, sessionmaker
from torch import from_numpy
from torchvision import tv_tensors

from domain.db.models import PromptDB, PromptType
from domain.errors import ServiceError
from domain.repositories.frame import FrameRepository
from domain.repositories.prompt import PromptRepository
from domain.repositories.supported_model import SupportedModelRepository
from domain.services.label import LabelService
from domain.services.schemas.annotation import AnnotationType
from domain.services.schemas.label import LabelInfo
from domain.services.schemas.mappers.annotation import annotations_db_to_schemas
from domain.services.schemas.mappers.mask import polygons_to_masks
from domain.services.schemas.pipeline import PipelineConfig
from domain.services.schemas.processor import ModelType
from settings import get_settings

logger = logging.getLogger(__name__)


class ReferenceBatchService:
    """Build reference batches from visual prompts for the inference pipeline.

    Encapsulates DB queries, frame I/O, per-prompt sample conversion, and batch collation.
    """

    def __init__(self, session_factory: sessionmaker[Session], frame_repository: FrameRepository) -> None:
        self._session_factory = session_factory
        self._frame_repository = frame_repository
        self._supported_model_repo = SupportedModelRepository()

    def build(self, config: PipelineConfig) -> tuple[Batch, dict[int, str]] | None:
        """Build a reference batch from the current pipeline config.

        Returns:
            (Batch, category_id → label_id mapping) or None when no batch can be built.
        """
        if config.processor is None:
            logger.debug("No active processor, skipping reference batch: project_id=%s", config.project_id)
            return None

        model_type = ModelType(config.processor.model_type)

        if model_type == ModelType.SAM3 and config.prompt_mode == PromptType.TEXT:
            # TODO: text-only batch for SAM3 — issue #758
            logger.warning("Text prompts not yet supported for SAM3: project_id=%s", config.project_id)
            return None

        supported_types = self._supported_model_repo.get_supported_annotation_types(model_type)
        needs_bboxes = AnnotationType.RECTANGLE in supported_types
        use_label_names = needs_bboxes and get_settings().sam3_hybrid_mode
        return self._build_visual_batch(
            project_id=config.project_id, output_bboxes=needs_bboxes, use_label_names=use_label_names
        )

    def _build_visual_batch(
        self, project_id: UUID, output_bboxes: bool = False, use_label_names: bool = True
    ) -> tuple[Batch, dict[int, str]] | None:
        """Query prompts, load frames, convert to samples, and collate.

        Args:
            project_id: Project to build the batch for.
            output_bboxes: Produce bboxes instead of masks.
            use_label_names: Pass real label names to samples; when False, ``"visual"``
                placeholder is used instead.

        Returns:
            (Batch, category_id → label_id mapping) or None if no valid samples.
        """
        result = self._query_prompts_and_labels(project_id, use_label_names)
        if result is None:
            return None

        db_prompts, label_info = result
        category_mappings = label_info.category_mappings
        samples = self._convert_prompts_to_samples(db_prompts, project_id, label_info, output_bboxes)

        if not samples:
            logger.info("No valid samples generated: project_id=%s", project_id)
            return None

        batch = Batch.collate(samples)
        logger.info(
            "Created reference batch: project_id=%s, samples=%d, categories=%d",
            project_id,
            len(batch.samples),
            len(category_mappings.label_to_category_id),
        )
        return batch, category_mappings.category_id_to_label_id

    def _query_prompts_and_labels(
        self, project_id: UUID, use_label_names: bool
    ) -> tuple[Sequence[PromptDB], LabelInfo] | None:
        """Fetch visual prompts and build label info inside a DB session.

        Returns:
            (prompts, label_info) or None when no visual prompts exist.
        """
        with self._session_factory() as session:
            prompt_repo = PromptRepository(session=session)
            label_svc = LabelService(session=session)

            db_prompts = prompt_repo.list_by_project_and_type(project_id=project_id, prompt_type=PromptType.VISUAL)
            if not db_prompts:
                logger.info("No visual prompts found: project_id=%s", project_id)
                return None

            all_label_ids: set[UUID] = set()
            for prompt in db_prompts:
                all_label_ids.update(ann.label_id for ann in prompt.annotations)

            category_mappings = label_svc.build_category_mappings(all_label_ids)

            if use_label_names:
                labels = label_svc.get_labels_by_ids(all_label_ids)
                db_id_to_name = {label.id: label.name for label in labels}
                label_id_to_name = {lid: db_id_to_name.get(lid, str(lid)) for lid in all_label_ids}
            else:
                label_id_to_name = dict.fromkeys(all_label_ids, "visual")

        return db_prompts, LabelInfo(category_mappings=category_mappings, label_id_to_name=label_id_to_name)

    def _convert_prompts_to_samples(
        self,
        db_prompts: Sequence[PromptDB],
        project_id: UUID,
        label_info: LabelInfo,
        output_bboxes: bool,
    ) -> list[Sample]:
        """Load frames and convert each prompt to a Sample."""
        samples: list[Sample] = []
        for prompt in db_prompts:
            if not prompt.frame_id:
                logger.warning("Visual prompt missing frame_id: prompt_id=%s", prompt.id)
                continue
            try:
                frame = self._frame_repository.read_frame(project_id, prompt.frame_id)
                if frame is None:
                    logger.warning("Frame not found: prompt_id=%s, frame_id=%s", prompt.id, prompt.frame_id)
                    continue

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                sample = self.visual_prompt_to_sample(
                    prompt=prompt,
                    frame=frame_rgb,
                    label_info=label_info,
                    output_bboxes=output_bboxes,
                )
                samples.append(sample)
            except Exception:
                logger.warning("Failed to convert prompt: prompt_id=%s", prompt.id, exc_info=True)
                continue
        return samples

    @staticmethod
    def visual_prompt_to_sample(  # noqa: C901
        prompt: PromptDB,
        frame: np.ndarray,
        label_info: LabelInfo,
        output_bboxes: bool = False,
    ) -> Sample:
        """Convert a visual prompt to a Sample with merged semantic masks.

        Args:
            prompt: Visual prompt with annotations
            frame: RGB image as numpy array (H, W, C)
            label_info: Bundled label context (category mappings and label names)
            output_bboxes: If True, produce bboxes from polygon vertices instead of masks

        Returns:
            Sample with either masks (N, H, W) or bboxes (N, 4) in [x1, y1, x2, y2] format.
        """
        if prompt.type != PromptType.VISUAL:
            raise ServiceError(f"Cannot convert non-visual prompt to sample: prompt type is {prompt.type}")

        annotations = annotations_db_to_schemas(prompt.annotations)
        if not annotations:
            raise ServiceError(
                f"Cannot convert visual prompt to sample: prompt {prompt.id} has no valid annotations with labels"
            )

        polygon_annotations = [(ann, ann.config) for ann in annotations if ann.config.type == AnnotationType.POLYGON]
        if not polygon_annotations:
            raise ServiceError(
                "Cannot create training sample: visual prompt must have at least one polygon annotation."
            )

        # Convert frame: HWC numpy → CHW tensor
        frame_chw = tv_tensors.Image(from_numpy(frame).permute(2, 0, 1))
        height, width = frame_chw.shape[-2:]

        # Group annotations by label_id
        label_groups: dict[UUID, list[Any]] = {}
        for ann, polygon in polygon_annotations:
            if ann.label_id not in label_groups:
                label_groups[ann.label_id] = []
            label_groups[ann.label_id].append(polygon)

        label_to_category_id = label_info.category_mappings.label_to_category_id
        label_id_to_name = label_info.label_id_to_name

        all_masks: list[np.ndarray] = []
        all_bboxes: list[list[float]] = []
        categories: list[str] = []
        category_ids: list[int] = []
        is_reference: list[bool] = []

        for label_id, polygons in sorted(label_groups.items(), key=lambda x: str(x[0])):
            if not polygons:
                continue

            category_id = label_to_category_id[label_id]
            category_name = label_id_to_name.get(label_id, str(label_id))

            if output_bboxes:
                for polygon in polygons:
                    xs = [pt.x for pt in polygon.points]
                    ys = [pt.y for pt in polygon.points]
                    all_bboxes.append([min(xs), min(ys), max(xs), max(ys)])
                    categories.append(category_name)
                    category_ids.append(category_id)
                    is_reference.append(True)
            else:
                instance_masks = polygons_to_masks(polygons, height, width)
                semantic_mask = np.any(instance_masks, axis=0).astype(np.uint8)
                all_masks.append(semantic_mask)
                categories.append(category_name)
                category_ids.append(category_id)
                is_reference.append(True)

        has_annotations = all_bboxes if output_bboxes else all_masks
        if not has_annotations:
            raise ServiceError(f"No valid annotations for prompt {prompt.id}")

        return Sample(
            image=frame_chw,
            masks=np.stack(all_masks, axis=0) if all_masks else None,
            bboxes=np.array(all_bboxes, dtype=np.float32) if all_bboxes else None,
            categories=categories,
            category_ids=np.array(category_ids, dtype=np.int32),
            is_reference=is_reference,
            image_path=str(prompt.frame_id),
        )
