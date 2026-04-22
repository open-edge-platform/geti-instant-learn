# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
from collections.abc import Iterable
from uuid import UUID, uuid4

import numpy as np

from domain.db.models import AnnotationDB, PromptDB, PromptType
from domain.services.schemas.annotation import AnnotationSchema, AnnotationType
from domain.services.schemas.mappers.mask import polygons_to_masks
from domain.services.schemas.prompt import (
    PromptCreateSchema,
    PromptListItemSchema,
    PromptSchema,
    PromptUpdateSchema,
    TextPromptCreateSchema,
    TextPromptSchema,
    TextPromptUpdateSchema,
    VisualPromptListItemSchema,
    VisualPromptSchema,
)

logger = logging.getLogger(__name__)


def prompt_db_to_schema(prompt: PromptDB, include_thumbnail: bool = False) -> PromptSchema | PromptListItemSchema:
    """
    Map a PromptDB instance to a PromptSchema or PromptListItemSchema object.

    Args:
        prompt: The prompt database entity
        include_thumbnail: If True, returns schema with thumbnail for list responses.
                          If False, returns schema without thumbnail for detail responses.
    """
    if prompt.type == PromptType.TEXT:
        return TextPromptSchema(
            id=prompt.id,
            type=PromptType.TEXT,
            content=prompt.text or "",
        )

    annotations: list[AnnotationSchema] = [
        AnnotationSchema(config=ann.config, label_id=ann.label_id) for ann in prompt.annotations
    ]

    if include_thumbnail:
        return VisualPromptListItemSchema(
            id=prompt.id,
            type=PromptType.VISUAL,
            frame_id=prompt.frame_id,
            annotations=annotations,
            thumbnail=prompt.thumbnail or "",
        )

    return VisualPromptSchema(
        id=prompt.id,
        type=PromptType.VISUAL,
        frame_id=prompt.frame_id,
        annotations=annotations,
    )


def prompts_db_to_schemas(
    prompts: Iterable[PromptDB], include_thumbnail: bool = False
) -> list[PromptSchema | PromptListItemSchema]:
    """
    Map a list of PromptDB instances to a list of PromptSchema or PromptListItemSchema objects.

    Args:
        prompts: Iterable of prompt database entities
        include_thumbnail: If True, includes thumbnails in the response (for lists)
    """
    return [prompt_db_to_schema(p, include_thumbnail=include_thumbnail) for p in prompts]


def prompt_create_schema_to_db(schema: PromptCreateSchema, project_id: UUID, thumbnail: str | None = None) -> PromptDB:
    """
    Create a new PromptDB (unpersisted) from schema.
    project_id should be injected by service layer.
    """
    if isinstance(schema, TextPromptCreateSchema):
        prompt_db = PromptDB(
            id=schema.id,
            type=PromptType.TEXT,
            text=schema.content,
            frame_id=None,
            project_id=project_id,
            annotations=[],
            thumbnail=None,
        )
    else:
        annotation_entities = [
            AnnotationDB(
                id=uuid4(),
                config=ann.config.model_dump(),
                label_id=ann.label_id,
                prompt_id=schema.id,
            )
            for ann in schema.annotations
        ]

        prompt_db = PromptDB(
            id=schema.id,
            type=PromptType.VISUAL,
            text=None,
            frame_id=schema.frame_id,
            project_id=project_id,
            annotations=annotation_entities,
            thumbnail=thumbnail,
        )

    return prompt_db


def prompt_update_schema_to_db(prompt_db: PromptDB, schema: PromptUpdateSchema) -> PromptDB:
    """
    Update an existing PromptDB instance from an update schema.
    For visual prompts, annotations are replaced if provided.
    """
    if isinstance(schema, TextPromptUpdateSchema):
        if schema.content is not None:
            prompt_db.text = schema.content
    else:
        if schema.frame_id is not None:
            prompt_db.frame_id = schema.frame_id

        if schema.annotations is not None:
            # replace existing annotations with new ones
            prompt_db.annotations.clear()
            for ann in schema.annotations:
                annotation_entity = AnnotationDB(
                    id=uuid4(),
                    config=ann.config.model_dump(),
                    label_id=ann.label_id,
                    prompt_id=prompt_db.id,
                )
                prompt_db.annotations.append(annotation_entity)
    return prompt_db


def deduplicate_annotations(
    annotations: list[AnnotationSchema], image_height: int, image_width: int, iou_threshold: float = 0.9
) -> list[AnnotationSchema]:
    """Remove duplicate or highly overlapping annotations based on polygon similarity.

    Uses IoU to identify similar masks. Keeps the first occurrence when duplicates are found.
    Only processes polygon annotations; other types are kept as-is.

    Args:
        annotations: List of annotations to deduplicate
        image_height: Height in pixels for mask generation
        image_width: Width in pixels for mask generation
        iou_threshold: IoU threshold above which polygons are considered duplicates (default: 0.9)

    Returns:
        List of unique annotations with duplicates removed
    """
    polygon_annotations = [ann for ann in annotations if ann.config.type == AnnotationType.POLYGON]
    other_annotations = [ann for ann in annotations if ann.config.type != AnnotationType.POLYGON]
    if len(polygon_annotations) <= 1:
        return annotations

    polygon_configs = [ann.config for ann in polygon_annotations]
    masks = polygons_to_masks(polygon_configs, image_height, image_width)

    unique_indices: list[int] = []
    for i in range(len(masks)):
        is_duplicate = False
        for j in unique_indices:
            iou = _calculate_mask_iou(masks[i], masks[j])
            if iou > iou_threshold:
                is_duplicate = True
                break
        if not is_duplicate:
            unique_indices.append(i)

    unique_polygon_annotations = [polygon_annotations[i] for i in unique_indices]

    return unique_polygon_annotations + other_annotations


def _calculate_mask_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """
    Calculate IoU between two binary masks.

    Args:
        mask1: First binary mask (H, W)
        mask2: Second binary mask (H, W)

    Returns:
        IoU score between 0 and 1
    """
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()

    if union == 0:
        return 0.0

    return float(intersection / union)
