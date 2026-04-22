# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any
from uuid import UUID, uuid4

import numpy as np
from instantlearn.data.base.sample import Sample
from torch import from_numpy
from torchvision import tv_tensors

from domain.db.models import AnnotationDB, PromptDB, PromptType
from domain.errors import ServiceError
from domain.services.schemas.annotation import AnnotationSchema, AnnotationType, RectangleAnnotation
from domain.services.schemas.mappers.annotation import annotations_db_to_schemas
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


def filter_prompts_by_annotation_type(
    prompts: Sequence[PromptDB], supported_types: set[AnnotationType]
) -> list[PromptDB]:
    """Return only prompts whose annotations match the supported annotation types."""
    filtered_prompts = []
    for prompt in prompts:
        matching_annotations = [
            ann for ann in prompt.annotations if AnnotationType(ann.config.get("type")) in supported_types
        ]
        if matching_annotations:
            filtered_prompts.append(prompt)
    return filtered_prompts


@dataclass
class AnnotationGroupResult:
    """Intermediate result from processing a group of annotations by type."""

    categories: list[str] = field(default_factory=list)
    category_ids: list[int] = field(default_factory=list)
    is_reference: list[bool] = field(default_factory=list)
    n_shot: list[int] = field(default_factory=list)
    masks: list[np.ndarray] = field(default_factory=list)
    bboxes: list[list[float]] = field(default_factory=list)


def _group_annotations_by_label(annotations: list[tuple[AnnotationSchema, Any]]) -> dict[UUID, list[Any]]:
    """Group annotation configs by label ID."""
    groups: dict[UUID, list[Any]] = {}
    for ann, config in annotations:
        groups.setdefault(ann.label_id, []).append(config)
    return groups


def _process_polygon_groups(
    label_groups: dict[UUID, list[Any]],
    label_to_category_id: dict[UUID, int],
    label_id_to_name: dict[UUID, str],
    label_shot_counts: dict[UUID, int],
    height: int,
    width: int,
) -> AnnotationGroupResult:
    """Convert polygon annotations grouped by label into masks with metadata.

    Args:
        label_groups: Mapping from label UUID to list of PolygonAnnotation configs
        label_to_category_id: Mapping from label UUID to category ID
        label_shot_counts: Current shot count per label (modified in-place)
        height: Image height in pixels
        width: Image width in pixels

    Returns:
        Result containing masks and associated metadata
    """
    result = AnnotationGroupResult()

    for label_id, polygons in sorted(label_groups.items(), key=lambda x: str(x[0])):
        if not polygons:
            continue

        instance_masks = polygons_to_masks(polygons, height, width)
        semantic_mask = np.any(instance_masks, axis=0).astype(np.uint8)

        category_id = label_to_category_id[label_id]
        current_shot = label_shot_counts.get(label_id, 0)

        result.masks.append(semantic_mask)
        result.categories.append(label_id_to_name.get(label_id, str(label_id)))
        result.category_ids.append(category_id)
        result.is_reference.append(True)
        result.n_shot.append(current_shot)

        label_shot_counts[label_id] = current_shot + 1

    return result


def _process_rectangle_groups(
    label_groups: dict[UUID, list[RectangleAnnotation]],
    label_to_category_id: dict[UUID, int],
    label_id_to_name: dict[UUID, str],
    label_shot_counts: dict[UUID, int],
) -> AnnotationGroupResult:
    """Convert rectangle annotations grouped by label into bounding boxes with metadata.

    Args:
        label_groups: Mapping from label UUID to list of RectangleAnnotation configs
        label_to_category_id: Mapping from label UUID to category ID
        label_shot_counts: Current shot count per label (modified in-place)

    Returns:
        Result containing bboxes and associated metadata
    """
    result = AnnotationGroupResult()

    for label_id, rects in sorted(label_groups.items(), key=lambda x: str(x[0])):
        if not rects:
            continue

        category_id = label_to_category_id[label_id]
        current_shot = label_shot_counts.get(label_id, 0)

        for rect in rects:
            result.bboxes.append([rect.points[0].x, rect.points[0].y, rect.points[1].x, rect.points[1].y])
            result.categories.append(label_id_to_name.get(label_id, str(label_id)))
            result.category_ids.append(category_id)
            result.is_reference.append(True)
            result.n_shot.append(current_shot)

        label_shot_counts[label_id] = current_shot + 1

    return result


def visual_prompt_to_sample(
    prompt: PromptDB,
    frame: np.ndarray,
    label_to_category_id: dict[UUID, int],
    label_id_to_name: dict[UUID, str],
    label_shot_counts: dict[UUID, int],
    supported_annotation_types: set[AnnotationType],
) -> Sample:
    """Convert a visual prompt to a Sample with masks and/or bounding boxes.

    Polygon annotations are merged into semantic masks (one per label).
    Rectangle annotations are converted to bounding boxes in [x1, y1, x2, y2] format.

    Args:
        prompt: Visual prompt with annotations
        frame: RGB image as numpy array (H, W, C)
        label_to_category_id: Mapping from label UUID to category ID (shared across batch)
        label_id_to_name: Mapping from label UUID to label name
        label_shot_counts: Current shot count per label (modified in-place)
        supported_annotation_types: Set of supported annotation types to include in the sample

    Returns:
        Sample with masks and/or bboxes, one entry per unique label in the prompt

    Example:
        Prompt with 3 car annotations + 2 person annotations:
        - Creates 2 masks (1 for cars, 1 for persons)
        - n_shot = [current_car_shot, current_person_shot]
        - Updates label_shot_counts for both labels
    """
    if prompt.type != PromptType.VISUAL:
        raise ServiceError(f"Cannot convert non-visual prompt to sample: prompt type is {prompt.type}")

    annotations = annotations_db_to_schemas(prompt.annotations)
    if not annotations:
        raise ServiceError(
            f"Cannot convert visual prompt to sample: prompt {prompt.id} has no valid annotations with labels"
        )

    supported_annotations = [ann for ann in annotations if ann.config.type in supported_annotation_types]

    if not supported_annotations:
        raise ServiceError("Cannot create training sample: visual prompt must have at least one annotation.")

    polygon_annotations = [
        (ann, ann.config) for ann in supported_annotations if ann.config.type == AnnotationType.POLYGON
    ]
    rectangle_annotations = [
        (ann, ann.config) for ann in supported_annotations if ann.config.type == AnnotationType.RECTANGLE
    ]

    # Convert frame: HWC numpy → CHW tensor
    frame_chw = tv_tensors.Image(from_numpy(frame).permute(2, 0, 1))
    height, width = frame_chw.shape[-2:]

    # Process each annotation type
    polygon_result = _process_polygon_groups(
        _group_annotations_by_label(polygon_annotations),
        label_to_category_id,
        label_id_to_name,
        label_shot_counts,
        height,
        width,
    )
    rect_result = _process_rectangle_groups(
        _group_annotations_by_label(rectangle_annotations),
        label_to_category_id,
        label_id_to_name,
        label_shot_counts,
    )

    # Merge results
    categories = polygon_result.categories + rect_result.categories
    category_ids = polygon_result.category_ids + rect_result.category_ids
    is_reference = polygon_result.is_reference + rect_result.is_reference
    n_shot = polygon_result.n_shot + rect_result.n_shot

    if not categories:
        raise ServiceError(f"No valid annotations for prompt {prompt.id} after processing")

    masks = np.stack(polygon_result.masks, axis=0) if polygon_result.masks else None
    bboxes = np.array(rect_result.bboxes, dtype=np.float32) if rect_result.bboxes else None

    return Sample(
        image=frame_chw,
        masks=masks,
        bboxes=bboxes,
        categories=categories,
        category_ids=np.array(category_ids, dtype=np.int32),
        is_reference=is_reference,
        n_shot=n_shot,
        image_path=str(prompt.frame_id),
    )


def deduplicate_annotations(
    annotations: list[AnnotationSchema], image_height: int, image_width: int, iou_threshold: float = 0.9
) -> list[AnnotationSchema]:
    """
    Remove duplicate or highly overlapping annotations.

    Uses IoU (Intersection over Union) to identify similar shapes.
    Keeps the first occurrence when duplicates are found.

    Args:
        annotations: List of annotations to deduplicate
        image_height: Height in pixels for mask generation
        image_width: Width in pixels for mask generation
        iou_threshold: IoU threshold above which annotations are considered duplicates (default: 0.9)

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


def _calculate_rect_iou(rect1: RectangleAnnotation, rect2: RectangleAnnotation) -> float:
    """Calculate IoU between two rectangle annotations.

    Args:
        rect1: First rectangle annotation (two points: top-left and bottom-right)
        rect2: Second rectangle annotation (two points: top-left and bottom-right)

    Returns:
        IoU score between 0 and 1
    """
    x1_min, y1_min = rect1.points[0].x, rect1.points[0].y
    x1_max, y1_max = rect1.points[1].x, rect1.points[1].y

    x2_min, y2_min = rect2.points[0].x, rect2.points[0].y
    x2_max, y2_max = rect2.points[1].x, rect2.points[1].y

    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)

    inter_area = max(0.0, inter_x_max - inter_x_min) * max(0.0, inter_y_max - inter_y_min)
    area1 = (x1_max - x1_min) * (y1_max - y1_min)
    area2 = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = area1 + area2 - inter_area

    if union_area == 0:
        return 0.0

    return inter_area / union_area
