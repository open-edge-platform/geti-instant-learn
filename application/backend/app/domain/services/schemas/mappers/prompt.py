# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Iterable
from uuid import UUID, uuid4

import numpy as np
from getiprompt.data.base.sample import Sample
from torch import from_numpy
from torchvision import tv_tensors

from domain.db.models import AnnotationDB, PromptDB, PromptType
from domain.errors import ServiceError
from domain.services.schemas.annotation import AnnotationSchema, AnnotationType
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


def visual_prompt_to_sample(prompt: PromptDB, frame: np.ndarray) -> Sample:
    """
    Convert a visual PromptDB and its frame to a training Sample.
    """
    if prompt.type != PromptType.VISUAL:
        raise ServiceError(f"Cannot convert non-visual prompt to sample: prompt type is {prompt.type}")

    annotations = annotations_db_to_schemas(prompt.annotations)

    if not annotations:
        raise ServiceError(
            f"Cannot convert visual prompt to sample: prompt {prompt.id} has no valid annotations with labels"
        )

    polygons = [ann.config for ann in annotations if ann.config.type == AnnotationType.POLYGON]

    if not polygons:
        raise ServiceError(
            "Cannot create training sample: visual prompt must have at least one polygon annotation to generate masks."
        )

    # Convert frame: HWC numpy → CHW tensor
    frame_chw = tv_tensors.Image(from_numpy(frame).permute(2, 0, 1))

    # Convert polygons to binary masks
    height, width = frame.shape[:2]
    masks = polygons_to_masks(polygons, height, width)

    categories = sorted([str(ann.label_id) for ann in annotations])
    category_ids = np.arange(len(categories), dtype=np.int32)

    return Sample(
        image=frame_chw,
        masks=masks,
        categories=categories,
        category_ids=category_ids,
    )
