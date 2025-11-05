# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Iterable
from typing import Any
from uuid import UUID, uuid4

from db.models import AnnotationDB, PromptDB, PromptType
from services.schemas.prompt import (
    PromptCreateSchema,
    PromptSchema,
    TextPromptCreateSchema,
    TextPromptSchema,
    VisualPromptSchema,
)


def prompt_db_to_schema(prompt: PromptDB) -> PromptSchema:
    """
    Map a PromptDB instance to PromptSchema.
    Pydantic will discriminate by `type` field.
    """
    if prompt.type == PromptType.TEXT:
        return TextPromptSchema(
            id=prompt.id,
            type=PromptType.TEXT,
            name=prompt.name,
            content=prompt.text or "",
            project_id=prompt.project_id,
        )
    # PromptType.VISUAL
    # Extract annotations from the relationship - config is already a dict that Pydantic will parse
    annotations: list[Any] = [ann.config for ann in prompt.annotations]
    return VisualPromptSchema(
        id=prompt.id,
        type=PromptType.VISUAL,
        name=prompt.name,
        frame_id=prompt.frame_id or UUID(int=0),  # Should never be None due to constraint
        annotations=annotations,  # type: ignore[arg-type]
        project_id=prompt.project_id,
    )


def prompts_db_to_schemas(prompts: Iterable[PromptDB]) -> list[PromptSchema]:
    """
    Map a list of PromptDB instances to a list of PromptSchema objects.
    """
    return [prompt_db_to_schema(p) for p in prompts]


def prompt_create_schema_to_db(schema: PromptCreateSchema, project_id: UUID) -> PromptDB:
    """
    Create a new PromptDB (unpersisted) from schema.
    project_id should be injected by service layer.
    """
    prompt_id = uuid4()

    if isinstance(schema, TextPromptCreateSchema):
        prompt_db = PromptDB(
            id=prompt_id,
            type=PromptType.TEXT,
            name=schema.name,
            text=schema.content,
            frame_id=None,
            project_id=project_id,
            annotations=[],
        )
    else:  # VisualPromptCreateSchema
        # Create annotation entities from the schema
        annotation_entities = [
            AnnotationDB(
                id=uuid4(),
                config=ann.model_dump(),
                prompt_id=prompt_id,
            )
            for ann in schema.annotations
        ]

        prompt_db = PromptDB(
            id=prompt_id,
            type=PromptType.VISUAL,
            name=schema.name,
            text=None,
            frame_id=schema.frame_id,
            project_id=project_id,
            annotations=annotation_entities,
        )

    return prompt_db
