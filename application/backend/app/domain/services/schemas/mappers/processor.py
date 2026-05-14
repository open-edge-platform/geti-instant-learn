# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Iterable
from uuid import UUID

from domain.db.models import ProcessorDB
from domain.services.schemas.base import Pagination
from domain.services.schemas.processor import ProcessorCreateSchema, ProcessorListSchema, ProcessorSchema


def processor_db_to_schema(processor: ProcessorDB) -> ProcessorSchema:
    """
    Map a ProcessorDB instance to ProcessorSchema.
    Pydantic will discriminate ModelConfig by its `model_type` inside config.
    """
    return ProcessorSchema(
        id=processor.id,
        active=processor.active,
        config=processor.config,
        name=processor.name,
    )


def processor_schema_to_db(schema: ProcessorCreateSchema, project_id: UUID) -> ProcessorDB:
    """
    Create a new ProcessorDB (unpersisted) from schema, project_id should be injected by service layer.
    """
    return ProcessorDB(
        id=schema.id,
        config=schema.config.model_dump(),
        active=schema.active,
        project_id=project_id,
        name=schema.name,
    )


def processors_db_to_list_items(
    processors: Iterable[ProcessorDB], total: int, offset: int = 0, limit: int = 20
) -> ProcessorListSchema:
    """
    Map an iterable of ProcessorDB entities to ProcessorListSchema with pagination metadata.

    Parameters:
        processors: Iterable of ProcessorDB entities to map
        total: Total number of processors available
        offset: Starting index of the returned items
        limit: Maximum number of items requested

    Returns:
        ProcessorListSchema with mapped processors and pagination metadata
    """
    items = [processor_db_to_schema(processor) for processor in processors]

    pagination = Pagination(
        count=len(items),
        total=total,
        offset=offset,
        limit=limit,
    )

    return ProcessorListSchema(models=items, pagination=pagination)
