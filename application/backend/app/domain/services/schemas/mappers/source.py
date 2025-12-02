# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Iterable
from uuid import UUID

from domain.db.models import SourceDB
from domain.services.schemas.base import Pagination
from domain.services.schemas.source import (
    SourceCreateSchema,
    SourceSchema,
    SourcesListSchema,
)


def source_db_to_schema(source: SourceDB) -> SourceSchema:
    """
    Map a SourceDB instance to SourceSchema.
    Pydantic will discriminate ReaderConfig by its `source_type` inside config.
    """
    return SourceSchema(
        id=source.id,
        connected=source.active,
        config=source.config,
    )


def sources_db_to_schemas(sources: Iterable[SourceDB]) -> list[SourceSchema]:
    """
    Map a list of SourceDB instances to a list of SourceSchema objects.
    """
    return [source_db_to_schema(s) for s in sources]


def source_schema_to_db(schema: SourceCreateSchema, project_id: UUID) -> SourceDB:
    """
    Create a new SourceDB (unpersisted) from schema, project_id should be injected by service layer.
    """
    return SourceDB(
        id=schema.id,
        config=schema.config.model_dump(),
        active=schema.connected,
        project_id=project_id,
    )


def sources_db_to_list_items(
    sources: Iterable[SourceDB], total: int, offset: int = 0, limit: int = 20
) -> SourcesListSchema:
    """
    Map an iterable of SourceDB entities to SourcesListSchema with pagination metadata.
    """
    items = [source_db_to_schema(source) for source in sources]

    pagination = Pagination(
        count=len(items),
        total=total,
        offset=offset,
        limit=limit,
    )

    return SourcesListSchema(sources=items, pagination=pagination)
