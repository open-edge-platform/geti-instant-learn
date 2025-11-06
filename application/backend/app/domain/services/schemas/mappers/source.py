# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Iterable
from uuid import UUID

from domain.db.models import SourceDB
from domain.services.schemas.source import SourceCreateSchema, SourceSchema


def source_db_to_schema(source: SourceDB) -> SourceSchema:
    """
    Map a SourceDB instance to SourceSchema.
    Pydantic will discriminate ReaderConfig by its `source_type` inside config.
    """
    return SourceSchema(
        id=source.id,
        connected=source.connected,
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
        connected=schema.connected,
        project_id=project_id,
    )
