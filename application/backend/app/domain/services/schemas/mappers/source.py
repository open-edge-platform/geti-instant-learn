# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Iterable
from uuid import UUID

from pydantic import TypeAdapter, ValidationError

from domain.db.models import SourceDB
from domain.services.schemas.base import Pagination
from domain.services.schemas.reader import ReaderConfig
from domain.services.schemas.source import (
    SourceCreateSchema,
    SourceSchema,
    SourcesListSchema,
)

# Create type adapter for ReaderConfig union validation
_reader_config_adapter = TypeAdapter(ReaderConfig)


def source_db_to_schema(source: SourceDB) -> SourceSchema:
    """
    Map a SourceDB instance to SourceSchema.

    Skips filesystem validation when deserializing from database, but checks
    file availability separately to populate the 'available' and 'unavailable_reason' fields.
    """
    # First, deserialize config with filesystem validation skipped
    try:
        config = _reader_config_adapter.validate_python(source.config, context={"skip_file_validation": True})
    except ValidationError as e:
        # If even basic validation fails, return with structural error
        error_msg = "; ".join([f"{err['loc'][0]}: {err['msg']}" for err in e.errors()])
        return SourceSchema(
            id=source.id,
            active=source.active,
            config=source.config,
            available=False,
            unavailable_reason=f"Invalid configuration: {error_msg}",
        )

    # Now check if the source is actually available (file exists, etc.)
    # by attempting validation without skip flag
    try:
        _reader_config_adapter.validate_python(source.config, context={"skip_file_validation": False})
        available = True
        unavailable_reason = None
    except ValidationError as e:
        # File doesn't exist or other filesystem validation failed
        available = False
        # Extract the most relevant error message
        errors = e.errors()
        if errors:
            unavailable_reason = errors[0].get("msg", "Source is unavailable")
        else:
            unavailable_reason = "Source is unavailable"

    return SourceSchema(
        id=source.id,
        active=source.active,
        config=config,
        available=available,
        unavailable_reason=unavailable_reason,
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
        config=schema.config.model_dump(mode="json"),
        active=schema.active,
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
