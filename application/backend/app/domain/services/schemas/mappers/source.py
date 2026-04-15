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

    Validates with filesystem checks enabled first (common case: files exist).
    On ValidationError, retries with skip_file_validation=True to deserialize
    config and mark source as unavailable.
    """
    # Try validation with filesystem checks enabled (optimistic path)
    try:
        config = _reader_config_adapter.validate_python(source.config, context={"skip_file_validation": False})
        # Success - source is available
        return SourceSchema(
            id=source.id,
            active=source.active,
            config=config,
            available=True,
            unavailable_reason=None,
        )
    except ValidationError as e:
        # Filesystem or other validation failed
        # Extract error message for unavailable_reason
        errors = e.errors()
        if errors:
            unavailable_reason = errors[0].get("msg", "Source is unavailable")
        else:
            unavailable_reason = "Source is unavailable"

        # Retry with filesystem validation skipped to get config
        try:
            config = _reader_config_adapter.validate_python(source.config, context={"skip_file_validation": True})
            return SourceSchema(
                id=source.id,
                active=source.active,
                config=config,
                available=False,
                unavailable_reason=unavailable_reason,
            )
        except ValidationError as structural_error:
            # Structural validation failed (e.g., missing required fields, wrong types)
            # Use model_construct to bypass all validation including ReaderConfig validation
            # This allows returning raw config dict for debugging without re-raising ValidationError
            error_msg = "; ".join([f"{err['loc'][0]}: {err['msg']}" for err in structural_error.errors()])
            return SourceSchema.model_construct(
                id=source.id,
                active=source.active,
                config=source.config,  # Raw dict - model_construct bypasses ReaderConfig validation
                available=False,
                unavailable_reason=f"Invalid configuration: {error_msg}",
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
