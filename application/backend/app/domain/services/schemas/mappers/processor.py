# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Iterable
from uuid import UUID

from domain.db.models import ProcessorDB
from domain.services.schemas.processor import ProcessorCreateSchema, ProcessorSchema


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


def processors_db_to_schemas(processors: Iterable[ProcessorDB]) -> list[ProcessorSchema]:
    """
    Map a list of ProcessorDB instances to a list of ProcessorSchema objects.
    """
    return [processor_db_to_schema(p) for p in processors]


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
