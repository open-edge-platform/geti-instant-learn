# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Iterable
from uuid import UUID

from domain.db.models import SinkDB
from domain.services.schemas.sink import SinkCreateSchema, SinkSchema


def sink_db_to_schema(sink: SinkDB) -> SinkSchema:
    """
    Map a SinkDB instance to SinkSchema.
    Pydantic will discriminate WriterConfig by its `sink_type` inside config.
    """
    return SinkSchema(
        id=sink.id,
        active=sink.active,
        config=sink.config,
    )


def sinks_db_to_schemas(sinks: Iterable[SinkDB]) -> list[SinkSchema]:
    """
    Map a list of SinkDB instances to a list of SinkSchema objects.
    """
    return [sink_db_to_schema(s) for s in sinks]


def sink_schema_to_db(schema: SinkCreateSchema, project_id: UUID) -> SinkDB:
    """
    Create a new SinkDB (unpersisted) from schema, project_id should be injected by service layer.
    """
    return SinkDB(
        id=schema.id,
        config=schema.config.model_dump(),
        active=schema.active,
        project_id=project_id,
    )
