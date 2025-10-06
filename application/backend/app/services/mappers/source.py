# Copyright (C) 2022-2025 Intel Corporation
# LIMITED EDGE SOFTWARE DISTRIBUTION LICENSE


from collections.abc import Iterable
from uuid import UUID

from db.models import SourceDB
from rest.schemas.source import SourcePayloadSchema, SourceSchema

COMMON_FIELDS = {"source_type", "name", "id"}


def source_db_to_schema(source: SourceDB) -> SourceSchema:
    """
    Map a SourceDB instance to SourceSchema (discriminated by source_type).
    Merges stored config (e.g. device_id) into top-level schema fields.
    """
    data = {
        "source_type": source.type,
        "id": source.id,
        "name": source.name,
    }
    # merge config (expected to contain additional fields like device_id etc.)
    if isinstance(source.config, dict):
        data.update(source.config)
    return SourceSchema(**data, connected=source.connected)


def sources_db_to_schemas(sources: Iterable[SourceDB]) -> list[SourceSchema]:
    """
    Map a list of SourceDB instances to a list of SourceSchema objects.
    """
    return [source_db_to_schema(s) for s in sources]


def source_payload_to_db(payload: SourcePayloadSchema, project_id: UUID) -> SourceDB:
    """
    Create a new SourceDB (unpersisted) from payload, project_id should be injected by service layer.
    """
    variant_dict = payload.model_dump()
    config = {k: v for k, v in variant_dict.items() if k not in {"source_type", "name"}}
    return SourceDB(
        type=variant_dict["source_type"],
        name=variant_dict["name"],
        config=config,
        connected=False,
        project_id=project_id,
    )
