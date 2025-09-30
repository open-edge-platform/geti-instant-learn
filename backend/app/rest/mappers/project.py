# Copyright (C) 2022-2025 Intel Corporation
# LIMITED EDGE SOFTWARE DISTRIBUTION LICENSE

from collections.abc import Iterable

from db.models import ProjectDB
from rest.schemas.processor import ProcessorSchema
from rest.schemas.project import ProjectListItem, ProjectPostPayload, ProjectSchema
from rest.schemas.sink import SinkSchema
from rest.schemas.source import SourceSchema


def project_db_to_schema(project: ProjectDB) -> ProjectSchema:
    """
    Map a ProjectDB ORM instance (and its optional related entities) to a ProjectSchema.
    """
    return ProjectSchema(
        id=project.id,
        name=project.name,
        source=(
            SourceSchema(
                id=project.source.id,
                type=project.source.type,
                config=project.source.config,
            )
            if project.source
            else None
        ),
        processor=(
            ProcessorSchema(
                id=project.processor.id,
                type=project.processor.type,
                config=project.processor.config,
                name=project.processor.name,
            )
            if project.processor
            else None
        ),
        sink=(
            SinkSchema(
                id=project.sink.id,
                config=project.sink.config,
            )
            if project.sink
            else None
        ),
    )


def project_db_to_list_item(project: ProjectDB) -> ProjectListItem:
    """
    Map a ProjectDB instance to a lightweight list item representation.
    """
    return ProjectListItem(id=project.id, name=project.name)


def projects_db_to_list_items(projects: Iterable[ProjectDB]) -> list[ProjectListItem]:
    """
    Bulk map an iterable of ProjectDB entities to list item schemas.
    """
    return [project_db_to_list_item(p) for p in projects]


def project_post_payload_to_db(payload: ProjectPostPayload) -> ProjectDB:
    """
    Create a new (unpersisted) ProjectDB entity from a ProjectPostPayload.
    The caller (service layer) is responsible for adding it to the session,
    flushing, activation handling, and committing.
    """
    project = ProjectDB(name=payload.name)
    if payload.id:
        project.id = payload.id
    return project
