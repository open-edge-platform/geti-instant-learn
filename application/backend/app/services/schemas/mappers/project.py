# Copyright (C) 2022-2025 Intel Corporation
# LIMITED EDGE SOFTWARE DISTRIBUTION LICENSE

from collections.abc import Iterable

from db.models import ProjectDB
from services.schemas.project import ProjectCreateSchema, ProjectSchema


def project_db_to_schema(project: ProjectDB) -> ProjectSchema:
    """
    Map a ProjectDB ORM instance to a ProjectSchema.
    """
    return ProjectSchema(id=project.id, name=project.name)


def projects_db_to_list_items(projects: Iterable[ProjectDB]) -> list[ProjectSchema]:
    """
    Bulk map an iterable of ProjectDB entities to list item schemas.
    """
    return [project_db_to_schema(p) for p in projects]


def project_schema_to_db(payload: ProjectCreateSchema) -> ProjectDB:
    """
    Create a new (unpersisted) ProjectDB entity from a ProjectCreateSchema.
    The caller (service layer) is responsible for adding it to the session,
    flushing, activation handling, and committing.
    """
    return ProjectDB(id=payload.id, name=payload.name)
