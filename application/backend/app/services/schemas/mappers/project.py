# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Iterable

from db.models import ProjectDB
from services.schemas.base import Pagination
from services.schemas.project import ProjectCreateSchema, ProjectSchema, ProjectsListSchema


def project_db_to_schema(project: ProjectDB) -> ProjectSchema:
    """
    Map a ProjectDB ORM instance to a ProjectSchema.
    """
    return ProjectSchema(id=project.id, name=project.name, active=project.active)


def projects_db_to_list_items(
    projects: Iterable[ProjectDB], total: int, offset: int = 0, limit: int = 20
) -> ProjectsListSchema:
    """
    Map an iterable of ProjectDB entities to ProjectsListSchema with pagination metadata.

    Parameters:
        projects: Iterable of ProjectDB entities to map
        total: Total number of projects available
        offset: Starting index of the returned items
        limit: Maximum number of items requested

    Returns:
        ProjectsListSchema with mapped projects and pagination metadata
    """
    items = [project_db_to_schema(p) for p in projects]

    pagination = Pagination(
        count=len(items),
        total=total,
        offset=offset,
        limit=limit,
    )

    return ProjectsListSchema(projects=items, pagination=pagination)


def project_schema_to_db(payload: ProjectCreateSchema) -> ProjectDB:
    """
    Create a new (unpersisted) ProjectDB entity from a ProjectCreateSchema.
    The caller (service layer) is responsible for adding it to the session,
    flushing, activation handling, and committing.
    """
    return ProjectDB(id=payload.id, name=payload.name)
