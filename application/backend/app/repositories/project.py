# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
from collections.abc import Sequence
from uuid import UUID

from sqlalchemy import func, literal, select
from sqlalchemy.orm import Session

from db.models import ProjectDB
from repositories.base import BaseRepository

logger = logging.getLogger(__name__)


class ProjectRepository(BaseRepository):
    """
    Repository responsible for low-level persistence of `ProjectDB` entities.

    Responsibilities:
      - Build and execute SQLAlchemy queries.
      - Add / delete ORM entities to the session.
      - No business logic, no commits, no domain exceptions.
    """

    def __init__(self, session: Session):
        """Initialize the repository."""
        super().__init__(session)

    def add(self, project: ProjectDB) -> None:
        """Add a new project instance to the session (not committed)."""
        logger.debug(f"Adding project entity {project.id} (name={project.name})")
        self.session.add(project)

    def get_by_id(self, project_id: UUID) -> ProjectDB | None:
        """Retrieve a project by its ID."""
        logger.debug(f"Fetching project by id={project_id}")
        return self.session.scalars(select(ProjectDB).where(ProjectDB.id == project_id)).first()

    def get_all(self) -> Sequence[ProjectDB]:
        """Retrieve all projects."""
        logger.debug("Fetching all projects")
        return self.session.scalars(select(ProjectDB)).all()

    def get_active(self) -> ProjectDB | None:
        """Retrieve the currently active project."""
        logger.debug("Fetching active project")
        return self.session.scalars(select(ProjectDB).where(ProjectDB.active.is_(True))).first()

    def exists_by_name(self, name: str) -> bool:
        """Check whether a project with the given name exists."""
        logger.debug(f"Checking existence by name={name}")
        return self.session.scalars(select(literal(True)).where(ProjectDB.name == name).limit(1)).first() is not None

    def exists_by_id(self, project_id: UUID) -> bool:
        """Check whether a project with the given ID exists."""
        logger.debug(f"Checking existence by id={project_id}")
        return (
            self.session.scalars(select(literal(True)).where(ProjectDB.id == project_id).limit(1)).first() is not None
        )

    def delete(self, project: ProjectDB) -> None:
        """Mark a project entity for deletion (not committed)."""
        logger.debug(f"Deleting project id={project.id} name={project.name}")
        self.session.delete(project)

    def get_paginated(self, offset: int = 0, limit: int = 20) -> tuple[Sequence[ProjectDB], int]:
        """
        Retrieve projects with pagination.

        Returns:
            A tuple of (projects, total_count)
        """
        logger.debug(f"Fetching projects with offset={offset}, limit={limit}")

        # Get total count
        total_count = self.session.scalar(select(func.count()).select_from(ProjectDB)) or 0

        # Get paginated results
        projects = self.session.scalars(select(ProjectDB).order_by(ProjectDB.name).offset(offset).limit(limit)).all()

        return projects, total_count
