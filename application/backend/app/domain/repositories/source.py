# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
from collections.abc import Sequence
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from domain.db.models import SourceDB
from domain.repositories.base import BaseRepository
from runtime.core.components.schemas.reader import SourceType

logger = logging.getLogger(__name__)


class SourceRepository(BaseRepository):
    """
    Repository responsible for low-level persistence of `SourceDB` entities.

    Responsibilities:
      - Build and execute SQLAlchemy queries.
      - Add / delete ORM entities to the session.
      - No business logic, no commits, no domain exceptions.
    """

    def __init__(self, session: Session):
        """Initialize the repository."""
        super().__init__(session)

    def add(self, source: SourceDB) -> None:
        """
        Add a new SourceDB entity to the session (not committed).
        """
        logger.debug(f"Adding source id={source.id} project_id={source.project_id}")
        self.session.add(source)

    def get_by_id(self, source_id: UUID) -> SourceDB | None:
        """
        Retrieve a source by primary key.
        """
        logger.debug(f"Fetching source by id={source_id}")
        return self.session.scalars(select(SourceDB).where(SourceDB.id == source_id)).first()

    def get_by_id_and_project(self, source_id: UUID, project_id: UUID) -> SourceDB | None:
        """
        Retrieve a source by id constrained to a project.
        """
        logger.debug(f"Fetching source id={source_id} in project_id={project_id}")
        stmt = select(SourceDB).where(SourceDB.id == source_id, SourceDB.project_id == project_id)
        return self.session.scalars(stmt).first()

    def get_all_by_project(self, project_id: UUID) -> Sequence[SourceDB]:
        """
        Retrieve all sources belonging to a project.
        """
        logger.debug(f"Fetching all sources for project_id={project_id}")
        stmt = select(SourceDB).where(SourceDB.project_id == project_id)
        return self.session.scalars(stmt).all()

    def delete(self, source: SourceDB) -> None:
        """
        Mark a SourceDB entity for deletion (not committed).
        """
        logger.debug(f"Deleting source id={source.id} project_id={source.project_id}")
        self.session.delete(source)

    def get_connected_in_project(self, project_id: UUID) -> SourceDB | None:
        """
        Retrieve the connected source in a project (if any).
        """
        stmt = select(SourceDB).where(SourceDB.project_id == project_id, SourceDB.connected.is_(True))
        return self.session.scalars(stmt).first()

    def get_by_type_in_project(self, project_id: UUID, source_type: SourceType) -> SourceDB | None:
        """
        Retrieve source of a given source_type in a project.
        """
        stmt = select(SourceDB).where(
            SourceDB.project_id == project_id,
            func.json_extract(SourceDB.config, "$.source_type") == source_type,
        )
        return self.session.scalars(stmt).first()
