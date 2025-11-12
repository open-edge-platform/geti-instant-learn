# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
from collections.abc import Sequence
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from domain.db.models import ProcessorDB
from domain.repositories.base import BaseRepository

logger = logging.getLogger(__name__)


class ProcessorRepository(BaseRepository):
    """
    Repository responsible for low-level persistence of `ProcessorDB` entities.

    Responsibilities:
      - Build and execute SQLAlchemy queries.
      - Add / delete ORM entities to the session.
      - No business logic, no commits, no domain exceptions.
    """

    def __init__(self, session: Session):
        """Initialize the repository."""
        super().__init__(session)

    def add(self, processor: ProcessorDB) -> None:
        """
        Add a new ProcessorDB entity to the session (not committed).
        """
        logger.debug(f"Adding processor id={processor.id} project_id={processor.project_id}")
        self.session.add(processor)

    def get_by_id(self, processor_id: UUID) -> ProcessorDB | None:
        """
        Retrieve a model configuration by primary key.
        """
        logger.debug(f"Fetching model configuration by id={processor_id}")
        return self.session.scalars(select(ProcessorDB).where(ProcessorDB.id == processor_id)).first()

    def get_by_id_and_project(self, processor_id: UUID, project_id: UUID) -> ProcessorDB | None:
        """
        Retrieve a model configuration by id constrained to a project.
        """
        logger.debug(f"Fetching model configuration id={processor_id} in project_id={project_id}")
        stmt = select(ProcessorDB).where(ProcessorDB.id == processor_id, ProcessorDB.project_id == project_id)
        return self.session.scalars(stmt).first()

    def get_all_by_project(self, project_id: UUID) -> Sequence[ProcessorDB]:
        """
        Retrieve all model configurations belonging to a project.
        """
        logger.debug(f"Fetching all model configurations for project_id={project_id}")
        stmt = select(ProcessorDB).where(ProcessorDB.project_id == project_id)
        return self.session.scalars(stmt).all()

    def delete(self, processor: ProcessorDB) -> None:
        """
        Mark a ProcessorDB entity for deletion (not committed).
        """
        logger.debug(f"Deleting model configuration id={processor.id} project_id={processor.project_id}")
        self.session.delete(processor)

    def get_activated_in_project(self, project_id: UUID) -> ProcessorDB | None:
        """
        Retrieve the connected source in a project (if any).
        """
        logger.debug(f"Get active model configuration for project_id={project_id}")
        stmt = select(ProcessorDB).where(ProcessorDB.project_id == project_id, ProcessorDB.active.is_(True))
        return self.session.scalars(stmt).first()
