# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
from collections.abc import Sequence
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from domain.db.models import SinkDB
from domain.repositories.base import BaseRepository

logger = logging.getLogger(__name__)


class SinkRepository(BaseRepository):
    """
    Data access layer for SinkDB entities.

    Provides methods to add, retrieve, and delete sinks in the database, scoped by project.
    Contains no business logic or transaction management.
    """

    def __init__(self, session: Session):
        """Initialize the repository."""
        super().__init__(session)

    def add(self, sink: SinkDB) -> None:
        """
        Add a new SinkDB entity to the session (not committed).
        """
        logger.debug(f"Adding sink id={sink.id} project_id={sink.project_id}")
        self.session.add(sink)

    def get_by_id(self, sink_id: UUID) -> SinkDB | None:
        """
        Retrieve a sink by primary key.
        """
        logger.debug(f"Fetching sink by id={sink_id}")
        return self.session.scalars(select(SinkDB).where(SinkDB.id == sink_id)).first()

    def get_by_id_and_project(self, sink_id: UUID, project_id: UUID) -> SinkDB | None:
        """
        Retrieve a sink by id constrained to a project.
        """
        logger.debug(f"Fetching sink id={sink_id} in project_id={project_id}")
        stmt = select(SinkDB).where(SinkDB.id == sink_id, SinkDB.project_id == project_id)
        return self.session.scalars(stmt).first()

    def get_all_by_project(self, project_id: UUID) -> Sequence[SinkDB]:
        """
        Retrieve all sinks belonging to a project.
        """
        logger.debug(f"Fetching all sinks for project_id={project_id}")
        stmt = select(SinkDB).where(SinkDB.project_id == project_id)
        return self.session.scalars(stmt).all()

    def delete(self, sink: SinkDB) -> None:
        """
        Mark a SinkDB entity for deletion (not committed).
        """
        logger.debug(f"Deleting sink id={sink.id} project_id={sink.project_id}")
        self.session.delete(sink)

    def get_active_in_project(self, project_id: UUID) -> SinkDB | None:
        """
        Retrieve the active sink in a project (if any).
        """
        stmt = select(SinkDB).where(SinkDB.project_id == project_id, SinkDB.active.is_(True))
        return self.session.scalars(stmt).first()
