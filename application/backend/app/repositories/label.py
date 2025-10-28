# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
from collections.abc import Sequence
from uuid import UUID

from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session

from db.models import LabelDB
from repositories.base import BaseRepository

logger = logging.getLogger(__name__)


class LabelRepository(BaseRepository):
    """
    Repository responsible for low-level persistence of `LabelDB` entities.

    Responsibilities:
      - Build and execute SQLAlchemy queries.
      - Add / delete ORM entities to the session.
      - No business logic, no commits, no domain exceptions.
    """

    def __init__(self, session: Session):
        """Initialize the repository."""
        super().__init__(session)

    def add(self, label: LabelDB) -> None:
        """Add a new label instance to the session (not committed)."""
        logger.debug(f"Adding label entity {label.id} (name={label.name})")
        self.session.add(label)

    def get_by_id(self, project_id: UUID, label_id: UUID) -> LabelDB | None:
        """Retrieve a label by its ID."""
        logger.debug(f"Fetching label by id={label_id} from project_id={project_id}")
        return self.session.scalars(select(LabelDB).filter_by(id=label_id, project_id=project_id)).first()

    def get_all(self, project_id: UUID) -> Sequence[LabelDB]:
        """Retrieve all labels."""
        logger.debug("Fetching all labels")
        return self.session.scalars(select(LabelDB).where(LabelDB.project_id == project_id)).all()

    def delete(self, project_id: UUID, label: LabelDB) -> None:
        """Mark a label entity for deletion (not committed)."""
        logger.debug(f"Deleting label id={label.id} name={label.name}")
        self.session.execute(delete(LabelDB).where(LabelDB.id == label.id, LabelDB.project_id == project_id))

    def get_paginated(self, project_id: UUID, offset: int = 0, limit: int = 20) -> tuple[Sequence[LabelDB], int]:
        """
        Retrieve labels with pagination.

        Returns:
            A tuple of (labels, total_count)
        """
        logger.debug(f"Fetching labels for project id {project_id} with offset={offset}, limit={limit}")

        # Fetch total count and paginated results in one query
        labels_query = (
            select(LabelDB).where(LabelDB.project_id == project_id).order_by(LabelDB.name).offset(offset).limit(limit)
        )
        total_count_query = select(func.count()).select_from(LabelDB).where(LabelDB.project_id == project_id)

        labels, total_count = (
            self.session.scalars(labels_query).all(),
            self.session.scalar(total_count_query) or 0,
        )

        return labels, total_count
