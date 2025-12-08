# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging

from sqlalchemy import select
from sqlalchemy.orm import Session

from domain.db.models import ProjectDB
from domain.repositories.base import BaseRepository

logger = logging.getLogger(__name__)


class ProjectRepository(BaseRepository[ProjectDB]):
    """
    Repository responsible for low-level persistence of `ProjectDB` entities.
    """

    def __init__(self, session: Session):
        """Initialize the repository."""
        super().__init__(session=session, model=ProjectDB)

    def get_active(self) -> ProjectDB | None:
        """Retrieve the currently active project."""
        stmt = select(ProjectDB).where(ProjectDB.active.is_(True))
        return self.session.execute(stmt).scalar_one_or_none()
