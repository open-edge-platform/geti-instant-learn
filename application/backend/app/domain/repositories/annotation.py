# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
from uuid import UUID

from sqlalchemy import exists, select
from sqlalchemy.orm import Session

from domain.db.models import AnnotationDB
from domain.repositories.base import BaseRepository

logger = logging.getLogger(__name__)


class AnnotationRepository(BaseRepository[AnnotationDB]):
    """
    Repository responsible for low-level persistence of `AnnotationDB` entities.
    """

    def __init__(self, session: Session):
        """Initialize the repository."""
        super().__init__(session=session, model=AnnotationDB)

    def is_label_in_use(self, label_id: UUID) -> bool:
        """Check if a label is referenced by any annotations."""
        stmt = select(exists().where(AnnotationDB.label_id == label_id))
        return self.session.scalar(stmt) or False
