# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging

from sqlalchemy.orm import Session

from domain.db.models import LabelDB
from domain.repositories.base import ProjectComponentRepository

logger = logging.getLogger(__name__)


class LabelRepository(ProjectComponentRepository[LabelDB]):
    """
    Repository responsible for low-level persistence of `LabelDB` entities.
    """

    def __init__(self, session: Session):
        """Initialize the repository."""
        super().__init__(session=session, model=LabelDB)
