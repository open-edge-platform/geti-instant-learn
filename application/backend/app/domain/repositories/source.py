# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging

from sqlalchemy.orm import Session

from domain.db.models import SourceDB
from domain.repositories.base import PipelineComponentRepository

logger = logging.getLogger(__name__)


class SourceRepository(PipelineComponentRepository[SourceDB]):
    """
    Repository responsible for low-level persistence of `SourceDB` entities.
    """

    def __init__(self, session: Session):
        """Initialize the repository."""
        super().__init__(session=session, model=SourceDB)
