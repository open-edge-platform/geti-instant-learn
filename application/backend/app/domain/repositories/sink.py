# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging

from sqlalchemy.orm import Session

from domain.db.models import SinkDB
from domain.repositories.base import PipelineComponentRepository

logger = logging.getLogger(__name__)


class SinkRepository(PipelineComponentRepository[SinkDB]):
    """
    Repository responsible for low-level persistence of `SinkDB` entities.
    """

    def __init__(self, session: Session):
        """Initialize the repository."""
        super().__init__(session=session, model=SinkDB)
