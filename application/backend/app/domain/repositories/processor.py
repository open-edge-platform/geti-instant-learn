# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging

from sqlalchemy.orm import Session

from domain.db.models import ProcessorDB
from domain.repositories.base import PipelineComponentRepository

logger = logging.getLogger(__name__)


class ProcessorRepository(PipelineComponentRepository[ProcessorDB]):
    """
    Repository responsible for low-level persistence of `ProcessorDB` entities.
    """

    def __init__(self, session: Session):
        """Initialize the repository."""
        super().__init__(session=session, model=ProcessorDB)
