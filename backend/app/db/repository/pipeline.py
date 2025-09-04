# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging

from sqlalchemy.orm import Session

from db.repository.common import BaseRepository

logger = logging.getLogger(__name__)


class PipelineRepository(BaseRepository):
    def __init__(self, session: Session):
        self.session = session
