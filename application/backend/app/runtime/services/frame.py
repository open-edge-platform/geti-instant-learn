# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
from pathlib import Path
from uuid import UUID

from domain.repositories.frame import FrameRepository

logger = logging.getLogger(__name__)


class FrameService:
    def __init__(
        self,
        frame_repo: FrameRepository,
    ):
        self._frame_repo = frame_repo

    def get_frame_path(self, project_id: UUID, frame_id: UUID) -> Path | None:
        """Get the path to a stored frame."""
        return self._frame_repo.get_frame_path(project_id, frame_id)
