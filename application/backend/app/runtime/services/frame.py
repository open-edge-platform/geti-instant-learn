# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
import uuid
from pathlib import Path
from queue import Empty, Queue
from uuid import UUID

from domain.errors import ResourceNotFoundError, ResourceType, ServiceError
from domain.repositories.frame import FrameRepository
from domain.repositories.project import ProjectRepository
from domain.repositories.source import SourceRepository
from domain.services.schemas.processor import InputData
from runtime.errors import PipelineNotActiveError

logger = logging.getLogger(__name__)


class FrameService:
    def __init__(
        self,
        frame_repo: FrameRepository,
        project_repo: ProjectRepository,
        source_repo: SourceRepository,
        inbound_queue: Queue[InputData] | None = None,
    ):
        self._inbound_queue = inbound_queue
        self._frame_repo = frame_repo
        self._project_repo = project_repo
        self._source_repo = source_repo

    def capture_frame(self, project_id: UUID) -> UUID:
        """
        Capture the latest frame from the running pipeline's inbound stream.

        Args:
            project_id: The project ID to capture a frame from.
        Returns:
            The UUID of the saved frame.
        Raises:
            ResourceNotFoundError: If project not found or has no active source.
            PipelineNotActiveError: If project is not active.
            ServiceError: If frame capture fails.
        """
        if self._inbound_queue is None:
            raise ServiceError("Frame capture service has not been properly initialized with inbound queue.")

        project = self._project_repo.get_by_id(project_id)
        if not project:
            raise ResourceNotFoundError(
                resource_type=ResourceType.PROJECT,
                resource_id=str(project_id),
            )

        if not project.active:
            raise PipelineNotActiveError(
                f"Cannot capture frame: project {project_id} is not active. "
                "Please activate the project before capturing frames."
            )

        active_source = self._source_repo.get_active_in_project(project_id)
        if not active_source:
            raise ResourceNotFoundError(
                resource_type=ResourceType.SOURCE,
                resource_id=None,
                message=f"Project {project_id} has no active source. Please connect a source before capturing frames.",
            )

        try:
            # wait for a frame from the pipeline
            input_data = self._inbound_queue.get(timeout=5.0)
            frame = input_data.frame

            frame_id = uuid.uuid4()
            self._frame_repo.save_frame(project_id, frame_id, frame)
            logger.info(f"Captured frame {frame_id} for project {project_id}")

            return frame_id

        except Empty:
            raise ServiceError("No frame received within 5 seconds timeout. Pipeline may not be running.")
        except Exception as e:
            logger.exception(f"Failed to capture frame for project {project_id}.")
            raise ServiceError(f"Frame capture failed: {str(e)}")

    def get_frame_path(self, project_id: UUID, frame_id: UUID) -> Path | None:
        """Get the path to a stored frame."""
        return self._frame_repo.get_frame_path(project_id, frame_id)
