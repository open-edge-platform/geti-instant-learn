# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
import uuid
from pathlib import Path
from queue import Empty, Queue
from uuid import UUID

from core.components.schemas.processor import InputData
from core.runtime.pipeline_manager import PipelineManager
from repositories.frame import FrameRepository
from repositories.project import ProjectRepository
from repositories.source import SourceRepository
from services.errors import ResourceNotFoundError, ResourceType, ServiceError

logger = logging.getLogger(__name__)


class FrameService:
    def __init__(
        self,
        pipeline_manager: PipelineManager,
        frame_repo: FrameRepository,
        project_repo: ProjectRepository,
        source_repo: SourceRepository,
    ):
        self._pipeline_manager = pipeline_manager
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
            ResourceNotFoundError: If project not found or has no connected source.
            ServiceError: If project is not active or frame capture fails.
        """
        project = self._project_repo.get_by_id(project_id)
        if not project:
            raise ResourceNotFoundError(
                resource_type=ResourceType.PROJECT,
                resource_id=str(project_id),
            )

        if not project.active:
            raise ServiceError(
                f"Cannot capture frame: project {project_id} is not active. "
                "Please activate the project before capturing frames."
            )

        connected_source = self._source_repo.get_connected_in_project(project_id)
        if not connected_source:
            raise ResourceNotFoundError(
                resource_type=ResourceType.SOURCE,
                resource_id=None,
                message=f"Project {project_id} has no connected source. "
                "Please connect a source before capturing frames.",
            )

        consumer_queue: Queue[InputData] | None = None

        try:
            consumer_queue = self._pipeline_manager.register_inbound_consumer(project_id)

            # wait for a frame from the pipeline
            input_data = consumer_queue.get(timeout=5.0)
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
        finally:
            if consumer_queue is not None:
                try:
                    self._pipeline_manager.unregister_inbound_consumer(project_id, consumer_queue)
                except Exception as e:
                    logger.warning(f"Failed to unregister consumer: {e}")

    def get_frame_path(self, project_id: UUID, frame_id: UUID) -> Path | None:
        """Get the path to a stored frame."""
        return self._frame_repo.get_frame_path(project_id, frame_id)
