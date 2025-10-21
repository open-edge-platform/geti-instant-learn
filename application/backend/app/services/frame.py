# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from repositories.frame import FrameRepository
from core.runtime.pipeline_manager import PipelineManager
import logging
import uuid
from uuid import UUID
from typing import Optional
import numpy as np
import threading

from core.runtime.pipeline_manager import PipelineManager
from repositories.frame import FrameRepository
from services.errors import ServiceError

logger = logging.getLogger(__name__)


class FrameCaptureError(ServiceError):
    """Raised when frame capture fails."""
    pass


class FrameCapture():
    """Temporary consumer to capture a single frame."""

    # TODO use code from core (broadcasters?)


class FrameService:
    def __init__(self, pipeline_manager: PipelineManager, frame_repo: FrameRepository):
        self._pipeline_manager = pipeline_manager
        self._frame_repo = frame_repo

    def capture_frame(self, project_id: UUID) -> UUID:
        """Capture the latest frame from the running pipeline's inbound queue."""
        # Create one-shot consumer
        capture = FrameCapture()

        try:
            # Register consumer
            self._pipeline_manager.register_inbound_consumer(project_id, capture)

            # Wait for frame
            frame = capture.wait(timeout=5.0)

            # Save frame
            frame_id = uuid.uuid4()
            self._frame_repo.save_frame(project_id, frame_id, frame)
            logger.info(f"Captured frame {frame_id} for project {project_id}")

            return frame_id

        except Exception as e:
            logger.error(f"Failed to capture frame for project {project_id}: {e}")
            raise FrameCaptureError(f"Frame capture failed: {e}")
        finally:
            # Always unregister
            try:
                self._pipeline_manager.unregister_inbound_consumer(project_id, capture)
            except Exception as e:
                logger.warning(f"Failed to unregister consumer: {e}")

    def get_frame(self, project_id: UUID, frame_id: UUID) -> Optional[bytes]:
        """Retrieve a stored frame by ID."""
        return self._frame_repo.load_frame(project_id, frame_id)



