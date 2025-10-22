# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
import uuid
from pathlib import Path
from queue import Empty, Queue
from threading import Event
from uuid import UUID

import numpy as np

from core.components.schemas.processor import InputData
from core.runtime.pipeline_manager import PipelineManager
from repositories.frame import FrameRepository
from services.errors import ServiceError

logger = logging.getLogger(__name__)


class FrameCaptureError(ServiceError):
    """Raised when frame capture fails."""


class FrameCapture:
    """Temporary consumer to capture a single frame from the inbound broadcaster."""

    def __init__(self):
        self._queue: Queue[InputData] | None = None
        self._frame: np.ndarray | None = None
        self._event = Event()

    def set_queue(self, queue: Queue[InputData]) -> None:
        """Set the queue for receiving frames."""
        self._queue = queue

    def wait(self, timeout: float = 5.0) -> np.ndarray:
        """
        Wait for and return a captured frame.

        Args:
            timeout: Maximum time to wait for a frame in seconds.
        Returns:
            The captured frame as numpy array.
        Raises:
            FrameCaptureError: If no frame is received within timeout or queue not set.
        """
        if self._queue is None:
            raise FrameCaptureError("Queue not set. Call set_queue() first.")

        try:
            input_data = self._queue.get(timeout=timeout)
            self._frame = input_data.frame
            self._event.set()
            return self._frame
        except Empty:
            raise FrameCaptureError(f"No frame received within {timeout} seconds")


class FrameService:
    def __init__(self, pipeline_manager: PipelineManager, frame_repo: FrameRepository):
        self._pipeline_manager = pipeline_manager
        self._frame_repo = frame_repo

    def capture_frame(self, project_id: UUID) -> UUID:
        """
        Capture the latest frame from the running pipeline's inbound stream.

        Args:
            project_id: The project ID to capture a frame from.
        Returns:
            The UUID of the saved frame.
        Raises:
            FrameCaptureError: If frame capture or saving fails.
        """
        capture = FrameCapture()
        consumer_queue: Queue[InputData] | None = None

        try:
            consumer_queue = self._pipeline_manager.register_inbound_consumer(project_id)
            capture.set_queue(consumer_queue)

            frame = capture.wait(timeout=5.0)

            frame_id = uuid.uuid4()
            self._frame_repo.save_frame(project_id, frame_id, frame)
            logger.info(f"Captured frame {frame_id} for project {project_id}")

            return frame_id

        except Exception as e:
            logger.error(f"Failed to capture frame for project {project_id}: {e}")
            raise FrameCaptureError(f"Frame capture failed: {e}")
        finally:
            if consumer_queue is not None:
                try:
                    self._pipeline_manager.unregister_inbound_consumer(project_id, consumer_queue)
                except Exception as e:
                    logger.warning(f"Failed to unregister consumer: {e}")

    def get_frame_path(self, project_id: UUID, frame_id: UUID) -> Path | None:
        """Get the path to a stored frame."""
        return self._frame_repo.get_frame_path(project_id, frame_id)
