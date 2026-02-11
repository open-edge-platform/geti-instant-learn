# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
from enum import StrEnum
from pathlib import Path
from uuid import UUID

import cv2
import numpy as np

from settings import get_settings

logger = logging.getLogger(__name__)

settings = get_settings()


class ColorFormat(StrEnum):
    """Color format for frame reading."""

    BGR = "bgr"
    RGB = "rgb"


class FrameRepository:
    def __init__(self, base_dir: Path | None = None):
        self._base_dir = base_dir or Path(settings.db_data_dir) / "projects"

    def _frame_path(self, project_id: UUID, frame_id: UUID) -> Path:
        """Construct the filesystem path for a given frame."""
        return self._base_dir / str(project_id) / "frames" / f"{frame_id}.jpg"

    def save_frame(self, project_id: UUID, frame_id: UUID, frame: np.ndarray) -> Path:
        """Save a frame as JPEG to the filesystem."""
        path = self._frame_path(project_id, frame_id)
        path.parent.mkdir(parents=True, exist_ok=True)

        try:
            # Convert RGB to BGR for OpenCV encoding
            bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            success, buffer = cv2.imencode(".jpg", bgr_frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            if not success:
                logger.error(f"Failed to encode frame {frame_id} for project {project_id}")
                raise RuntimeError(f"Failed to encode frame {frame_id}")
        except cv2.error as e:
            logger.exception(f"OpenCV error while encoding frame {frame_id} for project {project_id}.")
            raise RuntimeError(f"Failed to encode frame {frame_id}: {str(e)}")

        path.write_bytes(buffer.tobytes())
        return path

    def get_frame_path(self, project_id: UUID, frame_id: UUID) -> Path | None:
        """Get the filesystem path for a frame if it exists."""
        path = self._frame_path(project_id, frame_id)
        return path if path.exists() else None

    def delete_frame(self, project_id: UUID, frame_id: UUID) -> bool:
        """Delete a frame file."""
        path = self._frame_path(project_id, frame_id)
        if path.exists():
            path.unlink()
            logger.debug(f"Deleted frame {frame_id} from project {project_id}")
            return True
        return False

    def read_frame(
        self, project_id: UUID, frame_id: UUID, color_format: ColorFormat = ColorFormat.BGR
    ) -> np.ndarray | None:
        """Load a frame from disk.

        Args:
            project_id: The project UUID.
            frame_id: The frame UUID.
            color_format: Output color format (BGR or RGB). Defaults to BGR.

        Returns:
            Frame as numpy array in the specified color format, or None if not found.
        """
        path = self._frame_path(project_id, frame_id)
        if not path.exists():
            logger.warning(f"Frame file not found: {path}")
            return None
        try:
            frame = cv2.imread(str(path))
            if frame is None:
                logger.error(f"Failed to read frame from {path}")
                return None
            if color_format == ColorFormat.RGB:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return frame
        except cv2.error:
            logger.exception(f"OpenCV error while reading frame {frame_id} from {path}")
            return None
