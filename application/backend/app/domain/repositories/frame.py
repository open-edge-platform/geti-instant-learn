# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
from pathlib import Path
from uuid import UUID

import cv2
import numpy as np

from settings import get_settings

logger = logging.getLogger(__name__)

settings = get_settings()


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
            success, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
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
