#  Copyright (C) 2025 Intel Corporation
#  SPDX-License-Identifier: Apache-2.0

import base64
import logging
import re
import time
from pathlib import Path

import cv2
import numpy as np  # noqa: TC002

from domain.services.schemas.base import Pagination
from domain.services.schemas.processor import InputData
from domain.services.schemas.reader import FrameListResponse, FrameMetadata, ReaderConfig
from runtime.core.components.base import StreamReader

logger = logging.getLogger(__name__)


class ImageFolderReader(StreamReader):
    """
    A reader implementation for loading images from a folder.

    This reader iterates through image files in a specified directory,
    supporting common image formats (jpg, jpeg, png, bmp, tiff).
    """

    def __init__(self, config: ReaderConfig, supported_extensions: set[str]) -> None:
        self._config = config
        self._supported_extensions = supported_extensions
        self._image_paths: list[Path] = []
        self._current_index: int = 0
        self._last_image: np.ndarray | None = None
        self._last_image_path: Path | None = None
        self._thumbnail_cache: dict[int, str] = {}
        super().__init__()

    @staticmethod
    def _generate_thumbnail(image_path: Path, max_size: int = 150) -> str | None:
        """Generate a base64-encoded thumbnail for an image."""
        try:
            image = cv2.imread(str(image_path))
            if image is None:
                return None

            # Resize while maintaining aspect ratio
            h, w = image.shape[:2]
            scale = min(max_size / w, max_size / h)
            new_w, new_h = int(w * scale), int(h * scale)
            thumbnail = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

            # Encode to base64 with data URI prefix
            _, buffer = cv2.imencode(".jpg", thumbnail, [cv2.IMWRITE_JPEG_QUALITY, 80])
            base64_string = base64.b64encode(buffer).decode("utf-8")
            return f"data:image/jpeg;base64,{base64_string}"
        except Exception as e:
            logger.warning(f"Failed to generate thumbnail for {image_path}: {e}")
            return None

    def _get_image_files(self, folder_path: Path) -> list[Path]:
        """
        Filter and collect supported image files from the given folder.

        Args:
            folder_path: The directory to scan for images.

        Returns:
            A list of Path objects pointing to supported image files.
        """
        return [
            path
            for path in folder_path.iterdir()
            if path.is_file() and path.suffix.lower() in self._supported_extensions
        ]

    @staticmethod
    def _natural_sort_key(path: Path) -> list[str | int]:
        """
        Generate a natural sort key for filenames with numbers.

        Allows sorting like: img_1, img_2, img_10 instead of img_1, img_10, img_2.

        Args:
            path: The file path to generate a sort key for.

        Returns:
            A list of strings and integers for natural sorting.
        """
        return [int(segment) if segment.isdigit() else segment.lower() for segment in re.split(r"(\d+)", path.stem)]

    def connect(self) -> None:
        """Scan the folder and collect all supported image files."""
        folder_path = Path(self._config.images_folder_path)

        if not folder_path.exists() or not folder_path.is_dir():
            raise ValueError(f"Invalid folder path: {self._config.images_folder_path}")

        image_files = self._get_image_files(folder_path)
        if not image_files:
            logger.warning(f"No supported image files found in {folder_path}")

        self._image_paths = sorted(image_files, key=self._natural_sort_key)
        self._current_index = 0

        # Pre-generate thumbnails for first page (optimization)
        for idx, path in enumerate(self._image_paths[:30]):
            thumbnail = self._generate_thumbnail(path)
            if thumbnail:
                self._thumbnail_cache[idx] = thumbnail

    def seek(self, index: int) -> None:
        """
        Set the current position to a specific image index.

        Args:
            index (int): The target frame position to seek to.
        """
        if not self._image_paths:
            raise ValueError("No images loaded. Call connect() first.")

        if not 0 <= index < len(self._image_paths):
            raise IndexError(f"Index {index} out of range [0, {len(self._image_paths)})")

        self._current_index = index
        # clear cache to force reload on next read()
        self._last_image = None
        self._last_image_path = None

    def __len__(self) -> int:
        """Return the total number of images in the folder."""
        return len(self._image_paths)

    def index(self) -> int:
        """Return the current frame position."""
        return self._current_index

    def list_frames(self, offset: int = 0, limit: int = 30) -> FrameListResponse:
        """
        Return a paginated list of frames with thumbnails.

        Args:
            offset: Number of items to skip (0-based index).
            limit: Maximum number of frames to return.

        Returns:
            FrameListResponse with frame metadata including thumbnails and pagination info.
        """
        total = len(self._image_paths)
        end_idx = min(offset + limit, total)

        frames = []
        for idx in range(offset, end_idx):
            image_path = self._image_paths[idx]

            # Check cache first, generate if not cached
            thumbnail: str | None
            if idx in self._thumbnail_cache:
                thumbnail = self._thumbnail_cache[idx]
            else:
                thumbnail = self._generate_thumbnail(image_path)
                if thumbnail is not None:
                    self._thumbnail_cache[idx] = thumbnail

            if thumbnail is None:
                # Skip invalid images
                continue

            frames.append(FrameMetadata(index=idx, thumbnail=thumbnail))

        pagination = Pagination(
            count=len(frames),
            total=total,
            offset=offset,
            limit=limit,
        )

        return FrameListResponse(frames=frames, pagination=pagination)

    def read(self) -> InputData | None:
        """Read the current image."""
        if not self._image_paths:
            return None

        image_path = self._image_paths[self._current_index]

        # cache image to avoid repeated disk reads
        if self._last_image is None or self._last_image_path != image_path:
            image = cv2.imread(str(image_path))
            if image is None:
                logger.error(f"Failed to read image: {image_path}")
                return None
            self._last_image = image
            self._last_image_path = image_path

        time.sleep(0.033)  # a small delay (~30 FPS) to prevent overwhelming consumers

        image_rgb = cv2.cvtColor(self._last_image, cv2.COLOR_BGR2RGB)

        return InputData(
            timestamp=int(time.time() * 1000),
            frame=image_rgb,
            context={"path": str(image_path), "index": self._current_index},
        )

    def close(self) -> None:
        """Clean up resources."""
        self._image_paths = []
        self._current_index = 0
        self._last_image = None
        self._last_image_path = None
