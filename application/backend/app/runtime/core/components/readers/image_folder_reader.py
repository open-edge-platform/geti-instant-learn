#  Copyright (C) 2025 Intel Corporation
#  SPDX-License-Identifier: Apache-2.0

import base64
import logging
import re
import time
from abc import ABC
from pathlib import Path

import cv2

from runtime.core.components.base import StreamReader
from runtime.core.components.schemas.processor import InputData
from runtime.core.components.schemas.reader import FrameListResponse, FrameMetadata, ReaderConfig
from settings import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class ImageFolderReader(StreamReader, ABC):
    """
    A reader implementation for loading images from a folder.

    This reader iterates through image files in a specified directory,
    supporting common image formats (jpg, jpeg, png, bmp, tiff).
    """

    def __init__(self, config: ReaderConfig) -> None:
        self._config = config
        self._image_paths: list[Path] = []
        self._current_index: int = 0
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

            # Encode to base64
            _, buffer = cv2.imencode(".jpg", thumbnail, [cv2.IMWRITE_JPEG_QUALITY, 80])
            return base64.b64encode(buffer).decode("utf-8")
        except Exception as e:
            logger.warning(f"Failed to generate thumbnail for {image_path}: {e}")
            return None

    @staticmethod
    def _get_image_files(folder_path: Path) -> list[Path]:
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
            if path.is_file() and path.suffix.lower() in settings.supported_extension
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

    def __len__(self) -> int:
        """Return the total number of images in the folder."""
        return len(self._image_paths)

    def index(self) -> int:
        """Return the current frame position."""
        return self._current_index

    def list_frames(self, page: int = 1, page_size: int = 30) -> FrameListResponse:
        """
        Return a paginated list of frames with thumbnails.

        Args:
            page: The page number (1-based).
            page_size: Number of frames per page.

        Returns:
            FrameListResponse with frame metadata including thumbnails.
        """
        start_idx = (page - 1) * page_size
        end_idx = min(start_idx + page_size, len(self._image_paths))

        frames = []
        for idx in range(start_idx, end_idx):
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
                # Skip invalid images or provide placeholder
                continue

            frames.append(
                FrameMetadata(
                    index=idx,
                    thumbnail=thumbnail,
                    path=str(image_path),
                )
            )

        return FrameListResponse(total=len(self._image_paths), page=page, page_size=page_size, frames=frames)

    def read(self) -> InputData | None:
        """Read the current image and advance to the next."""
        if not self._image_paths or self._current_index >= len(self._image_paths):
            return None

        image_path = self._image_paths[self._current_index]
        image = cv2.imread(str(image_path))

        current_idx = self._current_index
        self._current_index += 1

        if image is None:
            # Log the error but maintain index synchronization
            logger.warning(f"Failed to load image: {image_path}")
            return None

        return InputData(
            timestamp=int(time.time() * 1000),
            frame=image,
            context={"path": str(image_path), "index": current_idx},
        )

    def close(self) -> None:
        """Clean up resources."""
        self._image_paths = []
        self._current_index = 0
