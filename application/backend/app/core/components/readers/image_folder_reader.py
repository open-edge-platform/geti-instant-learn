#  Copyright (C) 2025 Intel Corporation
#  SPDX-License-Identifier: Apache-2.0

import re
import time
from abc import ABC
from pathlib import Path

import cv2

from core.components.base import StreamReader
from core.components.schemas.processor import InputData
from core.components.schemas.reader import ReaderConfig


class ImageFolderReader(StreamReader, ABC):
    """
    A reader implementation for loading images from a folder.

    This reader iterates through image files in a specified directory,
    supporting common image formats (jpg, jpeg, png, bmp, tiff).
    """

    SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}

    def __init__(self, config: ReaderConfig) -> None:
        self._config = config
        self._image_paths: list[Path] = []
        self._current_index = 0
        super().__init__()

    def connect(self) -> None:
        """Scan the folder and collect all supported image files."""
        folder_path = Path(self._config.images_folder_path)

        if not folder_path.exists() or not folder_path.is_dir():
            raise ValueError(f"Invalid folder path: {self._config.images_folder_path}")

        self._image_paths = sorted(
            [p for p in folder_path.iterdir() if p.is_file() and p.suffix.lower() in self.SUPPORTED_EXTENSIONS],
            key=lambda p: [int(s) if s.isdigit() else s.lower() for s in re.split(r"(\d+)", p.stem)],
        )
        self._current_index = 0

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

    def input_data(self) -> int:
        """Return the total number of images in the folder."""
        return len(self._image_paths)

    def index(self) -> int:
        """Return the current frame position."""
        return self._current_index

    def list_frames(self, page: int = 1, page_size: int = 30) -> dict:
        """
        Return a paginated list of image paths.

        Args:
            page (int): The page number to retrieve (1-based).
            page_size (int): The number of frames per page.

        Returns:
            dict: A dictionary with the following structure:
                {
                    "frames": list,  # List of frame metadata or identifiers
                    "page": int,     # Current page number
                    "page_size": int,# Number of frames per page
                    "total": int     # Total number of frames available
                }
        """
        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size

        return {
            "total": len(self._image_paths),
            "page": page,
            "page_size": page_size,
            "frames": [str(p) for p in self._image_paths[start_idx:end_idx]],
        }

    def read(self) -> InputData | None:
        """Read the current image and advance to the next."""
        while self._image_paths and self._current_index < len(self._image_paths):
            image_path = self._image_paths[self._current_index]
            image = cv2.imread(str(image_path))
            if image is not None:
                current_idx = self._current_index
                self._current_index += 1
                return InputData(
                    timestamp=int(time.time() * 1000),  # Current time in milliseconds
                    frame=image,
                    context={"path": str(image_path), "index": current_idx},
                )
            self._current_index += 1
        return None

    def close(self) -> None:
        """Clean up resources."""
        self._image_paths = []
        self._current_index = 0
