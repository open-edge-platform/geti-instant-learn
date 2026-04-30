# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import cv2

from domain.services.schemas.reader import MaxResolution


MAX_RESOLUTION_BOUNDS: dict[MaxResolution, tuple[int, int]] = {
    MaxResolution.FULLHD: (1920, 1080),
}


def resize_frame_to_max_resolution(frame: cv2.typing.MatLike, max_resolution: MaxResolution | None) -> cv2.typing.MatLike:
    """Downscale oversized frames while preserving aspect ratio."""
    if max_resolution is None:
        return frame

    height, width = frame.shape[:2]
    max_width, max_height = MAX_RESOLUTION_BOUNDS[max_resolution]
    if height > width and max_width > max_height:
        max_width, max_height = max_height, max_width

    scale = min(max_width / width, max_height / height)
    if scale >= 1.0:
        return frame

    new_width = max(1, int(width * scale))
    new_height = max(1, int(height * scale))
    return cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)