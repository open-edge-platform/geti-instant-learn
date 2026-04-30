# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import patch

import numpy as np

from domain.services.schemas.reader import MaxResolution
from runtime.services.frame_resolution import resize_frame_to_max_resolution


class TestResizeFrameToMaxResolution:
    def test_downscales_oversized_frame_for_fullhd_cap(self) -> None:
        frame = np.zeros((2160, 3840, 3), dtype=np.uint8)
        resized = np.zeros((1080, 1920, 3), dtype=np.uint8)

        with patch("runtime.services.frame_resolution.cv2.resize", return_value=resized) as mock_resize:
            result = resize_frame_to_max_resolution(frame, MaxResolution.FULLHD)

        assert result is resized
        mock_resize.assert_called_once_with(frame, (1920, 1080), interpolation=3)

    def test_downscales_oversized_portrait_frame_for_fullhd_cap(self) -> None:
        frame = np.zeros((3840, 2160, 3), dtype=np.uint8)
        resized = np.zeros((1920, 1080, 3), dtype=np.uint8)

        with patch("runtime.services.frame_resolution.cv2.resize", return_value=resized) as mock_resize:
            result = resize_frame_to_max_resolution(frame, MaxResolution.FULLHD)

        assert result is resized
        mock_resize.assert_called_once_with(frame, (1080, 1920), interpolation=3)

    def test_preserves_frame_when_within_cap(self) -> None:
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)

        with patch("runtime.services.frame_resolution.cv2.resize") as mock_resize:
            result = resize_frame_to_max_resolution(frame, MaxResolution.FULLHD)

        assert result is frame
        mock_resize.assert_not_called()

    def test_preserves_frame_when_cap_is_disabled(self) -> None:
        frame = np.zeros((2160, 3840, 3), dtype=np.uint8)

        with patch("runtime.services.frame_resolution.cv2.resize") as mock_resize:
            result = resize_frame_to_max_resolution(frame, None)

        assert result is frame
        mock_resize.assert_not_called()