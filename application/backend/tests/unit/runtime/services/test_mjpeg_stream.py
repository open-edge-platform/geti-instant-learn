# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import asyncio
from unittest.mock import MagicMock

import numpy as np
import pytest

from domain.services.schemas.processor import OutputData
from runtime.core.components.broadcaster import FrameSlot
from runtime.services.mjpeg_stream import BOUNDARY, MjpegStreamService, _encode_jpeg


class TestEncodeJpeg:
    def test_encode_jpeg_returns_bytes(self):
        bgr = np.zeros((10, 10, 3), dtype=np.uint8)
        result = _encode_jpeg(bgr, quality=80)

        assert isinstance(result, bytes)
        assert len(result) > 0

    def test_encode_jpeg_starts_with_jpeg_magic(self):
        bgr = np.zeros((10, 10, 3), dtype=np.uint8)
        result = _encode_jpeg(bgr, quality=80)

        assert result[:2] == b"\xff\xd8"

    def test_encode_jpeg_quality_affects_size(self):
        bgr = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        low = _encode_jpeg(bgr, quality=10)
        high = _encode_jpeg(bgr, quality=95)

        assert len(low) < len(high)


class TestMjpegStreamService:
    @pytest.fixture
    def service(self) -> MjpegStreamService:
        return MjpegStreamService(quality=80, max_fps=30)

    @pytest.fixture
    def output_slot(self) -> FrameSlot[OutputData]:
        return FrameSlot[OutputData]()

    @pytest.fixture
    def mock_visualizer(self) -> MagicMock:
        visualizer = MagicMock()
        visualizer.visualize.return_value = np.zeros((10, 10, 3), dtype=np.uint8)
        return visualizer

    @pytest.mark.asyncio
    async def test_stream_yields_multipart_frame(self, service, output_slot, mock_visualizer):
        frame = np.zeros((10, 10, 3), dtype=np.uint8)
        output_data = OutputData(frame=frame, results=[])
        output_slot.update(output_data)

        gen = service.stream(output_slot, mock_visualizer, lambda: None)
        chunk = await gen.__anext__()

        assert chunk.startswith(f"--{BOUNDARY}\r\n".encode())
        assert b"Content-Type: image/jpeg\r\n" in chunk
        assert b"Content-Length: " in chunk

    @pytest.mark.asyncio
    async def test_stream_calls_visualizer(self, service, output_slot, mock_visualizer):
        frame = np.zeros((10, 10, 3), dtype=np.uint8)
        output_data = OutputData(frame=frame, results=[])
        output_slot.update(output_data)

        gen = service.stream(output_slot, mock_visualizer, lambda: None)
        await gen.__anext__()

        mock_visualizer.visualize.assert_called_once_with(output_data=output_data, visualization_info=None)

    @pytest.mark.asyncio
    async def test_stream_passes_vis_info(self, service, output_slot, mock_visualizer):
        frame = np.zeros((10, 10, 3), dtype=np.uint8)
        output_data = OutputData(frame=frame, results=[])
        output_slot.update(output_data)

        vis_info = MagicMock()
        gen = service.stream(output_slot, mock_visualizer, lambda: vis_info)
        await gen.__anext__()

        mock_visualizer.visualize.assert_called_once_with(output_data=output_data, visualization_info=vis_info)

    @pytest.mark.asyncio
    async def test_stream_skips_duplicate_frame(self, service, output_slot, mock_visualizer):
        frame = np.zeros((10, 10, 3), dtype=np.uint8)
        output_data = OutputData(frame=frame, results=[])
        output_slot.update(output_data)

        gen = service.stream(output_slot, mock_visualizer, lambda: None)
        await gen.__anext__()

        # Same object in slot — generator should not yield again within timeout
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(gen.__anext__(), timeout=0.05)

    @pytest.mark.asyncio
    async def test_stream_yields_again_for_new_frame(self, output_slot, mock_visualizer):
        # Use high max_fps so throttle doesn't interfere
        service = MjpegStreamService(quality=80, max_fps=1000)

        frame1 = np.zeros((10, 10, 3), dtype=np.uint8)
        output1 = OutputData(frame=frame1, results=[])
        output_slot.update(output1)

        gen = service.stream(output_slot, mock_visualizer, lambda: None)
        await gen.__anext__()

        frame2 = np.ones((10, 10, 3), dtype=np.uint8)
        output2 = OutputData(frame=frame2, results=[])
        output_slot.update(output2)

        chunk = await asyncio.wait_for(gen.__anext__(), timeout=1.0)
        assert chunk.startswith(f"--{BOUNDARY}\r\n".encode())

    @pytest.mark.asyncio
    async def test_stream_throttles_by_fps(self, output_slot, mock_visualizer):
        service = MjpegStreamService(quality=80, max_fps=10)  # 100ms interval

        frame = np.zeros((10, 10, 3), dtype=np.uint8)
        output1 = OutputData(frame=frame, results=[])
        output_slot.update(output1)

        gen = service.stream(output_slot, mock_visualizer, lambda: None)
        await gen.__anext__()

        # Push new frame immediately — should be throttled
        output2 = OutputData(frame=frame, results=[])
        output_slot.update(output2)

        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(gen.__anext__(), timeout=0.05)

    @pytest.mark.asyncio
    async def test_stream_waits_for_first_frame(self, service, output_slot, mock_visualizer):
        gen = service.stream(output_slot, mock_visualizer, lambda: None)

        # No frame in slot — should not yield
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(gen.__anext__(), timeout=0.05)
