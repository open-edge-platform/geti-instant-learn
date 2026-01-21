# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from queue import Queue
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from av import VideoFrame

from domain.services.schemas.processor import OutputData
from runtime.webrtc.stream import InferenceVideoStreamTrack


@pytest.fixture
def fxt_stream_queue():
    return Queue()


@pytest.fixture
def fxt_sample_frame():
    return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)


@pytest.fixture
def fxt_output_data(fxt_sample_frame):
    output_data = MagicMock(spec=OutputData)
    output_data.frame = fxt_sample_frame
    output_data.results = []
    return output_data


@pytest.fixture
def fxt_visualization_patches():
    def _passthrough(*, output_data: OutputData, visualization_info=None):  # noqa: ANN001
        return output_data.frame

    with patch("runtime.webrtc.stream.InferenceVisualizer.visualize", side_effect=_passthrough):
        yield


class TestInferenceVideoStreamTrack:
    @pytest.mark.asyncio
    async def test_recv_with_frame_in_queue(self, fxt_stream_queue, fxt_output_data, fxt_visualization_patches):
        fxt_stream_queue.put(fxt_output_data)
        track = InferenceVideoStreamTrack(stream_queue=fxt_stream_queue)

        frame = await track.recv()

        assert isinstance(frame, VideoFrame)
        assert frame.width == 640
        assert frame.height == 480
        assert frame.pts is not None
        assert frame.time_base is not None

    @pytest.mark.asyncio
    async def test_recv_with_empty_queue_no_cache(self, fxt_stream_queue):
        track = InferenceVideoStreamTrack(stream_queue=fxt_stream_queue)

        frame = await track.recv()

        assert isinstance(frame, VideoFrame)
        assert frame.width == 64
        assert frame.height == 64

    @pytest.mark.asyncio
    async def test_recv_with_empty_queue_uses_cache(self, fxt_stream_queue, fxt_output_data, fxt_visualization_patches):
        fxt_stream_queue.put(fxt_output_data)
        track = InferenceVideoStreamTrack(stream_queue=fxt_stream_queue)

        frame1 = await track.recv()
        assert frame1.width == 640
        assert frame1.height == 480

        frame2 = await track.recv()
        assert isinstance(frame2, VideoFrame)
        assert frame2.width == 640
        assert frame2.height == 480

    @pytest.mark.asyncio
    async def test_recv_multiple_frames(self, fxt_stream_queue, fxt_sample_frame, fxt_visualization_patches):
        track = InferenceVideoStreamTrack(stream_queue=fxt_stream_queue)

        for _ in range(3):
            output_data = MagicMock(spec=OutputData)
            output_data.frame = fxt_sample_frame
            output_data.results = []
            fxt_stream_queue.put(output_data)

        frames = [await track.recv() for _ in range(3)]

        assert len(frames) == 3
        for frame in frames:
            assert isinstance(frame, VideoFrame)
            assert frame.width == 640
            assert frame.height == 480

    @pytest.mark.asyncio
    async def test_timestamps_increment(self, fxt_stream_queue, fxt_output_data, fxt_visualization_patches):
        track = InferenceVideoStreamTrack(stream_queue=fxt_stream_queue)

        for _ in range(3):
            fxt_stream_queue.put(fxt_output_data)

        pts_values = [(await track.recv()).pts for _ in range(3)]

        assert pts_values[1] > pts_values[0]
        assert pts_values[2] > pts_values[1]

    @pytest.mark.asyncio
    async def test_recv_calls_visualization_info_provider(
        self, fxt_stream_queue, fxt_output_data, fxt_visualization_patches
    ):
        fxt_stream_queue.put(fxt_output_data)
        provider = MagicMock(return_value=None)

        track = InferenceVideoStreamTrack(
            stream_queue=fxt_stream_queue,
            enable_visualization=True,
            visualization_info_provider=provider,
        )

        await track.recv()

        provider.assert_called_once()
