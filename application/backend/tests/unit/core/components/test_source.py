import time
from unittest.mock import MagicMock

import pytest

from core.components.base import StreamReader
from core.components.broadcaster import FrameBroadcaster
from core.components.schemas.processor import InputData
from core.components.schemas.reader import FrameListResponse, FrameMetadata
from core.components.source import Source

test_cases = [
    ("happy_path", ["frame1", "frame2", "frame3"], ["frame1", "frame2", "frame3"]),
    ("handles_nones", [None, "frame1", None, "frame2", "frame3", None], ["frame1", "frame2", "frame3"]),
    ("broadcasts_all_frames", ["frame1", "frame2", "frame3", "frame4"], ["frame1", "frame2", "frame3", "frame4"]),
]


class TestSource:
    def setup_method(self, method):
        self.mock_stream_reader = MagicMock(spec=StreamReader)
        self.mock_stream_reader.__enter__.return_value = self.mock_stream_reader
        self.mock_broadcaster = MagicMock(spec=FrameBroadcaster)
        self.source = Source(self.mock_stream_reader, self.mock_broadcaster)

    @pytest.mark.parametrize(
        "test_id, input_data, expected_broadcasts", test_cases, ids=[case[0] for case in test_cases]
    )
    def test_source_run_logic(self, test_id, input_data, expected_broadcasts):
        iterator = iter(input_data)

        def read_and_then_stop(*args, **kwargs):
            try:
                return next(iterator)
            except StopIteration:
                self.source.stop()
                return None

        self.mock_stream_reader.read.side_effect = read_and_then_stop
        self.source.run()

        self.mock_stream_reader.__enter__.assert_called_once()
        self.mock_stream_reader.connect.assert_called_once()
        self.mock_stream_reader.__exit__.assert_called_once()

        broadcast_calls = [call.args[0] for call in self.mock_broadcaster.broadcast.call_args_list]
        assert broadcast_calls == expected_broadcasts

    def test_pause_stops_polling_reader(self):
        """Test that pausing stops the reader from being polled."""
        frames_read = []

        def read_and_track(*args, **kwargs):
            frame = f"frame{len(frames_read) + 1}"
            frames_read.append(frame)
            if len(frames_read) == 2:
                self.source.pause()
                # Stop after pausing to break the loop
                import threading

                threading.Timer(0.05, self.source.stop).start()
            time.sleep(0.01)
            return frame

        self.mock_stream_reader.read.side_effect = read_and_track
        self.source.run()

        # Should only read 2 frames before pausing, then stop
        assert len(frames_read) == 2

    def test_resume_continues_polling_reader(self):
        """Test that resuming continues polling the reader."""
        frames_read = []
        paused = False

        def read_and_control(*args, **kwargs):
            nonlocal paused
            frame = f"frame{len(frames_read) + 1}"
            frames_read.append(frame)

            if len(frames_read) == 2 and not paused:
                self.source.pause()
                paused = True
                # Simulate resume after a delay
                import threading

                threading.Timer(0.05, self.source.resume).start()
            elif len(frames_read) == 5:
                self.source.stop()

            time.sleep(0.01)
            return frame

        self.mock_stream_reader.read.side_effect = read_and_control
        self.source.run()

        # Should read 2 frames, pause, resume, then read 3 more
        assert len(frames_read) == 5

    def test_stop_while_paused(self):
        """Test that source can be stopped while paused."""
        frames_read = []

        def read_and_pause(*args, **kwargs):
            frame = f"frame{len(frames_read) + 1}"
            frames_read.append(frame)
            if len(frames_read) == 1:
                self.source.pause()
                # Stop after a delay
                import threading

                threading.Timer(0.05, self.source.stop).start()
            time.sleep(0.01)
            return frame

        self.mock_stream_reader.read.side_effect = read_and_pause
        self.source.run()

        # Should only read 1 frame, then pause and stop
        assert len(frames_read) == 1

    def test_seek_delegates_to_reader(self):
        """Test that seek() calls the reader's seek method."""
        expected_frame = InputData(frame=MagicMock(), timestamp=1, context={})
        self.mock_stream_reader.seek.return_value = expected_frame

        result = self.source.seek(42)

        self.mock_stream_reader.seek.assert_called_once_with(42)
        assert result == expected_frame

    def test_index_delegates_to_reader(self):
        """Test that index() calls the reader's index method."""
        self.mock_stream_reader.index.return_value = 10

        result = self.source.index()

        self.mock_stream_reader.index.assert_called_once()
        assert result == 10

    def test_list_frames_delegates_to_reader(self):
        """Test that list_frames() calls the reader's list_frames method."""
        expected_response = FrameListResponse(
            frames=[FrameMetadata(index=1, thumbnail="base64string", path="/path/to/frame1")],
            total=1,
            page=1,
            page_size=100,
        )
        self.mock_stream_reader.list_frames.return_value = expected_response

        result = self.source.list_frames(page=1, page_size=100)

        self.mock_stream_reader.list_frames.assert_called_once_with(1, 100)
        assert result == expected_response

    def test_pause_resume_state(self):
        """Test that pause and resume correctly set the pause event."""
        assert not self.source.is_paused()

        self.source.pause()
        assert self.source.is_paused()

        self.source.resume()
        assert not self.source.is_paused()
