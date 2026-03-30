from unittest.mock import MagicMock

import numpy as np
import pytest

from domain.services.schemas.base import Pagination
from domain.services.schemas.processor import InputData
from runtime.core.components.base import StreamReader
from runtime.core.components.broadcaster import FrameBroadcaster
from runtime.core.components.source import Source


def make_input(name: str) -> InputData:
    return InputData(timestamp=0, frame=np.zeros((2, 2, 3), dtype=np.uint8), context={"name": name})


def happy_path():
    f1, f2, f3 = make_input("frame1"), make_input("frame2"), make_input("frame3")
    return [f1, f2, f3], [f1, f2, f3]


def handles_nones():
    f1, f2, f3 = make_input("frame1"), make_input("frame2"), make_input("frame3")
    return [None, f1, None, f2, f3, None], [f1, f2, f3]


def broadcasts_all():
    f1, f2, f3, f4 = make_input("frame1"), make_input("frame2"), make_input("frame3"), make_input("frame4")
    return [f1, f2, f3, f4], [f1, f2, f3, f4]


test_cases = [
    ("happy_path", happy_path),
    ("handles_nones", handles_nones),
    ("broadcasts_all_frames", broadcasts_all),
]


class TestSource:
    def setup_method(self, method):
        self.mock_stream_reader = MagicMock(spec=StreamReader)
        self.mock_stream_reader.requires_manual_control = False
        self.mock_broadcaster = MagicMock(spec=FrameBroadcaster)
        self.source = Source(self.mock_stream_reader)
        self.source.setup(self.mock_broadcaster)

    @pytest.mark.parametrize("test_id, data_factory", test_cases, ids=[case[0] for case in test_cases])
    def test_source_run_logic(self, test_id, data_factory):
        input_data, expected_broadcasts = data_factory()
        iterator = iter(input_data)

        def read_and_then_stop(*args, **kwargs):
            try:
                return next(iterator)
            except StopIteration:
                self.source.stop()
                return None

        self.mock_stream_reader.read.side_effect = read_and_then_stop
        self.source.run()

        self.mock_stream_reader.connect.assert_called_once()
        assert self.mock_stream_reader.close.call_count == 2

        broadcast_calls = [call.args[0] for call in self.mock_broadcaster.broadcast.call_args_list]
        assert len(broadcast_calls) == len(expected_broadcasts)
        for actual, expected in zip(broadcast_calls, expected_broadcasts):
            assert actual is expected
            assert actual.trace is not None

    def test_seek_delegates_to_reader(self):
        self.source.seek(42)
        self.mock_stream_reader.seek.assert_called_once_with(42)

    def test_index_delegates_to_reader(self):
        self.mock_stream_reader.index.return_value = 10

        result = self.source.index()

        self.mock_stream_reader.index.assert_called_once()
        assert result == 10

    def test_list_frames_delegates_to_reader(self):
        from domain.services.schemas.reader import FrameListResponse, FrameMetadata

        expected_response = FrameListResponse(
            frames=[FrameMetadata(index=1, thumbnail="data:image/jpeg;base64,base64string")],
            pagination=Pagination(count=1, total=1, offset=0, limit=100),
        )
        self.mock_stream_reader.list_frames.return_value = expected_response

        result = self.source.list_frames(offset=0, limit=100)

        self.mock_stream_reader.list_frames.assert_called_once_with(offset=0, limit=100)
        assert result == expected_response

    def test_source_requires_initialization(self):
        """Test that Source raises an error if run without initialization."""
        uninitialized_source = Source(self.mock_stream_reader)
        with pytest.raises(RuntimeError, match="The source should be initialized before being used"):
            uninitialized_source.run()

    def test_source_connect_called_in_run(self):
        """Test that Source calls connect() in the run method."""

        def read_once_and_stop(*args, **kwargs):
            self.source.stop()
            return None

        self.mock_stream_reader.read.side_effect = read_once_and_stop

        self.source.run()
        self.mock_stream_reader.connect.assert_called_once()
