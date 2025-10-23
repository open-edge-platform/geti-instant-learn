from unittest.mock import MagicMock

import pytest

from core.components.base import StreamReader
from core.components.broadcaster import FrameBroadcaster
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
