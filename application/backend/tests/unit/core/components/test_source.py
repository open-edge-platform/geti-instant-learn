from queue import Queue
from unittest.mock import MagicMock

import pytest

from core.components.base import StreamReader
from core.components.source import Source

test_cases = [
    ("happy_path", ["frame1", "frame2", "frame3"], 3, ["frame1", "frame2", "frame3"]),
    ("handles_nones", [None, "frame1", None, "frame2", "frame3", None], 3, ["frame1", "frame2", "frame3"]),
    (
        "drops_oldest_on_queue_overflow",
        ["frame1", "frame2", "frame3", "frame4"],
        3,
        ["frame2", "frame3", "frame4"],  # frame1 is dropped
    ),
]


class TestSource:
    def setup_method(self, method):
        self.in_queue = Queue(3)
        self.mock_stream_reader = MagicMock(spec=StreamReader)
        self.mock_stream_reader.__enter__.return_value = self.mock_stream_reader
        self.source = Source(self.in_queue, self.mock_stream_reader)

    @pytest.mark.parametrize(
        "test_id, input_data, expected_qsize, expected_output", test_cases, ids=[case[0] for case in test_cases]
    )
    def test_source_run_logic(self, test_id, input_data, expected_qsize, expected_output):
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

        assert self.in_queue.qsize() == expected_qsize

        results = []
        while not self.in_queue.empty():
            results.append(self.in_queue.get())
        assert results == expected_output
