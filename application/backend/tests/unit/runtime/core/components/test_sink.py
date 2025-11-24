from queue import Empty, Queue
from unittest.mock import MagicMock, call

import pytest

from runtime.core.components.base import StreamWriter
from runtime.core.components.broadcaster import FrameBroadcaster
from runtime.core.components.sink import Sink

test_cases = [
    ("writes_all_items", ["data1", "data2", "data3"], ["data1", "data2", "data3"]),
    ("handles_empty_queue", [], []),
    (
        "handles_intermittent_empty_exceptions",
        ["data1", Empty(), "data2", Empty(), Empty(), "data3"],
        ["data1", "data2", "data3"],
    ),
]


class TestSink:
    def setup_method(self, method):
        self.broadcaster = MagicMock(spec=FrameBroadcaster)
        self.out_queue = MagicMock(spec=Queue)
        self.broadcaster.register.return_value = self.out_queue
        self.mock_stream_writer = MagicMock(spec=StreamWriter)
        self.mock_stream_writer.__enter__.return_value = self.mock_stream_writer
        self.sink = Sink(self.mock_stream_writer)
        self.sink.setup(self.broadcaster)

    @pytest.mark.parametrize(
        "test_id, get_side_effects, expected_writes", test_cases, ids=[case[0] for case in test_cases]
    )
    def test_sink_run_logic(self, test_id, get_side_effects, expected_writes):
        iterator = iter(get_side_effects)

        def mock_get(*args, **kwargs):
            try:
                next_item = next(iterator)
                if isinstance(next_item, Exception):
                    raise next_item
                return next_item
            except StopIteration:
                self.sink.stop()
                raise Empty

        self.out_queue.get.side_effect = mock_get

        self.sink.run()

        self.mock_stream_writer.__enter__.assert_called_once()
        self.mock_stream_writer.__exit__.assert_called_once()

        expected_calls = [call(item) for item in expected_writes]
        assert self.mock_stream_writer.write.call_args_list == expected_calls
