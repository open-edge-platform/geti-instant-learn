from queue import Empty, Queue
from unittest.mock import MagicMock, call

import pytest

from runtime.core.components.broadcaster import FrameBroadcaster
from runtime.core.components.processor import Processor

runner_test_cases = [
    (
        "processes_and_broadcasts_all_data",
        ["data1", "data2"],
        ["data1", "data2"],
    ),
    (
        "skips_broadcasting_for_none_results",
        ["data1", None, "data3"],
        ["data1", "data3"],
    ),
    (
        "handles_intermittent_empty_queue",
        [Empty(), "data1", Empty(), "data2"],
        ["data1", "data2"],
    ),
    ("handles_empty_input", [], []),
]


class TestProcessor:
    def setup_method(self, method):
        self.mock_inbound_broadcaster = MagicMock(spec=FrameBroadcaster)
        self.mock_in_queue = MagicMock(spec=Queue)
        self.mock_inbound_broadcaster.register.return_value = self.mock_in_queue
        self.mock_outbound_broadcaster = MagicMock(spec=FrameBroadcaster)
        self.runner = Processor(self.mock_inbound_broadcaster, self.mock_outbound_broadcaster, None)

    @pytest.mark.parametrize(
        "test_id, queue_effects, expected_broadcasts",
        runner_test_cases,
        ids=[case[0] for case in runner_test_cases],
    )
    def test_pipeline_runner_logic(self, test_id, queue_effects, expected_broadcasts):
        iterator = iter(queue_effects)

        def mock_get(*args, **kwargs):
            try:
                next_item = next(iterator)
                if isinstance(next_item, Exception):
                    raise next_item
                return next_item
            except StopIteration:
                self.runner.stop()
                raise Empty

        self.mock_in_queue.get.side_effect = mock_get

        # Verify that register was called during __init__
        self.mock_inbound_broadcaster.register.assert_called_once()

        self.runner.run()

        # Check that the broadcaster was called with the correct processed data.
        expected_broadcast_calls = [call(item) for item in expected_broadcasts]
        assert self.mock_outbound_broadcaster.broadcast.call_args_list == expected_broadcast_calls

        # Verify that unregister was called during stop
        self.mock_inbound_broadcaster.unregister.assert_called_once_with(self.mock_in_queue)
