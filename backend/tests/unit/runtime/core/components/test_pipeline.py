from queue import Empty, Queue
from unittest.mock import MagicMock, call

import pytest

from runtime.core.components.base import Processor
from runtime.core.components.broadcaster import FrameBroadcaster
from runtime.core.components.pipeline import PipelineRunner

runner_test_cases = [
    (
        "processes_and_broadcasts_all_data",
        ["data1", "data2"],
        ["proc_data1", "proc_data2"],
        ["proc_data1", "proc_data2"],
    ),
    (
        "skips_broadcasting_for_none_results",
        ["data1", "data2", "data3"],
        ["proc_data1", None, "proc_data3"],
        ["proc_data1", "proc_data3"],
    ),
    (
        "handles_intermittent_empty_queue",
        [Empty(), "data1", Empty(), "data2"],
        ["proc_data1", "proc_data2"],
        ["proc_data1", "proc_data2"],
    ),
    ("handles_empty_input", [], [], []),
]


class TestPipelineRunner:
    def setup_method(self, method):
        self.mock_in_queue = MagicMock(spec=Queue)
        self.mock_broadcaster = MagicMock(spec=FrameBroadcaster)
        self.mock_processor = MagicMock(spec=Processor)
        self.runner = PipelineRunner(self.mock_in_queue, self.mock_broadcaster, self.mock_processor)

    @pytest.mark.parametrize(
        "test_id, queue_effects, processor_effects, expected_broadcasts",
        runner_test_cases,
        ids=[case[0] for case in runner_test_cases],
    )
    def test_pipeline_runner_logic(self, test_id, queue_effects, processor_effects, expected_broadcasts):
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
        self.mock_processor.process.side_effect = processor_effects

        self.runner.run()

        # Check that the processor was called for each non-Empty item from the queue.
        actual_data_items = [item for item in queue_effects if not isinstance(item, Empty)]
        expected_processor_calls = [call(item) for item in actual_data_items]
        assert self.mock_processor.process.call_args_list == expected_processor_calls

        # Check that the broadcaster was called with the correct processed data.
        expected_broadcast_calls = [call(item) for item in expected_broadcasts]
        assert self.mock_broadcaster.broadcast.call_args_list == expected_broadcast_calls
