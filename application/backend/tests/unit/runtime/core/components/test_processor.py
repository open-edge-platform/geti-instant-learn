from queue import Empty, Queue
from unittest.mock import MagicMock

import numpy as np
import pytest

from domain.services.schemas.processor import InputData, OutputData
from runtime.core.components.broadcaster import FrameBroadcaster
from runtime.core.components.processor import Processor


def create_input_data(frame_id: int) -> InputData:
    return InputData(
        timestamp=frame_id * 1000,
        frame=np.zeros((480, 640, 3), dtype=np.uint8),
        context={"frame_id": frame_id},
    )


def create_output_data(frame_id: int) -> OutputData:
    return OutputData(
        frame=np.zeros((480, 640, 3), dtype=np.uint8),
        results=[],
    )


runner_test_cases = [
    (
        "processes_and_broadcasts_all_data",
        [create_input_data(1), create_input_data(2)],
        [create_output_data(1), create_output_data(2)],
    ),
    (
        "handles_intermittent_empty_queue",
        [Empty(), create_input_data(1), Empty(), create_input_data(2)],
        [create_output_data(1), create_output_data(2)],
    ),
    ("handles_empty_input", [], []),
]


class TestProcessor:
    def setup_method(self, method):
        self.mock_inbound_broadcaster = MagicMock(spec=FrameBroadcaster)
        self.mock_in_queue = MagicMock(spec=Queue)
        self.mock_inbound_broadcaster.register.return_value = self.mock_in_queue
        self.mock_outbound_broadcaster = MagicMock(spec=FrameBroadcaster)
        self.mock_model_handler = MagicMock()
        self.mock_model_handler.predict.return_value = []  # Return empty results list
        self.mock_label_service = MagicMock()
        self.runner = Processor(self.mock_model_handler, self.mock_label_service)
        self.runner.setup(self.mock_inbound_broadcaster, self.mock_outbound_broadcaster)

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
        self.mock_inbound_broadcaster.register.assert_called_once()

        self.runner.run()

        assert self.mock_outbound_broadcaster.broadcast.call_count == len(expected_broadcasts)

        for i, expected_output in enumerate(expected_broadcasts):
            actual_call = self.mock_outbound_broadcaster.broadcast.call_args_list[i]
            actual_output = actual_call[0][0]

            assert isinstance(actual_output, OutputData)
            assert np.array_equal(actual_output.frame, expected_output.frame)
            assert isinstance(actual_output.results, list)

        self.mock_inbound_broadcaster.unregister.assert_called_once_with(self.mock_in_queue)
