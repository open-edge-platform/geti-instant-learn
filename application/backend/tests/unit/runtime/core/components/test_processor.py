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
        "processes_and_broadcasts_all_frames",
        [create_input_data(1), create_input_data(2)],
        2,  # Expect 2 broadcasts: 1 per frame (batched for inference, broadcast individually)
    ),
    (
        "handles_intermittent_empty_queue",
        [Empty(), create_input_data(1), Empty(), create_input_data(2)],
        2,  # Expect 2 broadcasts: 1 for frame 1, 1 for frame 2 (separated by Empty)
    ),
    ("handles_empty_input", [], 0),  # Expect 0 broadcasts for no input
]


class TestProcessor:
    def setup_method(self, method):
        self.mock_inbound_broadcaster = MagicMock(spec=FrameBroadcaster)
        self.mock_in_queue = MagicMock(spec=Queue)
        self.mock_inbound_broadcaster.register.return_value = self.mock_in_queue
        self.mock_outbound_broadcaster = MagicMock(spec=FrameBroadcaster)
        self.mock_model_handler = MagicMock()
        self.mock_model_handler.predict.side_effect = lambda batch: [{}] * len(batch)
        self.mock_label_service = MagicMock()
        self.runner = Processor(self.mock_model_handler, self.mock_label_service)
        self.runner.setup(self.mock_inbound_broadcaster, self.mock_outbound_broadcaster)

    @pytest.mark.parametrize(
        "test_id, queue_effects, expected_broadcast_count",
        runner_test_cases,
        ids=[case[0] for case in runner_test_cases],
    )
    def test_pipeline_runner_logic(self, test_id, queue_effects, expected_broadcast_count):
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

        assert self.mock_outbound_broadcaster.broadcast.call_count == expected_broadcast_count

        for call in self.mock_outbound_broadcaster.broadcast.call_args_list:
            actual_output = call[0][0]
            assert isinstance(actual_output, OutputData)
            assert isinstance(actual_output.frame, np.ndarray)
            assert isinstance(actual_output.results, list)

        self.mock_inbound_broadcaster.unregister.assert_called_once_with(self.mock_in_queue)

    def test_processor_breaks_batch_on_requires_manual_control(self):
        frame_with_manual_control = InputData(
            timestamp=1000,
            frame=np.zeros((480, 640, 3), dtype=np.uint8),
            context={"frame_id": 1, "requires_manual_control": True},
        )
        frame_without_flag = InputData(
            timestamp=2000,
            frame=np.zeros((480, 640, 3), dtype=np.uint8),
            context={"frame_id": 2},
        )

        queue_effects = [frame_with_manual_control, frame_without_flag]
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

        self.runner.run()

        # Should process 2 times: once for frame with manual control, once for frame without
        assert self.mock_outbound_broadcaster.broadcast.call_count == 2

        # First call should be with single frame batch (manual control breaks the batch)
        first_predict_call = self.mock_model_handler.predict.call_args_list[0]
        first_batch = first_predict_call[0][0]
        assert len(first_batch) == 1

    def test_processor_collects_full_batch_without_manual_control(self):
        frames = [
            InputData(
                timestamp=i * 1000,
                frame=np.zeros((480, 640, 3), dtype=np.uint8),
                context={"frame_id": i},
            )
            for i in range(3)
        ]

        queue_effects = frames.copy()
        iterator = iter(queue_effects)

        def mock_get(*args, **kwargs):
            try:
                next_item = next(iterator)
                return next_item
            except StopIteration:
                self.runner.stop()
                raise Empty

        self.mock_in_queue.get.side_effect = mock_get

        self.runner.run()

        # Should call predict once with batch of 3
        assert self.mock_model_handler.predict.call_count == 1
        predict_call = self.mock_model_handler.predict.call_args_list[0]
        batch = predict_call[0][0]
        assert len(batch) == 3

        # Should broadcast 3 individual results
        assert self.mock_outbound_broadcaster.broadcast.call_count == 3
