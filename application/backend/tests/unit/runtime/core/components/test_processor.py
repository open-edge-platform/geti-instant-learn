#  Copyright (C) 2025 Intel Corporation
#  SPDX-License-Identifier: Apache-2.0

import time
from queue import Queue
from threading import Thread
from unittest.mock import Mock

import numpy as np
import pytest

from domain.dispatcher import ComponentType
from domain.services.schemas.processor import ErrorData, InputData, OutputData
from runtime.core.components.base import ModelHandler
from runtime.core.components.broadcaster import FrameBroadcaster
from runtime.core.components.processor import FrameSkipPolicy, Processor
from runtime.core.components.scene_change import SceneChangeDetector


def make_input_data(requires_manual_control: bool = False, with_trace: bool = False) -> InputData:
    return InputData(
        timestamp=1000,
        frame=np.zeros((64, 64, 3), dtype=np.uint8),
        context={"requires_manual_control": requires_manual_control},
        trace=Mock() if with_trace else None,
    )


@pytest.fixture
def mock_model_handler() -> Mock:
    handler = Mock(spec=ModelHandler)
    handler.initialise = Mock()
    handler.predict = Mock(return_value=[{"masks": np.zeros((1, 64, 64))}])
    return handler


@pytest.fixture
def mock_inbound_broadcaster() -> Mock:
    broadcaster = Mock(spec=FrameBroadcaster)
    broadcaster.register = Mock(return_value=Queue())
    broadcaster.unregister = Mock()
    broadcaster.clear = Mock()
    return broadcaster


@pytest.fixture
def mock_outbound_broadcaster() -> Mock:
    broadcaster = Mock(spec=FrameBroadcaster)
    broadcaster.broadcast = Mock()
    broadcaster.clear = Mock()
    return broadcaster


@pytest.fixture
def processor(mock_model_handler) -> Processor:
    return Processor(
        model_handler=mock_model_handler,
        batch_size=1,
        frame_skip_interval=0,
        frame_skip_amount=0,
    )


@pytest.fixture
def configured_processor(
    processor, mock_inbound_broadcaster, mock_outbound_broadcaster
) -> (tuple)[Processor, Queue[InputData]]:
    queue: Queue[InputData] = Queue()
    mock_inbound_broadcaster.register.return_value = queue
    processor.setup(mock_inbound_broadcaster, mock_outbound_broadcaster)
    return processor, queue


class TestFrameSkipPolicy:
    def test_no_skipping_when_interval_is_zero(self) -> None:
        policy = FrameSkipPolicy(interval=0, skip_amount=0)
        for _ in range(20):
            assert policy.should_skip() is False

    def test_no_skipping_when_skip_amount_is_zero(self) -> None:
        policy = FrameSkipPolicy(interval=3, skip_amount=0)
        for _ in range(20):
            assert policy.should_skip() is False

    def test_skip_pattern_basic(self) -> None:
        # interval=3, skip_amount=1 -> process, process, DROP, process, process, DROP, ...
        policy = FrameSkipPolicy(interval=3, skip_amount=1)
        results = [policy.should_skip() for _ in range(6)]
        assert results == [False, False, True, False, False, True]

    def test_skip_pattern_skip_amount_two(self) -> None:
        # interval=4, skip_amount=2 -> process, process, DROP, DROP, ...
        policy = FrameSkipPolicy(interval=4, skip_amount=2)
        results = [policy.should_skip() for _ in range(8)]
        assert results == [False, False, True, True, False, False, True, True]

    def test_reset_restarts_counter(self) -> None:
        policy = FrameSkipPolicy(interval=3, skip_amount=1)
        policy.should_skip()
        policy.should_skip()
        policy.reset()
        results = [policy.should_skip() for _ in range(3)]
        assert results == [False, False, True]

    @pytest.mark.parametrize(
        "interval, skip_amount",
        [
            (-1, 0),
            (1, 0),  # interval == 1 is invalid
        ],
    )
    def test_invalid_interval_raises(self, interval: int, skip_amount: int) -> None:
        with pytest.raises(ValueError):
            FrameSkipPolicy(interval=interval, skip_amount=skip_amount)

    @pytest.mark.parametrize(
        "interval, skip_amount",
        [
            (3, -1),
            (3, 3),  # skip_amount must be < interval
            (3, 4),
        ],
    )
    def test_invalid_skip_amount_raises(self, interval: int, skip_amount: int) -> None:
        with pytest.raises(ValueError):
            FrameSkipPolicy(interval=interval, skip_amount=skip_amount)

    def test_interval_property(self) -> None:
        policy = FrameSkipPolicy(interval=5, skip_amount=2)
        assert policy.interval == 5

    def test_skip_amount_property(self) -> None:
        policy = FrameSkipPolicy(interval=5, skip_amount=2)
        assert policy.skip_amount == 2

    def test_scene_change_forces_processing_over_skip(self) -> None:
        """When scene detector reports a change, frame is processed even if cyclic says skip."""
        scene_detector = Mock(spec=SceneChangeDetector)
        scene_detector.is_new_scene.return_value = True  # force scene change on every call

        policy = FrameSkipPolicy(interval=3, skip_amount=1, scene_detector=scene_detector)

        # Advance counter to position where cyclic would skip (position 2)
        policy.should_skip(np.zeros((64, 64, 3)))  # pos 0 -> False
        policy.should_skip(np.zeros((64, 64, 3)))  # pos 1 -> False
        result = policy.should_skip(np.zeros((64, 64, 3)))  # pos 2 would be True (skip)

        assert result is False  # scene change overrides the skip
        scene_detector.is_new_scene.assert_called()

    def test_scene_change_does_not_alter_cyclic_counter(self) -> None:
        """Scene detection override should not affect the cyclic counter's own state."""
        scene_detector = Mock(spec=SceneChangeDetector)
        scene_detector.is_new_scene.return_value = True  # always force processing

        policy = FrameSkipPolicy(interval=3, skip_amount=1, scene_detector=scene_detector)

        # Call should_skip multiple times with scene changes forcing processing
        for _ in range(6):
            policy.should_skip(np.zeros((64, 64, 3)))

        # Counter should have advanced normally (6 increments)
        assert policy._counter == 6

        # Next cycle should still follow the pattern: process, process, DROP
        result = policy.should_skip(np.zeros((64, 64, 3)))  # pos 0 -> False
        assert result is False
        result = policy.should_skip(np.zeros((64, 64, 3)))  # pos 1 -> False
        assert result is False
        result = policy.should_skip(np.zeros((64, 64, 3)))  # pos 2 -> True (skip) but overridden
        assert result is False  # still overridden by scene change

    def test_no_scene_change_falls_back_to_cyclic(self) -> None:
        """When scene detector reports no change, cyclic decision applies normally."""
        scene_detector = Mock(spec=SceneChangeDetector)
        scene_detector.is_new_scene.return_value = False  # no scene change

        policy = FrameSkipPolicy(interval=3, skip_amount=1, scene_detector=scene_detector)

        results = [policy.should_skip(np.zeros((64, 64, 3))) for _ in range(6)]
        assert results == [False, False, True, False, False, True]

    def test_scene_detector_not_called_when_frame_is_none(self) -> None:
        """Scene detector should not be called if frame is None."""
        scene_detector = Mock(spec=SceneChangeDetector)
        policy = FrameSkipPolicy(interval=3, skip_amount=1, scene_detector=scene_detector)

        # Call with None frame - should not invoke the detector
        result = policy.should_skip(frame=None)

        assert result is False  # no frame means no scene change detection
        scene_detector.is_new_scene.assert_not_called()

    def test_no_scene_detector_ignores_frame_argument(self) -> None:
        """When no scene detector is configured, the frame argument is ignored."""
        policy = FrameSkipPolicy(interval=3, skip_amount=1, scene_detector=None)

        results = [policy.should_skip(np.zeros((64, 64, 3))) for _ in range(6)]
        assert results == [False, False, True, False, False, True]

    def test_scene_change_in_middle_of_cycle_resets_expectation(self) -> None:
        """Scene change during a skip position allows processing but counter continues."""
        scene_detector = Mock(spec=SceneChangeDetector)

        # First 2 frames: no scene change (process)
        # 3rd frame: scene change (would be skipped, but overridden)
        # 4th frame: no scene change (process, start of new cycle)
        scene_detector.is_new_scene.side_effect = [False, False, True, False]

        policy = FrameSkipPolicy(interval=3, skip_amount=1, scene_detector=scene_detector)

        results = [policy.should_skip(np.zeros((64, 64, 3))) for _ in range(4)]
        assert results == [False, False, False, False]  # all processed due to overrides

        # Verify is_new_scene was called the expected number of times
        assert scene_detector.is_new_scene.call_count == 4


class TestProcessorInit:
    def test_run_raises_if_not_set_up(self, processor: Processor) -> None:
        with pytest.raises(RuntimeError, match="must be set up"):
            processor.run()

    def test_setup_registers_with_inbound_broadcaster(
        self,
        processor: Processor,
        mock_inbound_broadcaster: Mock,
        mock_outbound_broadcaster: Mock,
    ) -> None:
        processor.setup(mock_inbound_broadcaster, mock_outbound_broadcaster)
        mock_inbound_broadcaster.register.assert_called_once_with(Processor.__name__)

    def test_setup_stores_broadcasters(
        self,
        processor: Processor,
        mock_inbound_broadcaster: Mock,
        mock_outbound_broadcaster: Mock,
    ) -> None:
        processor.setup(mock_inbound_broadcaster, mock_outbound_broadcaster)
        assert processor._inbound_broadcaster is mock_inbound_broadcaster
        assert processor._outbound_broadcaster is mock_outbound_broadcaster

    def test_stop_unregisters_from_inbound_broadcaster(
        self,
        processor: Processor,
        mock_inbound_broadcaster: Mock,
        mock_outbound_broadcaster: Mock,
    ) -> None:
        processor.setup(mock_inbound_broadcaster, mock_outbound_broadcaster)
        processor._stop()
        mock_inbound_broadcaster.unregister.assert_called_once_with(Processor.__name__)

    def test_stop_calls_close_on_model_handler(
        self,
        processor: Processor,
        mock_model_handler: Mock,
        mock_inbound_broadcaster: Mock,
        mock_outbound_broadcaster: Mock,
    ) -> None:
        processor.setup(mock_inbound_broadcaster, mock_outbound_broadcaster)
        processor._stop()
        mock_model_handler.close.assert_called_once()


class TestProcessorRun:
    def _run_processor_with_frames(
        self,
        processor: Processor,
        queue: Queue,
        frames: list[InputData],
        stop_after: float = 0.2,
    ) -> None:
        """Put frames into the queue, start the processor, stop it after a delay."""
        for frame in frames:
            queue.put(frame)

        thread = Thread(target=processor.run, daemon=True)
        thread.start()
        time.sleep(stop_after)
        processor.stop()
        thread.join(timeout=2)

    def test_model_handler_initialised_on_run(
        self,
        configured_processor: tuple[Processor, Queue],
        mock_model_handler: Mock,
    ) -> None:
        processor, queue = configured_processor
        self._run_processor_with_frames(processor, queue, [])
        mock_model_handler.initialise.assert_called_once()

    def test_single_frame_is_processed(
        self,
        configured_processor: tuple[Processor, Queue],
        mock_model_handler: Mock,
        mock_outbound_broadcaster: Mock,
    ) -> None:
        processor, queue = configured_processor
        frame = make_input_data()
        self._run_processor_with_frames(processor, queue, [frame])
        mock_model_handler.predict.assert_called_once()
        mock_outbound_broadcaster.broadcast.assert_called_once()

    def test_output_data_contains_frame(
        self,
        configured_processor: tuple[Processor, Queue],
        mock_model_handler: Mock,
        mock_outbound_broadcaster: Mock,
    ) -> None:
        processor, queue = configured_processor
        frame = make_input_data()
        mock_model_handler.predict.return_value = [{"masks": np.zeros((1, 64, 64))}]

        self._run_processor_with_frames(processor, queue, [frame])

        call_args = mock_outbound_broadcaster.broadcast.call_args
        output: OutputData = call_args[0][0]
        assert isinstance(output, OutputData)
        assert output.frame is frame.frame

    def test_empty_result_when_predict_returns_fewer_results(
        self,
        configured_processor: tuple[Processor, Queue],
        mock_model_handler: Mock,
        mock_outbound_broadcaster: Mock,
    ) -> None:
        processor, queue = configured_processor
        # predict returns empty list -> result falls back to EMPTY_RESULT
        mock_model_handler.predict.return_value = []

        frame = make_input_data()
        self._run_processor_with_frames(processor, queue, [frame])

        call_args = mock_outbound_broadcaster.broadcast.call_args
        output: OutputData = call_args[0][0]
        assert output.results == []

    def test_frame_skipping_applied(
        self,
        mock_model_handler: Mock,
        mock_inbound_broadcaster: Mock,
        mock_outbound_broadcaster: Mock,
    ) -> None:
        # interval=3, skip_amount=1: frames 0,1 processed; frame 2 dropped per cycle
        queue: Queue = Queue()
        mock_inbound_broadcaster.register.return_value = queue

        processor = Processor(
            model_handler=mock_model_handler,
            batch_size=1,
            frame_skip_interval=3,
            frame_skip_amount=1,
        )
        processor.setup(mock_inbound_broadcaster, mock_outbound_broadcaster)

        for _ in range(3):
            queue.put(make_input_data())

        self._run_processor_with_frames(processor, queue, [], stop_after=0.3)

        # 2 out of 3 frames should be broadcast (1 skipped)
        assert mock_outbound_broadcaster.broadcast.call_count == 2

    def test_manual_control_frame_bypasses_skip_policy(
        self,
        mock_model_handler: Mock,
        mock_inbound_broadcaster: Mock,
        mock_outbound_broadcaster: Mock,
    ) -> None:
        queue: Queue = Queue()
        mock_inbound_broadcaster.register.return_value = queue

        # skip policy would skip every 3rd frame
        processor = Processor(
            model_handler=mock_model_handler,
            batch_size=1,
            frame_skip_interval=3,
            frame_skip_amount=1,
        )
        processor.setup(mock_inbound_broadcaster, mock_outbound_broadcaster)

        # advance counter so next non-manual frame would be skipped (position 2)
        policy = processor._skip_policy
        policy.should_skip()  # pos 0 -> False
        policy.should_skip()  # pos 1 -> False
        # pos 2 would be True (skip), but manual bypasses

        queue.put(make_input_data(requires_manual_control=True))
        self._run_processor_with_frames(processor, queue, [], stop_after=0.2)

        mock_outbound_broadcaster.broadcast.assert_called_once()

    def test_trace_recorded_for_frame(
        self,
        configured_processor: tuple[Processor, Queue],
        mock_model_handler: Mock,
        mock_outbound_broadcaster: Mock,
    ) -> None:
        processor, queue = configured_processor
        frame = make_input_data(with_trace=True)
        self._run_processor_with_frames(processor, queue, [frame])

        frame.trace.record_start.assert_called_once_with("processor")
        frame.trace.record_end.assert_called_once_with("processor")

    def test_no_trace_calls_when_trace_is_none(
        self,
        configured_processor: tuple[Processor, Queue],
        mock_model_handler: Mock,
        mock_outbound_broadcaster: Mock,
    ) -> None:
        processor, queue = configured_processor
        frame = make_input_data(with_trace=False)
        self._run_processor_with_frames(processor, queue, [frame])
        mock_outbound_broadcaster.broadcast.assert_called_once()

    def test_exception_in_predict_does_not_crash_loop(
        self,
        configured_processor: tuple[Processor, Queue],
        mock_model_handler: Mock,
        mock_outbound_broadcaster: Mock,
    ) -> None:
        processor, queue = configured_processor
        mock_model_handler.predict.side_effect = [RuntimeError("GPU error"), [{"masks": np.zeros((1,))}]]

        # First frame triggers exception, second should still be broadcast
        queue.put(make_input_data())
        queue.put(make_input_data())

        self._run_processor_with_frames(processor, queue, [], stop_after=0.3)

        mock_outbound_broadcaster.broadcast.assert_called_once()

    def test_batch_size_two_collects_two_frames(
        self,
        mock_model_handler: Mock,
        mock_inbound_broadcaster: Mock,
        mock_outbound_broadcaster: Mock,
    ) -> None:
        queue: Queue = Queue()
        mock_inbound_broadcaster.register.return_value = queue
        mock_model_handler.predict.return_value = [
            {"masks": np.zeros((1,))},
            {"masks": np.zeros((1,))},
        ]

        processor = Processor(
            model_handler=mock_model_handler,
            batch_size=2,
            frame_skip_interval=0,
            frame_skip_amount=0,
        )
        processor.setup(mock_inbound_broadcaster, mock_outbound_broadcaster)

        queue.put(make_input_data())
        queue.put(make_input_data())

        self._run_processor_with_frames(processor, queue, [], stop_after=0.3)

        # predict called once with batch of 2; broadcast called twice (once per result)
        mock_model_handler.predict.assert_called_once()
        batch_arg: list[InputData] = mock_model_handler.predict.call_args[0][0]
        assert len(batch_arg) == 2
        assert mock_outbound_broadcaster.broadcast.call_count == 2

    def test_partial_batch_processed_when_queue_becomes_empty(
        self,
        mock_model_handler: Mock,
        mock_inbound_broadcaster: Mock,
        mock_outbound_broadcaster: Mock,
    ) -> None:
        queue: Queue = Queue()
        mock_inbound_broadcaster.register.return_value = queue
        mock_model_handler.predict.return_value = [{"masks": np.zeros((1,))}]

        processor = Processor(
            model_handler=mock_model_handler,
            batch_size=4,
            frame_skip_interval=0,
            frame_skip_amount=0,
        )
        processor.setup(mock_inbound_broadcaster, mock_outbound_broadcaster)

        # Only 1 frame in queue; batch_size=4 but partial batch should still be processed
        queue.put(make_input_data())
        self._run_processor_with_frames(processor, queue, [], stop_after=0.3)

        mock_model_handler.predict.assert_called()
        mock_outbound_broadcaster.broadcast.assert_called_once()

    def test_manual_frame_breaks_batch_collection(
        self,
        mock_model_handler: Mock,
        mock_inbound_broadcaster: Mock,
        mock_outbound_broadcaster: Mock,
    ) -> None:
        queue: Queue = Queue()
        mock_inbound_broadcaster.register.return_value = queue
        mock_model_handler.predict.return_value = [{"masks": np.zeros((1,))}]

        processor = Processor(
            model_handler=mock_model_handler,
            batch_size=4,
            frame_skip_interval=0,
            frame_skip_amount=0,
        )
        processor.setup(mock_inbound_broadcaster, mock_outbound_broadcaster)

        # Manual frame should break batch collection immediately
        queue.put(make_input_data(requires_manual_control=True))
        self._run_processor_with_frames(processor, queue, [], stop_after=0.3)

        call_args = mock_model_handler.predict.call_args[0][0]
        assert len(call_args) == 1

    def test_stop_event_exits_run_loop(
        self,
        configured_processor: tuple[Processor, Queue],
        mock_model_handler: Mock,
    ) -> None:
        processor, queue = configured_processor

        thread = Thread(target=processor.run, daemon=True)
        thread.start()
        time.sleep(0.05)
        processor.stop()
        thread.join(timeout=2)

        assert not thread.is_alive()

    def test_stop_calls_close_on_model_handler_after_run(
        self,
        configured_processor: tuple[Processor, Queue],
        mock_model_handler: Mock,
    ) -> None:
        processor, queue = configured_processor
        self._run_processor_with_frames(processor, queue, [])
        mock_model_handler.close.assert_called_once()

    def test_inbound_error_propagates_to_outbound(
        self,
        configured_processor: tuple[Processor, Queue],
        mock_outbound_broadcaster: Mock,
    ) -> None:
        processor, queue = configured_processor
        error = ErrorData(message="Source connection failed", component=ComponentType.SOURCE)
        queue.put(error)

        self._run_processor_with_frames(processor, queue, [], stop_after=0.1)

        broadcast_args = [call.args[0] for call in mock_outbound_broadcaster.broadcast.call_args_list]
        assert any(isinstance(a, ErrorData) and a.message == "Source connection failed" for a in broadcast_args)

    def test_error_state_prevents_frame_processing(
        self,
        configured_processor: tuple[Processor, Queue],
        mock_model_handler: Mock,
        mock_outbound_broadcaster: Mock,
    ) -> None:
        processor, queue = configured_processor
        queue.put(ErrorData(message="Source error", component=ComponentType.SOURCE))

        self._run_processor_with_frames(processor, queue, [], stop_after=0.2)

        # ErrorData is forwarded downstream, not passed to predict
        mock_model_handler.predict.assert_not_called()
        broadcast_args = [call.args[0] for call in mock_outbound_broadcaster.broadcast.call_args_list]
        assert any(isinstance(a, ErrorData) for a in broadcast_args)
