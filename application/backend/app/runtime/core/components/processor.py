#  Copyright (C) 2025 Intel Corporation
#  SPDX-License-Identifier: Apache-2.0

import logging
from queue import Empty, Queue

import numpy as np

from domain.services.schemas.processor import InputData, OutputData
from runtime.core.components.base import ModelHandler, PipelineComponent
from runtime.core.components.broadcaster import FrameBroadcaster

logger = logging.getLogger(__name__)

EMPTY_RESULT: dict[str, np.ndarray] = {}


class FrameSkipPolicy:
    """
    Decides whether to skip (drop) a frame based on a cyclic counter.

    Skip pattern with interval N (N >= 1):
        Process frames 1..N-1, drop frame N, repeat.
        Example (N=3): process, process, DROP, process, process, DROP, ...

    If interval is 0, no frames are ever skipped.

    Args:
        interval: Total cycle length (process and skip). 0 disables skipping.
        skip_amount: Number of consecutive frames to skip per cycle. Must be < interval.

    Raises:
        ValueError: If frame_skip_interval is negative.
    """

    def __init__(self, interval: int = 3, skip_amount: int = 1) -> None:
        if interval < 0 or interval == 1:
            raise ValueError(f"frame_skip_interval must be > 1 or 0 for no skipping, got {interval}")
        if interval > 0 and (skip_amount < 0 or skip_amount >= interval):
            raise ValueError(f"skip_amount must be >= 0 and < interval, got {skip_amount} and {interval}")
        self._interval = interval
        self._skip_amount = skip_amount
        self._counter = 0

    @property
    def interval(self) -> int:
        return self._interval

    @property
    def skip_amount(self) -> int:
        return self._skip_amount

    def should_skip(self) -> bool:
        """Return True if the current frame should be dropped. Advances the internal counter on every call."""
        if self._interval == 0 or self._skip_amount == 0:
            return False

        position = self._counter % self._interval
        self._counter += 1

        # process the first (interval - skip_count) frames, skip the rest
        process_count = self._interval - self._skip_amount
        return position >= process_count

    def reset(self) -> None:
        """Reset the internal counter."""
        self._counter = 0


class Processor(PipelineComponent):
    """
    A job component responsible for retrieving raw frames from the inbound broadcaster,
    sending them to a processor for inference, and broadcasting the processed results to subscribed consumers.

    Supports frame skipping to align model throughput with source frame rate.
    """

    def __init__(
        self, model_handler: ModelHandler, batch_size: int = 1, frame_skip_interval: int = 3, frame_skip_amount: int = 1
    ) -> None:
        super().__init__()
        self._model_handler = model_handler
        self._batch_size = batch_size
        self._skip_policy = FrameSkipPolicy(interval=frame_skip_interval, skip_amount=frame_skip_amount)
        self._initialized = False

    def setup(
        self, inbound_broadcaster: FrameBroadcaster[InputData], outbound_broadcaster: FrameBroadcaster[OutputData]
    ) -> None:
        self._inbound_broadcaster = inbound_broadcaster
        self._outbound_broadcaster = outbound_broadcaster
        self._in_queue: Queue[InputData] = inbound_broadcaster.register(self.__class__.__name__)
        self._initialized = True

    def run(self) -> None:
        if not self._initialized:
            raise RuntimeError("Processor must be set up before running")
        logger.debug("Starting a pipeline runner loop")

        model_initialized = False

        while not self._stop_event.is_set():
            if self._handle_upstream_error():
                continue

            if not model_initialized:
                if not self._initialize_model():
                    continue
                model_initialized = True

            try:
                batch_data = self._collect_batch_data()
                if not batch_data:
                    continue

                self._process_batch(batch_data)

            except Exception as e:
                logger.exception("Error in pipeline runner loop: %s", e)
                continue

        logger.debug("Stopping the pipeline runner loop")

    def _handle_upstream_error(self) -> bool:
        """Check for errors from upstream and propagate to downstream.

        Returns:
            True if error was found and handled, False otherwise.
        """
        inbound_error = self._inbound_broadcaster.slot.error
        if inbound_error:
            # Silently propagate - error already logged by Source
            self._outbound_broadcaster.slot.set_error(inbound_error)
            self._stop_event.wait(timeout=0.5)
            return True
        return False

    def _initialize_model(self) -> bool:
        """Initialize the model handler.

        Returns:
            True if initialization succeeded, False otherwise.
        """
        try:
            self._model_handler.initialise()
            logger.info(
                "Pipeline model handler initialized, batch size: %d, frame skip interval: %d, skip amount: %d",
                self._batch_size,
                self._skip_policy.interval,
                self._skip_policy.skip_amount,
            )
            return True
        except Exception as e:
            error_msg = f"Failed to initialize model: {e}"
            logger.exception("Model initialization failed")
            self._outbound_broadcaster.slot.set_error(error_msg)
            self._stop_event.wait(timeout=0.5)
            return False

    def _collect_batch_data(self) -> list[InputData]:
        """Collect a batch of input data from the queue.

        Returns:
            List of InputData items, empty if no data available or error occurred.
        """
        batch_data: list[InputData] = []

        while len(batch_data) < self._batch_size and not self._stop_event.is_set():
            # Check for errors while collecting batch data
            if self._inbound_broadcaster.slot.error:
                break

            try:
                input_data: InputData = self._in_queue.get(timeout=0.1)
                if input_data.trace:
                    input_data.trace.record_start("processor")
            except Empty:
                if batch_data:  # if we have partial batch data, process what we have
                    break
                continue

            is_manual = input_data.context.get("requires_manual_control", False)

            if not is_manual and self._skip_policy.should_skip():
                logger.debug("Frame skipped (timestamp=%s)", input_data.timestamp)
                continue

            batch_data.append(input_data)

            if is_manual:
                break

        return batch_data if not self._stop_event.is_set() else []

    def _process_batch(self, batch_data: list[InputData]) -> None:
        """Process a batch of input data and broadcast results.

        Args:
            batch_data: List of InputData items to process.
        """
        results = self._model_handler.predict(batch_data)

        for i, data in enumerate(batch_data):
            result = results[i] if i < len(results) else EMPTY_RESULT
            if data.trace:
                data.trace.record_end("processor")
            output_data = OutputData(frame=data.frame, results=[result] if result else [], trace=data.trace)
            self._outbound_broadcaster.broadcast(output_data)

    def _stop(self) -> None:
        self._inbound_broadcaster.unregister(self.__class__.__name__)
        self._model_handler.close()
