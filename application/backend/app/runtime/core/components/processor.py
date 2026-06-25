#  Copyright (C) 2025 Intel Corporation
#  SPDX-License-Identifier: Apache-2.0

import logging
import time
from queue import Empty, Queue

import numpy as np

from domain.services.schemas.processor import ErrorData, InputData, OutputData
from runtime.core.components.base import ModelHandler, PipelineComponent
from runtime.core.components.broadcaster import FrameBroadcaster
from settings import get_settings

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
        settings = get_settings()
        self._stats_enabled = settings.enable_cpu_monitoring
        self._stats_interval_secs = settings.cpu_monitoring_interval_secs
        self._stats_started_at_s = time.perf_counter()
        self._stats_empty_poll_count = 0
        self._stats_input_frame_count = 0
        self._stats_skipped_frame_count = 0
        self._stats_batch_count = 0
        self._stats_output_frame_count = 0
        self._stats_error_count = 0
        self._stats_batch_wall_time_s = 0.0
        self._stats_batch_cpu_time_s = 0.0

    def setup(
        self,
        inbound_broadcaster: FrameBroadcaster[InputData | ErrorData],
        outbound_broadcaster: FrameBroadcaster[OutputData | ErrorData],
    ) -> None:
        self._inbound_broadcaster = inbound_broadcaster
        self._outbound_broadcaster = outbound_broadcaster
        self._in_queue: Queue[InputData | ErrorData] = inbound_broadcaster.register(self.__class__.__name__)
        self._initialized = True

    def run(self) -> None:
        if not self._initialized:
            raise RuntimeError("Processor must be set up before running")
        logger.debug("Starting a pipeline runner loop")

        self._model_handler.initialise()
        logger.info(
            "Pipeline model handler initialized, batch size: %d, frame skip interval: %d, skip amount: %d",
            self._batch_size,
            self._skip_policy.interval,
            self._skip_policy.skip_amount,
        )

        while not self._stop_event.is_set():
            try:
                batch_data = self._collect_batch_data()
                if not batch_data:
                    continue

                if isinstance(batch_data, ErrorData):
                    self._record_processor_error()
                    self._outbound_broadcaster.broadcast(batch_data)
                    continue

                batch_wall_started_at_s = time.perf_counter()
                batch_cpu_started_at_s = time.thread_time()
                self._process_batch(batch_data)
                self._record_processor_batch(
                    batch_size=len(batch_data),
                    wall_time_s=time.perf_counter() - batch_wall_started_at_s,
                    cpu_time_s=time.thread_time() - batch_cpu_started_at_s,
                )

            except Exception as e:
                logger.exception("Error in pipeline runner loop: %s", e)
                continue

        logger.debug("Stopping the pipeline runner loop")

    def _collect_batch_data(self) -> list[InputData] | ErrorData:
        """Collect a batch of input data from the queue.

        Returns:
            List of InputData items, empty if no data available, or ErrorData if an error was received.
        """
        batch_data: list[InputData] = []

        # Fetch first item; check for startup errors (e.g. missing video file)
        try:
            initial_data: InputData | ErrorData = self._in_queue.get(timeout=0.1)
        except Empty:
            self._record_processor_empty_poll()
            return []

        if isinstance(initial_data, ErrorData):
            return initial_data

        while len(batch_data) < self._batch_size and not self._stop_event.is_set():
            if initial_data is not None:
                input_data, initial_data = initial_data, None
            else:
                try:
                    input_data = self._in_queue.get(timeout=0.1)
                except Empty:
                    if batch_data:  # if we have partial batch data, process what we have
                        break
                    continue

            if input_data.trace:
                input_data.trace.record_start("processor")

            is_manual = input_data.context.get("requires_manual_control", False)
            self._record_processor_input_frame()

            if not is_manual and self._skip_policy.should_skip():
                self._record_processor_skipped_frame()
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

    def _record_processor_empty_poll(self) -> None:
        if not self._stats_enabled:
            return
        self._stats_empty_poll_count += 1
        self._log_processor_stats_if_needed()

    def _record_processor_input_frame(self) -> None:
        if self._stats_enabled:
            self._stats_input_frame_count += 1

    def _record_processor_skipped_frame(self) -> None:
        if self._stats_enabled:
            self._stats_skipped_frame_count += 1

    def _record_processor_error(self) -> None:
        if not self._stats_enabled:
            return
        self._stats_error_count += 1
        self._log_processor_stats_if_needed()

    def _record_processor_batch(self, batch_size: int, wall_time_s: float, cpu_time_s: float) -> None:
        if not self._stats_enabled:
            return
        self._stats_batch_count += 1
        self._stats_output_frame_count += batch_size
        self._stats_batch_wall_time_s += wall_time_s
        self._stats_batch_cpu_time_s += cpu_time_s
        self._log_processor_stats_if_needed()

    def _log_processor_stats_if_needed(self) -> None:
        now_s = time.perf_counter()
        elapsed_s = now_s - self._stats_started_at_s
        if elapsed_s < self._stats_interval_secs:
            return

        batch_rate = self._stats_batch_count / elapsed_s if elapsed_s > 0 else 0.0
        output_rate = self._stats_output_frame_count / elapsed_s if elapsed_s > 0 else 0.0
        avg_batch_wall_ms = (
            (self._stats_batch_wall_time_s / self._stats_batch_count) * 1000 if self._stats_batch_count else 0.0
        )
        avg_batch_cpu_ms = (
            (self._stats_batch_cpu_time_s / self._stats_batch_count) * 1000 if self._stats_batch_count else 0.0
        )
        logger.info(
            "processor_stats interval_secs=%.1f batches=%d batch_rate=%.1f input_frames=%d skipped_frames=%d "
            "output_frames=%d output_rate=%.1f empty_polls=%d errors=%d avg_batch_wall_ms=%.3f avg_batch_cpu_ms=%.3f",
            elapsed_s,
            self._stats_batch_count,
            batch_rate,
            self._stats_input_frame_count,
            self._stats_skipped_frame_count,
            self._stats_output_frame_count,
            output_rate,
            self._stats_empty_poll_count,
            self._stats_error_count,
            avg_batch_wall_ms,
            avg_batch_cpu_ms,
        )
        self._reset_processor_stats(now_s)

    def _reset_processor_stats(self, started_at_s: float) -> None:
        self._stats_started_at_s = started_at_s
        self._stats_empty_poll_count = 0
        self._stats_input_frame_count = 0
        self._stats_skipped_frame_count = 0
        self._stats_batch_count = 0
        self._stats_output_frame_count = 0
        self._stats_error_count = 0
        self._stats_batch_wall_time_s = 0.0
        self._stats_batch_cpu_time_s = 0.0

    def _stop(self) -> None:
        self._inbound_broadcaster.unregister(self.__class__.__name__)
        self._model_handler.close()
