# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
from queue import Queue, Empty

from runtime.core.components.base import JobComponent
from runtime.core.components.base import Processor
from runtime.core.components.broadcaster import FrameBroadcaster

logger = logging.getLogger(__name__)


class PipelineRunner(JobComponent):
    """A component that delegates processing logic to a processor."""

    def __init__(self, in_queue: Queue, broadcaster: FrameBroadcaster, processor: Processor):

        super().__init__()
        self._processor = processor
        self._in_queue = in_queue
        self._broadcaster = broadcaster

    def run(self) -> None:
        logger.debug("Starting a pipeline runner loop")
        while not self._stop_event.is_set():
            try:
                data = self._in_queue.get(timeout=0.1)
                processed_data = self._processor.process(data)
                if processed_data is not None:
                    self._broadcaster.broadcast(processed_data)

            except Empty:
                continue
            except Exception as e:
                logger.error(f"Error in pipeline runner loop: {e}")

        logger.debug("Stopping the pipeline runner loop")
