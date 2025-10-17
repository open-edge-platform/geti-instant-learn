#  Copyright (C) 2025 Intel Corporation
#  SPDX-License-Identifier: Apache-2.0

import logging
from queue import Empty, Queue
from typing import Any

from core.components.base import PipelineComponent
from core.components.broadcaster import FrameBroadcaster

logger = logging.getLogger(__name__)


class Processor(PipelineComponent):
    """
    A job component responsible retrieving raw frames from an input queue, sending them to a processor for inference,
    and broadcasting the processed results to subscribed consumers.
    """

    def __init__(self, in_queue: Queue, broadcaster: FrameBroadcaster, model: Any):
        super().__init__()
        # it is a placeholder for a vision prompt model instance
        self._model = model
        self._in_queue = in_queue
        self._broadcaster = broadcaster

    def run(self) -> None:
        logger.debug("Starting a pipeline runner loop")
        while not self._stop_event.is_set():
            try:
                data = self._in_queue.get(timeout=0.1)
                processed_data = data
                if processed_data is not None:
                    self._broadcaster.broadcast(processed_data)

            except Empty:
                continue
            except Exception as e:
                logger.exception("Error in pipeline runner loop: %s", e)

        logger.debug("Stopping the pipeline runner loop")
