#  Copyright (C) 2025 Intel Corporation
#  SPDX-License-Identifier: Apache-2.0

import logging
from queue import Empty, Queue
from typing import Any

from runtime.core.components.base import PipelineComponent
from runtime.core.components.broadcaster import FrameBroadcaster
from runtime.core.components.schemas.processor import InputData, OutputData

logger = logging.getLogger(__name__)


class Processor(PipelineComponent):
    """
    A job component responsible for retrieving raw frames from the inbound broadcaster,
    sending them to a processor for inference, and broadcasting the processed results to subscribed consumers.
    """

    def __init__(
        self,
        inbound_broadcaster: FrameBroadcaster[InputData],
        outbound_broadcaster: FrameBroadcaster[OutputData],
        model: Any,
    ):
        super().__init__()
        # it is a placeholder for a vision prompt model instance
        self._model = model
        self.inbound_broadcaster = inbound_broadcaster
        self._in_queue: Queue[InputData] = inbound_broadcaster.register()
        self._outbound_broadcaster = outbound_broadcaster

    def run(self) -> None:
        logger.debug("Starting a pipeline runner loop")
        while not self._stop_event.is_set():
            try:
                data = self._in_queue.get(timeout=0.1)
                processed_data = data
                if processed_data is not None:
                    self._outbound_broadcaster.broadcast(processed_data)
            except Empty:
                continue
            except Exception as e:
                logger.exception("Error in pipeline runner loop: %s", e)

        logger.debug("Stopping the pipeline runner loop")

    def _stop(self) -> None:
        self.inbound_broadcaster.unregister(self._in_queue)
