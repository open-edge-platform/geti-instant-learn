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


class Processor(PipelineComponent):
    """
    A job component responsible for retrieving raw frames from the inbound broadcaster,
    sending them to a processor for inference, and broadcasting the processed results to subscribed consumers.
    """

    def __init__(
        self,
        model_handler: ModelHandler,
        batch_size: int = 3,
        category_id_to_label_id: dict[int, str] | None = None,
    ) -> None:
        super().__init__()
        self._model_handler = model_handler
        self._batch_size = batch_size
        self._category_id_to_label_id = category_id_to_label_id or {}

    def setup(
        self,
        inbound_broadcaster: FrameBroadcaster[InputData],
        outbound_broadcaster: FrameBroadcaster[OutputData],
    ) -> None:
        self._inbound_broadcaster = inbound_broadcaster
        self._outbound_broadcaster = outbound_broadcaster
        self._in_queue: Queue[InputData] = inbound_broadcaster.register()
        self._initialized = True

    def run(self) -> None:
        logger.debug("Starting a pipeline runner loop")
        self._model_handler.initialise()
        logger.info("Pipeline model handler initialized")

        while not self._stop_event.is_set():
            try:
                batch_data: list[InputData] = []
                for _ in range(self._batch_size):
                    try:
                        input_data = self._in_queue.get(timeout=0.1)
                        batch_data.append(input_data)

                        if input_data.context.get("requires_manual_control", False):
                            break
                    except Empty:
                        break

                if not batch_data:
                    continue

                batch_results = self._model_handler.predict(batch_data)

                for i, data in enumerate(batch_data):
                    results: dict[str, np.ndarray] = batch_results[i] if i < len(batch_results) else EMPTY_RESULT
                    output_data = OutputData(
                        frame=data.frame,
                        results=[results],
                    )
                    self._outbound_broadcaster.broadcast(output_data)

            except Exception as e:
                logger.exception("Error in pipeline runner loop: %s", e)
                continue

        logger.debug("Stopping the pipeline runner loop")

    def _stop(self) -> None:
        self._inbound_broadcaster.unregister(self._in_queue)
