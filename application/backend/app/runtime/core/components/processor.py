#  Copyright (C) 2025 Intel Corporation
#  SPDX-License-Identifier: Apache-2.0

import logging
from queue import Empty, Queue

from getiprompt.models.base import Model
from getiprompt.types.results import Results
from torchvision import tv_tensors

from runtime.core.components.base import PipelineComponent
from runtime.core.components.broadcaster import FrameBroadcaster
from runtime.core.components.schemas.processor import InputData, ModelContext, OutputData

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
        model_context: ModelContext | None = None,
    ):
        super().__init__()
        self._model_context = model_context
        self.inbound_broadcaster = inbound_broadcaster
        self._in_queue: Queue[InputData] = inbound_broadcaster.register()
        self._outbound_broadcaster = outbound_broadcaster

    def _train_model(self) -> Model | None:
        if self._model_context is None:
            logger.debug("No model context provided, running in pass-through mode")
            return None
        model = self._model_context.model
        model.learn(self._model_context.images, self._model_context.priors)
        return model

    def run(self) -> None:
        logger.debug("Starting a pipeline runner loop")

        model = self._train_model()

        while not self._stop_event.is_set():
            try:
                data = self._in_queue.get(timeout=0.1)
                if model is not None:
                    results = model.infer([tv_tensors.Image(data.frame)])
                    output_data = OutputData(frame=data.frame, results=results)
                else:
                    output_data = OutputData(frame=data.frame, results=Results())

                self._outbound_broadcaster.broadcast(output_data)
            except Empty:
                continue
            except Exception as e:
                logger.exception("Error in pipeline runner loop: %s", e)
                continue
        logger.debug("Stopping the pipeline runner loop")

    def _stop(self) -> None:
        self.inbound_broadcaster.unregister(self._in_queue)
