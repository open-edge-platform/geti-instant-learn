#  Copyright (C) 2025 Intel Corporation
#  SPDX-License-Identifier: Apache-2.0

import logging
from queue import Empty, Queue

import torch
from getiprompt.data.base.batch import Batch
from getiprompt.data.base.sample import Sample
from torchvision import tv_tensors

from domain.services.schemas.processor import InputData, OutputData
from runtime.core.components.base import ModelHandler, PipelineComponent
from runtime.core.components.broadcaster import FrameBroadcaster

logger = logging.getLogger(__name__)


class Processor(PipelineComponent):
    """
    A job component responsible for retrieving raw frames from the inbound broadcaster,
    sending them to a processor for inference, and broadcasting the processed results to subscribed consumers.
    """

    def __init__(self, model_handler: ModelHandler):
        super().__init__()
        self._model_handler = model_handler

    def setup(
        self, inbound_broadcaster: FrameBroadcaster[InputData], outbound_broadcaster: FrameBroadcaster[OutputData]
    ) -> None:
        self._inbound_broadcaster = inbound_broadcaster
        self._outbound_broadcaster = outbound_broadcaster
        self._in_queue: Queue[InputData] = inbound_broadcaster.register()
        self._initialized = True

    def run(self) -> None:
        logger.debug("Starting a pipeline runner loop")
        self._model_handler.initialise()
        while not self._stop_event.is_set():
            try:
                data = self._in_queue.get(timeout=0.1)
                batch = self._create_batch(data)
                results = self._model_handler.infer(batch)
                output_data = OutputData(frame=data.frame, results=results)
                self._outbound_broadcaster.broadcast(output_data)
            except Empty:
                continue
            except Exception as e:
                logger.exception("Error in pipeline runner loop: %s", e)
                continue
        logger.debug("Stopping the pipeline runner loop")

    def _create_batch(self, data: InputData):
        # Convert HWC (numpy) to CHW (tensor) for model
        image_chw = tv_tensors.Image(torch.from_numpy(data.frame).permute(2, 0, 1))
        sample = Sample(image=image_chw)
        return Batch.collate([sample])

    def _stop(self) -> None:
        self._inbound_broadcaster.unregister(self._in_queue)
