#  Copyright (C) 2025 Intel Corporation
#  SPDX-License-Identifier: Apache-2.0

import logging
import time
from queue import Empty, Queue

import torch
from getiprompt.data.base.batch import Batch
from getiprompt.data.base.sample import Sample
from getiprompt.models.base import Model
from getiprompt.types.results import Results
from torchvision import tv_tensors

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
        model: Model | None = None,
        reference_batch: Batch | None = None,
    ):
        super().__init__()
        self._model = model
        self._reference_batch = reference_batch
        self.inbound_broadcaster = inbound_broadcaster
        self._in_queue: Queue[InputData] = inbound_broadcaster.register()
        self._outbound_broadcaster = outbound_broadcaster

    def _train_model(self) -> Model | None:
        if self._model is None or self._reference_batch is None:
            logger.debug("No model context provided, running in pass-through mode")
            return None
        model = self._model
        model.learn(self._reference_batch)
        return model

    def run(self) -> None:
        logger.debug("Starting a pipeline runner loop")

        model = self._train_model()

        while not self._stop_event.is_set():
            try:
                data = self._in_queue.get(timeout=0.1)
                if model is not None:
                    # Convert HWC (numpy) to CHW (tensor) for model
                    image_chw = tv_tensors.Image(torch.from_numpy(data.frame).permute(2, 0, 1))
                    sample = Sample(image=image_chw)
                    batch = Batch.collate([sample])

                    start_time = time.perf_counter()
                    results = model.infer(batch)
                    inference_time_ms = (time.perf_counter() - start_time) * 1000
                    logger.info("Model inference took %.2f ms", inference_time_ms)
                    print(f"Model inference took {inference_time_ms} ms")

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
