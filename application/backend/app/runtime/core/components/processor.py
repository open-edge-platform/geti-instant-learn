#  Copyright (C) 2025 Intel Corporation
#  SPDX-License-Identifier: Apache-2.0

import logging
from queue import Empty, Queue
from uuid import UUID

import torch
from getiprompt.data.base.batch import Batch
from getiprompt.data.base.sample import Sample
from torchvision import tv_tensors

from domain.services.label import LabelService
from domain.services.schemas.processor import InputData, OutputData
from runtime.core.components.base import ModelHandler, PipelineComponent
from runtime.core.components.broadcaster import FrameBroadcaster

logger = logging.getLogger(__name__)


class Processor(PipelineComponent):
    """
    A job component responsible for retrieving raw frames from the inbound broadcaster,
    sending them to a processor for inference, and broadcasting the processed results to subscribed consumers.
    """

    def __init__(self, model_handler: ModelHandler, label_service: LabelService, project_id: UUID | None = None):
        super().__init__()
        self._model_handler = model_handler
        self._label_service = label_service
        self._project_id: UUID | None = project_id

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
                data = self._in_queue.get(timeout=0.1)
                batch = self._create_batch(data)
                results = self._model_handler.infer(batch)
                # output_data = OutputData(frame=data.frame, results=results, labels_colors=self._get_labels_colors())
                output_data = OutputData(frame=data.frame, results=results, labels_colors=None)
                if results:
                    logger.info("Received INFERENCE results: %s", results)
                    logger.info(
                        "Prepared OUTPUT DATA results: %s, label colors %s",
                        output_data.results,
                        output_data.labels_colors,
                    )
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

    def _get_labels_colors(self) -> dict[str, tuple[int, int, int]]:
        if not self._project_id:
            return {}
        try:
            labels = self._label_service.get_all_labels(self._project_id).labels  # todo without pag limit
            return {str(label.id): self._hex_to_rgb(label.color) for label in labels}
        except Exception as e:
            logger.warning("Failed to retrieve label colors: %s", e)
            return {}

    @staticmethod
    def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
        hex_color = hex_color.lstrip("#")
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)
        return r, g, b

    def _stop(self) -> None:
        self._inbound_broadcaster.unregister(self._in_queue)
