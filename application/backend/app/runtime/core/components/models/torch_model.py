# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging

import numpy as np
import torch
from instantlearn.data.base.batch import Batch
from instantlearn.data.base.sample import Sample
from instantlearn.models.base import Model
from torchvision import tv_tensors

from domain.services.schemas.processor import InputData
from runtime.core.components.base import ModelHandler

logger = logging.getLogger(__name__)


class TorchModelHandler(ModelHandler):
    def __init__(self, model: Model, reference_batch: Batch) -> None:
        self._model = model
        self._reference_batch = reference_batch

    def initialise(self) -> None:
        logger.info(
            "Initialising TorchModelHandler: model=%s, reference batch size=%d",
            type(self._model).__name__,
            len(self._reference_batch.samples),
        )
        self._model.fit(self._reference_batch)

    def predict(self, inputs: list[InputData]) -> list[dict[str, np.ndarray]]:
        logger.debug("Inference started: model=%s batch size=%d", type(self._model).__name__, len(inputs))
        batch = self._build_batch(inputs)
        torch_results = self._model.predict(batch)
        results = []
        for result in torch_results:
            batch_result = {}
            for k, v in result.items():
                tensor = v
                if isinstance(tensor, torch.Tensor) and tensor.dtype == torch.bfloat16:
                    tensor = tensor.to(torch.float32)
                batch_result[k] = tensor.detach().cpu().numpy()
            results.append(batch_result)
        return results

    @staticmethod
    def _build_batch(inputs: list[InputData]) -> Batch:
        samples = [Sample(image=tv_tensors.Image(torch.from_numpy(data.frame).permute(2, 0, 1))) for data in inputs]
        return Batch.collate(samples)
