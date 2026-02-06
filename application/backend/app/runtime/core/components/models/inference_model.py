import logging

import torch
from instantlearn.data.base.batch import Batch
from instantlearn.models.base import Model

from runtime.core.components.base import ModelHandler

logger = logging.getLogger(__name__)


class InferenceModelHandler(ModelHandler):
    def __init__(self, model: Model, reference_batch: Batch) -> None:
        self._model = model
        self._reference_batch = reference_batch

    def initialise(self) -> None:
        logger.info(
            "Initialising InferenceModelHandler: model=%s, reference batch size=%d",
            type(self._model).__name__,
            len(self._reference_batch.samples),
        )
        self._model.fit(self._reference_batch)

    def predict(self, batch: Batch) -> list[dict[str, torch.Tensor]]:
        logger.debug("Inference started: model=%s batch size=%d", type(self._model).__name__, len(batch.samples))
        return self._model.predict(batch)
