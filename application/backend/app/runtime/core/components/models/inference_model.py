import logging
from uuid import UUID

import torch
from getiprompt.data.base.batch import Batch
from getiprompt.models.base import Model

from runtime.core.components.base import ModelHandler

logger = logging.getLogger(__name__)


class InferenceModelHandler(ModelHandler):
    def __init__(self, model: Model, reference_batch: Batch) -> None:
        self._model = model
        self._reference_batch = reference_batch
        self._category_to_label_id = {
            idx: UUID(cat)
            for idx, cat in enumerate(reference_batch.categories[0])  # todo
        }

    def initialise(self) -> None:
        self._model.fit(self._reference_batch)

    def predict(self, batch: Batch) -> list[dict[str, torch.Tensor]]:
        logger.info("InferenceModelHandler predict called with batch of size %d", len(batch.samples))
        return self._model.predict(batch)
