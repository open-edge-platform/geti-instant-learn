import logging

import torch
from instantlearn.data.base.batch import Batch
from instantlearn.models.base import Model

from runtime.core.components.base import ModelHandler

logger = logging.getLogger(__name__)


class OpenVINOModelHandler(ModelHandler):
    def __init__(self, model: Model, reference_batch: Batch) -> None:
        self._model = model
        self._reference_batch = reference_batch

    def initialise(self) -> None:
        self._model.export()

    def predict(self, batch: Batch) -> list[dict[str, torch.Tensor]]: ...
