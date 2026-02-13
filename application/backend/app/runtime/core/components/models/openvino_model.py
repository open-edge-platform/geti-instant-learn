import logging

import numpy as np
import openvino
from instantlearn.data.base.batch import Batch
from instantlearn.models.base import Model
from instantlearn.utils.constants import Backend

from domain.services.schemas.processor import InputData
from runtime.core.components.base import ModelHandler

logger = logging.getLogger(__name__)


class OpenVINOModelHandler(ModelHandler):
    def __init__(self, model: Model, reference_batch: Batch) -> None:
        self._model = model
        self._reference_batch = reference_batch

    def initialise(self) -> None:
        self._model.fit(self._reference_batch)
        path = self._model.export(".", Backend.OPENVINO)

        core = openvino.Core()
        ov_model = core.read_model(str(path))

        self._compiled_model = core.compile_model(ov_model, "GPU")

    def predict(self, inputs: list[InputData]) -> list[dict[str, np.ndarray]]:
        logger.debug("Inference started: model=%s batch size=%d", type(self._model).__name__, len(inputs))
        # TODO: Implement proper OpenVINO inference with InputData
        raise NotImplementedError("OpenVINO inference with InputData is not yet implemented")
