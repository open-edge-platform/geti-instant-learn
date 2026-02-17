# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging

import numpy as np
from instantlearn.data.base.batch import Batch
from instantlearn.models.base import Model

from domain.services.schemas.processor import InputData
from runtime.core.components.base import ModelHandler

logger = logging.getLogger(__name__)


class OpenVINOModelHandler(ModelHandler):
    def __init__(self, model: Model, reference_batch: Batch) -> None:
        self._model = model
        self._reference_batch = reference_batch

    def initialise(self) -> None: ...

    def predict(self, inputs: list[InputData]) -> list[dict[str, np.ndarray]]:
        raise NotImplementedError("OpenVINO inference is not yet implemented")
