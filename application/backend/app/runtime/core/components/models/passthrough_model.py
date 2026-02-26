# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging

import numpy as np

from domain.services.schemas.processor import InputData
from runtime.core.components.base import ModelHandler

logger = logging.getLogger(__name__)


class PassThroughModelHandler(ModelHandler):
    def initialise(self) -> None:
        pass

    def predict(self, inputs: list[InputData]) -> list[dict[str, np.ndarray]]:  # noqa: ARG002
        logger.debug("Using PassThroughModelHandler, returning empty results.")
        return []
