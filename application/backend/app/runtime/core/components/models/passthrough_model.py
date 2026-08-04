# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging

from instantlearn.data.base.prediction import Prediction

from domain.services.schemas.processor import InputData
from runtime.core.components.base import ModelHandler

logger = logging.getLogger(__name__)


class PassThroughModelHandler(ModelHandler):
    """No-op handler that lets frames flow through the pipeline unannotated.

    Used when inference is disabled, no model is configured, or the user has not
    created any prompts yet, so the ``Processor`` can still forward frames.
    """

    def predict(self, inputs: list[InputData]) -> list[Prediction]:  # noqa: ARG002
        logger.debug("Using PassThroughModelHandler, returning empty results.")
        return []
