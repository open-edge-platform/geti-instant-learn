import logging

import torch
from getiprompt.data.base.batch import Batch

from runtime.core.components.base import ModelHandler

logger = logging.getLogger(__name__)


class PassThroughModelHandler(ModelHandler):
    def initialise(self) -> None:
        pass

    def predict(self, batch: Batch) -> list[dict[str, torch.Tensor]]:  # noqa: ARG002
        logger.debug("Using PassThroughModelHandler, returning empty results.")
        return []
