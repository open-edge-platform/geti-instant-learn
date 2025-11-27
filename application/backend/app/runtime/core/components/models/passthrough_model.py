import torch
from getiprompt.data.base.batch import Batch
import logging
from runtime.core.components.base import ModelHandler

logger = logging.getLogger(__name__)


class PassThroughModelHandler(ModelHandler):
    def initialise(self) -> None:
        pass

    def infer(self, batch: Batch) -> list[dict[str, torch.Tensor]]:  # noqa: ARG002
        logger.info("Using PassThroughModelHandler, returning empty results.")
        return []
