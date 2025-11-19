import torch
from getiprompt.data.base.batch import Batch

from runtime.core.components.base import ModelHandler


class PassThroughModelHandler(ModelHandler):
    def initialise(self) -> None:
        pass

    def infer(self, batch: Batch) -> list[dict[str, torch.Tensor]]:  # noqa: ARG002
        return []
