# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Inference handler.

Wraps any :class:`instantlearn.models.base.Model` — PyTorch- or OpenVINO-backed —
behind the pipeline's :class:`~runtime.core.components.base.ModelHandler`
interface. The model arrives built and fitted from the
:class:`~runtime.core.components.factories.model.ModelFactory`, so this class
only adapts pipeline frames to ``Sample`` objects and returns ``Prediction``
objects unchanged.
"""

import logging

from instantlearn.data.base.batch import Batch
from instantlearn.data.base.prediction import Prediction
from instantlearn.data.base.sample import Sample
from instantlearn.models.base import Model

from domain.services.schemas.processor import InputData
from runtime.core.components.base import ModelHandler

logger = logging.getLogger(__name__)


class InferenceModelHandler(ModelHandler):
    def __init__(self, model: Model) -> None:
        self._model: Model | None = model
        logger.info(
            "Inference handler ready: model=%s backend=%s",
            type(model).__name__,
            getattr(model, "backend", None),
        )

    def predict(self, inputs: list[InputData]) -> list[Prediction]:
        """Run inference on a batch of frames.

        Args:
            inputs: Pipeline frames in RGB HWC ``uint8`` format.

        Returns:
            One ``Prediction`` per input frame.

        Raises:
            RuntimeError: If called after :meth:`close`.
        """
        if self._model is None:
            msg = "Model handler is closed. predict() cannot be called."
            raise RuntimeError(msg)

        batch = Batch.collate([Sample(image=data.frame) for data in inputs])
        logger.debug("Inference started: model=%s batch size=%d", type(self._model).__name__, len(inputs))

        return self._model.predict(batch)

    def close(self) -> None:
        """Drop the model reference and free accelerator memory."""
        logger.info(
            "Closing inference handler and releasing resources: model=%s",
            type(self._model).__name__ if self._model else "None",
        )
        self._model = None
        empty_accelerator_cache()


def empty_accelerator_cache() -> None:
    """Release cached accelerator memory when torch is available.

    Used after dropping a torch model, both on pipeline shutdown and once an
    OpenVINO export has made the torch graph redundant.
    """
    try:
        import torch
    except ImportError:  # pragma: no cover - torch is always installed today
        return

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        torch.xpu.empty_cache()
