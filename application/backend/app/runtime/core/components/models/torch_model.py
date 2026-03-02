# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import gc
import logging

import numpy as np
import torch
from instantlearn.data.base.batch import Batch
from instantlearn.data.base.sample import Sample
from instantlearn.models.base import Model
from torchvision import tv_tensors

from domain.services.schemas.processor import InputData
from runtime.core.components.base import ModelHandler

logger = logging.getLogger(__name__)


def release_device_memory(device: str) -> None:
    """Clear the device memory cache after model cleanup.

    Calls the appropriate cache-clearing function depending on the device
    type (CUDA, XPU, or CPU). This should be called after deleting model
    references and running ``gc.collect()`` to ensure freed tensors are
    returned to the device allocator.

    Args:
        device: The device string ("cpu", "cuda", "xpu").
    """
    device_type = device.split(":")[0]
    if device_type == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.info("CUDA memory cache cleared")
    elif device_type == "xpu" and hasattr(torch, "xpu") and torch.xpu.is_available():
        torch.xpu.empty_cache()
        logger.info("XPU memory cache cleared")
    else:
        logger.debug("No device cache to clear for device=%s", device_type)


class TorchModelHandler(ModelHandler):
    def __init__(self, model: Model, reference_batch: Batch, device: str = "cpu") -> None:
        self._model = model
        self._reference_batch = reference_batch
        self._device = device

    def initialise(self) -> None:
        logger.info(
            "Initialising TorchModelHandler: model=%s, reference batch size=%d",
            type(self._model).__name__,
            len(self._reference_batch.samples),
        )
        self._model.fit(self._reference_batch)

    def predict(self, inputs: list[InputData]) -> list[dict[str, np.ndarray]]:
        logger.debug("Inference started: model=%s batch size=%d", type(self._model).__name__, len(inputs))
        batch = self._build_batch(inputs)
        torch_results = self._model.predict(batch)
        results = []
        for result in torch_results:
            batch_result = {}
            for k, v in result.items():
                tensor = v
                if isinstance(tensor, torch.Tensor) and tensor.dtype == torch.bfloat16:
                    tensor = tensor.to(torch.float32)
                batch_result[k] = tensor.detach().cpu().numpy()
            results.append(batch_result)
        return results

    @staticmethod
    def _build_batch(inputs: list[InputData]) -> Batch:
        samples = [Sample(image=tv_tensors.Image(torch.from_numpy(data.frame).permute(2, 0, 1))) for data in inputs]
        return Batch.collate(samples)

    def cleanup(self) -> None:
        """Release model and reference batch, then free device memory."""
        model_name = type(self._model).__name__ if self._model else "unknown"
        logger.info("Cleaning up TorchModelHandler (model=%s, device=%s)", model_name, self._device)

        self._model = None
        self._reference_batch = None

        gc.collect()
        release_device_memory(self._device)
        logger.debug("TorchModelHandler cleanup complete")
