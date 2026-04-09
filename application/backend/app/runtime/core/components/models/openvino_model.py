# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
import tempfile
import time

import numpy as np
import openvino
from instantlearn.data.base.batch import Batch
from instantlearn.models.base import Model
from instantlearn.utils.constants import Backend
from instantlearn.utils.utils import device_to_openvino_device, precision_to_openvino_type
from openvino import properties

from domain.services.schemas.processor import InputData
from runtime.core.components.base import ModelHandler

logger = logging.getLogger(__name__)


class OpenVINOModelHandler(ModelHandler):
    def __init__(self, model: Model, reference_batch: Batch, precision: str) -> None:
        self._model = model
        self._reference_batch = reference_batch
        self._precision = precision
        self._compiled_model = None

    def initialise(self) -> None:
        self._model.fit(self._reference_batch)

        # Export on CPU to avoid XPU/CUDA compilation issues during tracing
        # The exported OpenVINO model can then be run on any device (CPU, GPU, etc.)
        original_device = next(self._model.parameters()).device
        self._model.cpu()
        try:
            with tempfile.TemporaryDirectory() as tmp_dir:
                path = self._model.export(tmp_dir, Backend.OPENVINO)

                core = openvino.Core()
                ov_device = device_to_openvino_device("CPU")
                core.set_property(
                    ov_device, {properties.hint.inference_precision: precision_to_openvino_type(self._precision)}
                )

                logger.debug(f"Compiling exported model from {path} for device {ov_device}...")
                logger.debug(f"Reading model {path}...")
                ov_model = core.read_model(str(path))
                logger.debug(f"Compiling model to {ov_device} (this may take a few minutes)...")

                start_time = time.time()
                self._compiled_model = core.compile_model(ov_model, ov_device)
                logger.debug(f"Model compilation finished in {time.time() - start_time:.2f}s.")
        finally:
            self._model.to(original_device)

    def predict(self, inputs: list[InputData]) -> list[dict[str, np.ndarray]]:
        if self._compiled_model is None:
            raise RuntimeError("Model not initialised. Call initialise() first.")

        logger.debug("Inference started: model=%s batch size=%d", type(self._model).__name__, len(inputs))

        results: list[dict[str, np.ndarray]] = []

        for input_data in inputs:
            image = np.expand_dims(input_data.frame.transpose(2, 0, 1), axis=0)  # HWC -> 1CHW
            output = self._compiled_model(image)
            masks = output["masks"]
            scores = output["scores"]
            labels = output["labels"]
            boxes = _masks_to_boxes(masks, scores)
            results.append(
                {
                    "pred_masks": masks,
                    "pred_scores": scores,
                    "pred_labels": labels,
                    "pred_boxes": boxes,
                }
            )
        return results

    def close(self) -> None:
        logger.info("Closing OpenVINOModelHandler and releasing resources")
        self._compiled_model = None
        self._model = None
        self._reference_batch = None


def _masks_to_boxes(masks: np.ndarray, scores: np.ndarray) -> np.ndarray:
    """Derive [x1, y1, x2, y2, score] bounding boxes from binary masks.

    Args:
        masks: Binary masks ``[N, H, W]``.
        scores: Confidence scores ``[N]``.

    Returns:
        Bounding boxes ``[N, 5]`` with ``(x1, y1, x2, y2, score)``.
    """
    n = masks.shape[0]
    if n == 0:
        return np.empty((0, 5), dtype=np.float32)

    boxes = np.empty((n, 5), dtype=np.float32)
    for i in range(n):
        rows = np.any(masks[i], axis=1)
        cols = np.any(masks[i], axis=0)
        if not rows.any():
            boxes[i] = [0, 0, 0, 0, scores[i]]
            continue
        y1, y2 = np.where(rows)[0][[0, -1]]
        x1, x2 = np.where(cols)[0][[0, -1]]
        boxes[i] = [x1, y1, x2 + 1, y2 + 1, scores[i]]

    return boxes
