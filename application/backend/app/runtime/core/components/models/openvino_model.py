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
        self._compiled_model: openvino.CompiledModel | None = None
        self._infer_request: openvino.InferRequest | None = None
        self._input_port: openvino.ConstOutput | None = None
        self._masks_output_port: openvino.ConstOutput | None = None
        self._scores_output_port: openvino.ConstOutput | None = None
        self._labels_output_port: openvino.ConstOutput | None = None

    @staticmethod
    def _resize_masks_to_frame(masks: np.ndarray, frame_h: int, frame_w: int) -> np.ndarray:
        """Resize predicted masks to original frame spatial size (H, W)."""
        if masks.ndim == 4 and masks.shape[0] == 1:
            masks = masks[0]

        if masks.ndim == 3 and (masks.shape[1] != frame_h or masks.shape[2] != frame_w):
            # Nearest-neighbor resize via index mapping — no cv2 needed.
            src_h, src_w = masks.shape[1], masks.shape[2]
            row_idx = (np.arange(frame_h) * src_h // frame_h).clip(0, src_h - 1)
            col_idx = (np.arange(frame_w) * src_w // frame_w).clip(0, src_w - 1)
            masks = masks[:, row_idx][:, :, col_idx]

        return masks > 0.5

    def initialise(self) -> None:
        self._model.fit(self._reference_batch)

        # Export on CPU to avoid XPU/CUDA compilation issues during tracing.
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

                logger.debug("Compiling exported model from %s for device %s...", path, ov_device)
                logger.debug("Reading model %s...", path)
                ov_model = core.read_model(str(path))

                input_size = getattr(self._model, "input_size", None)
                if input_size is not None:
                    input_name = ov_model.inputs[0].get_any_name()
                    ov_model.reshape({input_name: [1, 3, input_size, input_size]})

                logger.debug("Compiling model to %s (this may take a few minutes)...", ov_device)

                start_time = time.time()
                self._compiled_model = core.compile_model(ov_model, ov_device)
                logger.debug("Model compilation finished in %.2fs.", time.time() - start_time)

                self._infer_request = self._compiled_model.create_infer_request()
                self._input_port = self._compiled_model.input(0)

                outputs = list(self._compiled_model.outputs)
                output_by_name = {}
                for output in outputs:
                    for name in output.get_names():
                        output_by_name[name] = output

                # Fall back to positional outputs if names are not available.
                self._masks_output_port = output_by_name.get("masks", outputs[0])
                self._scores_output_port = output_by_name.get("scores", outputs[1])
                self._labels_output_port = output_by_name.get("labels", outputs[2])
        finally:
            self._model.to(original_device)

    def predict(self, inputs: list[InputData]) -> list[dict[str, np.ndarray]]:
        if self._compiled_model is None or self._infer_request is None:
            msg = "Model not initialised. Call initialise() before predict()."
            raise RuntimeError(msg)

        logger.debug("Inference started: model=%s batch size=%d", type(self._model).__name__, len(inputs))

        results: list[dict[str, np.ndarray]] = []

        for input_data in inputs:
            image = self._model.prepare_openvino_input(input_data.frame, self._model.input_size)
            output = self._infer_request.infer({self._input_port: image})
            pred_masks = np.asarray(output[self._masks_output_port])
            # Resize masks from model input size back to original frame dimensions.
            # Bounding boxes are derived from these resized masks below, so they
            # are already in original frame coordinates — no separate box resize needed.
            pred_masks = self._resize_masks_to_frame(pred_masks, input_data.frame.shape[0], input_data.frame.shape[1])
            pred_scores = np.asarray(output[self._scores_output_port])
            boxes = _masks_to_boxes(pred_masks, pred_scores)
            results.append(
                {
                    "pred_masks": pred_masks,
                    "pred_scores": pred_scores,
                    "pred_labels": np.asarray(output[self._labels_output_port]),
                    "pred_boxes": boxes,
                }
            )
        return results

    def close(self) -> None:
        logger.info("Closing OpenVINOModelHandler and releasing resources")
        self._compiled_model = None
        self._infer_request = None
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
