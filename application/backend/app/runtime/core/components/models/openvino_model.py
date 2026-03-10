# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
import tempfile
import time

import numpy as np
import openvino
import torch
from torch.nn import functional
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
        self._infer_request = None
        self._input_port = None
        self._masks_output_port = None
        self._scores_output_port = None
        self._labels_output_port = None

    def _get_target_size(self) -> int | None:
        """Return target square input size from wrapped model when available."""
        encoder = getattr(self._model, "encoder", None)
        input_size = getattr(encoder, "input_size", None)
        if isinstance(input_size, int) and input_size > 0:
            return input_size
        return None

    def _prepare_input(self, frame: np.ndarray) -> np.ndarray:
        """Prepare frame as contiguous float32 NCHW with stable spatial size."""
        image = np.expand_dims(frame.transpose(2, 0, 1), axis=0).astype(np.float32, copy=False)

        target_size = self._get_target_size()
        if target_size is not None and (image.shape[2] != target_size or image.shape[3] != target_size):
            image_tensor = torch.from_numpy(image)
            image_tensor = functional.interpolate(
                image_tensor,
                size=(target_size, target_size),
                mode="bilinear",
                align_corners=False,
            )
            image = image_tensor.numpy()

        return np.ascontiguousarray(image, dtype=np.float32)

    @staticmethod
    def _resize_masks_to_frame(masks: np.ndarray, frame_shape: tuple[int, int, int]) -> np.ndarray:
        """Resize predicted masks to original frame spatial size (H, W)."""
        frame_h, frame_w = frame_shape[:2]

        if masks.ndim == 4 and masks.shape[0] == 1:
            masks = masks[0]

        if masks.ndim != 3 or masks.shape[1] == frame_h and masks.shape[2] == frame_w:
            return masks

        mask_tensor = torch.from_numpy(masks.astype(np.float32, copy=False)).unsqueeze(1)  # [N, 1, H, W]
        mask_tensor = functional.interpolate(mask_tensor, size=(frame_h, frame_w), mode="nearest")
        resized = mask_tensor.squeeze(1).numpy()
        return resized > 0.5

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
                ov_device = device_to_openvino_device("GPU")
                core.set_property(
                    ov_device, {properties.hint.inference_precision: precision_to_openvino_type(self._precision)}
                )

                logger.info("Compiling exported model from %s for device %s...", path, ov_device)
                logger.info("Reading model %s...", path)
                ov_model = core.read_model(str(path))

                target_size = self._get_target_size()
                if target_size is not None:
                    input_name = ov_model.inputs[0].get_any_name()
                    ov_model.reshape({input_name: [1, 3, target_size, target_size]})

                logger.info("Compiling model to %s (this may take a few minutes)...", ov_device)

                start_time = time.time()
                self._compiled_model = core.compile_model(ov_model, ov_device)
                logger.info("Model compilation finished in %.2fs.", time.time() - start_time)

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
        if (
            self._compiled_model is None
            or self._infer_request is None
            or self._input_port is None
            or self._masks_output_port is None
            or self._scores_output_port is None
            or self._labels_output_port is None
        ):
            raise RuntimeError("Model not initialised. Call initialise() first.")

        logger.info("Inference started: model=%s batch size=%d", type(self._model).__name__, len(inputs))

        results: list[dict[str, np.ndarray]] = []

        for input_data in inputs:
            image = self._prepare_input(input_data.frame)
            output = self._infer_request.infer({self._input_port: image})
            pred_masks = np.asarray(output[self._masks_output_port])
            pred_masks = self._resize_masks_to_frame(pred_masks, input_data.frame.shape)
            results.append(
                {
                    "pred_masks": pred_masks,
                    "pred_scores": np.asarray(output[self._scores_output_port]),
                    "pred_labels": np.asarray(output[self._labels_output_port]),
                }
            )
        return results
