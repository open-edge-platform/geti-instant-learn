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
        print(f"DEBUG: Used device for model export: {original_device}")
        self._model.cpu()
        try:
            with tempfile.TemporaryDirectory() as tmp_dir:
                path = self._model.export(tmp_dir, Backend.OPENVINO)

                core = openvino.Core()
                ov_device = device_to_openvino_device("CPU")
                core.set_property(
                    ov_device, {properties.hint.inference_precision: precision_to_openvino_type(self._precision)}
                )

                print(f"DEBUG: Compiling exported model from {path} for device {ov_device}...")
                print(f"DEBUG: Reading model {path}...")
                ov_model = core.read_model(str(path))
                print(f"DEBUG: Compiling model to {ov_device} (this may take a few minutes)...")

                start_time = time.time()
                self._compiled_model = core.compile_model(ov_model, ov_device)
                print(f"DEBUG: Model compilation finished in {time.time() - start_time:.2f}s.")
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
            results.append(
                {
                    "pred_masks": output["masks"],
                    "pred_scores": output["scores"],
                    "pred_labels": output["labels"],
                }
            )
        print(f"results ->>> {results}")
        return results
