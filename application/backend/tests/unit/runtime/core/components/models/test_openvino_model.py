# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock

import numpy as np
import pytest

pytest.importorskip("openvino")

from domain.services.schemas.processor import InputData
from runtime.core.components.models import openvino_model
from runtime.core.components.models.openvino_model import OpenVINOModelHandler


class TestOpenVINOModelHandler:
    @pytest.fixture
    def mock_model(self):
        return MagicMock()

    @pytest.fixture
    def mock_reference_batch(self):
        return MagicMock()

    def test_initialise_exports_and_compiles_model_on_cpu(self, mock_model, mock_reference_batch, monkeypatch):
        handler = OpenVINOModelHandler(mock_model, mock_reference_batch, precision="fp16")

        original_device = MagicMock()
        parameter = MagicMock()
        parameter.device = original_device
        mock_model.parameters.return_value = iter((parameter,))
        mock_model.export.return_value = "/tmp/model.xml"

        ov_model = MagicMock()
        compiled_model = MagicMock()
        compiled_model.outputs = [
            MagicMock(get_names=MagicMock(return_value={"masks"})),
            MagicMock(get_names=MagicMock(return_value={"scores"})),
            MagicMock(get_names=MagicMock(return_value={"labels"})),
        ]
        mock_core = MagicMock()
        mock_core.read_model.return_value = ov_model
        mock_core.compile_model.return_value = compiled_model

        mock_device_to_openvino_device = MagicMock(return_value="CPU")
        mock_precision_to_openvino_type = MagicMock(return_value="f16")

        monkeypatch.setattr(openvino_model.openvino, "Core", MagicMock(return_value=mock_core))
        monkeypatch.setattr(openvino_model, "device_to_openvino_device", mock_device_to_openvino_device)
        monkeypatch.setattr(openvino_model, "precision_to_openvino_type", mock_precision_to_openvino_type)

        handler.initialise()

        mock_model.fit.assert_called_once_with(mock_reference_batch)
        mock_model.cpu.assert_called_once_with()
        mock_model.export.assert_called_once()
        assert mock_model.export.call_args.args[1] == openvino_model.Backend.OPENVINO

        mock_device_to_openvino_device.assert_called_once_with("CPU")
        mock_precision_to_openvino_type.assert_called_once_with("fp16")
        mock_core.set_property.assert_called_once_with(
            "CPU", {openvino_model.properties.hint.inference_precision: "f16"}
        )
        mock_core.read_model.assert_called_once_with("/tmp/model.xml")
        mock_core.compile_model.assert_called_once_with(ov_model, "CPU")

        assert handler._compiled_model is compiled_model
        assert handler._infer_request is not None
        mock_model.to.assert_called_once_with(original_device)

    def test_predict_raises_when_not_initialised(self, mock_model, mock_reference_batch):
        handler = OpenVINOModelHandler(mock_model, mock_reference_batch, precision="fp16")

        input_data = InputData(
            timestamp=0,
            frame=np.zeros((10, 10, 3), dtype=np.uint8),
            context={},
        )

        with pytest.raises(RuntimeError, match="Model not initialised"):
            handler.predict([input_data])

    def test_predict_formats_input_and_maps_outputs(self, mock_model, mock_reference_batch):
        handler = OpenVINOModelHandler(mock_model, mock_reference_batch, precision="fp16")

        masks = np.array([[[1, 0, 1], [0, 1, 0]]], dtype=np.float32)  # [1, H, W] matches frame (H=2, W=3)
        scores = np.array([0.5], dtype=np.float32)
        labels = np.array([1], dtype=np.int64)

        masks_port, scores_port, labels_port = MagicMock(), MagicMock(), MagicMock()
        handler._infer_request = MagicMock()
        handler._infer_request.infer.return_value = {
            masks_port: masks,
            scores_port: scores,
            labels_port: labels,
        }
        handler._compiled_model = MagicMock()
        handler._input_port = MagicMock()
        handler._masks_output_port = masks_port
        handler._scores_output_port = scores_port
        handler._labels_output_port = labels_port

        frame = np.arange(2 * 3 * 3, dtype=np.uint8).reshape(2, 3, 3)
        input_data = InputData(timestamp=0, frame=frame, context={})

        results = handler.predict([input_data])

        assert len(results) == 1
        np.testing.assert_array_equal(results[0]["pred_masks"], masks > 0.5)
        np.testing.assert_array_equal(results[0]["pred_scores"], scores)
        np.testing.assert_array_equal(results[0]["pred_labels"], labels)
        assert results[0]["pred_boxes"].shape[1] == 5
