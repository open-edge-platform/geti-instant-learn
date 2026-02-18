# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from domain.services.schemas.processor import InputData
from runtime.core.components.models.torch_model import TorchModelHandler


class TestTorchModelHandler:
    @pytest.fixture
    def mock_model(self):
        return MagicMock()

    @pytest.fixture
    def mock_reference_batch(self):
        return MagicMock()

    def test_predict_converts_bfloat16_to_float32(self, mock_model, mock_reference_batch):
        handler = TorchModelHandler(mock_model, mock_reference_batch)

        input_data = InputData(
            timestamp=0,
            frame=np.zeros((10, 10, 3), dtype=np.uint8),
            context={},
        )
        inputs = [input_data]

        bfloat16_tensor = torch.tensor([1.0, 2.0], dtype=torch.bfloat16)

        mock_model.predict.return_value = [{"scores": bfloat16_tensor}]
        results = handler.predict(inputs)

        assert len(results) == 1
        assert "scores" in results[0]
        assert isinstance(results[0]["scores"], np.ndarray)
        assert results[0]["scores"].dtype == np.float32
        np.testing.assert_array_equal(results[0]["scores"], np.array([1.0, 2.0], dtype=np.float32))

    def test_predict_handles_standard_tensors(self, mock_model, mock_reference_batch):
        handler = TorchModelHandler(mock_model, mock_reference_batch)
        input_data = InputData(
            timestamp=0,
            frame=np.zeros((10, 10, 3), dtype=np.uint8),
            context={},
        )
        inputs = [input_data]

        float32_tensor = torch.tensor([1.0, 2.0], dtype=torch.float32)
        mock_model.predict.return_value = [{"scores": float32_tensor}]

        results = handler.predict(inputs)

        assert len(results) == 1
        assert results[0]["scores"].dtype == np.float32
