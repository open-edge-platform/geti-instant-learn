# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from domain.services.schemas.processor import InputData
from runtime.core.components.models.torch_model import TorchModelHandler, release_device_memory


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

    def test_cleanup_frees_references_and_collects_garbage(self, mock_model, mock_reference_batch):
        handler = TorchModelHandler(mock_model, mock_reference_batch, device="cpu")

        with patch("runtime.core.components.models.torch_model.gc") as mock_gc:
            handler.cleanup()

        mock_gc.collect.assert_called_once()
        assert handler._model is None
        assert handler._reference_batch is None

    def test_cleanup_calls_cuda_empty_cache_for_cuda_device(self, mock_model, mock_reference_batch):
        handler = TorchModelHandler(mock_model, mock_reference_batch, device="cuda")

        with (
            patch("runtime.core.components.models.torch_model.gc"),
            patch("runtime.core.components.models.torch_model.torch") as mock_torch,
        ):
            mock_torch.cuda.is_available.return_value = True
            handler.cleanup()

        mock_torch.cuda.empty_cache.assert_called_once()

    def test_cleanup_calls_xpu_empty_cache_for_xpu_device(self, mock_model, mock_reference_batch):
        handler = TorchModelHandler(mock_model, mock_reference_batch, device="xpu")

        with (
            patch("runtime.core.components.models.torch_model.gc"),
            patch("runtime.core.components.models.torch_model.torch") as mock_torch,
        ):
            mock_torch.xpu.is_available.return_value = True
            handler.cleanup()

        mock_torch.xpu.empty_cache.assert_called_once()

    def test_cleanup_is_safe_to_call_twice(self, mock_model, mock_reference_batch):
        handler = TorchModelHandler(mock_model, mock_reference_batch, device="cpu")

        handler.cleanup()
        # Second call should not raise even though _model is already None
        handler.cleanup()

        assert handler._model is None
        assert handler._reference_batch is None


class TestReleaseDeviceMemory:
    def test_release_cuda_memory(self):
        with patch("runtime.core.components.models.torch_model.torch") as mock_torch:
            mock_torch.cuda.is_available.return_value = True
            release_device_memory("cuda")
            mock_torch.cuda.empty_cache.assert_called_once()

    def test_release_cuda_with_device_index(self):
        with patch("runtime.core.components.models.torch_model.torch") as mock_torch:
            mock_torch.cuda.is_available.return_value = True
            release_device_memory("cuda:0")
            mock_torch.cuda.empty_cache.assert_called_once()

    def test_release_xpu_memory(self):
        with patch("runtime.core.components.models.torch_model.torch") as mock_torch:
            mock_torch.xpu.is_available.return_value = True
            release_device_memory("xpu")
            mock_torch.xpu.empty_cache.assert_called_once()

    def test_release_cpu_is_noop(self):
        with patch("runtime.core.components.models.torch_model.torch") as mock_torch:
            release_device_memory("cpu")
            mock_torch.cuda.empty_cache.assert_not_called()
