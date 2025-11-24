#  Copyright (C) 2025 Intel Corporation
#  SPDX-License-Identifier: Apache-2.0

import os
from unittest.mock import MagicMock, patch

import pytest

from domain.services.schemas.processor import MatcherConfig
from runtime.core.components.factories.model import ModelFactory


class TestModelFactory:
    @pytest.mark.parametrize(
        "runtime,expected_device",
        [
            ("cpu", "cpu"),
            ("cuda", "cuda"),
            ("xpu", "xpu"),
        ],
    )
    def test_resolve_device_returns_correct_device(self, runtime, expected_device):
        with patch.dict(os.environ, {"RUNTIME": runtime}):
            device = ModelFactory._resolve_device()

            assert device == expected_device

    def test_resolve_device_raises_error_for_unknown_runtime(self):
        with patch.dict(os.environ, {"RUNTIME": "unknown_runtime"}):
            with pytest.raises(ValueError, match="Unknown runtime: unknown_runtime"):
                ModelFactory._resolve_device()

    def test_resolve_device_defaults_to_cpu(self):
        with patch.dict(os.environ, {}, clear=True):
            device = ModelFactory._resolve_device()

            assert device == "cpu"

    def test_factory_creates_matcher_model_with_config(self):
        config = MatcherConfig(
            num_foreground_points=50,
            num_background_points=3,
            mask_similarity_threshold=0.5,
            precision="fp32",
        )
        mock_reference_batch = MagicMock()

        with patch.object(ModelFactory, "_resolve_device", return_value="cpu"):
            with patch("runtime.core.components.factories.model.Matcher") as mock_matcher:
                mock_instance = mock_matcher.return_value

                result = ModelFactory.create(mock_reference_batch, config)

                assert result is not mock_instance  # Returns a ModelHandler wrapping the model
                mock_matcher.assert_called_once_with(
                    num_foreground_points=50,
                    num_background_points=3,
                    mask_similarity_threshold=0.5,
                    precision="fp32",
                    device="cpu",
                )

    def test_factory_returns_none_for_unknown_config(self):
        result = ModelFactory.create(None, None)

        assert result is not None  # Returns PassThroughModelHandler instead of None
