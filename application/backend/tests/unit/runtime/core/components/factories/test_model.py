#  Copyright (C) 2025 Intel Corporation
#  SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock, patch

from domain.services.schemas.processor import MatcherConfig
from runtime.core.components.factories.model import ModelFactory


class TestModelFactory:
    def test_factory_creates_matcher_model_with_config(self):
        config = MatcherConfig(
            num_foreground_points=50,
            num_background_points=3,
            mask_similarity_threshold=0.5,
            precision="fp32",
        )
        mock_reference_batch = MagicMock()
        mock_settings = MagicMock()
        mock_settings.device = "cpu"

        with patch("runtime.core.components.factories.model.get_settings", return_value=mock_settings):
            with patch("runtime.core.components.factories.model.Matcher") as mock_matcher:
                mock_instance = mock_matcher.return_value

                result = ModelFactory.create(mock_reference_batch, config)

                assert result is not mock_instance  # Returns a ModelHandler wrapping the model
                mock_matcher.assert_called_once_with(
                    num_foreground_points=50,
                    num_background_points=3,
                    encoder_model="dinov3_small",
                    mask_similarity_threshold=0.5,
                    precision="fp32",
                    device="cpu",
                    use_mask_refinement=False,
                )

    def test_factory_returns_none_for_unknown_config(self):
        result = ModelFactory.create(None, None)

        assert result is not None  # Returns PassThroughModelHandler instead of None
