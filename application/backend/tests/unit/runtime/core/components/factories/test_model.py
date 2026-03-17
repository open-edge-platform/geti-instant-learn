#  Copyright (C) 2025 Intel Corporation
#  SPDX-License-Identifier: Apache-2.0

from unittest.mock import DEFAULT, MagicMock, patch

import pytest
from instantlearn.utils.constants import SAMModelName

from domain.services.schemas.processor import MatcherConfig, PerDinoConfig, SoftMatcherConfig
from runtime.core.components.factories.model import DeviceResolver, ModelFactory
from runtime.core.components.models.passthrough_model import PassThroughModelHandler


class TestDeviceResolver:
    @pytest.mark.parametrize(
        ("has_intel_gpu", "has_nvidia_gpu", "expected_device"),
        [
            (True, True, "xpu"),
            (False, True, "cuda"),
            (False, False, "cpu"),
        ],
    )
    def test_resolve_device_auto_priority(self, has_intel_gpu, has_nvidia_gpu, expected_device):
        resolver = DeviceResolver()
        with (
            patch.object(DeviceResolver, "_has_intel_gpu", return_value=has_intel_gpu),
            patch.object(DeviceResolver, "_has_nvidia_gpu", return_value=has_nvidia_gpu),
        ):
            assert resolver.resolve_device("auto") == expected_device

    def test_resolve_device_keeps_explicit_device(self):
        resolver = DeviceResolver()
        assert resolver.resolve_device("cuda") == "cuda"

    @pytest.mark.parametrize(
        ("has_intel_gpu", "has_nvidia_gpu", "expected_device"),
        [
            (True, True, "xpu"),
            (False, True, "cuda"),
            (False, False, "cpu"),
        ],
    )
    def test_resolve_device_none_behaves_like_auto(self, has_intel_gpu, has_nvidia_gpu, expected_device):
        resolver = DeviceResolver()
        with (
            patch.object(DeviceResolver, "_has_intel_gpu", return_value=has_intel_gpu),
            patch.object(DeviceResolver, "_has_nvidia_gpu", return_value=has_nvidia_gpu),
        ):
            assert resolver.resolve_device(None) == expected_device


class TestModelFactory:
    @pytest.fixture
    def mock_reference_batch(self):
        return MagicMock()

    @pytest.fixture
    def mock_settings(self):
        settings = MagicMock()
        settings.processor_inference_enabled = True
        settings.processor_openvino_enabled = False
        return settings

    @pytest.fixture
    def mock_device_resolver(self):
        resolver = MagicMock()
        resolver.resolve_device.return_value = "cpu"
        return resolver

    @pytest.fixture
    def model_factory(self, mock_device_resolver):
        return ModelFactory(device_resolver=mock_device_resolver)

    @pytest.mark.parametrize(
        ("resolved_device", "expected_precision", "use_torch_handler"),
        [
            ("cuda", "bf16", True),
            ("cpu", "bf16", True),
            ("xpu", "fp32", False),
        ],
    )
    def test_factory_create_auto_uses_resolved_device_and_handler_branch(
        self,
        mock_reference_batch,
        mock_settings,
        model_factory,
        mock_device_resolver,
        resolved_device,
        expected_precision,
        use_torch_handler,
    ):
        config = MatcherConfig(
            num_foreground_points=5,
            num_background_points=3,
            confidence_threshold=0.5,
            precision="bf16",
            sam_model=SAMModelName.SAM_HQ_TINY,
            encoder_model="dinov3_small",
            use_mask_refinement=True,
            use_nms=True,
        )
        mock_settings.device = "auto"
        mock_settings.processor_openvino_enabled = not use_torch_handler

        with (
            patch.multiple(
                "runtime.core.components.factories.model",
                get_settings=DEFAULT,
                Matcher=DEFAULT,
                TorchModelHandler=DEFAULT,
                OpenVINOModelHandler=DEFAULT,
            ) as mocks,
        ):
            mocks["get_settings"].return_value = mock_settings
            mock_device_resolver.resolve_device.return_value = resolved_device
            mock_matcher = mocks["Matcher"]
            mock_torch_handler = mocks["TorchModelHandler"]
            mock_openvino_handler = mocks["OpenVINOModelHandler"]

            mock_model_instance = MagicMock()
            mock_matcher.return_value = mock_model_instance

            mock_torch_handler_instance = MagicMock()
            mock_openvino_handler_instance = MagicMock()
            mock_torch_handler.return_value = mock_torch_handler_instance
            mock_openvino_handler.return_value = mock_openvino_handler_instance

            result = model_factory.create(mock_reference_batch, config)

            mock_device_resolver.resolve_device.assert_called_once_with(None)
            assert mock_matcher.call_args.kwargs["device"] == resolved_device
            assert mock_matcher.call_args.kwargs["precision"] == expected_precision

            if use_torch_handler:
                assert result is mock_torch_handler_instance
                mock_torch_handler.assert_called_once_with(mock_model_instance, mock_reference_batch)
                mock_openvino_handler.assert_not_called()
            else:
                assert result is mock_openvino_handler_instance
                mock_openvino_handler.assert_called_once_with(
                    mock_model_instance,
                    mock_reference_batch,
                    precision="fp32",
                )
                mock_torch_handler.assert_not_called()

    def test_factory_creates_matcher_model_with_config(self, mock_reference_batch, mock_settings, model_factory):
        mock_settings.processor_openvino_enabled = True
        config = MatcherConfig(
            num_foreground_points=50,
            num_background_points=3,
            confidence_threshold=0.5,
            precision="fp32",
            sam_model=SAMModelName.SAM_HQ_TINY,
            encoder_model="dinov3_small",
            use_mask_refinement=True,
        )

        with patch.multiple(
            "runtime.core.components.factories.model",
            get_settings=DEFAULT,
            Matcher=DEFAULT,
            OpenVINOModelHandler=DEFAULT,
        ) as mocks:
            mocks["get_settings"].return_value = mock_settings
            mock_matcher = mocks["Matcher"]
            mock_handler = mocks["OpenVINOModelHandler"]

            mock_model_instance = MagicMock()
            mock_matcher.return_value = mock_model_instance

            model_factory.create(mock_reference_batch, config)

            mock_matcher.assert_called_once_with(
                num_foreground_points=50,
                num_background_points=3,
                confidence_threshold=0.5,
                precision="fp32",
                device="cpu",
                use_mask_refinement=True,
                sam=SAMModelName.SAM_HQ_TINY,
                encoder_model="dinov3_small",
            )
            mock_handler.assert_called_once_with(mock_model_instance, mock_reference_batch, precision="fp32")

    def test_factory_creates_perdino_model_with_config(self, mock_reference_batch, mock_settings, model_factory):
        config = PerDinoConfig(
            sam_model=SAMModelName.SAM_HQ_TINY,
            encoder_model="dinov3_large",
            num_foreground_points=80,
            num_background_points=2,
            num_grid_cells=16,
            point_selection_threshold=0.65,
            confidence_threshold=0.42,
            precision="bf16",
        )

        with patch.multiple(
            "runtime.core.components.factories.model",
            get_settings=DEFAULT,
            PerDino=DEFAULT,
            TorchModelHandler=DEFAULT,
        ) as mocks:
            mocks["get_settings"].return_value = mock_settings
            mock_perdino = mocks["PerDino"]
            mock_handler = mocks["TorchModelHandler"]

            mock_model_instance = MagicMock()
            mock_perdino.return_value = mock_model_instance

            model_factory.create(mock_reference_batch, config)

            mock_perdino.assert_called_once_with(
                sam=SAMModelName.SAM_HQ_TINY,
                encoder_model="dinov3_large",
                num_foreground_points=80,
                num_background_points=2,
                num_grid_cells=16,
                point_selection_threshold=0.65,
                confidence_threshold=0.42,
                precision="bf16",
                device="cpu",
            )
            mock_handler.assert_called_once_with(mock_model_instance, mock_reference_batch)

    def test_factory_creates_softmatcher_model_with_config(self, mock_reference_batch, mock_settings, model_factory):
        config = SoftMatcherConfig(
            sam_model=SAMModelName.SAM_HQ_TINY,
            encoder_model="dinov3_large",
            num_foreground_points=40,
            num_background_points=2,
            confidence_threshold=0.42,
            use_sampling=True,
            use_spatial_sampling=True,
            approximate_matching=True,
            softmatching_score_threshold=0.5,
            softmatching_bidirectional=True,
            precision="bf16",
        )

        with patch.multiple(
            "runtime.core.components.factories.model",
            get_settings=DEFAULT,
            SoftMatcher=DEFAULT,
            TorchModelHandler=DEFAULT,
        ) as mocks:
            mocks["get_settings"].return_value = mock_settings
            mock_softmatcher = mocks["SoftMatcher"]
            mock_handler = mocks["TorchModelHandler"]

            mock_model_instance = MagicMock()
            mock_softmatcher.return_value = mock_model_instance

            model_factory.create(mock_reference_batch, config)

            mock_softmatcher.assert_called_once_with(
                sam=SAMModelName.SAM_HQ_TINY,
                encoder_model="dinov3_large",
                num_foreground_points=40,
                num_background_points=2,
                confidence_threshold=0.42,
                use_sampling=True,
                use_spatial_sampling=True,
                approximate_matching=True,
                softmatching_score_threshold=0.5,
                softmatching_bidirectional=True,
                precision="bf16",
                device="cpu",
            )
            mock_handler.assert_called_once_with(mock_model_instance, mock_reference_batch)

    def test_factory_returns_passthrough_for_none_reference_batch(self, model_factory):
        config = MatcherConfig(
            num_foreground_points=5,
            num_background_points=3,
            confidence_threshold=0.5,
            precision="fp32",
            sam_model=SAMModelName.SAM_HQ_TINY,
            encoder_model="dinov3_small",
        )

        result = model_factory.create(None, config)

        assert isinstance(result, PassThroughModelHandler)

    def test_factory_returns_passthrough_for_none_config(
        self, mock_reference_batch, mock_settings, model_factory, mock_device_resolver
    ):
        with patch.multiple("runtime.core.components.factories.model", get_settings=DEFAULT) as mocks:
            mocks["get_settings"].return_value = mock_settings
            result = model_factory.create(mock_reference_batch, None)

        assert isinstance(result, PassThroughModelHandler)
        mock_device_resolver.resolve_device.assert_not_called()

    def test_factory_returns_passthrough_when_both_none(self, model_factory):
        result = model_factory.create(None, None)

        assert isinstance(result, PassThroughModelHandler)

    def test_factory_returns_passthrough_when_inference_disabled(
        self, mock_reference_batch, model_factory, mock_device_resolver
    ):
        config = MatcherConfig(
            num_foreground_points=1,
            num_background_points=1,
            confidence_threshold=0.5,
            precision="fp32",
            sam_model=SAMModelName.SAM_HQ_TINY,
            encoder_model="dinov3_small",
        )
        mock_settings = MagicMock()
        mock_settings.processor_inference_enabled = False

        with (
            patch.multiple(
                "runtime.core.components.factories.model",
                get_settings=DEFAULT,
                TorchModelHandler=DEFAULT,
                Matcher=DEFAULT,
            ) as mocks,
        ):
            mocks["get_settings"].return_value = mock_settings
            mock_handler = mocks["TorchModelHandler"]
            mock_matcher = mocks["Matcher"]
            result = model_factory.create(mock_reference_batch, config)

        assert isinstance(result, PassThroughModelHandler)
        mock_handler.assert_not_called()
        mock_matcher.assert_not_called()
        mock_device_resolver.resolve_device.assert_not_called()

    @pytest.mark.parametrize(
        "config_class,model_patch_name",
        [
            (MatcherConfig, "Matcher"),
            (PerDinoConfig, "PerDino"),
            (SoftMatcherConfig, "SoftMatcher"),
        ],
    )
    def test_factory_returns_inference_handler_for_valid_configs(
        self, mock_reference_batch, mock_settings, model_factory, config_class, model_patch_name
    ):
        mock_settings.processor_openvino_enabled = config_class == MatcherConfig
        if config_class == MatcherConfig:
            config = MatcherConfig(
                num_foreground_points=5,
                num_background_points=3,
                confidence_threshold=0.5,
                sam_model=SAMModelName.SAM_HQ_TINY,
                encoder_model="dinov3_small",
            )
        elif config_class == PerDinoConfig:
            config = PerDinoConfig(
                num_foreground_points=80,
                num_background_points=2,
                sam_model=SAMModelName.SAM_HQ_TINY,
                encoder_model="dinov3_large",
            )
        else:
            config = SoftMatcherConfig(
                num_foreground_points=40,
                num_background_points=2,
                sam_model=SAMModelName.SAM_HQ_TINY,
                encoder_model="dinov3_large",
            )

        with patch.multiple(
            "runtime.core.components.factories.model",
            get_settings=DEFAULT,
            **{
                model_patch_name: DEFAULT,
                "TorchModelHandler": DEFAULT,
                "OpenVINOModelHandler": DEFAULT,
            },
        ) as mocks:
            mocks["get_settings"].return_value = mock_settings
            mock_torch_handler = mocks["TorchModelHandler"]
            mock_openvino_handler = mocks["OpenVINOModelHandler"]

            mock_torch_handler_instance = MagicMock()
            mock_openvino_handler_instance = MagicMock()
            mock_torch_handler.return_value = mock_torch_handler_instance
            mock_openvino_handler.return_value = mock_openvino_handler_instance

            result = model_factory.create(mock_reference_batch, config)

            if config_class == MatcherConfig:
                assert result is mock_openvino_handler_instance
                mock_openvino_handler.assert_called_once()
                mock_torch_handler.assert_not_called()
            else:
                assert result is mock_torch_handler_instance
                mock_torch_handler.assert_called_once()
                mock_openvino_handler.assert_not_called()
