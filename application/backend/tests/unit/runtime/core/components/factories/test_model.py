#  Copyright (C) 2025 Intel Corporation
#  SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from unittest.mock import DEFAULT, MagicMock, patch

import pytest
from instantlearn.models.sam3.sam3 import MODEL_NAMES as SAM3_MODEL_NAMES
from instantlearn.utils.constants import CompressionMode, SAMModelName

from domain.services.schemas.device import DeviceInfo, DeviceType
from domain.services.schemas.processor import (
    CompressionPreset,
    MatcherConfig,
    PerDinoConfig,
    Sam3Config,
    SoftMatcherConfig,
)
from runtime.core.components.factories.model import _IR_COMPLETE_MARKER, ModelFactory, _sam3_ir_complete
from runtime.core.components.models.inference_model import InferenceModelHandler
from runtime.core.components.models.passthrough_model import PassThroughModelHandler

FACTORY_MODULE = "runtime.core.components.factories.model"


def _cpu() -> DeviceInfo:
    return DeviceInfo(type=DeviceType.CPU, name="CPU", memory=None, index=None)


def _write_sam3_ir(ir_dir: Path, *, complete: bool = True) -> Path:
    """Create a SAM3 IR directory, published (marker present) by default."""
    ir_dir.mkdir(parents=True, exist_ok=True)
    for name in SAM3_MODEL_NAMES:
        (ir_dir / f"{name}.xml").write_text("<net/>")
        (ir_dir / f"{name}.bin").write_bytes(b"\x00")
    (ir_dir / "tokenizer.json").write_text("{}")
    if complete:
        (ir_dir / _IR_COMPLETE_MARKER).touch()
    return ir_dir


def _sam3_cache_dir(settings, compression: CompressionMode = CompressionMode.INT8_SYM) -> Path:
    """Return the published cache path the factory uses for the default SAM3 config."""
    return Path(settings.ir_cache_dir) / "sam3-facebook-sam3.1-r1008" / f"openvino-{compression.value}"


def _fake_export(export_root, export_config) -> Path:
    """Stand in for ``SAM3.to_openvino``, writing an IR into the staging directory."""
    return _write_sam3_ir(Path(export_root) / f"openvino-{export_config.compression.value}", complete=False)


def _fake_export_with_intermediates(export_root, export_config) -> Path:
    """Like :func:`_fake_export` but also leaves the library's intermediate artefacts."""
    export_root = Path(export_root)
    (export_root / "onnx").mkdir(parents=True, exist_ok=True)
    (export_root / "onnx" / "vision_encoder.onnx").write_bytes(b"\x00")
    (export_root / f"openvino-{export_config.compression.value}-fp16-source").mkdir(parents=True, exist_ok=True)
    return _fake_export(export_root, export_config)


class TestModelFactory:
    @pytest.fixture
    def mock_reference_batch(self):
        batch = MagicMock()
        batch.samples = [MagicMock(bboxes=None)]
        return batch

    @pytest.fixture
    def mock_settings(self, tmp_path):
        settings = MagicMock()
        settings.processor_inference_enabled = True
        settings.processor_openvino_enabled = False
        settings.ir_cache_dir = tmp_path / "ir-cache"
        return settings

    @pytest.fixture
    def mock_device_service(self):
        service = MagicMock()
        service.resolve.return_value = _cpu()
        service.resolve_auto.return_value = _cpu()
        return service

    @pytest.fixture
    def model_factory(self, mock_device_service):
        return ModelFactory(device_service=mock_device_service)

    # --- pass-through branches ---

    def test_factory_returns_passthrough_for_none_reference_batch(self, model_factory):
        config = MatcherConfig(sam_model=SAMModelName.SAM_HQ_TINY, encoder_model="dinov3_small")

        result = model_factory.create(None, config)

        assert isinstance(result, PassThroughModelHandler)

    def test_factory_returns_passthrough_for_none_config(
        self, mock_reference_batch, mock_settings, model_factory, mock_device_service
    ):
        with patch(f"{FACTORY_MODULE}.get_settings", return_value=mock_settings):
            result = model_factory.create(mock_reference_batch, None)

        assert isinstance(result, PassThroughModelHandler)
        mock_device_service.resolve.assert_not_called()

    def test_factory_returns_passthrough_when_both_none(self, model_factory):
        assert isinstance(model_factory.create(None, None), PassThroughModelHandler)

    def test_factory_returns_passthrough_when_inference_disabled(
        self, mock_reference_batch, mock_settings, model_factory, mock_device_service
    ):
        mock_settings.processor_inference_enabled = False
        config = MatcherConfig(sam_model=SAMModelName.SAM_HQ_TINY, encoder_model="dinov3_small")

        with patch.multiple(FACTORY_MODULE, get_settings=DEFAULT, Matcher=DEFAULT) as mocks:
            mocks["get_settings"].return_value = mock_settings
            result = model_factory.create(mock_reference_batch, config)

        assert isinstance(result, PassThroughModelHandler)
        mocks["Matcher"].assert_not_called()
        mock_device_service.resolve.assert_not_called()

    # --- device resolution ---

    @pytest.mark.parametrize(
        ("resolved_device", "use_openvino", "expected_precision"),
        [
            (DeviceInfo(type=DeviceType.CUDA, name="NVIDIA", memory=1, index=0), False, "bf16"),
            (_cpu(), False, "bf16"),
            # OpenVINO always traces in fp32 regardless of the configured precision
            (DeviceInfo(type=DeviceType.XPU, name="Intel", memory=1, index=0), True, "fp32"),
        ],
    )
    def test_factory_uses_resolved_device_and_precision(
        self,
        mock_reference_batch,
        mock_settings,
        model_factory,
        mock_device_service,
        resolved_device,
        use_openvino,
        expected_precision,
    ):
        config = MatcherConfig(precision="bf16", sam_model=SAMModelName.SAM_HQ_TINY, encoder_model="dinov3_small")
        mock_settings.processor_openvino_enabled = use_openvino
        mock_device_service.resolve.return_value = resolved_device

        with patch.multiple(FACTORY_MODULE, get_settings=DEFAULT, Matcher=DEFAULT, MatcherOpenVINO=DEFAULT) as mocks:
            mocks["get_settings"].return_value = mock_settings
            mocks["MatcherOpenVINO"].__name__ = "MatcherOpenVINO"

            result = model_factory.create(mock_reference_batch, config)

        mock_device_service.resolve.assert_called_once_with("auto")
        assert mocks["Matcher"].call_args.kwargs["device"] == resolved_device.as_torch
        assert mocks["Matcher"].call_args.kwargs["precision"] == expected_precision
        assert isinstance(result, InferenceModelHandler)

    # --- torch backend ---

    def test_factory_creates_matcher_with_config_and_fits_it(self, mock_reference_batch, mock_settings, model_factory):
        config = MatcherConfig(
            num_foreground_points=50,
            num_background_points=3,
            confidence_threshold=0.5,
            precision="fp32",
            sam_model=SAMModelName.SAM_HQ_TINY,
            encoder_model="dinov3_small",
            use_mask_refinement=True,
        )

        with patch.multiple(FACTORY_MODULE, get_settings=DEFAULT, Matcher=DEFAULT) as mocks:
            mocks["get_settings"].return_value = mock_settings
            model_factory.create(mock_reference_batch, config)

        mocks["Matcher"].assert_called_once_with(
            num_foreground_points=50,
            num_background_points=3,
            confidence_threshold=0.5,
            precision="fp32",
            device="cpu",
            use_mask_refinement=True,
            similarity_threshold=None,
            num_grid_cells=8,
            sam=SAMModelName.SAM_HQ_TINY,
            encoder_model="dinov3_small",
        )
        mocks["Matcher"].return_value.fit.assert_called_once_with(mock_reference_batch)

    def test_factory_creates_perdino_with_config(self, mock_reference_batch, mock_settings, model_factory):
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

        with patch.multiple(FACTORY_MODULE, get_settings=DEFAULT, PerDino=DEFAULT) as mocks:
            mocks["get_settings"].return_value = mock_settings
            result = model_factory.create(mock_reference_batch, config)

        mocks["PerDino"].assert_called_once_with(
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
        mocks["PerDino"].return_value.fit.assert_called_once_with(mock_reference_batch)
        assert isinstance(result, InferenceModelHandler)

    def test_factory_creates_softmatcher_with_config(self, mock_reference_batch, mock_settings, model_factory):
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

        with patch.multiple(FACTORY_MODULE, get_settings=DEFAULT, SoftMatcher=DEFAULT) as mocks:
            mocks["get_settings"].return_value = mock_settings
            model_factory.create(mock_reference_batch, config)

        mocks["SoftMatcher"].assert_called_once_with(
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
        mocks["SoftMatcher"].return_value.fit.assert_called_once_with(mock_reference_batch)

    def test_factory_wraps_model_in_inference_handler(self, mock_reference_batch, mock_settings, model_factory):
        config = MatcherConfig(sam_model=SAMModelName.SAM_HQ_TINY, encoder_model="dinov3_small")

        with patch.multiple(FACTORY_MODULE, get_settings=DEFAULT, Matcher=DEFAULT) as mocks:
            mocks["get_settings"].return_value = mock_settings
            result = model_factory.create(mock_reference_batch, config)

        assert isinstance(result, InferenceModelHandler)
        assert result._model is mocks["Matcher"].return_value

    # --- OpenVINO backend ---

    @pytest.mark.parametrize(
        ("config_factory", "torch_name", "ov_name"),
        [
            (
                lambda: MatcherConfig(sam_model=SAMModelName.SAM_HQ_TINY, encoder_model="dinov3_small"),
                "Matcher",
                "MatcherOpenVINO",
            ),
            (
                lambda: PerDinoConfig(sam_model=SAMModelName.SAM_HQ_TINY, encoder_model="dinov3_small"),
                "PerDino",
                "PerDinoOpenVINO",
            ),
            (
                lambda: SoftMatcherConfig(sam_model=SAMModelName.SAM_HQ_TINY, encoder_model="dinov3_small"),
                "SoftMatcher",
                "SoftMatcherOpenVINO",
            ),
        ],
    )
    def test_factory_exports_and_loads_openvino_sibling(
        self, mock_reference_batch, mock_settings, model_factory, config_factory, torch_name, ov_name
    ):
        mock_settings.processor_openvino_enabled = True
        config = config_factory()

        with patch.multiple(FACTORY_MODULE, get_settings=DEFAULT, **{torch_name: DEFAULT, ov_name: DEFAULT}) as mocks:
            mocks["get_settings"].return_value = mock_settings
            mocks[ov_name].__name__ = ov_name
            torch_model = mocks[torch_name].return_value

            result = model_factory.create(mock_reference_batch, config)

            # The torch model is fitted, moved to CPU, and exported...
            torch_model.fit.assert_called_once_with(mock_reference_batch)
            torch_model.cpu.assert_called_once()
            torch_model.to_openvino.assert_called_once()
            # ...and the OpenVINO sibling loads the resulting IR directory.
            mocks[ov_name].assert_called_once_with(
                model_dir=torch_model.to_openvino.return_value,
                device="CPU",
            )
            assert isinstance(result, InferenceModelHandler)
            assert result._model is mocks[ov_name].return_value

    @pytest.mark.parametrize(
        ("preset", "expected_compression"),
        [
            (CompressionPreset.THROUGHPUT, CompressionMode.INT8_SYM),
            (CompressionPreset.ACCURACY, CompressionMode.FP16),
        ],
    )
    def test_factory_passes_preset_compression_to_export(
        self, mock_reference_batch, mock_settings, model_factory, preset, expected_compression
    ):
        mock_settings.processor_openvino_enabled = True
        config = MatcherConfig(sam_model=SAMModelName.SAM_HQ_TINY, encoder_model="dinov3_small", preset=preset)

        with patch.multiple(FACTORY_MODULE, get_settings=DEFAULT, Matcher=DEFAULT, MatcherOpenVINO=DEFAULT) as mocks:
            mocks["get_settings"].return_value = mock_settings
            mocks["MatcherOpenVINO"].__name__ = "MatcherOpenVINO"
            model_factory.create(mock_reference_batch, config)

            export_config = mocks["Matcher"].return_value.to_openvino.call_args[0][1]
            assert export_config.compression == expected_compression

    # --- SAM3 ---

    def test_factory_creates_sam3_torch_in_classic_mode_without_bboxes(
        self, mock_reference_batch, mock_settings, model_factory
    ):
        config = Sam3Config(confidence_threshold=0.5, resolution=1008, precision="fp32")

        with patch.multiple(FACTORY_MODULE, get_settings=DEFAULT, SAM3=DEFAULT, Sam3PromptMode=DEFAULT) as mocks:
            mocks["get_settings"].return_value = mock_settings
            model_factory.create(mock_reference_batch, config)

        assert mocks["SAM3"].call_args.kwargs["prompt_mode"] == mocks["Sam3PromptMode"].CLASSIC
        assert mocks["SAM3"].call_args.kwargs["device"] == "cpu"
        mocks["SAM3"].return_value.fit.assert_called_once_with(mock_reference_batch)

    def test_factory_creates_sam3_torch_in_canvas_mode_with_bboxes(self, mock_settings, model_factory):
        reference_batch = MagicMock()
        reference_batch.samples = [MagicMock(bboxes=[[0, 0, 1, 1]])]
        config = Sam3Config()

        with patch.multiple(FACTORY_MODULE, get_settings=DEFAULT, SAM3=DEFAULT, Sam3PromptMode=DEFAULT) as mocks:
            mocks["get_settings"].return_value = mock_settings
            model_factory.create(reference_batch, config)

        assert mocks["SAM3"].call_args.kwargs["prompt_mode"] == mocks["Sam3PromptMode"].CANVAS

    def test_factory_exports_sam3_ir_on_cache_miss(self, mock_reference_batch, mock_settings, model_factory):
        mock_settings.processor_openvino_enabled = True
        config = Sam3Config(resolution=1008)
        expected_dir = _sam3_cache_dir(mock_settings)

        with patch.multiple(FACTORY_MODULE, get_settings=DEFAULT, SAM3=DEFAULT, SAM3OpenVINO=DEFAULT) as mocks:
            mocks["get_settings"].return_value = mock_settings
            exporter = mocks["SAM3"].return_value
            exporter.to_openvino.side_effect = _fake_export

            result = model_factory.create(mock_reference_batch, config)

            # A throwaway torch SAM3 is built solely to produce the IR.
            exporter.to_openvino.assert_called_once()
            exporter.fit.assert_not_called()
            mocks["SAM3OpenVINO"].assert_called_once()
            # The IR is published into the cache, not loaded from the staging directory.
            assert mocks["SAM3OpenVINO"].call_args.kwargs["model_dir"] == expected_dir
            mocks["SAM3OpenVINO"].return_value.fit.assert_called_once_with(mock_reference_batch)
            assert isinstance(result, InferenceModelHandler)

        assert _sam3_ir_complete(expected_dir)

    def test_factory_exports_sam3_on_cpu(self, mock_reference_batch, mock_settings, model_factory):
        mock_settings.processor_openvino_enabled = True
        config = Sam3Config(resolution=1008)

        with patch.multiple(FACTORY_MODULE, get_settings=DEFAULT, SAM3=DEFAULT, SAM3OpenVINO=DEFAULT) as mocks:
            mocks["get_settings"].return_value = mock_settings
            mocks["SAM3"].return_value.to_openvino.side_effect = _fake_export

            model_factory.create(mock_reference_batch, config)

        # Tracing runs on CPU regardless of the accelerator chosen for inference.
        assert mocks["SAM3"].call_args.kwargs["device"] == "cpu"

    def test_factory_removes_staging_directory_after_export(self, mock_reference_batch, mock_settings, model_factory):
        mock_settings.processor_openvino_enabled = True
        config = Sam3Config(resolution=1008)

        with patch.multiple(FACTORY_MODULE, get_settings=DEFAULT, SAM3=DEFAULT, SAM3OpenVINO=DEFAULT) as mocks:
            mocks["get_settings"].return_value = mock_settings
            mocks["SAM3"].return_value.to_openvino.side_effect = _fake_export_with_intermediates

            model_factory.create(mock_reference_batch, config)

        # Only the model-keyed cache entry survives; ONNX dumps and fp16 sources are gone.
        cache_root = Path(mock_settings.ir_cache_dir)
        assert [p.name for p in cache_root.iterdir()] == ["sam3-facebook-sam3.1-r1008"]

    def test_factory_reuses_cached_sam3_ir(self, mock_reference_batch, mock_settings, model_factory):
        mock_settings.processor_openvino_enabled = True
        config = Sam3Config(resolution=1008)
        cached_dir = _write_sam3_ir(_sam3_cache_dir(mock_settings))

        with patch.multiple(FACTORY_MODULE, get_settings=DEFAULT, SAM3=DEFAULT, SAM3OpenVINO=DEFAULT) as mocks:
            mocks["get_settings"].return_value = mock_settings

            model_factory.create(mock_reference_batch, config)

            # No export at all: the torch SAM3 is never constructed.
            mocks["SAM3"].assert_not_called()
            assert mocks["SAM3OpenVINO"].call_args.kwargs["model_dir"] == cached_dir

    def test_factory_re_exports_when_completion_marker_is_missing(
        self, mock_reference_batch, mock_settings, model_factory
    ):
        mock_settings.processor_openvino_enabled = True
        config = Sam3Config(resolution=1008)
        # Every file is present but the export never finished: must not be trusted.
        _write_sam3_ir(_sam3_cache_dir(mock_settings), complete=False)

        with patch.multiple(FACTORY_MODULE, get_settings=DEFAULT, SAM3=DEFAULT, SAM3OpenVINO=DEFAULT) as mocks:
            mocks["get_settings"].return_value = mock_settings
            mocks["SAM3"].return_value.to_openvino.side_effect = _fake_export

            model_factory.create(mock_reference_batch, config)

            mocks["SAM3"].return_value.to_openvino.assert_called_once()

    def test_factory_cleans_up_partial_sam3_ir_on_export_failure(
        self, mock_reference_batch, mock_settings, model_factory
    ):
        mock_settings.processor_openvino_enabled = True
        config = Sam3Config(resolution=1008)
        ir_dir = _write_sam3_ir(_sam3_cache_dir(mock_settings))

        with patch.multiple(FACTORY_MODULE, get_settings=DEFAULT, SAM3=DEFAULT, SAM3OpenVINO=DEFAULT) as mocks:
            mocks["get_settings"].return_value = mock_settings
            mocks["SAM3"].return_value.to_openvino.side_effect = RuntimeError("export blew up")
            # Force a cache miss by removing one required sub-model.
            (ir_dir / f"{SAM3_MODEL_NAMES[0]}.xml").unlink()

            with pytest.raises(RuntimeError, match="export blew up"):
                model_factory.create(mock_reference_batch, config)

        assert not ir_dir.exists(), "a half-written IR directory must not be left behind"
        # The staging directory must not survive the failure either.
        assert list(Path(mock_settings.ir_cache_dir).glob("*openvino*")) == []

    def test_factory_rejects_incomplete_export(self, mock_reference_batch, mock_settings, model_factory):
        mock_settings.processor_openvino_enabled = True
        config = Sam3Config(resolution=1008)

        def _truncated_export(export_root, export_config):
            ir_dir = _fake_export(export_root, export_config)
            (ir_dir / f"{SAM3_MODEL_NAMES[0]}.bin").write_bytes(b"")
            return ir_dir

        with patch.multiple(FACTORY_MODULE, get_settings=DEFAULT, SAM3=DEFAULT, SAM3OpenVINO=DEFAULT) as mocks:
            mocks["get_settings"].return_value = mock_settings
            mocks["SAM3"].return_value.to_openvino.side_effect = _truncated_export

            with pytest.raises(FileNotFoundError, match="Incomplete SAM3 OpenVINO export"):
                model_factory.create(mock_reference_batch, config)

        assert not _sam3_cache_dir(mock_settings).exists()


class TestSam3IrComplete:
    def test_missing_directory_is_incomplete(self, tmp_path):
        assert _sam3_ir_complete(tmp_path / "nope") is False

    def test_complete_directory_is_complete(self, tmp_path):
        _write_sam3_ir(tmp_path)
        assert _sam3_ir_complete(tmp_path) is True

    def test_missing_marker_is_incomplete(self, tmp_path):
        _write_sam3_ir(tmp_path, complete=False)
        assert _sam3_ir_complete(tmp_path) is False

    def test_empty_file_is_incomplete(self, tmp_path):
        _write_sam3_ir(tmp_path)
        (tmp_path / f"{SAM3_MODEL_NAMES[1]}.bin").write_bytes(b"")
        assert _sam3_ir_complete(tmp_path) is False

    def test_missing_submodel_is_incomplete(self, tmp_path):
        _write_sam3_ir(tmp_path)
        (tmp_path / f"{SAM3_MODEL_NAMES[2]}.xml").unlink()
        assert _sam3_ir_complete(tmp_path) is False
