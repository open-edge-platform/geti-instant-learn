# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for SAM3 OpenVINO export wiring."""

from pathlib import Path
from unittest.mock import MagicMock, patch

from instantlearn.models.sam3.sam3 import SAM3, CanvasConfig, Sam3PromptMode
from instantlearn.models.torch_base import ExportConfig
from instantlearn.scripts.sam3.export_sam3 import export_sam3_openvino


def test_export_sam3_openvino_runs_default_int8_sym_pipeline(tmp_path: Path) -> None:
    """The library export helper runs ONNX, IR conversion, INT8_SYM compression, and validation."""
    onnx_dir = tmp_path / "onnx"
    compressed_dir = tmp_path / "openvino-int8_sym"

    with (
        patch(
            "instantlearn.scripts.sam3.export_sam3.export_to_onnx",
            return_value=(onnx_dir, {"vision-encoder": onnx_dir / "vision-encoder.onnx"}),
        ) as mock_onnx,
        patch("instantlearn.scripts.sam3.export_sam3.convert_to_openvino") as mock_convert,
        patch(
            "instantlearn.scripts.sam3.export_sam3.apply_weight_compression",
            return_value=compressed_dir,
        ) as mock_compress,
        patch("instantlearn.scripts.sam3.export_sam3.validate_openvino_models") as mock_validate,
        patch("instantlearn.scripts.sam3.export_sam3._ensure_openvino_export_complete") as mock_ensure,
    ):
        result = export_sam3_openvino(
            model_id="facebook/sam3.1",
            output_dir=tmp_path,
            resolution=1008,
            validate=True,
            device="CPU",
        )

    assert result == compressed_dir
    mock_onnx.assert_called_once_with(
        model_id="facebook/sam3.1",
        output_dir=tmp_path,
        resolution=1008,
        opset_version=17,
    )
    mock_convert.assert_called_once_with(onnx_dir=onnx_dir, output_dir=tmp_path / "openvino-fp16", precision="fp16")
    mock_compress.assert_called_once_with(tmp_path / "openvino-fp16", tmp_path, "int8_sym")
    assert mock_ensure.call_count == 2
    mock_validate.assert_called_once_with(compressed_dir, device="CPU", resolution=1008)


def test_export_sam3_openvino_can_skip_compression(tmp_path: Path) -> None:
    """Passing compression_mode=None returns the uncompressed IR directory."""
    onnx_dir = tmp_path / "onnx"

    with (
        patch("instantlearn.scripts.sam3.export_sam3.export_to_onnx", return_value=(onnx_dir, {})),
        patch("instantlearn.scripts.sam3.export_sam3.convert_to_openvino"),
        patch("instantlearn.scripts.sam3.export_sam3.apply_weight_compression") as mock_compress,
        patch("instantlearn.scripts.sam3.export_sam3.validate_openvino_models") as mock_validate,
        patch("instantlearn.scripts.sam3.export_sam3._ensure_openvino_export_complete"),
    ):
        result = export_sam3_openvino(
            model_id="facebook/sam3.1",
            output_dir=tmp_path,
            precision="fp32",
            compression_mode=None,
        )

    assert result == tmp_path / "openvino-fp32"
    mock_compress.assert_not_called()
    mock_validate.assert_not_called()


def _sam3_stub() -> SAM3:
    """Create a SAM3 instance shell for export method tests."""
    model = object.__new__(SAM3)
    model.model_id = "facebook/sam3.1"
    model.resolution = 1008
    model.device = "cpu"
    model.confidence_threshold = 0.5
    model.prompt_mode = Sam3PromptMode.CLASSIC
    model.drop_spatial_bias = True
    model.canvas_config = CanvasConfig()
    return model


def test_sam3_export_delegates_to_openvino_helper(tmp_path: Path) -> None:
    """SAM3.export() writes the default INT8_SYM OpenVINO artifact bundle."""
    model = _sam3_stub()
    exported_dir = tmp_path / "openvino-int8_sym"

    with patch("instantlearn.scripts.sam3.export_sam3.export_sam3_openvino", return_value=exported_dir) as mock_export:
        result = SAM3.export(model, tmp_path)

    assert result == exported_dir
    mock_export.assert_called_once_with(
        model_id="facebook/sam3.1",
        output_dir=tmp_path,
        resolution=1008,
        precision="fp16",
        compression_mode="int8_sym",
    )


def test_sam3_to_openvino_exports_and_loads_sibling(tmp_path: Path) -> None:
    """SAM3.to_openvino() exports artifacts and constructs SAM3OpenVINO from them."""
    model = _sam3_stub()
    exported_dir = tmp_path / "openvino-fp32"
    ov_model = MagicMock()

    with (
        patch.object(SAM3, "_export_openvino", return_value=exported_dir) as mock_export,
        patch("instantlearn.models.sam3.sam3_openvino.SAM3OpenVINO", return_value=ov_model) as mock_ov_cls,
    ):
        result = SAM3.to_openvino(model, export_path=tmp_path, config=ExportConfig(precision="fp32", opset=18))

    assert result is ov_model
    mock_export.assert_called_once_with(
        output_dir=tmp_path,
        precision="fp32",
        opset_version=18,
        compression_mode=None,
    )
    mock_ov_cls.assert_called_once_with(
        model_dir=exported_dir,
        model_id="facebook/sam3.1",
        device="cpu",
        confidence_threshold=0.5,
        resolution=1008,
        prompt_mode=Sam3PromptMode.CLASSIC,
        drop_spatial_bias=True,
        canvas_config=model.canvas_config,
    )
