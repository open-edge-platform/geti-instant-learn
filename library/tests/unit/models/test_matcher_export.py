# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for Matcher.export() method."""

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import torch

from instantlearn.components.feature_extractors import ReferenceFeatures
from instantlearn.models.matcher import Matcher
from instantlearn.utils.constants import Backend


class TestMatcherExport:
    """Unit tests for Matcher.export() method."""

    @pytest.fixture
    def mock_reference_features(self) -> ReferenceFeatures:
        """Create mock reference features for testing."""
        num_categories = 2
        num_patches = 1024
        embed_dim = 384

        return ReferenceFeatures(
            ref_embeddings=torch.randn(num_categories, num_patches, embed_dim),
            masked_ref_embeddings=torch.randn(num_categories, embed_dim),
            flatten_ref_masks=torch.randn(num_categories, num_patches),
            category_ids=[1, 2],
        )

    @pytest.fixture
    def mock_components(self) -> dict[str, Any]:
        """Create mock components for Matcher."""
        return {
            "sam_predictor": MagicMock(),
            "encoder": MagicMock(),
            "masked_feature_extractor": MagicMock(),
            "prompt_generator": MagicMock(),
            "segmenter": MagicMock(),
        }

    @patch("instantlearn.models.matcher.load_sam_model")
    @patch("instantlearn.models.matcher.ImageEncoder")
    def test_export_creates_directory(
        self,
        mock_image_encoder: MagicMock,
        mock_sam_predictor: MagicMock,
        mock_components: dict[str, Any],
        mock_reference_features: ReferenceFeatures,
        tmp_path: Path,
    ) -> None:
        """Test that export() creates the export directory if it doesn't exist."""
        mock_sam_predictor.return_value = mock_components["sam_predictor"]
        mock_image_encoder.return_value = mock_components["encoder"]
        mock_components["encoder"].input_size = 512

        model = Matcher(device="cpu")
        model.encoder = mock_components["encoder"]

        export_dir = tmp_path / "new_export_dir"
        assert not export_dir.exists()

        # Mock torch.onnx.export to avoid actual export
        with patch("torch.onnx.export"):
            model.export(
                reference_features=mock_reference_features,
                export_dir=export_dir,
                backend=Backend.ONNX,
            )

        assert export_dir.exists()

    @patch("instantlearn.models.matcher.load_sam_model")
    @patch("instantlearn.models.matcher.ImageEncoder")
    def test_export_onnx_returns_correct_path(
        self,
        mock_image_encoder: MagicMock,
        mock_sam_predictor: MagicMock,
        mock_components: dict[str, Any],
        mock_reference_features: ReferenceFeatures,
        tmp_path: Path,
    ) -> None:
        """Test that export() with ONNX backend returns correct path."""
        mock_sam_predictor.return_value = mock_components["sam_predictor"]
        mock_image_encoder.return_value = mock_components["encoder"]
        mock_components["encoder"].input_size = 512

        model = Matcher(device="cpu")
        model.encoder = mock_components["encoder"]

        with patch("torch.onnx.export"):
            result = model.export(
                reference_features=mock_reference_features,
                export_dir=tmp_path,
                backend=Backend.ONNX,
            )

        assert result == tmp_path / "matcher.onnx"

    @patch("instantlearn.models.matcher.load_sam_model")
    @patch("instantlearn.models.matcher.ImageEncoder")
    def test_export_openvino_returns_correct_path(
        self,
        mock_image_encoder: MagicMock,
        mock_sam_predictor: MagicMock,
        mock_components: dict[str, Any],
        mock_reference_features: ReferenceFeatures,
        tmp_path: Path,
    ) -> None:
        """Test that export() with OpenVINO backend returns correct path."""
        mock_sam_predictor.return_value = mock_components["sam_predictor"]
        mock_image_encoder.return_value = mock_components["encoder"]
        mock_components["encoder"].input_size = 512

        model = Matcher(device="cpu")
        model.encoder = mock_components["encoder"]

        with (
            patch("torch.onnx.export"),
            patch("openvino.convert_model") as mock_convert,
            patch("openvino.save_model"),
        ):
            mock_convert.return_value = MagicMock()
            result = model.export(
                reference_features=mock_reference_features,
                export_dir=tmp_path,
                backend=Backend.OPENVINO,
            )

        assert result == tmp_path / "matcher.xml"

    @patch("instantlearn.models.matcher.load_sam_model")
    @patch("instantlearn.models.matcher.ImageEncoder")
    def test_export_onnx_calls_torch_export(
        self,
        mock_image_encoder: MagicMock,
        mock_sam_predictor: MagicMock,
        mock_components: dict[str, Any],
        mock_reference_features: ReferenceFeatures,
        tmp_path: Path,
    ) -> None:
        """Test that export() with ONNX backend calls torch.onnx.export."""
        mock_sam_predictor.return_value = mock_components["sam_predictor"]
        mock_image_encoder.return_value = mock_components["encoder"]
        mock_components["encoder"].input_size = 512

        model = Matcher(device="cpu")
        model.encoder = mock_components["encoder"]

        with patch("torch.onnx.export") as mock_export:
            model.export(
                reference_features=mock_reference_features,
                export_dir=tmp_path,
                backend=Backend.ONNX,
            )

        mock_export.assert_called_once()
        call_kwargs = mock_export.call_args.kwargs
        assert call_kwargs["input_names"] == ["target_image"]
        assert call_kwargs["output_names"] == ["masks", "scores", "labels"]
        assert "dynamic_axes" in call_kwargs

    @patch("instantlearn.models.matcher.load_sam_model")
    @patch("instantlearn.models.matcher.ImageEncoder")
    def test_export_openvino_calls_convert_model(
        self,
        mock_image_encoder: MagicMock,
        mock_sam_predictor: MagicMock,
        mock_components: dict[str, Any],
        mock_reference_features: ReferenceFeatures,
        tmp_path: Path,
    ) -> None:
        """Test that export() with OpenVINO backend calls openvino.convert_model."""
        mock_sam_predictor.return_value = mock_components["sam_predictor"]
        mock_image_encoder.return_value = mock_components["encoder"]
        mock_components["encoder"].input_size = 512

        model = Matcher(device="cpu")
        model.encoder = mock_components["encoder"]

        with (
            patch("torch.onnx.export"),
            patch("openvino.convert_model") as mock_convert,
            patch("openvino.save_model") as mock_save,
        ):
            mock_convert.return_value = MagicMock()
            model.export(
                reference_features=mock_reference_features,
                export_dir=tmp_path,
                backend=Backend.OPENVINO,
            )

        mock_convert.assert_called_once()
        mock_save.assert_called_once()

    @patch("instantlearn.models.matcher.load_sam_model")
    @patch("instantlearn.models.matcher.ImageEncoder")
    def test_export_string_path_converted_to_path(
        self,
        mock_image_encoder: MagicMock,
        mock_sam_predictor: MagicMock,
        mock_components: dict[str, Any],
        mock_reference_features: ReferenceFeatures,
        tmp_path: Path,
    ) -> None:
        """Test that export() handles string paths correctly."""
        mock_sam_predictor.return_value = mock_components["sam_predictor"]
        mock_image_encoder.return_value = mock_components["encoder"]
        mock_components["encoder"].input_size = 512

        model = Matcher(device="cpu")
        model.encoder = mock_components["encoder"]

        str_path = str(tmp_path / "string_path_export")

        with patch("torch.onnx.export"):
            result = model.export(
                reference_features=mock_reference_features,
                export_dir=str_path,
                backend=Backend.ONNX,
            )

        assert isinstance(result, Path)
        assert Path(str_path).exists()

    @patch("instantlearn.models.matcher.load_sam_model")
    @patch("instantlearn.models.matcher.ImageEncoder")
    def test_export_default_backend_is_onnx(
        self,
        mock_image_encoder: MagicMock,
        mock_sam_predictor: MagicMock,
        mock_components: dict[str, Any],
        mock_reference_features: ReferenceFeatures,
        tmp_path: Path,
    ) -> None:
        """Test that export() uses ONNX backend by default."""
        mock_sam_predictor.return_value = mock_components["sam_predictor"]
        mock_image_encoder.return_value = mock_components["encoder"]
        mock_components["encoder"].input_size = 512

        model = Matcher(device="cpu")
        model.encoder = mock_components["encoder"]

        with patch("torch.onnx.export"):
            result = model.export(
                reference_features=mock_reference_features,
                export_dir=tmp_path,
            )

        # Default backend should be ONNX, returning .onnx file
        assert result.suffix == ".onnx"

    @patch("instantlearn.models.matcher.load_sam_model")
    @patch("instantlearn.models.matcher.ImageEncoder")
    def test_export_unsupported_backend_returns_export_dir(
        self,
        mock_image_encoder: MagicMock,
        mock_sam_predictor: MagicMock,
        mock_components: dict[str, Any],
        mock_reference_features: ReferenceFeatures,
        tmp_path: Path,
    ) -> None:
        """Test that export() returns export directory for unsupported backends."""
        mock_sam_predictor.return_value = mock_components["sam_predictor"]
        mock_image_encoder.return_value = mock_components["encoder"]
        mock_components["encoder"].input_size = 512

        model = Matcher(device="cpu")
        model.encoder = mock_components["encoder"]

        result = model.export(
            reference_features=mock_reference_features,
            export_dir=tmp_path,
            backend=Backend.TENSORRT,  # Unsupported backend
        )

        assert result == tmp_path
