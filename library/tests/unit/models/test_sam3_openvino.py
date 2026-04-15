# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for SAM3OpenVINO model.

These tests mock OpenVINO compiled models, tokenizer, and preprocessors
to validate SAM3OpenVINO logic without loading real weights.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from instantlearn.data.base.sample import Sample
from instantlearn.models.sam3.sam3 import Sam3PromptMode
from instantlearn.models.sam3.sam3_openvino import SAM3OpenVINO, _find_model_file, _require_model_file  # noqa: PLC2701

if TYPE_CHECKING:
    from collections.abc import Callable


_RNG = np.random.default_rng(42)


def _make_vision_result() -> MagicMock:
    """Create a mock vision encoder output."""
    result = MagicMock()
    result.__getitem__ = MagicMock(
        side_effect=lambda k: {
            "fpn_feat_0": _RNG.standard_normal((1, 256, 4, 4)).astype(np.float32),
            "fpn_feat_1": _RNG.standard_normal((1, 256, 2, 2)).astype(np.float32),
            "fpn_feat_2": _RNG.standard_normal((1, 256, 1, 1)).astype(np.float32),
            "fpn_pos_2": _RNG.standard_normal((1, 256, 1, 1)).astype(np.float32),
        }[k],
    )
    return result


def _make_text_result() -> MagicMock:
    """Create a mock text encoder output."""
    result = MagicMock()
    result.__getitem__ = MagicMock(
        side_effect=lambda k: {
            "text_features": _RNG.standard_normal((1, 32, 256)).astype(np.float32),
            "text_mask": np.ones((1, 32), dtype=bool),
        }[k],
    )
    return result


def _make_geometry_result(num_prompts: int = 2) -> MagicMock:
    """Create a mock geometry encoder output."""
    result = MagicMock()
    result.__getitem__ = MagicMock(
        side_effect=lambda k: {
            "geometry_features": _RNG.standard_normal((1, num_prompts, 256)).astype(np.float32),
            "geometry_mask": np.ones((1, num_prompts), dtype=bool),
        }[k],
    )
    return result


def _make_decoder_result() -> MagicMock:
    """Create a mock (prompt) decoder output."""
    num_queries = 200
    h_feat, w_feat = 63, 63
    result = MagicMock()
    result.__getitem__ = MagicMock(
        side_effect=lambda k: {
            "pred_masks": _RNG.standard_normal((1, num_queries, h_feat, w_feat)).astype(np.float32),
            "pred_boxes": _RNG.random((1, num_queries, 4)).astype(np.float32),
            "pred_logits": _RNG.standard_normal((1, num_queries)).astype(np.float32),
            "presence_logits": _RNG.standard_normal((1, 1)).astype(np.float32),
        }[k],
    )
    return result


def _make_compiled_model(result_factory: Callable[[], MagicMock]) -> MagicMock:
    """Create a mock ov.CompiledModel that returns results from a factory."""
    model = MagicMock()
    model.side_effect = lambda _inputs: result_factory()

    # Mock create_infer_request() for the InferRequest-based runner path
    def _create_infer_request() -> MagicMock:
        request = MagicMock()
        result_holder: list[MagicMock] = []

        def _infer(_inputs: list) -> None:
            result_holder.clear()
            result_holder.append(result_factory())

        def _get_tensor(name: str) -> MagicMock:
            tensor_mock = MagicMock()
            tensor_mock.data = result_holder[0][name]
            return tensor_mock

        request.infer = MagicMock(side_effect=_infer)
        request.get_tensor = MagicMock(side_effect=_get_tensor)
        return request

    model.create_infer_request = _create_infer_request
    return model


def _make_mock_tokenizer() -> MagicMock:
    """Create a mock CLIPTokenizerFast."""
    tokenizer = MagicMock()
    tokenizer.return_value = MagicMock(
        input_ids=np.ones((1, 32), dtype=np.int64),
        attention_mask=np.ones((1, 32), dtype=np.int64),
    )
    return tokenizer


def _make_mock_preprocessor() -> MagicMock:
    """Create a mock Sam3Preprocessor."""
    pre = MagicMock()
    pre.return_value = (
        torch.randn(1, 3, 1008, 1008),
        torch.tensor([[224, 224]], dtype=torch.int32),
    )
    return pre


def _make_mock_prompt_preprocessor() -> MagicMock:
    """Create a mock Sam3PromptPreprocessor."""
    pre = MagicMock()
    pre.return_value = (
        torch.tensor([[[0.3, 0.3, 0.1, 0.1]]]),
        torch.tensor([[[0.5, 0.5]]]),
    )
    return pre


def _make_mock_postprocessor() -> MagicMock:
    """Create a mock Sam3Postprocessor."""
    post = MagicMock()
    post.return_value = [
        {
            "scores": torch.tensor([0.9, 0.8]),
            "boxes": torch.tensor([[10, 10, 50, 50], [60, 60, 100, 100]], dtype=torch.float32),
            "masks": torch.ones(2, 224, 224, dtype=torch.int64),
        },
    ]
    return post


@pytest.fixture
def tmp_model_dir(tmp_path: Path) -> Path:
    """Create a temp directory with dummy model files."""
    for name in [
        "vision-encoder.xml",
        "text-encoder.xml",
        "geometry-encoder.xml",
        "geometry-encoder-exemplar.xml",
        "prompt-decoder.xml",
    ]:
        (tmp_path / name).touch()
    return tmp_path


def _build_sam3_openvino(
    model_dir: Path,
    prompt_mode: Sam3PromptMode = Sam3PromptMode.CLASSIC,
    *,
    geo_prompts: int = 2,
) -> SAM3OpenVINO:
    """Build SAM3OpenVINO with mocked internals.

    Args:
        model_dir: Path to temp model directory.
        prompt_mode: Prompt mode.
        geo_prompts: Number of geometry prompts for mock output shape.

    Returns:
        Mocked SAM3OpenVINO instance.
    """
    mock_core = MagicMock()
    vision_cm = _make_compiled_model(_make_vision_result)
    text_cm = _make_compiled_model(_make_text_result)
    decoder_cm = _make_compiled_model(_make_decoder_result)
    geo_cm = _make_compiled_model(lambda: _make_geometry_result(geo_prompts))
    geo_ex_cm = _make_compiled_model(lambda: _make_geometry_result(geo_prompts))

    def compile_model_side_effect(path: str | Path, _device: str) -> MagicMock:
        name = Path(path).stem
        mapping = {
            "vision-encoder": vision_cm,
            "text-encoder": text_cm,
            "prompt-decoder": decoder_cm,
            "geometry-encoder": geo_cm,
            "geometry-encoder-exemplar": geo_ex_cm,
        }
        return mapping.get(name, MagicMock())

    mock_core.compile_model = MagicMock(side_effect=compile_model_side_effect)

    with (
        patch("instantlearn.models.sam3.sam3_openvino.ov") as mock_ov,
        patch("instantlearn.models.sam3.sam3_openvino.CLIPTokenizerFast") as mock_tok_cls,
        patch("instantlearn.models.sam3.sam3_openvino.Sam3Preprocessor") as mock_pre_cls,
        patch("instantlearn.models.sam3.sam3_openvino.Sam3PromptPreprocessor") as mock_ppre_cls,
        patch("instantlearn.models.sam3.sam3_openvino.Sam3Postprocessor") as mock_post_cls,
    ):
        mock_ov.Core.return_value = mock_core
        mock_tok_cls.from_pretrained.return_value = _make_mock_tokenizer()
        mock_pre_cls.return_value = _make_mock_preprocessor()
        mock_ppre_cls.return_value = _make_mock_prompt_preprocessor()
        mock_post_cls.return_value = _make_mock_postprocessor()

        return SAM3OpenVINO(model_dir=model_dir, prompt_mode=prompt_mode)


class TestSAM3OpenVINOInit:
    """Test SAM3OpenVINO initialization and model detection."""

    def test_v3_init(self, tmp_model_dir: Path) -> None:
        """Test initialization loads all models correctly."""
        model = _build_sam3_openvino(tmp_model_dir)

        assert model.prompt_mode == Sam3PromptMode.CLASSIC
        assert model.geometry_model is not None
        assert model.geometry_exemplar_model is not None

    def test_missing_decoder_raises(self, tmp_path: Path) -> None:
        """Test missing decoder raises FileNotFoundError."""
        (tmp_path / "vision-encoder.xml").touch()
        (tmp_path / "text-encoder.xml").touch()
        with pytest.raises(FileNotFoundError):
            _build_sam3_openvino(tmp_path)


class TestClassicTextPrompts:
    """Test classic mode with text-only prompts."""

    def test_fit_stores_category_mapping(self, tmp_model_dir: Path) -> None:
        """Test fit() stores category mapping."""
        model = _build_sam3_openvino(tmp_model_dir)
        ref = Sample(categories=["cat", "dog"], category_ids=[0, 1])
        model.fit(ref)

        assert model.category_mapping == {"cat": 0, "dog": 1}

    def test_predict_text_only(self, tmp_model_dir: Path) -> None:
        """Test prediction with text-only prompts returns valid structure."""
        model = _build_sam3_openvino(tmp_model_dir)
        model.fit(Sample(categories=["shoe"], category_ids=[0]))

        target = Sample(image=torch.randn(3, 224, 224))
        results = model.predict(target)

        assert len(results) == 1
        assert "pred_masks" in results[0]
        assert "pred_boxes" in results[0]
        assert "pred_labels" in results[0]

    def test_predict_multiple_categories(self, tmp_model_dir: Path) -> None:
        """Test prediction with multiple categories."""
        model = _build_sam3_openvino(tmp_model_dir)
        model.fit(Sample(categories=["cat", "dog"], category_ids=[0, 1]))

        target = Sample(image=torch.randn(3, 224, 224))
        results = model.predict(target)

        assert len(results) == 1
        # Labels should contain cat_ids from fit
        labels = results[0]["pred_labels"]
        unique_labels = set(labels.numpy().tolist())
        assert unique_labels <= {0, 1}


class TestClassicBoxPrompts:
    """Test classic mode with box prompts."""

    def test_predict_with_bbox(self, tmp_model_dir: Path) -> None:
        """Test prediction with a bounding box prompt."""
        model = _build_sam3_openvino(tmp_model_dir)

        target = Sample(
            image=torch.randn(3, 224, 224),
            bboxes=np.array([[10, 10, 50, 50]]),
            categories=["shoe"],
            category_ids=[0],
        )
        results = model.predict(target)

        assert len(results) == 1
        assert results[0]["pred_boxes"].shape[1] == 5  # [x1, y1, x2, y2, score]


class TestClassicPointPrompts:
    """Test classic mode with point prompts."""

    def test_predict_with_point(self, tmp_model_dir: Path) -> None:
        """Test prediction with a point prompt."""
        model = _build_sam3_openvino(tmp_model_dir)

        target = Sample(
            image=torch.randn(3, 224, 224),
            points=np.array([[100, 100]]),
            categories=["shoe"],
            category_ids=[0],
        )
        results = model.predict(target)

        assert len(results) == 1
        assert results[0]["pred_masks"].ndim == 3  # [N, H, W]


class TestVisualExemplarMode:
    """Test visual-exemplar mode flow."""

    def test_fit_caches_features(self, tmp_model_dir: Path) -> None:
        """Test fit() encodes and caches exemplar features."""
        model = _build_sam3_openvino(tmp_model_dir, prompt_mode=Sam3PromptMode.VISUAL_EXEMPLAR)

        ref = Sample(
            image=torch.randn(3, 224, 224),
            bboxes=np.array([[10, 10, 50, 50]]),
            categories=["shoe"],
            category_ids=[0],
        )
        model.fit(ref)

        assert model.exemplar_geometry_features is not None
        assert len(model.exemplar_geometry_features) == 1
        assert model.exemplar_category_ids == [0]

    def test_predict_uses_cached_features(self, tmp_model_dir: Path) -> None:
        """Test predict() uses cached exemplar features."""
        model = _build_sam3_openvino(tmp_model_dir, prompt_mode=Sam3PromptMode.VISUAL_EXEMPLAR)

        ref = Sample(
            image=torch.randn(3, 224, 224),
            bboxes=np.array([[10, 10, 50, 50]]),
            categories=["shoe"],
            category_ids=[0],
        )
        model.fit(ref)

        target = Sample(image=torch.randn(3, 224, 224))
        results = model.predict(target)

        assert len(results) == 1
        assert "pred_masks" in results[0]

    def test_predict_without_fit_raises(self, tmp_model_dir: Path) -> None:
        """Test predict() raises when fit() has not been called."""
        model = _build_sam3_openvino(tmp_model_dir, prompt_mode=Sam3PromptMode.VISUAL_EXEMPLAR)

        with pytest.raises(RuntimeError, match="No cached exemplar features"):
            model.predict(Sample(image=torch.randn(3, 224, 224)))

    def test_fit_no_prompts_raises(self, tmp_model_dir: Path) -> None:
        """Test fit() with no box/point prompts raises ValueError."""
        model = _build_sam3_openvino(tmp_model_dir, prompt_mode=Sam3PromptMode.VISUAL_EXEMPLAR)

        # Sample with image but no prompts
        ref = Sample(
            image=torch.randn(3, 224, 224),
            categories=["shoe"],
            category_ids=[0],
        )
        with pytest.raises(ValueError, match="at least one reference sample"):
            model.fit(ref)

    def test_multi_category_exemplar(self, tmp_model_dir: Path) -> None:
        """Test multi-category exemplar with separate reference samples."""
        model = _build_sam3_openvino(tmp_model_dir, prompt_mode=Sam3PromptMode.VISUAL_EXEMPLAR)

        refs = [
            Sample(
                image=torch.randn(3, 224, 224),
                bboxes=np.array([[10, 10, 50, 50]]),
                categories=["shoe"],
                category_ids=[0],
            ),
            Sample(
                image=torch.randn(3, 224, 224),
                bboxes=np.array([[20, 20, 60, 60]]),
                categories=["hat"],
                category_ids=[1],
            ),
        ]
        model.fit(refs)

        assert len(model.exemplar_geometry_features) == 2
        assert set(model.exemplar_category_ids) == {0, 1}


class TestUtilities:
    """Test utility methods."""

    def test_pad_or_truncate_pads(self) -> None:
        """Test _pad_or_truncate pads short sequences."""
        arr = np.array([[1, 2, 3]], dtype=np.int64)
        result = SAM3OpenVINO._pad_or_truncate(arr, 5)  # noqa: SLF001
        assert result.shape == (1, 5)
        np.testing.assert_array_equal(result[0, :3], [1, 2, 3])
        np.testing.assert_array_equal(result[0, 3:], [0, 0])

    def test_pad_or_truncate_truncates(self) -> None:
        """Test _pad_or_truncate truncates long sequences."""
        arr = np.array([[1, 2, 3, 4, 5]], dtype=np.int64)
        result = SAM3OpenVINO._pad_or_truncate(arr, 3)  # noqa: SLF001
        assert result.shape == (1, 3)
        np.testing.assert_array_equal(result[0], [1, 2, 3])

    def test_pad_or_truncate_exact(self) -> None:
        """Test _pad_or_truncate returns same array when lengths match."""
        arr = np.array([[1, 2, 3]], dtype=np.int64)
        result = SAM3OpenVINO._pad_or_truncate(arr, 3)  # noqa: SLF001
        assert result is arr

    def test_build_category_mapping(self, tmp_model_dir: Path) -> None:
        """Test _build_category_mapping merges from multiple samples."""
        from instantlearn.data.base.batch import Batch  # noqa: PLC0415

        model = _build_sam3_openvino(tmp_model_dir)
        batch = Batch.collate([
            Sample(categories=["a", "b"], category_ids=[0, 1]),
            Sample(categories=["b", "c"], category_ids=[1, 2]),
        ])
        mapping = model._build_category_mapping(batch)  # noqa: SLF001
        assert mapping == {"a": 0, "b": 1, "c": 2}

    def test_aggregate_results_empty(self) -> None:
        """Test _aggregate_results with no detections."""
        result = SAM3OpenVINO._aggregate_results(  # noqa: SLF001
            all_masks=[torch.empty(0)],
            all_boxes=[torch.empty(0)],
            all_labels=[torch.empty(0)],
            img_size=(224, 224),
        )
        assert result["pred_masks"].shape == (0, 224, 224)
        assert result["pred_boxes"].shape == (0, 5)

    def test_export_returns_model_dir(self, tmp_model_dir: Path) -> None:
        """Test export() returns model directory."""
        model = _build_sam3_openvino(tmp_model_dir)
        result_path = model.export()
        assert result_path == model.model_dir


class TestModelFileDiscovery:
    """Test _find_model_file and _require_model_file."""

    def test_find_xml_preferred(self, tmp_path: Path) -> None:
        """Test .xml is preferred over .onnx."""
        (tmp_path / "vision-encoder.xml").touch()
        (tmp_path / "vision-encoder.onnx").touch()
        result = _find_model_file(tmp_path, "vision-encoder")
        assert result is not None
        assert result.suffix == ".xml"

    def test_find_onnx_fallback(self, tmp_path: Path) -> None:
        """Test .onnx is found when no .xml exists."""
        (tmp_path / "vision-encoder.onnx").touch()
        result = _find_model_file(tmp_path, "vision-encoder")
        assert result is not None
        assert result.suffix == ".onnx"

    def test_find_fp16_onnx(self, tmp_path: Path) -> None:
        """Test FP16 ONNX variant is found."""
        (tmp_path / "vision-encoder-fp16.onnx").touch()
        result = _find_model_file(tmp_path, "vision-encoder")
        assert result is not None
        assert result.name == "vision-encoder-fp16.onnx"

    def test_find_returns_none_when_missing(self, tmp_path: Path) -> None:
        """Test returns None for missing model."""
        result = _find_model_file(tmp_path, "nonexistent")
        assert result is None

    def test_require_raises_when_missing(self, tmp_path: Path) -> None:
        """Test _require_model_file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="not found"):
            _require_model_file(tmp_path, "nonexistent")
