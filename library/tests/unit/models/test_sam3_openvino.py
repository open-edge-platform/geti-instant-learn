# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the SAM3OpenVINO public model contract."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from instantlearn.data.base.prediction import Prediction
from instantlearn.data.base.sample import Category, Sample
from instantlearn.models.openvino_base import OpenVINOModel
from instantlearn.models.sam3 import SAM3, SAM3OpenVINO, Sam3PromptMode
from instantlearn.utils import Backend


@pytest.fixture
def openvino_model_dir(tmp_path: Path) -> Path:
    """Create a minimal local SAM3 OpenVINO artifact directory."""
    for model_name in [
        "vision-encoder",
        "text-encoder",
        "geometry-encoder",
        "prompt-decoder",
    ]:
        (tmp_path / f"{model_name}.xml").touch()
    (tmp_path / "tokenizer.json").write_text("{}")
    return tmp_path


def _mock_openvino_core() -> MagicMock:
    """Create a mock OpenVINO core with compilable model placeholders."""
    compiled_model = MagicMock()
    compiled_model.create_infer_request.return_value = MagicMock()
    core = MagicMock()
    core.compile_model.return_value = compiled_model
    return core


class TestSAM3OpenVINOInit:
    """Initialization and static contract tests."""

    def test_inherits_openvino_model(self, openvino_model_dir: Path) -> None:
        """SAM3OpenVINO uses OpenVINOModel as its backend abstraction."""
        mock_core = _mock_openvino_core()

        with (
            patch("instantlearn.models.openvino_base.ov.Core", return_value=mock_core),
            patch("instantlearn.models.sam3.sam3_openvino.CLIPTokenizerFast.from_pretrained") as mock_tokenizer,
        ):
            model = SAM3OpenVINO(model_dir=openvino_model_dir, device="cpu", prompt_mode=Sam3PromptMode.CLASSIC)

        assert isinstance(model, OpenVINOModel)
        assert model.backend == Backend.OPENVINO
        assert model.ov_device == "CPU"
        assert model.model_dir == openvino_model_dir
        assert model.compression_mode == "int8_sym"
        assert mock_core.compile_model.call_count == 4
        mock_tokenizer.assert_called_once_with(str(openvino_model_dir))

    def test_exports_when_model_dir_is_missing(self, openvino_model_dir: Path, tmp_path: Path) -> None:
        """SAM3OpenVINO exports official SAM3 to INT8_SYM artifacts when no model_dir is provided."""
        mock_core = _mock_openvino_core()

        with (
            patch("instantlearn.models.openvino_base.ov.Core", return_value=mock_core),
            patch("instantlearn.models.sam3.sam3_openvino.CLIPTokenizerFast.from_pretrained"),
            patch(
                "instantlearn.scripts.sam3.export_sam3.export_sam3_openvino",
                return_value=openvino_model_dir,
            ) as mock_export,
        ):
            model = SAM3OpenVINO(
                model_dir=None,
                model_id="facebook/sam3.1",
                device="cpu",
                export_dir=tmp_path / "sam3-export",
            )

        assert model.model_dir == openvino_model_dir
        mock_export.assert_called_once_with(
            model_id="facebook/sam3.1",
            output_dir=tmp_path / "sam3-export",
            resolution=1008,
            precision="fp16",
            compression_mode="int8_sym",
        )

    def test_export_compression_mode_is_configurable(self, openvino_model_dir: Path, tmp_path: Path) -> None:
        """SAM3OpenVINO forwards custom init-time export compression mode."""
        mock_core = _mock_openvino_core()

        with (
            patch("instantlearn.models.openvino_base.ov.Core", return_value=mock_core),
            patch("instantlearn.models.sam3.sam3_openvino.CLIPTokenizerFast.from_pretrained"),
            patch(
                "instantlearn.scripts.sam3.export_sam3.export_sam3_openvino",
                return_value=openvino_model_dir,
            ) as mock_export,
        ):
            model = SAM3OpenVINO(
                model_dir=None,
                device="cpu",
                export_dir=tmp_path / "sam3-export",
                compression_mode=None,
            )

        assert model.compression_mode is None
        mock_export.assert_called_once_with(
            model_id="facebook/sam3.1",
            output_dir=tmp_path / "sam3-export",
            resolution=1008,
            precision="fp16",
            compression_mode=None,
        )

    def test_card_delegates_to_sam3(self) -> None:
        """SAM3OpenVINO exposes the same model capabilities as SAM3."""
        assert SAM3OpenVINO.card() == SAM3.card()

    def test_from_pretrained_sets_model_id_and_export_dir(self, tmp_path: Path) -> None:
        """from_pretrained() forwards repo_id as model_id and cache_dir as export_dir."""
        expected = object()

        with patch.object(SAM3OpenVINO, "__init__", return_value=None) as mock_init:
            result = SAM3OpenVINO.from_pretrained(
                "facebook/sam3.1",
                cache_dir=tmp_path / "ov-export",
                device="CPU",
            )

        assert isinstance(result, SAM3OpenVINO)
        mock_init.assert_called_once_with(
            device="CPU",
            model_id="facebook/sam3.1",
            export_dir=tmp_path / "ov-export",
        )
        del expected

    def test_from_pretrained_rejects_revision(self) -> None:
        """from_pretrained() accepts revision for API shape but does not support it yet."""
        with pytest.raises(ValueError, match="revision"):
            SAM3OpenVINO.from_pretrained("facebook/sam3.1", revision="main")


class TestSAM3OpenVINOPredict:
    """Prediction return contract tests."""

    @pytest.mark.parametrize(
        ("prompt_mode", "method_name"),
        [
            (Sam3PromptMode.CLASSIC, "_predict_classic"),
            (Sam3PromptMode.VISUAL_EXEMPLAR, "_predict_visual_exemplar"),
            (Sam3PromptMode.CANVAS, "_predict_canvas"),
        ],
    )
    def test_predict_returns_prediction(self, prompt_mode: Sam3PromptMode, method_name: str) -> None:
        """Public predict() converts internal tensor dicts to Prediction objects."""
        model = object.__new__(SAM3OpenVINO)
        model.prompt_mode = prompt_mode
        model.category_mapping = {"shoe": 0}

        raw_prediction = {
            "pred_masks": torch.ones(1, 4, 4, dtype=torch.uint8),
            "pred_boxes": torch.tensor([[0.0, 1.0, 2.0, 3.0, 0.7]], dtype=torch.float32),
            "pred_labels": torch.tensor([0], dtype=torch.int64),
        }
        sample = Sample(
            image=np.zeros((4, 4, 3), dtype=np.uint8),
            categories=[Category(id=0, label="shoe")],
        )

        with patch.object(model, method_name, return_value=[raw_prediction]) as mock_predict:
            predictions = SAM3OpenVINO.predict(model, sample)

        assert len(predictions) == 1
        prediction = predictions[0]
        assert isinstance(prediction, Prediction)
        assert prediction.masks.shape == (1, 4, 4)
        assert prediction.boxes is not None
        np.testing.assert_allclose(prediction.boxes, np.array([[0.0, 1.0, 2.0, 3.0]], dtype=np.float32))
        np.testing.assert_allclose(prediction.scores, np.array([0.7], dtype=np.float32))
        np.testing.assert_array_equal(prediction.label_ids, np.array([0], dtype=np.int32))
        np.testing.assert_array_equal(prediction.label_names, np.array(["shoe"], dtype=object))
        private_batch = mock_predict.call_args.args[0]
        assert private_batch.samples[0].image.shape == (3, 4, 4)
