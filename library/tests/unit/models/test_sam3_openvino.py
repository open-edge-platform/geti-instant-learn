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
            model = SAM3OpenVINO(ir_path=openvino_model_dir, device="cpu", prompt_mode=Sam3PromptMode.CLASSIC)

        assert isinstance(model, OpenVINOModel)
        assert model.backend == Backend.OPENVINO
        assert model.ov_device == "CPU"
        assert model.model_dir == openvino_model_dir
        assert mock_core.compile_model.call_count == 4
        mock_tokenizer.assert_called_once_with(str(openvino_model_dir))

    def test_card_delegates_to_sam3(self) -> None:
        """SAM3OpenVINO exposes the same model capabilities as SAM3."""
        assert SAM3OpenVINO.card() == SAM3.card()

    def test_from_pretrained_loads_exported_model_dir(self, tmp_path: Path) -> None:
        """from_pretrained() forwards an exported OpenVINO artifact directory."""
        expected = object()
        ir_path = tmp_path / "openvino-int8_sym"

        with patch.object(SAM3OpenVINO, "__init__", return_value=None) as mock_init:
            result = SAM3OpenVINO.from_pretrained(
                ir_path,
                device="CPU",
            )

        assert isinstance(result, SAM3OpenVINO)
        mock_init.assert_called_once_with(
            ir_path=ir_path,
            device="CPU",
        )
        del expected

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
        tensor_samples = mock_predict.call_args.args[0]
        assert tensor_samples[0].image.shape == (3, 4, 4)
