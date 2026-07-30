# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock

import numpy as np
import pytest
from instantlearn.data.base.prediction import Prediction

from domain.services.schemas.processor import InputData
from runtime.core.components.models.inference_model import InferenceModelHandler, masks_to_boxes


def _input(frame: np.ndarray | None = None) -> InputData:
    return InputData(
        timestamp=0,
        frame=frame if frame is not None else np.zeros((10, 12, 3), dtype=np.uint8),
        context={},
    )


def _prediction(boxes: np.ndarray | None = None, masks: np.ndarray | None = None) -> Prediction:
    if masks is None:
        masks = np.zeros((1, 4, 4), dtype=bool)
        masks[0, 1:3, 2:4] = True
    return Prediction(
        masks=masks,
        scores=np.array([0.7], dtype=np.float32),
        label_ids=np.array([1], dtype=np.int32),
        label_names=np.array(["cat"], dtype=object),
        boxes=boxes,
    )


class TestInferenceModelHandler:
    @pytest.fixture
    def model(self):
        model = MagicMock()
        model.predict.return_value = [_prediction(boxes=np.array([[1, 2, 3, 4]], dtype=np.float32))]
        return model

    def test_predict_builds_numpy_samples_from_frames(self, model):
        handler = InferenceModelHandler(model)
        frame_a = np.full((10, 12, 3), 7, dtype=np.uint8)
        frame_b = np.full((10, 12, 3), 9, dtype=np.uint8)
        model.predict.return_value = [_prediction(), _prediction()]

        handler.predict([_input(frame_a), _input(frame_b)])

        batch = model.predict.call_args[0][0]
        assert len(batch.samples) == 2
        # Frames are passed through as HWC numpy, no torch conversion
        np.testing.assert_array_equal(batch.samples[0].image, frame_a)
        np.testing.assert_array_equal(batch.samples[1].image, frame_b)

    def test_predict_returns_model_predictions_unchanged(self, model):
        handler = InferenceModelHandler(model)

        results = handler.predict([_input()])

        assert results is model.predict.return_value
        assert isinstance(results[0], Prediction)

    def test_predict_keeps_model_provided_boxes(self, model):
        handler = InferenceModelHandler(model)

        results = handler.predict([_input()])

        np.testing.assert_array_equal(results[0].boxes, np.array([[1, 2, 3, 4]], dtype=np.float32))

    def test_predict_derives_boxes_from_masks_when_missing(self, model):
        model.predict.return_value = [_prediction(boxes=None)]
        handler = InferenceModelHandler(model)

        results = handler.predict([_input()])

        # mask covers rows 1-2 and cols 2-3 -> xyxy (2, 1, 4, 3)
        np.testing.assert_array_equal(results[0].boxes, np.array([[2, 1, 4, 3]], dtype=np.float32))

    def test_predict_raises_after_close(self, model):
        handler = InferenceModelHandler(model)
        handler.close()

        with pytest.raises(RuntimeError, match="closed"):
            handler.predict([_input()])

    def test_close_releases_the_model(self, model):
        handler = InferenceModelHandler(model)

        handler.close()

        assert handler._model is None

    def test_initialise_is_a_noop(self, model):
        handler = InferenceModelHandler(model)

        handler.initialise()

        model.fit.assert_not_called()
        model.predict.assert_not_called()


class TestMasksToBoxes:
    def test_returns_empty_for_no_masks(self):
        assert masks_to_boxes(np.zeros((0, 4, 4), dtype=bool)).shape == (0, 4)

    def test_returns_empty_for_none(self):
        assert masks_to_boxes(None).shape == (0, 4)

    def test_empty_mask_yields_zero_box(self):
        boxes = masks_to_boxes(np.zeros((1, 4, 4), dtype=bool))

        np.testing.assert_array_equal(boxes, np.zeros((1, 4), dtype=np.float32))

    def test_derives_tight_box_per_instance(self):
        masks = np.zeros((2, 6, 6), dtype=bool)
        masks[0, 1:3, 2:5] = True
        masks[1, 4:6, 0:2] = True

        boxes = masks_to_boxes(masks)

        np.testing.assert_array_equal(boxes, np.array([[2, 1, 5, 3], [0, 4, 2, 6]], dtype=np.float32))
        assert boxes.dtype == np.float32
