# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for models including PerDino, Matcher, SoftMatcher, and GroundedSAM."""

from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from torchvision.tv_tensors import Image

from instantlearn.data.base.prediction import Prediction
from instantlearn.data.base.sample import Category, Sample
from instantlearn.models.grounded_sam import GroundedSAM
from instantlearn.models.matcher import Matcher
from instantlearn.models.per_dino import PerDino
from instantlearn.models.soft_matcher import SoftMatcher
from tests import CPU_DEVICE


def _empty_prediction() -> Prediction:
    """Build an empty backend-neutral ``Prediction`` for mock returns."""
    return Prediction(
        masks=np.zeros((0, 224, 224), dtype=bool),
        scores=np.zeros((0,), dtype=np.float32),
        label_ids=np.zeros((0,), dtype=np.int32),
        label_names=np.array([], dtype=object),
        boxes=np.zeros((0, 4), dtype=np.float32),
        points=None,
    )


def _assert_prediction_list(predictions: list) -> None:
    """Assert a predict() result is a list of ``Prediction`` objects."""
    assert isinstance(predictions, list)
    assert len(predictions) == 1
    assert isinstance(predictions[0], Prediction)
    assert predictions[0].masks is not None
    assert predictions[0].scores is not None
    assert predictions[0].label_ids is not None
    assert predictions[0].label_names is not None


class TestPerDino:
    """Test PerDino model."""

    @pytest.fixture
    def mock_components(self) -> dict[str, Any]:
        """Create mock components for PerDino."""
        return {
            "sam_predictor": MagicMock(),
            "encoder": MagicMock(),
            "masked_feature_extractor": MagicMock(),
            "similarity_matcher": MagicMock(),
            "prompt_generator": MagicMock(),
            "segmenter": MagicMock(),
        }

    @patch("instantlearn.models.per_dino.per_dino.load_sam_model")
    @patch("instantlearn.models.per_dino.per_dino.ImageEncoder")
    def test_per_dino_initialization(
        self,
        mock_image_encoder: MagicMock,
        mock_load_sam: MagicMock,
        mock_components: dict[str, Any],
    ) -> None:
        """Test PerDino initialization."""
        mock_load_sam.return_value = mock_components["sam_predictor"]
        mock_image_encoder.return_value = mock_components["encoder"]

        model = PerDino(device=CPU_DEVICE)

        assert hasattr(model, "sam_predictor")
        assert hasattr(model, "encoder")
        assert hasattr(model, "masked_feature_extractor")
        assert hasattr(model, "similarity_matcher")
        assert hasattr(model, "prompt_generator")
        assert hasattr(model, "segmenter")

    def test_per_dino_card(self) -> None:
        """PerDino advertises its own capability card."""
        card = PerDino.card()
        assert card.name == "PerDino"
        assert card.family == "per_dino"

    @patch("instantlearn.models.per_dino.per_dino.load_sam_model")
    @patch("instantlearn.models.per_dino.per_dino.ImageEncoder")
    def test_per_dino_forward_pass(
        self,
        mock_image_encoder: MagicMock,
        mock_load_sam: MagicMock,
        mock_components: dict[str, Any],
    ) -> None:
        """Test PerDino predict() returns a list of Prediction objects."""
        mock_load_sam.return_value = mock_components["sam_predictor"]
        mock_image_encoder.return_value = mock_components["encoder"]

        model = PerDino(device=CPU_DEVICE)
        model.fit = MagicMock(return_value=None)
        model.predict = MagicMock(return_value=[_empty_prediction()])

        target_images = [Image(torch.zeros((3, 224, 224), dtype=torch.uint8))]
        predictions = model.predict(target_images)

        _assert_prediction_list(predictions)
        model.predict.assert_called_once_with(target_images)


class TestMatcher:
    """Test Matcher model."""

    @pytest.fixture
    def mock_components(self) -> dict[str, Any]:
        """Create mock components for Matcher."""
        return {
            "sam_predictor": MagicMock(),
            "encoder": MagicMock(),
            "masked_feature_extractor": MagicMock(),
            "prompt_generator": MagicMock(),
            "point_filter": MagicMock(),
            "segmenter": MagicMock(),
        }

    @patch("instantlearn.models.matcher.matcher.load_sam_model")
    @patch("instantlearn.models.matcher.matcher.ImageEncoder")
    def test_matcher_initialization(
        self,
        mock_image_encoder: MagicMock,
        mock_sam_predictor: MagicMock,
        mock_components: dict[str, Any],
    ) -> None:
        """Test Matcher initialization."""
        mock_sam_predictor.return_value = mock_components["sam_predictor"]
        mock_image_encoder.return_value = mock_components["encoder"]

        model = Matcher(device=CPU_DEVICE)

        assert hasattr(model, "sam_predictor")
        assert hasattr(model, "encoder")
        assert hasattr(model, "masked_feature_extractor")
        assert hasattr(model, "prompt_generator")
        assert hasattr(model, "segmenter")

    def test_matcher_card(self) -> None:
        """Matcher advertises its own capability card."""
        card = Matcher.card()
        assert card.name == "Matcher"
        assert card.family == "matcher"

    @patch("instantlearn.models.matcher.matcher.load_sam_model")
    @patch("instantlearn.models.matcher.matcher.ImageEncoder")
    def test_matcher_forward_pass(
        self,
        mock_image_encoder: MagicMock,
        mock_sam_predictor: MagicMock,
        mock_components: dict[str, Any],
    ) -> None:
        """Test Matcher predict() returns a list of Prediction objects."""
        mock_sam_predictor.return_value = mock_components["sam_predictor"]
        mock_image_encoder.return_value = mock_components["encoder"]

        model = Matcher(device=CPU_DEVICE)
        model.fit = MagicMock(return_value=None)
        model.predict = MagicMock(return_value=[_empty_prediction()])

        target_images = [Image(torch.zeros((3, 224, 224), dtype=torch.uint8))]
        predictions = model.predict(target_images)

        _assert_prediction_list(predictions)
        model.predict.assert_called_once_with(target_images)


class TestSoftMatcher:
    """Test SoftMatcher model."""

    @pytest.fixture
    def mock_components(self) -> dict[str, Any]:
        """Create mock components for SoftMatcher."""
        return {
            "sam_predictor": MagicMock(),
            "encoder": MagicMock(),
            "masked_feature_extractor": MagicMock(),
            "prompt_generator": MagicMock(),
            "point_filter": MagicMock(),
            "segmenter": MagicMock(),
        }

    @patch("instantlearn.models.matcher.matcher.load_sam_model")
    @patch("instantlearn.models.matcher.matcher.ImageEncoder")
    def test_soft_matcher_initialization(
        self,
        mock_image_encoder: MagicMock,
        mock_sam_predictor: MagicMock,
        mock_components: dict[str, Any],
    ) -> None:
        """Test SoftMatcher initialization with new components."""
        mock_sam_predictor.return_value = mock_components["sam_predictor"]
        mock_image_encoder.return_value = mock_components["encoder"]

        model = SoftMatcher(device=CPU_DEVICE)

        assert hasattr(model, "sam_predictor")
        assert hasattr(model, "encoder")
        assert hasattr(model, "masked_feature_extractor")
        assert hasattr(model, "prompt_generator")
        assert hasattr(model, "segmenter")

    def test_soft_matcher_card(self) -> None:
        """SoftMatcher advertises its own distinct capability card."""
        card = SoftMatcher.card()
        assert card.name == "SoftMatcher"
        assert card.family == "soft_matcher"

    @patch("instantlearn.models.matcher.matcher.load_sam_model")
    @patch("instantlearn.models.matcher.matcher.ImageEncoder")
    def test_soft_matcher_forward_pass(
        self,
        mock_image_encoder: MagicMock,
        mock_sam_predictor: MagicMock,
        mock_components: dict[str, Any],
    ) -> None:
        """Test SoftMatcher predict() returns a list of Prediction objects."""
        mock_sam_predictor.return_value = mock_components["sam_predictor"]
        mock_image_encoder.return_value = mock_components["encoder"]

        model = SoftMatcher(device=CPU_DEVICE)
        model.fit = MagicMock(return_value=None)
        model.predict = MagicMock(return_value=[_empty_prediction()])

        target_images = [Image(torch.zeros((3, 224, 224), dtype=torch.uint8))]
        predictions = model.predict(target_images)

        _assert_prediction_list(predictions)
        model.predict.assert_called_once_with(target_images)


class TestGroundedSAM:
    """Test GroundedSAM model."""

    @patch("instantlearn.models.grounded_sam.grounded.TextToBoxPromptGenerator._load_grounding_model_and_processor")
    @patch("instantlearn.models.grounded_sam.grounded_sam.load_sam_model")
    def test_grounded_sam_initialization(self, mock_load_sam: MagicMock, mock_grounding: MagicMock) -> None:
        """Test GroundedSAM initialization with new components."""
        mock_load_sam.return_value = MagicMock()
        mock_grounding.return_value = (MagicMock(), MagicMock())

        model = GroundedSAM(device=CPU_DEVICE)

        assert hasattr(model, "sam_predictor")
        assert hasattr(model, "prompt_generator")
        assert hasattr(model, "segmenter")
        assert hasattr(model, "prompt_filter")

    @patch("instantlearn.models.grounded_sam.grounded.TextToBoxPromptGenerator._load_grounding_model_and_processor")
    @patch("instantlearn.models.grounded_sam.grounded_sam.load_sam_model")
    def test_predict_returns_predictions(self, mock_load_sam: MagicMock, mock_grounding: MagicMock) -> None:
        """predict() converts segmenter dicts to Prediction using target categories."""
        mock_load_sam.return_value = MagicMock()
        mock_grounding.return_value = (MagicMock(), MagicMock())
        model = GroundedSAM(device=CPU_DEVICE)
        model.postprocessor = None
        model.prompt_generator = MagicMock(
            spec=torch.nn.Module,
            return_value=(torch.zeros(1, 1, 1, 5), torch.tensor([7])),
        )
        model.prompt_filter = MagicMock(spec=torch.nn.Module, side_effect=lambda box_prompts: box_prompts)
        model.segmenter = MagicMock(
            spec=torch.nn.Module,
            return_value=[
                {
                    "pred_masks": torch.ones(1, 4, 4),
                    "pred_labels": torch.tensor([7]),
                    "pred_scores": torch.tensor([0.9]),
                },
            ],
        )

        sample = Sample(image=np.zeros((4, 4, 3), dtype=np.uint8), categories=[Category(7, "cat")])
        predictions = model.predict(sample)

        assert len(predictions) == 1
        assert isinstance(predictions[0], Prediction)
        assert predictions[0].label_ids.tolist() == [7]
        assert predictions[0].label_names.tolist() == ["cat"]
        assert predictions[0].masks.shape == (1, 4, 4)

    @patch("instantlearn.models.grounded_sam.grounded.TextToBoxPromptGenerator._load_grounding_model_and_processor")
    @patch("instantlearn.models.grounded_sam.grounded_sam.load_sam_model")
    def test_predict_raises_without_categories(self, mock_load_sam: MagicMock, mock_grounding: MagicMock) -> None:
        """predict() raises when neither fit() nor the target provide categories."""
        mock_load_sam.return_value = MagicMock()
        mock_grounding.return_value = (MagicMock(), MagicMock())
        model = GroundedSAM(device=CPU_DEVICE)
        sample = Sample(image=np.zeros((4, 4, 3), dtype=np.uint8), categories=[])

        with pytest.raises(ValueError, match="requires categories"):
            model.predict(sample)

    @patch("instantlearn.models.grounded_sam.grounded.TextToBoxPromptGenerator._load_grounding_model_and_processor")
    @patch("instantlearn.models.grounded_sam.grounded_sam.load_sam_model")
    def test_predict_raises_without_image(self, mock_load_sam: MagicMock, mock_grounding: MagicMock) -> None:
        """predict() raises when a target sample has no image."""
        mock_load_sam.return_value = MagicMock()
        mock_grounding.return_value = (MagicMock(), MagicMock())
        model = GroundedSAM(device=CPU_DEVICE)
        sample = Sample(image=None, categories=[Category(7, "cat")])

        with pytest.raises(ValueError, match="each sample to contain an image"):
            model.predict(sample)


class TestMatcherZeroCoverageRegression:
    """Regression tests for the zero-coverage category bug.

    Scenario: fit() is called with two reference categories.  Category 1 ("cat")
    has a polygon large enough to survive patch-grid downsampling. Category 2
    ("led") has a 1-pixel polygon that covers zero encoder patch cells.

    Before the fix the pipeline crashed in two places:
      1. BidirectionalPromptGenerator._select_background_points — NaN from
         ``mean(dim=0)`` on a ``[0, N]`` similarity sub-matrix.
      2. SamDecoder._predict_masks_for_category — SAM called with 0 foreground
         prompt instances, which crashes the SAM forward pass.

    After the fix predict() must complete without error, and the zero-coverage
    category must produce no detections.
    """

    FEAT_SIZE = 8            # encoder feature-grid side
    NUM_PATCHES = FEAT_SIZE * FEAT_SIZE   # 64
    EMBED_DIM = 32
    IMG_H = 64
    IMG_W = 64

    def _build_matcher(self, mock_encoder_cls: MagicMock, mock_load_sam: MagicMock) -> Matcher:
        """Construct a Matcher whose heavyweight sub-components are mocked."""
        feat_size = self.FEAT_SIZE
        patch_size = 14
        input_size = feat_size * patch_size  # 112

        # SAM predictor mock ------------------------------------------------
        mock_sam_pred = MagicMock()
        mock_sam_pred.device = torch.device("cpu")
        mock_sam_pred.dtype = torch.float32
        mock_sam_pred.sam_model_name = "mock"
        mock_sam_pred.set_image.return_value = None

        h, w = self.IMG_H, self.IMG_W

        def _sam_forward(point_coords, point_labels, boxes, mask_input, multimask_output, **_):  # noqa: ANN202
            num_fg = point_coords.shape[0]
            # Return non-empty masks so the scoring path is exercised.
            masks = torch.ones(num_fg, 3, h, w, dtype=torch.float32)
            iou_preds = torch.ones(num_fg, 3, dtype=torch.float32) * 0.9
            low_res = torch.zeros(num_fg, 3, 256, 256, dtype=torch.float32)
            return masks, iou_preds, low_res

        mock_sam_pred.forward.side_effect = _sam_forward
        mock_load_sam.return_value = mock_sam_pred

        # Encoder mock -------------------------------------------------------
        mock_enc = MagicMock()
        mock_enc.input_size = input_size
        mock_enc.patch_size = patch_size
        mock_enc.feature_size = feat_size
        # __call__ returns target embeddings [1, N, D]
        mock_enc.return_value = torch.randn(1, self.NUM_PATCHES, self.EMBED_DIM)
        mock_encoder_cls.return_value = mock_enc

        return Matcher(
            device=CPU_DEVICE,
            num_foreground_points=5,
            num_background_points=2,
            confidence_threshold=0.0,
            use_mask_refinement=False,
            postprocessor=None,
        )

    def _inject_ref_features(self, matcher: Matcher) -> None:
        """Directly inject a two-category ReferenceFeatures where cat-2 is zero-coverage."""
        from instantlearn.components.feature_extractors.reference_features import ReferenceFeatures
        from instantlearn.models.torch_adapter import CategoryRegistry

        torch.manual_seed(0)
        n = self.NUM_PATCHES
        d = self.EMBED_DIM

        ref_embeddings = torch.randn(2, n, d)

        # Category 1: normal — non-zero masked embedding, some foreground patches.
        cat1_embed = torch.randn(1, d)
        cat1_embed = cat1_embed / cat1_embed.norm(dim=-1, keepdim=True)

        # Category 2: zero-coverage — zero masked embedding, all-zero mask.
        cat2_embed = torch.zeros(1, d)

        masked_ref_embeddings = torch.stack([cat1_embed, cat2_embed])  # [2, 1, D]

        # Strictly binary mask (required by ReferenceFeatures.__post_init__).
        flatten_ref_masks = torch.zeros(2, n)
        flatten_ref_masks[0, :10] = 1.0   # cat1 has 10 foreground patches
        # cat2 remains all-zero

        matcher.ref_features = ReferenceFeatures(
            ref_embeddings=ref_embeddings,
            masked_ref_embeddings=masked_ref_embeddings,
            flatten_ref_masks=flatten_ref_masks,
            category_ids=[1, 2],
        )
        matcher.categories = CategoryRegistry._from_id_to_name({1: "cat", 2: "led"})

    @patch("instantlearn.models.matcher.matcher.load_sam_model")
    @patch("instantlearn.models.matcher.matcher.ImageEncoder")
    def test_predict_does_not_crash(
        self,
        mock_encoder_cls: MagicMock,
        mock_load_sam: MagicMock,
    ) -> None:
        """predict() must not raise with a zero-coverage category present."""
        matcher = self._build_matcher(mock_encoder_cls, mock_load_sam)
        self._inject_ref_features(matcher)

        target = Sample(
            image=np.zeros((self.IMG_H, self.IMG_W, 3), dtype=np.uint8),
            categories=[Category(0, "dummy")],
        )
        predictions = matcher.predict(target)   # must not raise

        assert isinstance(predictions, list)
        assert len(predictions) == 1

    @patch("instantlearn.models.matcher.matcher.load_sam_model")
    @patch("instantlearn.models.matcher.matcher.ImageEncoder")
    def test_zero_coverage_category_produces_no_detections(
        self,
        mock_encoder_cls: MagicMock,
        mock_load_sam: MagicMock,
    ) -> None:
        """Category 2 ('led', zero-coverage) must not appear in the detections."""
        matcher = self._build_matcher(mock_encoder_cls, mock_load_sam)
        self._inject_ref_features(matcher)

        target = Sample(
            image=np.zeros((self.IMG_H, self.IMG_W, 3), dtype=np.uint8),
            categories=[Category(0, "dummy")],
        )
        predictions = matcher.predict(target)
        pred = predictions[0]

        detected_ids = pred.label_ids.tolist() if pred.label_ids is not None else []
        assert 2 not in detected_ids, (
            "Zero-coverage category (id=2, 'led') must produce no detections"
        )

    @patch("instantlearn.models.matcher.matcher.load_sam_model")
    @patch("instantlearn.models.matcher.matcher.ImageEncoder")
    def test_sam_not_called_for_zero_coverage_category(
        self,
        mock_encoder_cls: MagicMock,
        mock_load_sam: MagicMock,
    ) -> None:
        """SAM forward() must only be invoked for category 1 (valid foreground prompts),
        never for category 2 (zero-coverage, 0-row point_coords)."""
        matcher = self._build_matcher(mock_encoder_cls, mock_load_sam)
        self._inject_ref_features(matcher)

        # Count SAM forward calls before and after predict
        sam_pred_mock = mock_load_sam.return_value
        sam_pred_mock.forward.reset_mock()

        target = Sample(
            image=np.zeros((self.IMG_H, self.IMG_W, 3), dtype=np.uint8),
            categories=[Category(0, "dummy")],
        )
        matcher.predict(target)

        # SAM may be called once (for cat-1 with valid prompts).
        # It must never be called for cat-2 (would crash with 0-row point_coords).
        call_count = sam_pred_mock.forward.call_count
        assert call_count <= 1, (
            f"SAM was called {call_count} times; expected at most 1 (only for cat-1). "
            "Category 2 (zero-coverage) must bypass SAM entirely."
        )

    @patch("instantlearn.models.matcher.matcher.load_sam_model")
    @patch("instantlearn.models.matcher.matcher.ImageEncoder")
    def test_all_zero_coverage_categories_graceful(
        self,
        mock_encoder_cls: MagicMock,
        mock_load_sam: MagicMock,
    ) -> None:
        """Edge case: all categories have zero coverage. predict() must return an
        empty-detection Prediction without calling SAM at all."""
        from instantlearn.components.feature_extractors.reference_features import ReferenceFeatures
        from instantlearn.models.torch_adapter import CategoryRegistry

        matcher = self._build_matcher(mock_encoder_cls, mock_load_sam)

        n, d = self.NUM_PATCHES, self.EMBED_DIM
        matcher.ref_features = ReferenceFeatures(
            ref_embeddings=torch.randn(2, n, d),
            masked_ref_embeddings=torch.zeros(2, 1, d),   # both zero-coverage
            flatten_ref_masks=torch.zeros(2, n),          # both all-zero
            category_ids=[3, 7],
        )
        matcher.categories = CategoryRegistry._from_id_to_name({3: "a", 7: "b"})

        sam_pred_mock = mock_load_sam.return_value
        sam_pred_mock.forward.reset_mock()

        target = Sample(
            image=np.zeros((self.IMG_H, self.IMG_W, 3), dtype=np.uint8),
            categories=[Category(0, "dummy")],
        )
        predictions = matcher.predict(target)   # must not raise

        sam_pred_mock.forward.assert_not_called()
        pred = predictions[0]
        assert pred.label_ids is not None
        assert pred.label_ids.shape[0] == 0, "All-zero-coverage run must produce 0 detections"

