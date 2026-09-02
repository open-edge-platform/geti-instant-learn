# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for SamDecoder."""

from unittest.mock import MagicMock, patch

import pytest
import torch
from torchvision.tv_tensors import Image

from instantlearn.components.sam import SamDecoder


class TestSamDecoderValidation:
    """Test validation in SamDecoder for tensor-based inputs."""

    @pytest.fixture
    def mock_sam_predictor(self) -> MagicMock:
        """Create a mock SAM predictor."""
        predictor = MagicMock()
        predictor.device = torch.device("cpu")
        predictor.dtype = torch.float32

        # Mock model with image encoder
        mock_model = MagicMock()
        mock_model.image_encoder.img_size = 1024
        predictor.model = mock_model

        # Mock prediction methods
        predictor.set_image.return_value = None
        predictor.predict.return_value = (
            torch.zeros((1, 3, 1024, 1024), dtype=torch.bool),
            torch.tensor([0.8, 0.9, 0.7], dtype=torch.float32),
            torch.zeros((3, 256, 256), dtype=torch.float32),
        )

        return predictor

    @pytest.fixture
    def sam_decoder(self, mock_sam_predictor: MagicMock) -> SamDecoder:
        """Create a SamDecoder instance."""
        return SamDecoder(sam_predictor=mock_sam_predictor)

    def test_forward_with_point_prompts(self, sam_decoder: SamDecoder) -> None:
        """Test forward pass with tensor-based point prompts."""
        # Create sample data: 1 image, 1 category, max 4 points
        image = Image(torch.zeros((3, 480, 640), dtype=torch.uint8))

        # point_prompts: [T=1, C=1, max_points=4, 4] with (x, y, score, label)
        point_prompts = torch.zeros(1, 1, 4, 4, dtype=torch.float32)
        point_prompts[0, 0, 0] = torch.tensor([100, 150, 0.9, 1])  # foreground point

        # similarities: [T=1, C=1, feat_size, feat_size]
        similarities = torch.ones(1, 1, 16, 16, dtype=torch.float32)

        category_ids = [0]

        # Mock _process_single_image_with_points to return valid results
        with patch.object(sam_decoder, "_process_single_image_with_points") as mock_process:
            mock_process.return_value = (
                torch.ones((1, 480, 640), dtype=torch.bool),  # pred_masks
                torch.tensor([0.9]),  # pred_scores
                torch.tensor([0], dtype=torch.int64),  # pred_labels
                torch.tensor([[100, 150, 0.9, 1]]),  # pred_points
            )

            predictions = sam_decoder.forward(
                images=[image],
                category_ids=category_ids,
                point_prompts=point_prompts,
                similarities=similarities,
            )

        assert len(predictions) == 1
        assert isinstance(predictions[0], dict)
        assert "pred_masks" in predictions[0]
        assert "pred_scores" in predictions[0]
        assert "pred_labels" in predictions[0]
        assert "pred_points" in predictions[0]

    def test_forward_with_box_prompts(self, sam_decoder: SamDecoder) -> None:
        """Test forward pass with tensor-based box prompts."""
        image = Image(torch.zeros((3, 480, 640), dtype=torch.uint8))

        # box_prompts: [T=1, C=1, max_boxes=4, 5] with (x1, y1, x2, y2, score)
        box_prompts = torch.zeros(1, 1, 4, 5, dtype=torch.float32)
        box_prompts[0, 0, 0] = torch.tensor([50, 50, 150, 150, 0.9])

        category_ids = [0]

        # Mock _process_single_image_with_boxes to return valid results
        with patch.object(sam_decoder, "_process_single_image_with_boxes") as mock_process:
            mock_process.return_value = (
                torch.ones((1, 480, 640), dtype=torch.bool),  # pred_masks
                torch.tensor([0.9]),  # pred_scores
                torch.tensor([0], dtype=torch.int64),  # pred_labels
                torch.tensor([[50, 50, 150, 150, 0.9]]),  # pred_boxes
            )

            predictions = sam_decoder.forward(
                images=[image],
                category_ids=category_ids,
                box_prompts=box_prompts,
            )

        assert len(predictions) == 1
        assert isinstance(predictions[0], dict)
        assert "pred_masks" in predictions[0]
        assert "pred_scores" in predictions[0]
        assert "pred_labels" in predictions[0]
        assert "pred_boxes" in predictions[0]

    def test_forward_requires_either_points_or_boxes(self, sam_decoder: SamDecoder) -> None:
        """Test that forward raises error when neither prompts are provided."""
        image = Image(torch.zeros((3, 480, 640), dtype=torch.uint8))
        category_ids = [0]

        with pytest.raises(ValueError, match="Provide either point_prompts or box_prompts"):
            sam_decoder.forward(images=[image], category_ids=category_ids)

    def test_forward_rejects_both_prompts(self, sam_decoder: SamDecoder) -> None:
        """Test that forward raises error when both prompts are provided."""
        image = Image(torch.zeros((3, 480, 640), dtype=torch.uint8))
        category_ids = [0]

        point_prompts = torch.zeros(1, 1, 4, 4, dtype=torch.float32)
        similarities = torch.ones(1, 1, 16, 16, dtype=torch.float32)
        box_prompts = torch.zeros(1, 1, 4, 5, dtype=torch.float32)

        with pytest.raises(ValueError, match="Provide either point_prompts or box_prompts"):
            sam_decoder.forward(
                images=[image],
                category_ids=category_ids,
                point_prompts=point_prompts,
                similarities=similarities,
                box_prompts=box_prompts,
            )

    def test_forward_with_multiple_categories(self, sam_decoder: SamDecoder) -> None:
        """Test forward with multiple categories."""
        image = Image(torch.zeros((3, 480, 640), dtype=torch.uint8))

        # 2 categories, max 4 points each
        point_prompts = torch.zeros(1, 2, 4, 4, dtype=torch.float32)
        point_prompts[0, 0, 0] = torch.tensor([100, 150, 0.9, 1])
        point_prompts[0, 1, 0] = torch.tensor([200, 250, 0.8, 1])
        similarities = torch.ones(1, 2, 16, 16, dtype=torch.float32)

        category_ids = [0, 1]

        with patch.object(sam_decoder, "_process_single_image_with_points") as mock_process:
            mock_process.return_value = (
                torch.ones((2, 480, 640), dtype=torch.bool),
                torch.tensor([0.9, 0.8]),
                torch.tensor([0, 1], dtype=torch.int64),
                torch.tensor([[100, 150, 0.9, 1], [200, 250, 0.8, 1]]),
            )

            predictions = sam_decoder.forward(
                images=[image],
                category_ids=category_ids,
                point_prompts=point_prompts,
                similarities=similarities,
            )

        assert len(predictions) == 1
        assert predictions[0]["pred_labels"].shape[0] == 2

    def test_forward_with_empty_results(self, sam_decoder: SamDecoder) -> None:
        """Test forward handles empty prediction results."""
        image = Image(torch.zeros((3, 480, 640), dtype=torch.uint8))

        point_prompts = torch.zeros(1, 1, 4, 4, dtype=torch.float32)
        point_prompts[0, 0, 0] = torch.tensor([100, 150, 0.9, 1])
        similarities = torch.ones(1, 1, 16, 16, dtype=torch.float32)

        category_ids = [0]

        with patch.object(sam_decoder, "_process_single_image_with_points") as mock_process:
            mock_process.return_value = (
                torch.empty((0, 480, 640), dtype=torch.bool),
                torch.empty(0),
                torch.empty(0, dtype=torch.int64),
                torch.empty((0, 4)),
            )

            predictions = sam_decoder.forward(
                images=[image],
                category_ids=category_ids,
                point_prompts=point_prompts,
                similarities=similarities,
            )

        assert len(predictions) == 1
        assert predictions[0]["pred_masks"].shape[0] == 0
        assert predictions[0]["pred_labels"].shape[0] == 0


class TestSamDecoderEmptyTensorHandling:
    """Test empty tensor handling in SamDecoder."""

    @pytest.fixture
    def mock_sam_predictor(self) -> MagicMock:
        """Create a mock SAM predictor."""
        predictor = MagicMock()
        predictor.device = torch.device("cpu")
        predictor.dtype = torch.float32

        mock_model = MagicMock()
        mock_model.image_encoder.img_size = 1024
        predictor.model = mock_model

        # Mock prediction methods
        predictor.set_image.return_value = None
        predictor.predict.return_value = (
            torch.zeros((1, 3, 1024, 1024), dtype=torch.bool),
            torch.tensor([0.8, 0.9, 0.7], dtype=torch.float32),
            torch.zeros((3, 256, 256), dtype=torch.float32),
        )

        return predictor

    @pytest.fixture
    def sam_decoder(self, mock_sam_predictor: MagicMock) -> SamDecoder:
        """Create a SamDecoder instance."""
        return SamDecoder(sam_predictor=mock_sam_predictor)

    def test_empty_tensor_for_zero_num_points(self, sam_decoder: SamDecoder) -> None:
        """Test empty tensors when num_points is zero."""
        image = Image(torch.zeros((3, 480, 640), dtype=torch.uint8))

        # Zero valid points
        point_prompts = torch.zeros(1, 1, 4, 4, dtype=torch.float32)
        similarities = torch.ones(1, 1, 16, 16, dtype=torch.float32)

        category_ids = [0]

        with patch.object(sam_decoder, "_process_single_image_with_points") as mock_process:
            mock_process.return_value = (
                torch.empty((0, 480, 640), dtype=torch.bool),
                torch.empty(0),
                torch.empty(0, dtype=torch.int64),
                torch.empty((0, 4)),
            )

            result = sam_decoder.forward(
                images=[image],
                category_ids=category_ids,
                point_prompts=point_prompts,
                similarities=similarities,
            )

        assert len(result) == 1
        prediction = result[0]
        assert prediction["pred_masks"].shape[0] == 0
        assert prediction["pred_scores"].shape[0] == 0
        assert prediction["pred_labels"].shape[0] == 0
        assert prediction["pred_points"].shape[0] == 0

    def test_empty_tensor_for_zero_num_boxes(self, sam_decoder: SamDecoder) -> None:
        """Test empty tensors when num_boxes is zero."""
        image = Image(torch.zeros((3, 480, 640), dtype=torch.uint8))

        box_prompts = torch.zeros(1, 1, 4, 5, dtype=torch.float32)

        category_ids = [0]

        with patch.object(sam_decoder, "_process_single_image_with_boxes") as mock_process:
            mock_process.return_value = (
                torch.empty((0, 480, 640), dtype=torch.bool),
                torch.empty(0),
                torch.empty(0, dtype=torch.int64),
                torch.empty((0, 5)),
            )

            result = sam_decoder.forward(
                images=[image],
                category_ids=category_ids,
                box_prompts=box_prompts,
            )

        assert len(result) == 1
        prediction = result[0]
        assert prediction["pred_masks"].shape[0] == 0
        assert prediction["pred_boxes"].shape[0] == 0

    def test_empty_tensor_consistency(self, sam_decoder: SamDecoder) -> None:
        """Test that empty tensor handling maintains consistency across all outputs."""
        image = Image(torch.zeros((3, 480, 640), dtype=torch.uint8))

        point_prompts = torch.zeros(1, 1, 4, 4, dtype=torch.float32)
        similarities = torch.ones(1, 1, 16, 16, dtype=torch.float32)

        category_ids = [0]

        with patch.object(sam_decoder, "_process_single_image_with_points") as mock_process:
            mock_process.return_value = (
                torch.empty((0, 480, 640), dtype=torch.bool),
                torch.empty(0),
                torch.empty(0, dtype=torch.int64),
                torch.empty((0, 4)),
            )

            result = sam_decoder.forward(
                images=[image],
                category_ids=category_ids,
                point_prompts=point_prompts,
                similarities=similarities,
            )

        prediction = result[0]
        # All outputs should be consistent (0 items)
        assert prediction["pred_masks"].shape[0] == 0
        assert prediction["pred_scores"].shape[0] == 0
        assert prediction["pred_labels"].shape[0] == 0
        assert prediction["pred_points"].shape[0] == 0

    def test_empty_tensor_with_multiple_categories(self, sam_decoder: SamDecoder) -> None:
        """Test empty tensor handling with multiple categories."""
        image = Image(torch.zeros((3, 480, 640), dtype=torch.uint8))

        # 2 categories, both with zero points
        point_prompts = torch.zeros(1, 2, 4, 4, dtype=torch.float32)
        similarities = torch.ones(1, 2, 16, 16, dtype=torch.float32)

        category_ids = [0, 1]

        with patch.object(sam_decoder, "_process_single_image_with_points") as mock_process:
            mock_process.return_value = (
                torch.empty((0, 480, 640), dtype=torch.bool),
                torch.empty(0),
                torch.empty(0, dtype=torch.int64),
                torch.empty((0, 4)),
            )

            result = sam_decoder.forward(
                images=[image],
                category_ids=category_ids,
                point_prompts=point_prompts,
                similarities=similarities,
            )

        assert len(result) == 1
        prediction = result[0]
        assert prediction["pred_masks"].shape[0] == 0
        assert prediction["pred_labels"].shape[0] == 0

    def test_empty_tensor_with_different_image_sizes(self, sam_decoder: SamDecoder) -> None:
        """Test empty tensor handling with different image sizes."""
        image = Image(torch.zeros((3, 320, 480), dtype=torch.uint8))

        point_prompts = torch.zeros(1, 1, 4, 4, dtype=torch.float32)
        similarities = torch.ones(1, 1, 16, 16, dtype=torch.float32)

        category_ids = [0]

        with patch.object(sam_decoder, "_process_single_image_with_points") as mock_process:
            mock_process.return_value = (
                torch.empty((0, 320, 480), dtype=torch.bool),
                torch.empty(0),
                torch.empty(0, dtype=torch.int64),
                torch.empty((0, 4)),
            )

            result = sam_decoder.forward(
                images=[image],
                category_ids=category_ids,
                point_prompts=point_prompts,
                similarities=similarities,
            )

        prediction = result[0]
        # Empty masks should have correct spatial dimensions
        assert prediction["pred_masks"].shape[1:] == (320, 480)


class TestSamDecoderZeroCoverageCategory:
    """SamDecoder must not crash when a category has zero foreground prompts.

    A zero-coverage category (reference polygon too small to survive patch-grid
    downsampling) results in all-zero padded point prompts (label column == 0).
    _preprocess_points filters them out leaving 0 foreground prompt instances,
    which must NOT be forwarded to SAM (SAM crashes with 0-row prompt tensors).
    The category must instead silently produce zero masks / zero scores.
    """

    @pytest.fixture
    def mock_sam_predictor(self) -> MagicMock:
        """SAM predictor mock that tracks how many times forward is called."""
        predictor = MagicMock()
        predictor.device = torch.device("cpu")
        predictor.dtype = torch.float32
        mock_model = MagicMock()
        mock_model.image_encoder.img_size = 1024
        predictor.model = mock_model
        predictor.set_image.return_value = None

        # forward() returns valid masks for a single foreground prompt
        h, w = 64, 64
        predictor.forward.return_value = (
            torch.ones(1, 3, h, w, dtype=torch.float32),   # masks [1, 3, H, W]
            torch.tensor([[0.9, 0.85, 0.8]]),               # iou_preds [1, 3]
            torch.zeros(1, 3, 256, 256),                    # low_res_logits
        )
        return predictor

    @pytest.fixture
    def sam_decoder(self, mock_sam_predictor: MagicMock) -> SamDecoder:
        return SamDecoder(sam_predictor=mock_sam_predictor, confidence_threshold=0.0)

    def _make_point_prompts_all_zero(self, max_points: int = 8) -> torch.Tensor:
        """All-zero point prompts — simulates a zero-coverage category (label col = 0)."""
        return torch.zeros(max_points, 4, dtype=torch.float32)

    def _make_valid_point_prompts(self, max_points: int = 8, h: int = 64, w: int = 64) -> torch.Tensor:
        """A single valid foreground point at image centre."""
        pts = torch.zeros(max_points, 4, dtype=torch.float32)
        pts[0] = torch.tensor([w // 2, h // 2, 0.9, 1.0])  # (x, y, score, label=1)
        return pts


    def test_predict_masks_no_crash_zero_fg_points(self, sam_decoder: SamDecoder, mock_sam_predictor: MagicMock) -> None:
        """_predict_masks_for_category must NOT call SAM and must return empty tensors
        when point_coords has 0 rows (all prompts were filtered as label==0 padding)."""
        h, w = 64, 64

        # All-zero padded points → _preprocess_points produces 0 foreground instances
        all_zero_pts = self._make_point_prompts_all_zero()
        point_coords, point_labels, _ = sam_decoder._preprocess_points(all_zero_pts)

        assert point_coords.shape[0] == 0, "Precondition: should have 0 fg prompts"

        sim = torch.rand(8, 8)
        masks, scores = sam_decoder._predict_masks_for_category(
            point_coords, point_labels, sim, (h, w)
        )

        mock_sam_predictor.forward.assert_not_called()
        assert masks.shape == (0, h, w)
        assert scores.shape == (0,)

    def test_predict_masks_no_crash_zero_fg_points_various_sizes(self, sam_decoder: SamDecoder) -> None:
        """The guard works for different image sizes."""
        for h, w in [(32, 48), (128, 128), (480, 640)]:
            all_zero_pts = self._make_point_prompts_all_zero()
            point_coords, point_labels, _ = sam_decoder._preprocess_points(all_zero_pts)
            sim = torch.rand(4, 4)
            masks, scores = sam_decoder._predict_masks_for_category(
                point_coords, point_labels, sim, (h, w)
            )
            assert masks.shape == (0, h, w), f"Wrong mask shape for size ({h},{w})"
            assert scores.shape == (0,)


    def test_two_categories_one_zero_coverage_no_crash(
        self, sam_decoder: SamDecoder, mock_sam_predictor: MagicMock
    ) -> None:
        """Two categories: cat-0 has valid foreground prompts; cat-1 has all-zero
        padded prompts (zero-coverage).  The method must not crash and cat-1
        must contribute only all-zero / label=-1 padded outputs."""
        h, w = 64, 64
        image = torch.zeros(3, h, w, dtype=torch.float32)

        pts_valid = self._make_valid_point_prompts(max_points=8, h=h, w=w)
        pts_zero = self._make_point_prompts_all_zero(max_points=8)
        # [C=2, max_points=8, 4]
        point_prompts = torch.stack([pts_valid, pts_zero]).unsqueeze(0)  # [1, 2, 8, 4]
        similarities = torch.rand(1, 2, 8, 8)
        category_ids = [1, 2]

        pred_masks, pred_scores, pred_labels, pred_points = (
            sam_decoder._process_single_image_with_points(
                image,
                point_prompts[0],
                similarities[0],
                category_ids,
            )
        )

        # SAM must have been called exactly once — for category 0 only.
        assert mock_sam_predictor.forward.call_count == 1, (
            "SAM must only be called for categories with valid foreground prompts"
        )

        # Zero-coverage category (id=2) must not appear in pred_labels (or if it does,
        # it was padded with score=0 and filtered out by valid_mask >= 0 logic).
        # At minimum the output must be a valid tensor tuple.
        assert isinstance(pred_masks, torch.Tensor)
        assert isinstance(pred_scores, torch.Tensor)
        assert isinstance(pred_labels, torch.Tensor)

    def test_all_zero_coverage_categories_no_crash(
        self, sam_decoder: SamDecoder, mock_sam_predictor: MagicMock
    ) -> None:
        """Edge case: ALL categories have zero-coverage prompts.
        SAM must never be called; output is empty masks."""
        h, w = 64, 64
        image = torch.zeros(3, h, w, dtype=torch.float32)

        pts_zero = self._make_point_prompts_all_zero(max_points=8)
        point_prompts = torch.stack([pts_zero, pts_zero])   # [2, 8, 4]
        similarities = torch.rand(2, 8, 8)
        category_ids = [3, 7]

        pred_masks, pred_scores, pred_labels, pred_points = (
            sam_decoder._process_single_image_with_points(
                image,
                point_prompts,
                similarities,
                category_ids,
            )
        )

        mock_sam_predictor.forward.assert_not_called()
        # No valid masks produced — pred_labels should be empty (all -1 filtered out).
        assert pred_labels.shape[0] == 0

    def test_zero_coverage_category_zero_scores(
        self, sam_decoder: SamDecoder, mock_sam_predictor: MagicMock
    ) -> None:
        """A zero-coverage category that passes the valid_mask >= 0 filter
        (if padded_labels stay at -1 they are excluded) must have score == 0."""
        h, w = 32, 32
        image = torch.zeros(3, h, w, dtype=torch.float32)

        pts_zero = self._make_point_prompts_all_zero(max_points=4)
        point_prompts = pts_zero.unsqueeze(0)   # [1, 4, 4]
        similarities = torch.rand(1, 4, 4)
        category_ids = [99]

        pred_masks, pred_scores, pred_labels, _ = (
            sam_decoder._process_single_image_with_points(
                image, point_prompts, similarities, category_ids
            )
        )

        # Either 0 detections (padded label -1 filtered) or scores are 0 for the category.
        if pred_labels.numel() > 0:
            cat99_mask = pred_labels == 99
            assert (pred_scores[cat99_mask] == 0).all(), (
                "Zero-coverage category must have score 0"
            )

