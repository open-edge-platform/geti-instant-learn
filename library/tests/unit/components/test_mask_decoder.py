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
