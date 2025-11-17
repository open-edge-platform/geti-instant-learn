# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for SamDecoder."""

from unittest.mock import MagicMock

import pytest
import torch
from torchvision.tv_tensors import Image

from getiprompt.components.mask_decoder import SamDecoder


class TestSamDecoderValidation:
    """Test validation in SamDecoder for mask/point count matching."""

    @pytest.fixture
    def mock_sam_predictor(self) -> MagicMock:
        """Create a mock SAM predictor."""
        predictor = MagicMock()
        predictor.device = torch.device("cpu")

        # Mock model with image encoder
        mock_model = MagicMock()
        mock_model.image_encoder.img_size = 1024
        predictor.model = mock_model

        # Mock transform
        mock_transform = MagicMock()
        mock_transform.target_length = 1024
        mock_transform.apply_image_torch.return_value = torch.zeros((3, 1024, 1024), dtype=torch.uint8)
        mock_transform.apply_coords_torch.return_value = torch.tensor([[100, 150]], dtype=torch.float32)
        mock_transform.apply_boxes_torch.return_value = torch.tensor([[50, 50, 150, 150]], dtype=torch.float32)
        mock_transform.apply_inverse_coords_torch.return_value = torch.tensor([[100, 150]], dtype=torch.float32)
        mock_transform.apply_inverse_boxes.return_value = torch.tensor([[50, 50, 150, 150]], dtype=torch.float32)
        predictor.transform = mock_transform

        # Mock prediction methods
        predictor.set_torch_image.return_value = None
        predictor.predict_torch.return_value = (
            torch.zeros((1, 3, 1024, 1024), dtype=torch.bool),
            torch.tensor([0.8, 0.9, 0.7], dtype=torch.float32),
            torch.zeros((3, 256, 256), dtype=torch.float32),
        )

        return predictor

    @pytest.fixture
    def sam_decoder(self, mock_sam_predictor: MagicMock) -> SamDecoder:
        """Create a SamDecoder instance."""
        decoder = SamDecoder(sam_predictor=mock_sam_predictor)
        # Mock the transform that SamDecoder creates internally
        mock_transform = MagicMock()
        mock_transform.apply_image_torch.return_value = torch.zeros((3, 1024, 1024), dtype=torch.uint8)
        mock_transform.apply_coords_torch.return_value = torch.tensor([[100, 150]], dtype=torch.float32)
        mock_transform.apply_boxes_torch.return_value = torch.tensor([[50, 50, 150, 150]], dtype=torch.float32)
        mock_transform.apply_inverse_coords_torch.return_value = torch.tensor([[100, 150]], dtype=torch.float32)
        mock_transform.apply_inverse_boxes.return_value = torch.tensor([[50, 50, 150, 150]], dtype=torch.float32)
        decoder.transform = mock_transform
        return decoder

    def test_assertion_matching_masks_and_points(self, sam_decoder: SamDecoder) -> None:
        """Test that assertion passes when mask and point counts match."""
        # Create sample data
        image = Image(torch.zeros((3, 480, 640), dtype=torch.uint8))
        point_prompts = {0: torch.tensor([[100, 150, 0.9, 1]], dtype=torch.float32)}
        box_prompts = {}

        # Mock the predict method to return matching counts
        mock_masks = torch.ones((2, 480, 640), dtype=torch.bool)  # 2 masks
        # Format: [num_positive_points, 1_positive + N_negative, 3] where last dim is [x, y, score]
        # For 2 masks, we need 2 positive points, so shape should be [2, 1, 3]
        mock_points = torch.tensor([[[100, 150, 0.9]], [[200, 250, 0.8]]], dtype=torch.float32)  # 2 positive points
        mock_boxes = torch.empty((0, 6), dtype=torch.float32)

        sam_decoder.predict = MagicMock(return_value=(mock_masks, mock_points, mock_boxes))

        # This should not raise an assertion error
        predictions = sam_decoder.forward([image], [point_prompts], [box_prompts], None)

        # Should return a list of dictionaries (one per image)
        assert len(predictions) == 1
        assert isinstance(predictions[0], dict)
        assert "pred_masks" in predictions[0]
        assert "pred_points" in predictions[0]
        assert "pred_boxes" in predictions[0]
        assert "pred_labels" in predictions[0]

    def test_assertion_with_zero_masks_and_points(self, sam_decoder: SamDecoder) -> None:
        """Test that assertion passes with zero masks and points."""
        # Create sample data
        image = Image(torch.zeros((3, 480, 640), dtype=torch.uint8))
        point_prompts = {0: torch.tensor([[100, 150, 0.9, 1]], dtype=torch.float32)}
        box_prompts = {}

        # Mock the predict method to return zero counts
        mock_masks = torch.empty((0, 480, 640), dtype=torch.bool)  # 0 masks
        mock_points = torch.empty((0, 1, 3), dtype=torch.float32)  # 0 points
        mock_boxes = torch.empty((0, 6), dtype=torch.float32)

        sam_decoder.predict = MagicMock(return_value=(mock_masks, mock_points, mock_boxes))

        # This should not raise an assertion error
        result = sam_decoder.forward([image], [point_prompts], [box_prompts], None)

        assert len(result) == 1
        assert isinstance(result[0], dict)
        assert isinstance(result[0]["pred_masks"], torch.Tensor)
        assert isinstance(result[0]["pred_points"], torch.Tensor)
        assert isinstance(result[0]["pred_boxes"], torch.Tensor)
        assert isinstance(result[0]["pred_labels"], torch.Tensor)

    def test_assertion_with_multiple_classes(self, sam_decoder: SamDecoder) -> None:
        """Test assertion with multiple classes."""
        # Create sample data with multiple classes
        image = Image(torch.zeros((3, 480, 640), dtype=torch.uint8))
        point_prompts = {
            0: torch.tensor([[100, 150, 0.9, 1]], dtype=torch.float32),
            1: torch.tensor([[200, 250, 0.8, 1]], dtype=torch.float32),
        }
        box_prompts = {}

        # Mock the predict method to return matching counts for each class
        # For multiple classes, we need to mock the predict method to return different results for each class
        def mock_predict(
            points_per_class: torch.Tensor | None,
            *_args: object,
            **_kwargs: object,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            if points_per_class is not None and len(points_per_class) > 0:
                # Return 1 mask and 1 point for each class
                return (
                    torch.ones((1, 480, 640), dtype=torch.bool),  # 1 mask
                    torch.tensor([[[100, 150, 0.9]]], dtype=torch.float32),  # 1 point
                    torch.empty((0, 6), dtype=torch.float32),
                )
            return (
                torch.empty((0, 480, 640), dtype=torch.bool),
                torch.empty((0, 1, 3), dtype=torch.float32),
                torch.empty((0, 6), dtype=torch.float32),
            )

        sam_decoder.predict = mock_predict

        # This should not raise an assertion error
        result = sam_decoder.forward([image], [point_prompts], [box_prompts], None)

        assert len(result) == 1
        assert isinstance(result[0], dict)
        assert "pred_masks" in result[0]
        assert "pred_points" in result[0]
        assert "pred_boxes" in result[0]
        assert "pred_labels" in result[0]


class TestSamDecoderEmptyTensorHandling:
    """Test empty tensor handling in SamDecoder."""

    @pytest.fixture
    def mock_sam_predictor(self) -> MagicMock:
        """Create a mock SAM predictor."""
        predictor = MagicMock()
        predictor.device = torch.device("cpu")

        # Mock model with image encoder
        mock_model = MagicMock()
        mock_model.image_encoder.img_size = 1024
        predictor.model = mock_model

        # Mock transform
        mock_transform = MagicMock()
        mock_transform.target_length = 1024
        mock_transform.apply_image_torch.return_value = torch.zeros((3, 1024, 1024), dtype=torch.uint8)
        mock_transform.apply_coords_torch.return_value = torch.tensor([[100, 150]], dtype=torch.float32)
        mock_transform.apply_boxes_torch.return_value = torch.tensor([[50, 50, 150, 150]], dtype=torch.float32)
        mock_transform.apply_inverse_coords_torch.return_value = torch.tensor([[100, 150]], dtype=torch.float32)
        mock_transform.apply_inverse_boxes.return_value = torch.tensor([[50, 50, 150, 150]], dtype=torch.float32)
        predictor.transform = mock_transform

        # Mock prediction methods
        predictor.set_torch_image.return_value = None
        predictor.predict_torch.return_value = (
            torch.zeros((1, 3, 1024, 1024), dtype=torch.bool),
            torch.tensor([0.8, 0.9, 0.7], dtype=torch.float32),
            torch.zeros((3, 256, 256), dtype=torch.float32),
        )

        return predictor

    @pytest.fixture
    def sam_decoder(self, mock_sam_predictor: MagicMock) -> SamDecoder:
        """Create a SamDecoder instance."""
        decoder = SamDecoder(sam_predictor=mock_sam_predictor)
        # Mock the transform that SamDecoder creates internally
        mock_transform = MagicMock()
        mock_transform.apply_image_torch.return_value = torch.zeros((3, 1024, 1024), dtype=torch.uint8)
        mock_transform.apply_coords_torch.return_value = torch.tensor([[100, 150]], dtype=torch.float32)
        mock_transform.apply_boxes_torch.return_value = torch.tensor([[50, 50, 150, 150]], dtype=torch.float32)
        mock_transform.apply_inverse_coords_torch.return_value = torch.tensor([[100, 150]], dtype=torch.float32)
        mock_transform.apply_inverse_boxes.return_value = torch.tensor([[50, 50, 150, 150]], dtype=torch.float32)
        decoder.transform = mock_transform
        return decoder

    def test_empty_tensor_handling_for_empty_masks(self, sam_decoder: SamDecoder) -> None:
        """Test that empty tensors are properly handled when no masks are found."""
        # Create sample data
        image = Image(torch.zeros((3, 480, 640), dtype=torch.uint8))
        point_prompts = {0: torch.tensor([[100, 150, 0.9, 1]], dtype=torch.float32)}
        box_prompts = {}

        # Mock the predict method to return empty results
        mock_masks = torch.empty((0, 480, 640), dtype=torch.bool)  # No masks
        mock_points = torch.empty((0, 1, 3), dtype=torch.float32)  # No points
        mock_boxes = torch.empty((0, 6), dtype=torch.float32)  # No boxes

        sam_decoder.predict = MagicMock(return_value=(mock_masks, mock_points, mock_boxes))

        result = sam_decoder.forward([image], [point_prompts], [box_prompts], None)

        # Check that result is a list with one dictionary
        assert len(result) == 1
        prediction = result[0]
        assert isinstance(prediction, dict)

        # Check that empty tensors are properly added with correct shapes
        assert isinstance(prediction["pred_masks"], torch.Tensor)
        assert isinstance(prediction["pred_points"], torch.Tensor)
        assert isinstance(prediction["pred_boxes"], torch.Tensor)
        assert isinstance(prediction["pred_labels"], torch.Tensor)

        # Check that empty tensors have correct shapes
        assert prediction["pred_masks"].shape[0] == 0  # Empty masks
        assert prediction["pred_points"].shape[0] == 0  # Empty points
        assert prediction["pred_boxes"].shape[0] == 0  # Empty boxes
        assert prediction["pred_labels"].shape[0] == 0  # Empty labels

    def test_empty_tensor_handling_consistency(self, sam_decoder: SamDecoder) -> None:
        """Test that empty tensor handling maintains consistency across all outputs."""
        # Create sample data
        image = Image(torch.zeros((3, 480, 640), dtype=torch.uint8))
        point_prompts = {0: torch.tensor([[100, 150, 0.9, 1]], dtype=torch.float32)}
        box_prompts = {}

        # Mock the predict method to return empty results
        mock_masks = torch.empty((0, 480, 640), dtype=torch.bool)
        mock_points = torch.empty((0, 1, 3), dtype=torch.float32)
        mock_boxes = torch.empty((0, 6), dtype=torch.float32)

        sam_decoder.predict = MagicMock(return_value=(mock_masks, mock_points, mock_boxes))

        result = sam_decoder.forward([image], [point_prompts], [box_prompts], None)

        assert len(result) == 1
        prediction = result[0]

        # All outputs should be consistent - same number of items (0 in this case)
        assert prediction["pred_masks"].shape[0] == 0
        assert prediction["pred_points"].shape[0] == 0
        assert prediction["pred_boxes"].shape[0] == 0
        assert prediction["pred_labels"].shape[0] == 0

    def test_empty_tensor_handling_with_original_size(self, sam_decoder: SamDecoder) -> None:
        """Test that empty tensors are created with correct original size."""
        # Create sample data
        image = Image(torch.zeros((3, 480, 640), dtype=torch.uint8))
        point_prompts = {0: torch.tensor([[100, 150, 0.9, 1]], dtype=torch.float32)}
        box_prompts = {}

        # Mock the predict method to return empty results
        mock_masks = torch.empty((0, 480, 640), dtype=torch.bool)
        mock_points = torch.empty((0, 1, 3), dtype=torch.float32)
        mock_boxes = torch.empty((0, 6), dtype=torch.float32)

        sam_decoder.predict = MagicMock(return_value=(mock_masks, mock_points, mock_boxes))

        result = sam_decoder.forward([image], [point_prompts], [box_prompts], None)

        assert len(result) == 1
        prediction = result[0]

        # Check that empty masks have the correct original size
        empty_mask = prediction["pred_masks"]
        assert empty_mask.shape[1:] == (480, 640)  # Original size (H, W)

    def test_empty_tensor_handling_with_multiple_classes(self, sam_decoder: SamDecoder) -> None:
        """Test empty tensor handling with multiple classes."""
        # Create sample data with multiple classes
        image = Image(torch.zeros((3, 480, 640), dtype=torch.uint8))
        # Format: [x, y, score, label] where label=1 for foreground points
        point_prompts = {
            0: torch.tensor([[100, 150, 0.9, 1]], dtype=torch.float32),
            1: torch.tensor([[200, 250, 0.8, 1]], dtype=torch.float32),
        }
        box_prompts = {}

        # Mock the predict method to return empty results
        mock_masks = torch.empty((0, 480, 640), dtype=torch.bool)
        mock_points = torch.empty((0, 1, 3), dtype=torch.float32)
        mock_boxes = torch.empty((0, 6), dtype=torch.float32)

        sam_decoder.predict = MagicMock(return_value=(mock_masks, mock_points, mock_boxes))

        result = sam_decoder.forward([image], [point_prompts], [box_prompts], None)

        assert len(result) == 1
        prediction = result[0]

        # Check that all outputs are empty tensors
        assert prediction["pred_masks"].shape[0] == 0
        assert prediction["pred_points"].shape[0] == 0
        assert prediction["pred_boxes"].shape[0] == 0
        assert prediction["pred_labels"].shape[0] == 0

    def test_empty_tensor_handling_edge_cases(self, sam_decoder: SamDecoder) -> None:
        """Test edge cases in empty tensor handling."""
        # Create sample data
        image = Image(torch.zeros((3, 480, 640), dtype=torch.uint8))
        point_prompts = {0: torch.tensor([[100, 150, 0.9, 1]], dtype=torch.float32)}
        box_prompts = {}

        # Mock the predict method to return empty results
        mock_masks = torch.empty((0, 480, 640), dtype=torch.bool)
        mock_points = torch.empty((0, 1, 3), dtype=torch.float32)
        mock_boxes = torch.empty((0, 6), dtype=torch.float32)

        sam_decoder.predict = MagicMock(return_value=(mock_masks, mock_points, mock_boxes))

        result = sam_decoder.forward([image], [point_prompts], [box_prompts], None)

        # Test that the method doesn't crash with empty inputs
        assert len(result) == 1
        prediction = result[0]

        # Test that empty tensors are properly created
        assert isinstance(prediction["pred_masks"], torch.Tensor)
        assert isinstance(prediction["pred_points"], torch.Tensor)
        assert isinstance(prediction["pred_boxes"], torch.Tensor)
        assert isinstance(prediction["pred_labels"], torch.Tensor)

    def test_empty_tensor_handling_with_different_sizes(self, sam_decoder: SamDecoder) -> None:
        """Test empty tensor handling with different image sizes."""
        # Create sample data with different image size
        image = Image(torch.zeros((3, 320, 480), dtype=torch.uint8))
        point_prompts = {0: torch.tensor([[100, 150, 0.9, 1]], dtype=torch.float32)}
        box_prompts = {}

        # Mock the predict method to return empty results
        mock_masks = torch.empty((0, 320, 480), dtype=torch.bool)
        mock_points = torch.empty((0, 1, 3), dtype=torch.float32)
        mock_boxes = torch.empty((0, 6), dtype=torch.float32)

        sam_decoder.predict = MagicMock(return_value=(mock_masks, mock_points, mock_boxes))

        result = sam_decoder.forward([image], [point_prompts], [box_prompts], None)

        assert len(result) == 1
        prediction = result[0]

        # Check that empty masks have the correct original size
        empty_mask = prediction["pred_masks"]
        assert empty_mask.shape[1:] == (320, 480)  # Original size (H, W)
