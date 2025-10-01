# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Test cases for the refactored SamDecoder."""

from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from segment_anything_hq.predictor import SamPredictor as SamHQPredictor

from getiprompt.processes.segmenters.sam_decoder import SamDecoder
from getiprompt.types import Image, Masks, Points, Priors, Similarities


@pytest.fixture
def mock_sam_predictor() -> SamHQPredictor:
    """Create a mock SAM predictor for testing."""
    predictor = MagicMock(spec=SamHQPredictor)
    predictor.device = torch.device("cpu")

    # Create a proper mock for the transform attribute
    mock_transform = MagicMock()
    mock_transform.target_length = 1024
    mock_transform.apply_image.return_value = np.zeros((1024, 1024, 3), dtype=np.uint8)

    # Mock apply_coords_torch to return same shape as input (preserving batch dimension)
    def mock_apply_coords_torch(coords: torch.Tensor, original_size: tuple[int, int]) -> torch.Tensor:
        # Apply simple scaling transformation while preserving input shape
        scale_x, scale_y = 1024 / original_size[1], 1024 / original_size[0]
        transformed = coords.clone()
        transformed[..., 0] = coords[..., 0] * scale_x
        transformed[..., 1] = coords[..., 1] * scale_y
        return transformed

    mock_transform.apply_coords_torch.side_effect = mock_apply_coords_torch
    predictor.transform = mock_transform

    # Mock prediction methods
    predictor.set_torch_image.return_value = None

    # Mock predict_torch to return realistic output shapes
    masks = torch.zeros((1, 3, 1024, 1024), dtype=torch.bool)  # 3 masks from multimask_output
    mask_scores = torch.tensor([0.8, 0.9, 0.7], dtype=torch.float32)
    low_res_logits = torch.zeros((3, 256, 256), dtype=torch.float32)
    predictor.predict_torch.return_value = (masks, mask_scores, low_res_logits)

    return predictor


@pytest.fixture
def sam_decoder(mock_sam_predictor: SamHQPredictor) -> SamDecoder:
    """Create a SamDecoder instance with mock predictor."""
    return SamDecoder(sam_predictor=mock_sam_predictor, mask_similarity_threshold=0.38, nms_iou_threshold=0.1)


@pytest.fixture
def sample_image() -> Image:
    """Create a sample image for testing."""
    return Image(np.zeros((480, 640, 3), dtype=np.uint8))


@pytest.fixture
def sample_priors() -> Priors:
    """Create sample priors with points."""
    priors = Priors()
    # Points tensor: [x, y, score, label] where label 1=foreground, 0=background
    points = torch.tensor(
        [
            [100, 150, 0.9, 1],  # foreground point
            [200, 250, 0.8, 0],  # background point
            [300, 350, 0.7, 1],  # another foreground point
        ],
        dtype=torch.float32,
    )
    priors.points.add(points, class_id=0)
    return priors


@pytest.fixture
def sample_similarities() -> Similarities:
    """Create sample similarities for testing."""
    similarities = Similarities()
    similarity_map = torch.ones((1, 480, 640), dtype=torch.float32) * 0.5
    similarities.add(similarity_map, class_id=0)
    return similarities


class TestSamDecoderBasic:
    """Basic functionality tests for SamDecoder."""

    @staticmethod
    def test_initialization_with_defaults(mock_sam_predictor: SamHQPredictor) -> None:
        """Test initialization with default parameters."""
        decoder = SamDecoder(sam_predictor=mock_sam_predictor)

        assert decoder.predictor is mock_sam_predictor
        assert decoder.mask_similarity_threshold == 0.38
        assert decoder.nms_iou_threshold == 0.1

    @staticmethod
    def test_preprocess_inputs_basic(
        sam_decoder: SamDecoder,
        sample_image: Image,
        sample_priors: Priors,
    ) -> None:
        """Test input preprocessing."""
        preprocessed_images, preprocessed_points, original_sizes = sam_decoder.preprocess_inputs(
            [sample_image],
            [sample_priors],
        )

        assert len(preprocessed_images) == 1
        assert len(preprocessed_points) == 1
        assert len(original_sizes) == 1
        assert original_sizes[0] == (480, 640)

    @staticmethod
    def test_forward_basic(
        sam_decoder: SamDecoder,
        sample_image: Image,
        sample_priors: Priors,
        sample_similarities: Similarities,
    ) -> None:
        """Test basic forward pass."""
        # Mock the predict_by_points method
        mock_masks = Masks()
        mock_masks.add(torch.ones((100, 100), dtype=torch.bool), class_id=0)
        mock_points = Points()
        mock_points.add(torch.tensor([[100, 150, 0.9, 1]], dtype=torch.float32), class_id=0)

        sam_decoder.predict_by_points = MagicMock(return_value=(mock_masks, mock_points))

        masks_per_image, points_per_image, boxes_per_image = sam_decoder.forward(
            [sample_image], [sample_priors], [sample_similarities]
        )

        assert len(masks_per_image) == 1
        assert len(points_per_image) == 1
        assert len(boxes_per_image) == 1
        assert isinstance(masks_per_image[0], Masks)
        assert isinstance(points_per_image[0], Points)
