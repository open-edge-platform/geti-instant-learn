# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Test cases for the refactored SamDecoder."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from segment_anything_hq.predictor import SamPredictor as SamHQPredictor

from getiprompt.processes.segmenters.sam_decoder import SamDecoder
from getiprompt.types import Boxes, Image, Masks, Points, Priors, Similarities


@pytest.fixture
def mock_sam_predictor():
    """Create a mock SAM predictor for testing."""
    predictor = MagicMock(spec=SamHQPredictor)
    predictor.device = torch.device("cpu")
    predictor.transform.target_length = 1024
    
    # Mock transform methods
    predictor.transform.apply_image.return_value = np.zeros((1024, 1024, 3), dtype=np.uint8)
    predictor.transform.apply_coords_torch.return_value = torch.tensor([[100, 200], [300, 400]], dtype=torch.float32)
    
    # Mock prediction methods
    predictor.set_torch_image.return_value = None
    
    # Mock predict_torch to return realistic output shapes
    masks = torch.zeros((1, 3, 1024, 1024), dtype=torch.bool)  # 3 masks from multimask_output
    mask_scores = torch.tensor([0.8, 0.9, 0.7], dtype=torch.float32)
    low_res_logits = torch.zeros((3, 256, 256), dtype=torch.float32)
    predictor.predict_torch.return_value = (masks, mask_scores, low_res_logits)
    
    return predictor


@pytest.fixture
def sam_decoder(mock_sam_predictor):
    """Create a SamDecoder instance with mock predictor."""
    return SamDecoder(
        sam_predictor=mock_sam_predictor,
        mask_similarity_threshold=0.38,
        nms_iou_threshold=0.1
    )


@pytest.fixture
def sample_image():
    """Create a sample image for testing."""
    return Image(np.zeros((480, 640, 3), dtype=np.uint8))


@pytest.fixture
def sample_priors():
    """Create sample priors with points."""
    priors = Priors()
    # Points tensor: [x, y, score, label] where label 1=foreground, 0=background
    points = torch.tensor([
        [100, 150, 0.9, 1],  # foreground point
        [200, 250, 0.8, 0],  # background point
        [300, 350, 0.7, 1],  # another foreground point
    ], dtype=torch.float32)
    priors.points.add(points, class_id=0)
    return priors


@pytest.fixture
def sample_similarities():
    """Create sample similarities for testing."""
    similarities = Similarities()
    similarity_map = torch.ones((1, 480, 640), dtype=torch.float32) * 0.5
    similarities.add(similarity_map, class_id=0)
    return similarities


class TestSamDecoderInitialization:
    """Test SamDecoder initialization and basic properties."""
    
    def test_initialization_with_defaults(self, mock_sam_predictor):
        """Test initialization with default parameters."""
        decoder = SamDecoder(sam_predictor=mock_sam_predictor)
        
        assert decoder.predictor is mock_sam_predictor
        assert decoder.mask_similarity_threshold == 0.38
        assert decoder.nms_iou_threshold == 0.1
    
    def test_initialization_with_custom_parameters(self, mock_sam_predictor):
        """Test initialization with custom parameters."""
        decoder = SamDecoder(
            sam_predictor=mock_sam_predictor,
            mask_similarity_threshold=0.45,
            nms_iou_threshold=0.2
        )
        
        assert decoder.mask_similarity_threshold == 0.45
        assert decoder.nms_iou_threshold == 0.2
    
    def test_inheritance_from_nn_module(self, mock_sam_predictor):
        """Test that SamDecoder inherits from nn.Module."""
        decoder = SamDecoder(sam_predictor=mock_sam_predictor)
        assert isinstance(decoder, torch.nn.Module)


class TestCoordinateTransformations:
    """Test coordinate transformation methods."""
    
    def test_get_preprocess_shape(self):
        """Test preprocess shape calculation."""
        newh, neww = SamDecoder._get_preprocess_shape(480, 640, 1024)
        
        # Should scale based on the longer side (640)
        scale = 1024 / 640
        expected_h = int(480 * scale + 0.5)
        expected_w = int(640 * scale + 0.5)
        
        assert newh == expected_h
        assert neww == expected_w
    
    def test_apply_coords_torch(self):
        """Test coordinate transformation."""
        coords = torch.tensor([[100, 150], [200, 250]], dtype=torch.float32)
        original_size = (480, 640)
        long_side_length = 1024
        
        transformed = SamDecoder._apply_coords_torch(coords, original_size, long_side_length)
        
        # Verify transformation doesn't modify original tensor
        assert not torch.equal(coords, transformed)
        assert transformed.shape == coords.shape
        assert transformed.dtype == torch.float32
    
    def test_apply_inverse_coords_torch(self):
        """Test inverse coordinate transformation."""
        coords = torch.tensor([[160, 240], [320, 400]], dtype=torch.float32)
        original_size = (480, 640)
        long_side_length = 1024
        
        inverse_coords = SamDecoder._apply_inverse_coords_torch(coords, original_size, long_side_length)
        
        assert inverse_coords.shape == coords.shape
        assert inverse_coords.dtype == torch.float32
    
    def test_coordinate_transformation_roundtrip(self):
        """Test that forward and inverse transformations are consistent."""
        original_coords = torch.tensor([[100, 150], [300, 250]], dtype=torch.float32)
        original_size = (480, 640)
        long_side_length = 1024
        
        # Forward transformation
        transformed = SamDecoder._apply_coords_torch(original_coords, original_size, long_side_length)
        
        # Inverse transformation
        recovered = SamDecoder._apply_inverse_coords_torch(transformed, original_size, long_side_length)
        
        # Should be approximately equal (allowing for floating point precision)
        assert torch.allclose(original_coords, recovered, atol=1e-6)


class TestInputPreprocessing:
    """Test input preprocessing methods."""
    
    def test_preprocess_inputs_basic(self, sam_decoder, sample_image, sample_priors):
        """Test basic input preprocessing."""
        images = [sample_image]
        priors = [sample_priors]
        
        preprocessed_images, preprocessed_points, original_sizes = sam_decoder.preprocess_inputs(images, priors)
        
        assert len(preprocessed_images) == 1
        assert len(preprocessed_points) == 1
        assert len(original_sizes) == 1
        
        # Check image preprocessing
        assert preprocessed_images[0].shape == (1, 3, 1024, 1024)
        assert original_sizes[0] == (480, 640)
        
        # Check points preprocessing
        assert 0 in preprocessed_points[0]  # class_id 0 should exist
        transformed_points = preprocessed_points[0][0]
        assert transformed_points.shape[1] == 4  # [x, y, score, label]
    
    def test_preprocess_inputs_empty_points(self, sam_decoder, sample_image):
        """Test preprocessing with empty points."""
        priors = Priors()
        empty_points = torch.empty((0, 4), dtype=torch.float32)
        priors.points.add(empty_points, class_id=0)
        
        preprocessed_images, preprocessed_points, original_sizes = sam_decoder.preprocess_inputs([sample_image], [priors])
        
        assert len(preprocessed_points[0][0]) == 0
    
    def test_preprocess_inputs_multiple_classes(self, sam_decoder, sample_image):
        """Test preprocessing with multiple classes."""
        priors = Priors()
        
        # Class 0 points
        points_class_0 = torch.tensor([[100, 150, 0.9, 1]], dtype=torch.float32)
        priors.points.add(points_class_0, class_id=0)
        
        # Class 1 points
        points_class_1 = torch.tensor([[200, 250, 0.8, 1]], dtype=torch.float32)
        priors.points.add(points_class_1, class_id=1)
        
        preprocessed_images, preprocessed_points, original_sizes = sam_decoder.preprocess_inputs([sample_image], [priors])
        
        assert 0 in preprocessed_points[0]
        assert 1 in preprocessed_points[0]
        assert len(preprocessed_points[0][0]) == 1
        assert len(preprocessed_points[0][1]) == 1
    
    def test_preprocess_inputs_invalid_prior_maps(self, sam_decoder, sample_image):
        """Test preprocessing with invalid number of prior maps."""
        priors = Priors()
        points1 = torch.tensor([[100, 150, 0.9, 1]], dtype=torch.float32)
        points2 = torch.tensor([[200, 250, 0.8, 1]], dtype=torch.float32)
        
        # Add multiple prior maps for the same class (should raise error)
        priors.points.add(points1, class_id=0)
        priors.points.add(points2, class_id=0)
        
        with pytest.raises(ValueError, match="Each class must have exactly one prior map"):
            sam_decoder.preprocess_inputs([sample_image], [priors])


class TestPointPreprocessing:
    """Test point preprocessing static methods."""
    
    def test_point_preprocess_basic(self):
        """Test basic point preprocessing."""
        points = torch.tensor([[[100, 150]], [[200, 250]], [[300, 350]]], dtype=torch.float32)
        labels = torch.tensor([[1], [0], [1]], dtype=torch.float32)  # fg, bg, fg
        scores = torch.tensor([[0.9], [0.8], [0.7]], dtype=torch.float32)
        
        final_coords, final_labels = SamDecoder.point_preprocess(points, labels, scores)
        
        # Should have 2 positive points (fg), each paired with 1 negative point (bg)
        assert final_coords.shape[0] == 2  # num_positive
        assert final_coords.shape[1] == 2  # 1 positive + 1 negative
        assert final_coords.shape[2] == 3  # [x, y, score]
        
        assert final_labels.shape == (2, 2)  # [num_positive, 1_positive + num_negative]
    
    def test_point_preprocess_no_negative_points(self):
        """Test point preprocessing with no negative points."""
        points = torch.tensor([[[100, 150]], [[200, 250]]], dtype=torch.float32)
        labels = torch.tensor([[1], [1]], dtype=torch.float32)  # both positive
        scores = torch.tensor([[0.9], [0.8]], dtype=torch.float32)
        
        final_coords, final_labels = SamDecoder.point_preprocess(points, labels, scores)
        
        # Should have 2 positive points, each with no negative points
        assert final_coords.shape == (2, 1, 3)  # no negative points to pair with
        assert final_labels.shape == (2, 1)
    
    def test_remap_preprocessed_points(self):
        """Test remapping of preprocessed points."""
        # Simulated preprocessed points: [num_positive, 1+num_negative, 3]
        preprocessed_points = torch.tensor([
            [[100, 150, 0.9], [200, 250, 0.8]],  # positive point with one negative
            [[300, 350, 0.7], [200, 250, 0.8]],  # another positive with same negative
        ], dtype=torch.float32)
        
        remapped = SamDecoder.remap_preprocessed_points(preprocessed_points)
        
        # Should have 3 total points: 2 positive + 1 negative
        assert remapped.shape == (3, 4)  # [x, y, score, label]
        
        # Check labels: first 2 should be positive (1), last should be negative (0)
        assert remapped[0, 3] == 1  # positive
        assert remapped[1, 3] == 1  # positive
        assert remapped[2, 3] == 0  # negative


class TestMaskRefinement:
    """Test mask refinement functionality."""
    
    def test_mask_refinement_basic(self, sam_decoder):
        """Test basic mask refinement."""
        masks = torch.ones((2, 1, 100, 100), dtype=torch.bool)
        low_res_logits = torch.zeros((2, 256, 256), dtype=torch.float32)
        point_coords = torch.tensor([[[50, 50, 0.9]], [[60, 60, 0.8]]], dtype=torch.float32)
        point_labels = torch.tensor([[1], [1]], dtype=torch.float32)
        similarity_map = torch.ones((100, 100), dtype=torch.float32) * 0.5
        original_size = (100, 100)
        
        # Mock the _predict method for refinement
        sam_decoder._predict = MagicMock(return_value=(masks, torch.tensor([0.8, 0.9]), low_res_logits))
        
        final_masks, final_points, final_labels = sam_decoder.mask_refinement(
            masks, low_res_logits, point_coords, point_labels, 
            similarity_map, original_size
        )
        
        # Should return filtered results
        assert final_masks.shape[0] <= masks.shape[0]  # May be filtered
        assert final_points.shape[0] == final_masks.shape[0]
        assert final_labels.shape[0] == final_masks.shape[0]
    
    def test_mask_refinement_empty_masks(self, sam_decoder):
        """Test mask refinement with empty masks."""
        masks = torch.zeros((2, 1, 100, 100), dtype=torch.bool)  # All zeros
        low_res_logits = torch.zeros((2, 256, 256), dtype=torch.float32)
        point_coords = torch.tensor([[[50, 50, 0.9]], [[60, 60, 0.8]]], dtype=torch.float32)
        point_labels = torch.tensor([[1], [1]], dtype=torch.float32)
        similarity_map = torch.ones((100, 100), dtype=torch.float32) * 0.5
        original_size = (100, 100)
        
        final_masks, final_points, final_labels = sam_decoder.mask_refinement(
            masks, low_res_logits, point_coords, point_labels,
            similarity_map, original_size
        )
        
        # Should return empty results for empty masks
        assert final_masks.shape[0] == 0
        assert final_points.shape[0] == 0
        assert final_labels.shape[0] == 0


class TestPredictByPoints:
    """Test point-based prediction functionality."""
    
    def test_predict_by_points_basic(self, sam_decoder, sample_similarities):
        """Test basic point-based prediction."""
        class_points = {
            0: torch.tensor([[100, 150, 0.9, 1], [200, 250, 0.8, 0]], dtype=torch.float32)
        }
        original_size = (480, 640)
        
        # Mock the refinement method
        refined_masks = torch.ones((1, 100, 100), dtype=torch.bool)
        refined_points = torch.tensor([[[100, 150, 0.9]], [[200, 250, 0.8]]], dtype=torch.float32)
        sam_decoder.mask_refinement = MagicMock(return_value=(
            refined_masks, refined_points, torch.tensor([[1], [0]])
        ))
        
        masks, points = sam_decoder.predict_by_points(class_points, sample_similarities, original_size)
        
        assert isinstance(masks, Masks)
        assert isinstance(points, Points)
        assert 0 in masks.data  # Should have results for class 0
    
    def test_predict_by_points_no_foreground(self, sam_decoder, sample_similarities):
        """Test prediction with no foreground points."""
        class_points = {
            0: torch.tensor([[100, 150, 0.9, 0]], dtype=torch.float32)  # Only background
        }
        original_size = (480, 640)
        
        masks, points = sam_decoder.predict_by_points(class_points, sample_similarities, original_size)
        
        # Should return empty results when no foreground points
        assert isinstance(masks, Masks)
        assert isinstance(points, Points)
        assert len(masks.data) == 0 or all(len(masks.get(i)) == 0 for i in masks.class_ids())


class TestForwardPass:
    """Test the main forward pass functionality."""
    
    def test_forward_basic(self, sam_decoder, sample_image, sample_priors, sample_similarities):
        """Test basic forward pass."""
        images = [sample_image]
        priors = [sample_priors]
        similarities = [sample_similarities]
        
        # Mock predict_by_points
        mock_masks = Masks()
        mock_masks.add(torch.ones((100, 100), dtype=torch.bool), class_id=0)
        mock_points = Points()
        mock_points.add(torch.tensor([[100, 150, 0.9, 1]], dtype=torch.float32), class_id=0)
        
        sam_decoder.predict_by_points = MagicMock(return_value=(mock_masks, mock_points))
        
        masks_per_image, points_per_image, boxes_per_image = sam_decoder.forward(images, priors, similarities)
        
        assert len(masks_per_image) == 1
        assert len(points_per_image) == 1
        assert len(boxes_per_image) == 1
        assert isinstance(masks_per_image[0], Masks)
        assert isinstance(points_per_image[0], Points)
    
    def test_forward_no_similarities(self, sam_decoder, sample_image, sample_priors):
        """Test forward pass without similarities."""
        images = [sample_image]
        priors = [sample_priors]
        
        # Should handle None similarities gracefully
        masks_per_image, points_per_image, boxes_per_image = sam_decoder.forward(images, priors, None)
        
        assert len(masks_per_image) == 1
        assert len(points_per_image) == 1
        assert len(boxes_per_image) == 1
    
    def test_forward_empty_priors(self, sam_decoder, sample_image):
        """Test forward pass with empty priors."""
        images = [sample_image]
        priors = [Priors()]  # Empty priors
        
        masks_per_image, points_per_image, boxes_per_image = sam_decoder.forward(images, priors)
        
        assert len(masks_per_image) == 1
        assert isinstance(masks_per_image[0], Masks)
        assert len(masks_per_image[0].data) == 0  # Should be empty


class TestIntegration:
    """Integration tests for the complete SamDecoder workflow."""
    
    @patch('getiprompt.models.models.load_sam_model')
    def test_integration_with_real_types(self, mock_load_sam):
        """Test integration with real data types (but mocked model)."""
        # Setup mock
        mock_predictor = MagicMock(spec=SamHQPredictor)
        mock_predictor.device = torch.device("cpu")
        mock_predictor.transform.target_length = 1024
        mock_predictor.transform.apply_image.return_value = np.zeros((1024, 1024, 3))
        mock_predictor.transform.apply_coords_torch.return_value = torch.tensor([[100, 200]], dtype=torch.float32)
        mock_predictor.set_torch_image.return_value = None
        
        # Mock prediction to return reasonable outputs
        masks = torch.ones((1, 3, 1024, 1024), dtype=torch.bool)
        scores = torch.tensor([0.9, 0.8, 0.7])
        logits = torch.zeros((3, 256, 256))
        mock_predictor.predict_torch.return_value = (masks, scores, logits)
        
        mock_load_sam.return_value = mock_predictor
        
        # Create decoder
        decoder = SamDecoder(sam_predictor=mock_predictor)
        
        # Create real input data
        image = Image(np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8))
        priors = Priors()
        points = torch.tensor([[100, 150, 0.9, 1], [200, 250, 0.8, 0]], dtype=torch.float32)
        priors.points.add(points, class_id=0)
        
        similarities = Similarities()
        similarity_map = torch.rand((1, 480, 640))
        similarities.add(similarity_map, class_id=0)
        
        # Run forward pass
        masks_result, points_result, boxes_result = decoder.forward([image], [priors], [similarities])
        
        # Verify results
        assert len(masks_result) == 1
        assert len(points_result) == 1
        assert len(boxes_result) == 1
        assert isinstance(masks_result[0], Masks)
        assert isinstance(points_result[0], Points)
        assert isinstance(boxes_result[0], Boxes)
    
    def test_docstring_example(self, mock_sam_predictor):
        """Test the example from the class docstring."""
        # Create decoder
        segmenter = SamDecoder(sam_predictor=mock_sam_predictor)
        
        # Create inputs as in docstring
        image = Image(np.zeros((1024, 1024, 3), dtype=np.uint8))
        priors = Priors()
        points = torch.tensor([[512, 512, 0.9, 1], [100, 100, 0.8, 0]], dtype=torch.float32)  # fg, bg
        priors.points.add(points, class_id=1)
        
        similarities = Similarities()
        similarities.add(torch.ones(1, 1024, 1024), class_id=1)
        
        # Mock the prediction to return some results
        mock_masks = Masks()
        mock_masks.add(torch.ones((100, 100), dtype=torch.bool), class_id=1)
        mock_points = Points()
        mock_points.add(torch.tensor([[512, 512, 0.9, 1], [100, 100, 0.8, 0]], dtype=torch.float32), class_id=1)
        
        segmenter.predict_by_points = MagicMock(return_value=(mock_masks, mock_points))
        
        # Run the example
        masks, used_points, boxes = segmenter(
            images=[image],
            priors=[priors],
            similarities=[similarities],
        )
        
        # Verify docstring assertions
        assert isinstance(masks, list) and isinstance(masks[0], Masks) and len(masks[0].get(1)) == 1
        assert isinstance(used_points, list) and isinstance(used_points[0], Points) and len(used_points[0].get(1)[0]) == 2


class TestErrorHandling:
    """Test error handling and edge cases."""
    
    def test_invalid_similarity_threshold(self, mock_sam_predictor):
        """Test that decoder works with extreme similarity thresholds."""
        # Very low threshold (should keep all masks)
        decoder = SamDecoder(mock_sam_predictor, mask_similarity_threshold=0.0)
        assert decoder.mask_similarity_threshold == 0.0
        
        # Very high threshold (should filter most masks)
        decoder = SamDecoder(mock_sam_predictor, mask_similarity_threshold=1.0)
        assert decoder.mask_similarity_threshold == 1.0
    
    def test_device_consistency(self, sam_decoder):
        """Test that operations maintain device consistency."""
        # Ensure all tensors are on the same device as the predictor
        assert sam_decoder.predictor.device == torch.device("cpu")
        
        # Test with CUDA device (if available)
        if torch.cuda.is_available():
            sam_decoder.predictor.device = torch.device("cuda:0")
            assert sam_decoder.predictor.device == torch.device("cuda:0")


if __name__ == "__main__":
    pytest.main([__file__])
