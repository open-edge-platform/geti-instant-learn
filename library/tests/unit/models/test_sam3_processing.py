# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for SAM3 preprocessing and postprocessing modules.

Tests Sam3Preprocessor, Sam3PromptPreprocessor, and Sam3Postprocessor
with synthetic tensors — no model weights required.
"""

import pytest
import torch

from instantlearn.models.sam3.processing import (
    Sam3Postprocessor,
    Sam3Preprocessor,
    Sam3PromptPreprocessor,
)


# ---------------------------------------------------------------------------
# Sam3Preprocessor
# ---------------------------------------------------------------------------


class TestSam3Preprocessor:
    """Test image preprocessor."""

    @pytest.fixture
    def preprocessor(self) -> Sam3Preprocessor:
        """Create a preprocessor with default target size."""
        return Sam3Preprocessor(target_size=1008)

    @pytest.fixture
    def small_preprocessor(self) -> Sam3Preprocessor:
        """Create a smaller preprocessor for faster tests."""
        return Sam3Preprocessor(target_size=256)

    def test_output_shape(self, small_preprocessor: Sam3Preprocessor) -> None:
        """Test output tensor has target_size × target_size spatial dims."""
        image = torch.randint(0, 256, (1, 3, 480, 640), dtype=torch.uint8)
        pixel_values, original_sizes = small_preprocessor(image)

        assert pixel_values.shape == (1, 3, 256, 256)

    def test_original_sizes(self, small_preprocessor: Sam3Preprocessor) -> None:
        """Test original_sizes is correctly recorded."""
        image = torch.randint(0, 256, (1, 3, 480, 640), dtype=torch.uint8)
        _, original_sizes = small_preprocessor(image)

        assert original_sizes.shape == (1, 2)
        assert original_sizes[0, 0].item() == 480
        assert original_sizes[0, 1].item() == 640

    def test_batch_processing(self, small_preprocessor: Sam3Preprocessor) -> None:
        """Test batch of images produces correct batch dim."""
        images = torch.randint(0, 256, (3, 3, 320, 320), dtype=torch.uint8)
        pixel_values, original_sizes = small_preprocessor(images)

        assert pixel_values.shape[0] == 3
        assert original_sizes.shape == (3, 2)

    def test_normalization_range(self, small_preprocessor: Sam3Preprocessor) -> None:
        """Test output values are in [-1, 1] range (SAM3 normalization)."""
        image = torch.randint(0, 256, (1, 3, 100, 100), dtype=torch.uint8)
        pixel_values, _ = small_preprocessor(image)

        assert pixel_values.min() >= -1.0 - 1e-5
        assert pixel_values.max() <= 1.0 + 1e-5

    def test_float_input(self, small_preprocessor: Sam3Preprocessor) -> None:
        """Test preprocessor handles float input [0, 1]."""
        image = torch.rand(1, 3, 100, 100, dtype=torch.float32) * 255.0
        pixel_values, _ = small_preprocessor(image)

        assert pixel_values.shape == (1, 3, 256, 256)

    def test_invalid_ndim_raises(self, small_preprocessor: Sam3Preprocessor) -> None:
        """Test 3D input raises ValueError."""
        image = torch.randint(0, 256, (3, 100, 100), dtype=torch.uint8)

        with pytest.raises(ValueError, match="Expected input shape"):
            small_preprocessor(image)

    def test_get_preprocess_shape_landscape(self) -> None:
        """Test aspect-ratio preserving shape for landscape image."""
        h, w = Sam3Preprocessor.get_preprocess_shape(480, 640, 1008)

        assert w == 1008
        assert h < 1008
        # Aspect ratio roughly preserved
        assert abs(h / w - 480 / 640) < 0.01

    def test_get_preprocess_shape_portrait(self) -> None:
        """Test aspect-ratio preserving shape for portrait image."""
        h, w = Sam3Preprocessor.get_preprocess_shape(640, 480, 1008)

        assert h == 1008
        assert w < 1008

    def test_get_preprocess_shape_square(self) -> None:
        """Test aspect-ratio preserving shape for square image."""
        h, w = Sam3Preprocessor.get_preprocess_shape(500, 500, 1008)

        assert h == 1008
        assert w == 1008


# ---------------------------------------------------------------------------
# Sam3PromptPreprocessor
# ---------------------------------------------------------------------------


class TestSam3PromptPreprocessor:
    """Test prompt preprocessor for boxes and points."""

    @pytest.fixture
    def preprocessor(self) -> Sam3PromptPreprocessor:
        """Create a prompt preprocessor."""
        return Sam3PromptPreprocessor(target_size=1008)

    def test_box_normalization_single(self, preprocessor: Sam3PromptPreprocessor) -> None:
        """Test single box normalization from xyxy to cxcywh."""
        sizes = torch.tensor([[480, 640]], dtype=torch.int32)
        boxes, _ = preprocessor(sizes, input_boxes=[100, 100, 300, 300])

        assert boxes is not None
        assert boxes.shape == (1, 1, 4)
        # Center x should be 200/640 = 0.3125
        assert abs(boxes[0, 0, 0].item() - 200 / 640) < 1e-4
        # Center y should be 200/480 ≈ 0.4167
        assert abs(boxes[0, 0, 1].item() - 200 / 480) < 1e-4

    def test_point_normalization_single(self, preprocessor: Sam3PromptPreprocessor) -> None:
        """Test single point normalization."""
        sizes = torch.tensor([[480, 640]], dtype=torch.int32)
        _, points = preprocessor(sizes, input_points=[320, 240])

        assert points is not None
        assert points.shape == (1, 1, 2)
        # x: 320/640 = 0.5, y: 240/480 = 0.5
        assert abs(points[0, 0, 0].item() - 0.5) < 1e-4
        assert abs(points[0, 0, 1].item() - 0.5) < 1e-4

    def test_box_none_when_not_provided(self, preprocessor: Sam3PromptPreprocessor) -> None:
        """Test boxes returned as None when not provided."""
        sizes = torch.tensor([[480, 640]], dtype=torch.int32)
        boxes, points = preprocessor(sizes, input_points=[100, 100])

        assert boxes is None
        assert points is not None

    def test_points_none_when_not_provided(self, preprocessor: Sam3PromptPreprocessor) -> None:
        """Test points returned as None when not provided."""
        sizes = torch.tensor([[480, 640]], dtype=torch.int32)
        boxes, points = preprocessor(sizes, input_boxes=[10, 10, 50, 50])

        assert boxes is not None
        assert points is None

    def test_both_boxes_and_points(self, preprocessor: Sam3PromptPreprocessor) -> None:
        """Test both boxes and points can be provided simultaneously."""
        sizes = torch.tensor([[480, 640]], dtype=torch.int32)
        boxes, points = preprocessor(
            sizes,
            input_boxes=torch.tensor([[[100, 100, 200, 200]]]),
            input_points=torch.tensor([[[150, 200]]]),
        )

        assert boxes is not None
        assert points is not None

    def test_batch_of_boxes(self, preprocessor: Sam3PromptPreprocessor) -> None:
        """Test multiple boxes in a batch."""
        sizes = torch.tensor([[480, 640]], dtype=torch.int32)
        input_boxes = torch.tensor([[[10, 10, 50, 50], [100, 100, 200, 200]]], dtype=torch.float32)
        boxes, _ = preprocessor(sizes, input_boxes=input_boxes)

        assert boxes.shape == (1, 2, 4)

    def test_box_xyxy_to_cxcywh_conversion(self) -> None:
        """Test xyxy to cxcywh conversion."""
        boxes = torch.tensor([[10.0, 20.0, 50.0, 60.0]])
        cxcywh = Sam3PromptPreprocessor.box_xyxy_to_cxcywh(boxes)

        assert abs(cxcywh[0, 0].item() - 30.0) < 1e-4  # cx = (10+50)/2
        assert abs(cxcywh[0, 1].item() - 40.0) < 1e-4  # cy = (20+60)/2
        assert abs(cxcywh[0, 2].item() - 40.0) < 1e-4  # w = 50-10
        assert abs(cxcywh[0, 3].item() - 40.0) < 1e-4  # h = 60-20

    def test_list_input(self, preprocessor: Sam3PromptPreprocessor) -> None:
        """Test list inputs are accepted alongside tensors."""
        sizes = torch.tensor([[100, 100]], dtype=torch.int32)
        boxes, _ = preprocessor(sizes, input_boxes=[25, 25, 75, 75])

        assert boxes is not None
        assert boxes.shape == (1, 1, 4)


# ---------------------------------------------------------------------------
# Sam3Postprocessor
# ---------------------------------------------------------------------------


class TestSam3Postprocessor:
    """Test postprocessor for SAM3 outputs."""

    @pytest.fixture
    def postprocessor(self) -> Sam3Postprocessor:
        """Create a postprocessor with default params."""
        return Sam3Postprocessor(target_size=256, threshold=0.3, mask_threshold=0.5)

    def _make_outputs(
        self,
        batch_size: int = 1,
        num_queries: int = 10,
        mask_size: int = 64,
    ) -> dict[str, torch.Tensor]:
        """Create synthetic model outputs."""
        return {
            "pred_logits": torch.randn(batch_size, num_queries),
            "pred_boxes": torch.rand(batch_size, num_queries, 4),
            "pred_masks": torch.randn(batch_size, num_queries, mask_size, mask_size),
            "presence_logits": torch.randn(batch_size, 1),
        }

    def test_returns_list_of_dicts(self, postprocessor: Sam3Postprocessor) -> None:
        """Test postprocessor returns list of dicts in eager mode."""
        outputs = self._make_outputs()
        results = postprocessor(outputs, target_sizes=[(224, 224)])

        assert isinstance(results, list)
        assert len(results) == 1
        assert isinstance(results[0], dict)
        assert "scores" in results[0]
        assert "boxes" in results[0]
        assert "masks" in results[0]

    def test_score_filtering(self, postprocessor: Sam3Postprocessor) -> None:
        """Test that low-score predictions are filtered out."""
        outputs = self._make_outputs(num_queries=5)
        # Force all logits very negative → all scores < threshold
        outputs["pred_logits"] = torch.full((1, 5), -10.0)

        results = postprocessor(outputs, target_sizes=[(224, 224)])

        assert results[0]["scores"].numel() == 0

    def test_high_confidence_kept(self, postprocessor: Sam3Postprocessor) -> None:
        """Test that high-score predictions are kept."""
        outputs = self._make_outputs(num_queries=5)
        # Force all logits high → all scores above threshold
        outputs["pred_logits"] = torch.full((1, 5), 10.0)
        outputs["presence_logits"] = torch.full((1, 1), 10.0)

        results = postprocessor(outputs, target_sizes=[(224, 224)])

        assert results[0]["scores"].numel() == 5

    def test_mask_interpolation(self, postprocessor: Sam3Postprocessor) -> None:
        """Test masks are interpolated to target size."""
        outputs = self._make_outputs(num_queries=3, mask_size=32)
        outputs["pred_logits"] = torch.full((1, 3), 10.0)
        outputs["presence_logits"] = torch.full((1, 1), 10.0)

        target_h, target_w = 480, 640
        results = postprocessor(outputs, target_sizes=[(target_h, target_w)])

        masks = results[0]["masks"]
        assert masks.shape[-2:] == (target_h, target_w)

    def test_masks_are_binary(self, postprocessor: Sam3Postprocessor) -> None:
        """Test output masks are binarized to 0/1."""
        outputs = self._make_outputs(num_queries=3)
        outputs["pred_logits"] = torch.full((1, 3), 10.0)
        outputs["presence_logits"] = torch.full((1, 1), 10.0)

        results = postprocessor(outputs, target_sizes=[(224, 224)])

        masks = results[0]["masks"]
        unique_vals = torch.unique(masks)
        assert all(v in [0, 1] for v in unique_vals.tolist())

    def test_batch_processing(self, postprocessor: Sam3Postprocessor) -> None:
        """Test postprocessing handles batch > 1."""
        outputs = self._make_outputs(batch_size=2, num_queries=5)
        results = postprocessor(outputs, target_sizes=[(224, 224), (320, 320)])

        assert len(results) == 2

    def test_box_scaling(self, postprocessor: Sam3Postprocessor) -> None:
        """Test boxes are scaled to target image dimensions."""
        outputs = self._make_outputs(num_queries=1)
        outputs["pred_logits"] = torch.full((1, 1), 10.0)
        outputs["presence_logits"] = torch.full((1, 1), 10.0)
        # Set a box at exact center with known size in normalized coords
        outputs["pred_boxes"] = torch.tensor([[[0.5, 0.5, 0.2, 0.3]]])  # cxcywh

        results = postprocessor(outputs, target_sizes=[(100, 200)])

        boxes = results[0]["boxes"]
        # x coords scaled by width=200, y coords scaled by height=100
        assert boxes.numel() > 0

    def test_box_cxcywh_to_xyxy(self) -> None:
        """Test cxcywh to xyxy conversion."""
        boxes = torch.tensor([[30.0, 40.0, 40.0, 40.0]])  # cx, cy, w, h
        xyxy = Sam3Postprocessor.box_cxcywh_to_xyxy(boxes)

        assert abs(xyxy[0, 0].item() - 10.0) < 1e-4  # x1 = 30-20
        assert abs(xyxy[0, 1].item() - 20.0) < 1e-4  # y1 = 40-20
        assert abs(xyxy[0, 2].item() - 50.0) < 1e-4  # x2 = 30+20
        assert abs(xyxy[0, 3].item() - 60.0) < 1e-4  # y2 = 40+20

    def test_no_presence_logits(self) -> None:
        """Test postprocessor works without presence_logits."""
        postprocessor = Sam3Postprocessor(target_size=256, threshold=0.3, mask_threshold=0.5)
        outputs = {
            "pred_logits": torch.full((1, 3), 10.0),
            "pred_boxes": torch.rand(1, 3, 4),
            "pred_masks": torch.randn(1, 3, 32, 32),
        }

        results = postprocessor(outputs, target_sizes=[(224, 224)])

        assert isinstance(results, list)
        assert len(results) == 1
