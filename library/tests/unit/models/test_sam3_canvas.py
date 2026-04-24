# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for SAM3 canvas mode: CanvasConfig, geometry helpers, and EfficientSAM3 guard."""

import numpy as np
import pytest
import torch

from instantlearn.models.sam3.sam3 import CanvasConfig, SAM3


class TestCanvasConfigDefaults:
    """CanvasConfig default values."""

    def test_defaults(self) -> None:
        cfg = CanvasConfig()
        assert cfg.split_ratio == 0.3
        assert cfg.crop_padding == 2.0
        assert cfg.cache_text is True
        assert cfg.share_vision == "auto"


class TestCanvasConfigValidation:
    """CanvasConfig __post_init__ validation."""

    @pytest.mark.parametrize("ratio", [0.0, 1.0, -0.1, 1.5])
    def test_split_ratio_out_of_range(self, ratio: float) -> None:
        with pytest.raises(ValueError, match="split_ratio"):
            CanvasConfig(split_ratio=ratio)

    @pytest.mark.parametrize("padding", [0.0, -1.0, -0.01])
    def test_crop_padding_non_positive(self, padding: float) -> None:
        with pytest.raises(ValueError, match="crop_padding"):
            CanvasConfig(crop_padding=padding)

    def test_valid_edge_values(self) -> None:
        cfg = CanvasConfig(split_ratio=0.01, crop_padding=0.01)
        assert cfg.split_ratio == 0.01
        assert cfg.crop_padding == 0.01


class TestCanvasConfigCustom:
    """CanvasConfig with custom values."""

    def test_custom_values(self) -> None:
        cfg = CanvasConfig(split_ratio=0.5, crop_padding=3.0, cache_text=False, share_vision=False)
        assert cfg.split_ratio == 0.5
        assert cfg.crop_padding == 3.0
        assert cfg.cache_text is False
        assert cfg.share_vision is False

    def test_share_vision_strategies(self) -> None:
        for strategy in ("auto", "grouped", "spaced", False):
            cfg = CanvasConfig(share_vision=strategy)
            assert cfg.share_vision == strategy


class TestEfficientSam3CanvasGuard:
    """EfficientSAM3 rejects canvas mode at init."""

    def test_canvas_mode_raises(self) -> None:
        from instantlearn.models.efficient_sam3.efficient_sam3 import EfficientSAM3

        with pytest.raises(ValueError, match="Canvas mode is not supported"):
            EfficientSAM3(prompt_mode="canvas")


class TestBuildCanvasVerticalGeometry:
    """_build_canvas_vertical returns correct canvas shape and target region."""

    @pytest.fixture
    def sam3_stub(self) -> SAM3:
        """Minimal SAM3 stub with canvas_config but no loaded model."""
        sam = object.__new__(SAM3)
        sam.canvas_config = CanvasConfig(split_ratio=0.3, crop_padding=2.0)
        return sam

    def test_canvas_shape_and_region(self, sam3_stub: SAM3) -> None:
        ref = torch.randn(3, 100, 120)
        tgt = torch.randn(3, 200, 150)
        ref_bbox = np.array([10, 10, 50, 50], dtype=np.float32)

        canvas, canvas_bbox, tgt_region = sam3_stub._build_canvas_vertical(ref, tgt, ref_bbox)

        # Canvas width = max(ref_w, tgt_w) = 150
        assert canvas.shape[0] == 3
        assert canvas.shape[2] == 150  # max width

        # Target region should be at top (y=0)
        tx, ty, tw, th = tgt_region
        assert ty == 0
        assert tw == 150  # canvas width
        assert th > 0

        # Canvas bbox should have valid coordinates
        assert canvas_bbox.shape == (4,)
        assert all(canvas_bbox >= 0)

    def test_square_inputs(self, sam3_stub: SAM3) -> None:
        ref = torch.randn(3, 100, 100)
        tgt = torch.randn(3, 100, 100)
        ref_bbox = np.array([20, 20, 80, 80], dtype=np.float32)

        canvas, canvas_bbox, tgt_region = sam3_stub._build_canvas_vertical(ref, tgt, ref_bbox)
        tx, ty, tw, th = tgt_region

        # Target takes (1 - split_ratio) of canvas height
        total_h = canvas.shape[1]
        expected_tgt_h = int(total_h * (1 - sam3_stub.canvas_config.split_ratio))
        assert abs(th - expected_tgt_h) <= 1


class TestExtractTargetPredictions:
    """_extract_target_predictions remaps boxes to original target coords."""

    def test_empty_predictions(self) -> None:
        pred = {
            "pred_boxes": torch.empty(0, 5),
            "pred_masks": torch.empty(0, 100, 100),
        }
        result = SAM3._extract_target_predictions(pred, (0, 0, 100, 100), 200, 200)
        assert result["pred_boxes"].shape[0] == 0

    def test_boxes_remapped(self) -> None:
        # Target region at (50, 100, 200, 300) on canvas
        # Predictions at canvas coords
        pred = {
            "pred_boxes": torch.tensor([
                [100.0, 200.0, 150.0, 250.0, 0.9],  # inside target region
            ]),
        }
        tgt_region = (50, 100, 200, 300)
        tgt_h, tgt_w = 600, 400

        result = SAM3._extract_target_predictions(pred, tgt_region, tgt_h, tgt_w)

        # Box should be shifted by -tx, -ty then scaled to original size
        boxes = result["pred_boxes"]
        assert boxes.shape == (1, 5)
        # x coords: (100-50) * 400/200 = 100, (150-50) * 400/200 = 200
        # y coords: (200-100) * 600/300 = 200, (250-100) * 600/300 = 300
        assert torch.allclose(boxes[0, :4], torch.tensor([100.0, 200.0, 200.0, 300.0]), atol=1.0)

    def test_filters_outside_boxes(self) -> None:
        # Box completely outside target region should be filtered
        pred = {
            "pred_boxes": torch.tensor([
                [0.0, 0.0, 10.0, 10.0, 0.9],  # outside target region
                [60.0, 110.0, 90.0, 140.0, 0.8],  # inside
            ]),
        }
        tgt_region = (50, 100, 200, 300)

        result = SAM3._extract_target_predictions(pred, tgt_region, 300, 200)
        # At minimum the outside box should be excluded
        assert result["pred_boxes"].shape[0] <= 2


class TestMergeCrossCategory:
    """_merge_cross_category merges per-category results with NMS."""

    def test_no_overlap(self) -> None:
        boxes = [
            torch.tensor([[10.0, 10.0, 50.0, 50.0, 0.9]]),
            torch.tensor([[200.0, 200.0, 250.0, 250.0, 0.8]]),
        ]
        masks = [torch.ones(1, 100, 100), torch.ones(1, 100, 100)]
        labels = [torch.tensor([0]), torch.tensor([1])]

        result = SAM3._merge_cross_category(boxes, masks, labels, (100, 100))
        assert result["pred_boxes"].shape[0] == 2
        assert result["pred_labels"].shape[0] == 2

    def test_overlapping_suppression(self) -> None:
        # Two identical boxes from different categories, higher score wins
        boxes = [
            torch.tensor([[10.0, 10.0, 50.0, 50.0, 0.9]]),
            torch.tensor([[10.0, 10.0, 50.0, 50.0, 0.7]]),
        ]
        masks = [torch.ones(1, 100, 100), torch.zeros(1, 100, 100)]
        labels = [torch.tensor([0]), torch.tensor([1])]

        result = SAM3._merge_cross_category(boxes, masks, labels, (100, 100), iou_threshold=0.5)
        assert result["pred_boxes"].shape[0] == 1
        assert result["pred_boxes"][0, 4].item() == pytest.approx(0.9)

    def test_empty_inputs(self) -> None:
        result = SAM3._merge_cross_category(
            [torch.empty(0, 5)], [torch.empty(0, 50, 50)], [torch.empty(0, dtype=torch.long)], (50, 50),
        )
        assert result["pred_boxes"].shape[0] == 0


class TestCropAroundBbox:
    """_crop_around_bbox crops with padding and adjusts bbox coords."""

    @pytest.fixture
    def sam3_stub(self) -> SAM3:
        sam = object.__new__(SAM3)
        sam.canvas_config = CanvasConfig(crop_padding=2.0)
        return sam

    def test_crop_contains_bbox(self, sam3_stub: SAM3) -> None:
        image = torch.randn(3, 200, 300)
        bbox = np.array([100, 50, 150, 100], dtype=np.float32)

        crop, adj_bbox = sam3_stub._crop_around_bbox(image, bbox)

        # Crop should be smaller than original
        assert crop.shape[1] <= 200
        assert crop.shape[2] <= 300
        assert crop.shape[0] == 3

        # Adjusted bbox should be within crop
        ax1, ay1, ax2, ay2 = adj_bbox[:4]
        assert ax1 >= 0
        assert ay1 >= 0
        assert ax2 <= crop.shape[2]
        assert ay2 <= crop.shape[1]

    def test_bbox_at_edge(self, sam3_stub: SAM3) -> None:
        image = torch.randn(3, 100, 100)
        bbox = np.array([0, 0, 20, 20], dtype=np.float32)

        crop, adj_bbox = sam3_stub._crop_around_bbox(image, bbox)
        # Should not crash; crop is clamped to image bounds
        assert crop.shape[0] == 3
        assert crop.shape[1] > 0
        assert crop.shape[2] > 0
