# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for NegativeMaskToPoints utility."""

import torch

from instantlearn.components.negative_prompts import NegativeMaskToPoints


class TestNegativeMaskToPoints:
    """Tests for NegativeMaskToPoints module."""

    def test_single_mask_returns_correct_shape(self) -> None:
        """Test that a single mask produces the expected number of points."""
        converter = NegativeMaskToPoints(num_points_per_mask=5)
        mask = torch.zeros(1, 100, 100, dtype=torch.bool)
        mask[0, 20:60, 30:70] = True  # 40x40 block

        points, labels = converter(mask)

        assert points.shape == (5, 2)
        assert labels.shape == (5,)
        assert (labels == 0).all()

    def test_multiple_masks(self) -> None:
        """Test that multiple masks produce concatenated points."""
        converter = NegativeMaskToPoints(num_points_per_mask=3)
        masks = torch.zeros(2, 64, 64, dtype=torch.bool)
        masks[0, 10:30, 10:30] = True
        masks[1, 40:60, 40:60] = True

        points, labels = converter(masks)

        assert points.shape == (6, 2)
        assert labels.shape == (6,)
        assert (labels == 0).all()

    def test_empty_mask_returns_empty(self) -> None:
        """Test that an all-zero mask returns empty tensors."""
        converter = NegativeMaskToPoints(num_points_per_mask=5)
        mask = torch.zeros(1, 64, 64, dtype=torch.bool)

        points, labels = converter(mask)

        assert points.shape[0] == 0
        assert labels.shape[0] == 0

    def test_single_pixel_mask(self) -> None:
        """Test a mask with only one active pixel."""
        converter = NegativeMaskToPoints(num_points_per_mask=5)
        mask = torch.zeros(1, 100, 100, dtype=torch.bool)
        mask[0, 50, 60] = True

        points, labels = converter(mask)

        assert points.shape == (1, 2)
        # Point should be at (x=60, y=50) since mask uses (row, col) indexing
        assert points[0, 0].item() == 60.0  # x
        assert points[0, 1].item() == 50.0  # y

    def test_points_within_mask_bounds(self) -> None:
        """Test that all sampled points lie within the mask region."""
        converter = NegativeMaskToPoints(num_points_per_mask=10)
        mask = torch.zeros(1, 200, 200, dtype=torch.bool)
        mask[0, 50:100, 80:150] = True  # rectangular block

        points, labels = converter(mask)

        # Points are in (x, y) format, mask region is cols 80-149, rows 50-99
        assert (points[:, 0] >= 80).all() and (points[:, 0] <= 149).all()
        assert (points[:, 1] >= 50).all() and (points[:, 1] <= 99).all()

    def test_centroid_is_first_point(self) -> None:
        """Test that the first point is near the centroid of the mask."""
        converter = NegativeMaskToPoints(num_points_per_mask=5)
        mask = torch.zeros(1, 100, 100, dtype=torch.bool)
        mask[0, 40:60, 40:60] = True  # 20x20 centered block

        points, _ = converter(mask)

        # Centroid should be near (49.5, 49.5) in (x, y) coords
        assert abs(points[0, 0].item() - 49.5) < 2.0
        assert abs(points[0, 1].item() - 49.5) < 2.0

    def test_num_points_per_mask_respected(self) -> None:
        """Test different num_points_per_mask values."""
        mask = torch.zeros(1, 100, 100, dtype=torch.bool)
        mask[0, 10:90, 10:90] = True  # large region

        for n in [1, 3, 7, 15]:
            converter = NegativeMaskToPoints(num_points_per_mask=n)
            points, _ = converter(mask)
            assert points.shape[0] == n

    def test_output_dtype(self) -> None:
        """Test that outputs have correct dtypes."""
        converter = NegativeMaskToPoints(num_points_per_mask=3)
        mask = torch.zeros(1, 64, 64, dtype=torch.bool)
        mask[0, 10:30, 10:30] = True

        points, labels = converter(mask)

        assert points.dtype == torch.float32
        assert labels.dtype == torch.long
