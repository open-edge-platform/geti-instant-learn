# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for PerDino's export-safe prompt generation.

The exported graph is traced statically, so the export path must not read tensor
shapes into Python ints or branch on data-dependent values — doing so bakes the
trace-time values into the graph. These tests pin the export path to the eager
one and check it stays shape-agnostic.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
import torch

from instantlearn.models.per_dino.prompt_generators import GridPromptGenerator


@pytest.fixture
def generator() -> GridPromptGenerator:
    """Grid prompt generator with a small grid for readable assertions."""
    return GridPromptGenerator(
        num_grid_cells=4,
        point_selection_threshold=0.65,
        num_bg_points=2,
        num_foreground_points=10,
        max_points=12,
    )


def _similarity_map(peaks: list[tuple[int, int]], size: int = 8) -> torch.Tensor:
    """Build a similarity map that is above threshold only at ``peaks`` (y, x)."""
    similarity_map = torch.zeros(size, size)
    for score_rank, (y, x) in enumerate(peaks):
        similarity_map[y, x] = 0.9 - 0.01 * score_rank
    return similarity_map


class TestForegroundPointSelection:
    """The export path must agree with the eager grid-based selection."""

    def test_matches_eager_path(self, generator: GridPromptGenerator) -> None:
        """Both paths pick the same one-best-point-per-cell set."""
        # One peak per cell in three different cells of a 4x4 grid over an 8x8 map.
        similarity_map = _similarity_map([(0, 0), (0, 5), (6, 2)])

        eager = generator._get_foreground_points(similarity_map)  # noqa: SLF001
        exported = generator._get_foreground_points_export(similarity_map)  # noqa: SLF001

        assert torch.allclose(eager.sort(dim=0).values, exported.sort(dim=0).values)

    def test_keeps_only_best_point_per_cell(self, generator: GridPromptGenerator) -> None:
        """Two peaks in one cell collapse to the higher-scoring one."""
        similarity_map = torch.zeros(8, 8)
        similarity_map[0, 0] = 0.8
        similarity_map[1, 1] = 0.9  # same top-left cell, higher score

        points = generator._get_foreground_points_export(similarity_map)  # noqa: SLF001

        assert points.shape[0] == 1
        assert points[0, 2] == pytest.approx(0.9)

    def test_returns_empty_when_nothing_passes_threshold(self, generator: GridPromptGenerator) -> None:
        """No detections is a valid result; the export path must not invent one."""
        points = generator._get_foreground_points_export(torch.zeros(8, 8))  # noqa: SLF001
        assert points.shape == (0, 3)

    def test_sorted_by_descending_score(self, generator: GridPromptGenerator) -> None:
        """Downstream selection assumes score-ordered points."""
        points = generator._get_foreground_points_export(_similarity_map([(0, 0), (0, 5), (6, 2)]))  # noqa: SLF001
        scores = points[:, 2]
        assert torch.all(scores[:-1] >= scores[1:])


class TestExportPathIsShapeAgnostic:
    """Guard against trace-time values getting baked into the exported graph."""

    @pytest.mark.parametrize("num_points", [0, 1, 5, 15])
    def test_filter_handles_any_point_count(self, generator: GridPromptGenerator, num_points: int) -> None:
        """Filtering must work above and below the foreground budget (10)."""
        points = torch.zeros(num_points, 4)
        points[:, 2] = torch.linspace(0.9, 0.7, num_points) if num_points else torch.zeros(0)

        with patch("torch.onnx.is_in_onnx_export", return_value=True):
            filtered = generator._filter_foreground_points(points)  # noqa: SLF001

        assert filtered.shape[0] == min(num_points, generator.num_foreground_points)

    @pytest.mark.parametrize("num_points", [0, 3, 12, 20])
    def test_padding_always_yields_max_points(self, generator: GridPromptGenerator, num_points: int) -> None:
        """Padding must produce a fixed-size tensor regardless of input length."""
        padded = generator._pad_points(torch.ones(num_points, 4), torch.device("cpu"), torch.float32)  # noqa: SLF001
        assert padded.shape == (generator.max_points, 4)

    @pytest.mark.parametrize("num_peaks", [1, 3, 8])
    def test_foreground_selection_handles_any_peak_count(
        self,
        generator: GridPromptGenerator,
        num_peaks: int,
    ) -> None:
        """Selection must not depend on how many peaks the trace-time input had."""
        peaks = [(y, x) for y in range(0, 8, 2) for x in range(0, 8, 2)][:num_peaks]
        points = generator._get_foreground_points_export(_similarity_map(peaks))  # noqa: SLF001
        assert points.shape[0] == len(peaks)
