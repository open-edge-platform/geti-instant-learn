# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import patch
from uuid import UUID

import numpy as np
import pytest
import torch

from domain.services.schemas.label import CategoryMappings, RGBColor, VisualizationInfo, VisualizationLabel
from domain.services.schemas.processor import OutputData
from runtime.webrtc.visualizer import DEFAULT_FALLBACK_COLOR, InferenceVisualizer


@pytest.fixture
def fxt_frame() -> np.ndarray:
    return np.zeros((8, 8, 3), dtype=np.uint8)


@pytest.fixture
def fxt_visualizer() -> InferenceVisualizer:
    with patch("runtime.webrtc.visualizer.get_settings") as mock_get_settings:
        mock_get_settings.return_value.mask_alpha = 1.0
        mock_get_settings.return_value.mask_outline_thickness = 0
        yield InferenceVisualizer(enable_visualization=True)


def _single_pixel_mask(h: int, w: int, y: int, x: int) -> torch.Tensor:
    mask = torch.zeros((1, h, w), dtype=torch.float32)
    mask[0, y, x] = 1.0
    return mask


def _two_pixel_disjoint_masks(h: int, w: int) -> torch.Tensor:
    masks = torch.zeros((2, h, w), dtype=torch.float32)
    masks[0, 2, 2] = 1.0
    masks[1, 5, 5] = 1.0
    return masks


def _make_vis_info(
    *, category_id_to_label_id: dict[int, str], label_colors: dict[str, tuple[int, int, int]]
) -> VisualizationInfo:
    return VisualizationInfo(
        label_colors=[
            VisualizationLabel(
                id=UUID(label_id),
                color=RGBColor(*rgb),
                object_name=None,
            )
            for label_id, rgb in label_colors.items()
        ],
        category_mappings=CategoryMappings(label_to_category_id={}, category_id_to_label_id=category_id_to_label_id),
    )


def test_visualize_disabled_returns_original_frame(fxt_frame: np.ndarray) -> None:
    viz = InferenceVisualizer(enable_visualization=False)
    output = OutputData(
        frame=fxt_frame,
        results=[{"pred_masks": _single_pixel_mask(8, 8, 3, 3), "pred_labels": torch.tensor([0])}],
    )

    result = viz.visualize(output_data=output, visualization_info=None)

    assert result is output.frame


def test_visualize_no_results_returns_original_frame(
    fxt_visualizer: InferenceVisualizer, fxt_frame: np.ndarray
) -> None:
    result = fxt_visualizer.visualize(output_data=OutputData(frame=fxt_frame, results=[]), visualization_info=None)
    assert result is fxt_frame


@pytest.mark.parametrize(
    "labels, category_id_to_label_id, label_colors, expected",
    [
        # category -> label_id -> label_colors
        (
            torch.tensor([0], dtype=torch.int64),
            {0: "00000000-0000-0000-0000-000000000001"},
            {"00000000-0000-0000-0000-000000000001": (255, 0, 0)},
            (255, 0, 0),
        ),
        # no category->label_id mapping => deterministic per category
        (torch.tensor([7], dtype=torch.int64), {}, {}, "deterministic:7"),
        # missing labels => default fallback
        (None, {}, {}, DEFAULT_FALLBACK_COLOR),
    ],
)
def test_visualize_resolves_color_per_mask(
    fxt_visualizer: InferenceVisualizer,
    fxt_frame: np.ndarray,
    labels: torch.Tensor | None,
    category_id_to_label_id: dict[int, str],
    label_colors: dict[str, tuple[int, int, int]],
    expected: tuple[int, int, int] | str,
) -> None:
    output = OutputData(
        frame=fxt_frame,
        results=[{"pred_masks": _single_pixel_mask(8, 8, 4, 4), "pred_labels": labels}],
    )
    vis_info = _make_vis_info(category_id_to_label_id=category_id_to_label_id, label_colors=label_colors)

    result = fxt_visualizer.visualize(output_data=output, visualization_info=vis_info)

    if isinstance(expected, str) and expected.startswith("deterministic:"):
        category_id = int(expected.split(":", 1)[1])
        expected_color = fxt_visualizer._generate_deterministic_color(category_id)
    else:
        expected_color = expected

    assert tuple(result[4, 4].tolist()) == expected_color


def test_visualize_applies_correct_colors_for_multiple_categories_in_single_prediction(
    fxt_visualizer: InferenceVisualizer, fxt_frame: np.ndarray
) -> None:
    masks = _two_pixel_disjoint_masks(8, 8)
    labels = torch.tensor([0, 1], dtype=torch.int64)

    label_a = "00000000-0000-0000-0000-00000000000a"
    label_b = "00000000-0000-0000-0000-00000000000b"
    vis_info = _make_vis_info(
        category_id_to_label_id={0: label_a, 1: label_b},
        label_colors={label_a: (255, 0, 0), label_b: (0, 255, 0)},
    )

    output = OutputData(frame=fxt_frame, results=[{"pred_masks": masks, "pred_labels": labels}])

    result = fxt_visualizer.visualize(output_data=output, visualization_info=vis_info)

    assert tuple(result[2, 2].tolist()) == (255, 0, 0)
    assert tuple(result[5, 5].tolist()) == (0, 255, 0)


def test_visualize_ignores_pred_boxes(fxt_visualizer: InferenceVisualizer, fxt_frame: np.ndarray) -> None:
    label_0 = "00000000-0000-0000-0000-000000000001"
    vis_info = _make_vis_info(category_id_to_label_id={0: label_0}, label_colors={label_0: (0, 255, 0)})

    output = OutputData(
        frame=fxt_frame,
        results=[
            {
                "pred_boxes": torch.tensor([[0.0, 0.0, 1.0, 1.0]]),
                "pred_masks": _single_pixel_mask(8, 8, 0, 0),
                "pred_labels": torch.tensor([0], dtype=torch.int64),
            }
        ],
    )

    result = fxt_visualizer.visualize(output_data=output, visualization_info=vis_info)

    assert tuple(result[0, 0].tolist()) == (0, 255, 0)
