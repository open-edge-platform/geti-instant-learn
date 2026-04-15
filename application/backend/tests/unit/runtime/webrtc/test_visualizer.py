# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import patch
from uuid import UUID

import numpy as np
import pytest

from domain.services.schemas.label import (
    CategoryMappings,
    RGBColor,
    VisualizationInfo,
    VisualizationLabel,
)
from domain.services.schemas.processor import OutputData
from runtime.webrtc.visualizer import (
    DEFAULT_FALLBACK_COLOR,
    CategoryResolver,
    InferenceVisualizer,
    generate_deterministic_color,
)


@pytest.fixture
def fxt_frame() -> np.ndarray:
    return np.zeros((8, 8, 3), dtype=np.uint8)


@pytest.fixture
def fxt_large_frame() -> np.ndarray:
    """Larger frame for box visualization tests."""
    return np.zeros((100, 100, 3), dtype=np.uint8)


@pytest.fixture
def fxt_visualizer() -> InferenceVisualizer:
    with patch("runtime.webrtc.visualizer.get_settings") as mock_get_settings:
        mock_get_settings.return_value.visualize_masks = True
        mock_get_settings.return_value.visualize_boxes = False
        mock_get_settings.return_value.visualize_labels = False
        mock_get_settings.return_value.mask_alpha = 1.0
        mock_get_settings.return_value.mask_outline_thickness = 0
        mock_get_settings.return_value.box_thickness = 2
        mock_get_settings.return_value.label_font_scale = 0.5
        yield InferenceVisualizer(enable_visualization=True)


@pytest.fixture
def fxt_visualizer_boxes_only() -> InferenceVisualizer:
    """Visualizer configured to draw only boxes."""
    with patch("runtime.webrtc.visualizer.get_settings") as mock_get_settings:
        mock_get_settings.return_value.visualize_masks = False
        mock_get_settings.return_value.visualize_boxes = True
        mock_get_settings.return_value.visualize_labels = False
        mock_get_settings.return_value.mask_alpha = 1.0
        mock_get_settings.return_value.mask_outline_thickness = 0
        mock_get_settings.return_value.box_thickness = 2
        mock_get_settings.return_value.label_font_scale = 0.5
        yield InferenceVisualizer(enable_visualization=True)


@pytest.fixture
def fxt_visualizer_both() -> InferenceVisualizer:
    """Visualizer configured to draw both masks and boxes."""
    with patch("runtime.webrtc.visualizer.get_settings") as mock_get_settings:
        mock_get_settings.return_value.visualize_masks = True
        mock_get_settings.return_value.visualize_boxes = True
        mock_get_settings.return_value.visualize_labels = False
        mock_get_settings.return_value.mask_alpha = 1.0
        mock_get_settings.return_value.mask_outline_thickness = 0
        mock_get_settings.return_value.box_thickness = 2
        mock_get_settings.return_value.label_font_scale = 0.5
        yield InferenceVisualizer(enable_visualization=True)


@pytest.fixture
def fxt_visualizer_boxes_with_labels() -> InferenceVisualizer:
    """Visualizer configured to draw boxes with label captions."""
    with patch("runtime.webrtc.visualizer.get_settings") as mock_get_settings:
        mock_get_settings.return_value.visualize_masks = False
        mock_get_settings.return_value.visualize_boxes = True
        mock_get_settings.return_value.visualize_labels = True
        mock_get_settings.return_value.mask_alpha = 1.0
        mock_get_settings.return_value.mask_outline_thickness = 0
        mock_get_settings.return_value.box_thickness = 2
        mock_get_settings.return_value.label_font_scale = 0.5
        yield InferenceVisualizer(enable_visualization=True)


def _single_pixel_mask(h: int, w: int, y: int, x: int) -> np.ndarray:
    mask = np.zeros((1, h, w), dtype=np.float32)
    mask[0, y, x] = 1.0
    return mask


def _two_pixel_disjoint_masks(h: int, w: int) -> np.ndarray:
    masks = np.zeros((2, h, w), dtype=np.float32)
    masks[0, 2, 2] = 1.0
    masks[1, 5, 5] = 1.0
    return masks


def _make_vis_info(
    *,
    category_id_to_label_id: dict[int, str],
    label_colors: dict[str, tuple[int, int, int]],
    label_names: dict[str, str] | None = None,
) -> VisualizationInfo:
    return VisualizationInfo(
        label_colors=[
            VisualizationLabel(
                id=UUID(label_id),
                color=RGBColor(*rgb),
                object_name=(label_names or {}).get(label_id),
            )
            for label_id, rgb in label_colors.items()
        ],
        category_mappings=CategoryMappings(label_to_category_id={}, category_id_to_label_id=category_id_to_label_id),
    )


def test_visualize_disabled_returns_original_frame(fxt_frame: np.ndarray) -> None:
    viz = InferenceVisualizer(enable_visualization=False)
    output = OutputData(
        frame=fxt_frame,
        results=[{"pred_masks": _single_pixel_mask(8, 8, 3, 3), "pred_labels": np.array([0])}],
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
            np.array([0], dtype=np.int64),
            {0: "00000000-0000-0000-0000-000000000001"},
            {"00000000-0000-0000-0000-000000000001": (255, 0, 0)},
            (255, 0, 0),
        ),
        # no category->label_id mapping => deterministic per category
        (np.array([7], dtype=np.int64), {}, {}, "deterministic:7"),
        # missing labels => default fallback
        (None, {}, {}, DEFAULT_FALLBACK_COLOR),
    ],
)
def test_visualize_resolves_color_per_mask(
    fxt_visualizer: InferenceVisualizer,
    fxt_frame: np.ndarray,
    labels: np.ndarray | None,
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
        expected_color = generate_deterministic_color(category_id)
    else:
        expected_color = expected

    assert tuple(result[4, 4].tolist()) == expected_color


def test_visualize_applies_correct_colors_for_multiple_categories_in_single_prediction(
    fxt_visualizer: InferenceVisualizer, fxt_frame: np.ndarray
) -> None:
    masks = _two_pixel_disjoint_masks(8, 8)
    labels = np.array([0, 1], dtype=np.int64)

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


def test_visualize_masks_disabled_does_not_draw_masks(
    fxt_visualizer_boxes_only: InferenceVisualizer, fxt_frame: np.ndarray
) -> None:
    output = OutputData(
        frame=fxt_frame,
        results=[{"pred_masks": _single_pixel_mask(8, 8, 4, 4), "pred_labels": np.array([0])}],
    )

    result = fxt_visualizer_boxes_only.visualize(output_data=output, visualization_info=None)

    # Frame should remain unchanged (all zeros) since masks are disabled and no boxes provided
    assert tuple(result[4, 4].tolist()) == (0, 0, 0)


def test_visualize_draws_box_with_correct_color(
    fxt_visualizer_boxes_only: InferenceVisualizer, fxt_large_frame: np.ndarray
) -> None:
    label_id = "00000000-0000-0000-0000-000000000001"
    vis_info = _make_vis_info(
        category_id_to_label_id={0: label_id},
        label_colors={label_id: (255, 0, 0)},
    )

    # Box from (10, 10) to (50, 50)
    boxes = np.array([[10, 10, 50, 50]], dtype=np.float32)
    labels = np.array([0], dtype=np.int64)

    output = OutputData(
        frame=fxt_large_frame,
        results=[{"pred_boxes": boxes, "pred_labels": labels}],
    )

    result = fxt_visualizer_boxes_only.visualize(output_data=output, visualization_info=vis_info)

    # Check that pixels on the box edge are colored (top edge at y=10)
    assert tuple(result[10, 20].tolist()) == (255, 0, 0)
    # Check that pixels inside the box are not colored (box is not filled)
    assert tuple(result[30, 30].tolist()) == (0, 0, 0)


def test_visualize_draws_box_with_score_column(
    fxt_visualizer_boxes_only: InferenceVisualizer, fxt_large_frame: np.ndarray
) -> None:
    label_id = "00000000-0000-0000-0000-000000000001"
    vis_info = _make_vis_info(
        category_id_to_label_id={0: label_id},
        label_colors={label_id: (0, 255, 0)},
    )

    # Box with score: [x1, y1, x2, y2, score]
    boxes = np.array([[10, 10, 50, 50, 0.95]], dtype=np.float32)
    labels = np.array([0], dtype=np.int64)

    output = OutputData(
        frame=fxt_large_frame,
        results=[{"pred_boxes": boxes, "pred_labels": labels}],
    )

    result = fxt_visualizer_boxes_only.visualize(output_data=output, visualization_info=vis_info)

    # Check that the box is drawn
    assert tuple(result[10, 20].tolist()) == (0, 255, 0)


def test_visualize_draws_multiple_boxes_with_different_colors(
    fxt_visualizer_boxes_only: InferenceVisualizer, fxt_large_frame: np.ndarray
) -> None:
    label_a = "00000000-0000-0000-0000-00000000000a"
    label_b = "00000000-0000-0000-0000-00000000000b"
    vis_info = _make_vis_info(
        category_id_to_label_id={0: label_a, 1: label_b},
        label_colors={label_a: (255, 0, 0), label_b: (0, 0, 255)},
    )

    # Two non-overlapping boxes
    boxes = np.array(
        [
            [10, 10, 30, 30],  # First box
            [60, 60, 90, 90],  # Second box
        ],
        dtype=np.float32,
    )
    labels = np.array([0, 1], dtype=np.int64)

    output = OutputData(
        frame=fxt_large_frame,
        results=[{"pred_boxes": boxes, "pred_labels": labels}],
    )

    result = fxt_visualizer_boxes_only.visualize(output_data=output, visualization_info=vis_info)

    # First box should be red
    assert tuple(result[10, 15].tolist()) == (255, 0, 0)
    # Second box should be blue
    assert tuple(result[60, 70].tolist()) == (0, 0, 255)


def test_visualize_box_without_labels_uses_fallback_color(
    fxt_visualizer_boxes_only: InferenceVisualizer, fxt_large_frame: np.ndarray
) -> None:
    boxes = np.array([[10, 10, 50, 50]], dtype=np.float32)

    output = OutputData(
        frame=fxt_large_frame,
        results=[{"pred_boxes": boxes, "pred_labels": None}],
    )

    result = fxt_visualizer_boxes_only.visualize(output_data=output, visualization_info=None)

    # Should use DEFAULT_FALLBACK_COLOR
    assert tuple(result[10, 20].tolist()) == DEFAULT_FALLBACK_COLOR


def test_visualize_box_with_unmapped_category_uses_deterministic_color(
    fxt_visualizer_boxes_only: InferenceVisualizer, fxt_large_frame: np.ndarray
) -> None:
    vis_info = _make_vis_info(category_id_to_label_id={}, label_colors={})

    boxes = np.array([[10, 10, 50, 50]], dtype=np.float32)
    labels = np.array([42], dtype=np.int64)  # Category 42 has no mapping

    output = OutputData(
        frame=fxt_large_frame,
        results=[{"pred_boxes": boxes, "pred_labels": labels}],
    )

    result = fxt_visualizer_boxes_only.visualize(output_data=output, visualization_info=vis_info)

    expected_color = generate_deterministic_color(42)
    assert tuple(result[10, 20].tolist()) == expected_color


def test_visualize_empty_boxes_array_does_not_crash(
    fxt_visualizer_boxes_only: InferenceVisualizer, fxt_large_frame: np.ndarray
) -> None:
    output = OutputData(
        frame=fxt_large_frame,
        results=[{"pred_boxes": np.array([]).reshape(0, 4), "pred_labels": np.array([])}],
    )

    result = fxt_visualizer_boxes_only.visualize(output_data=output, visualization_info=None)

    # Frame should remain unchanged
    np.testing.assert_array_equal(result, fxt_large_frame)


def test_visualize_boxes_disabled_does_not_draw_boxes(
    fxt_visualizer: InferenceVisualizer, fxt_large_frame: np.ndarray
) -> None:
    label_id = "00000000-0000-0000-0000-000000000001"
    vis_info = _make_vis_info(
        category_id_to_label_id={0: label_id},
        label_colors={label_id: (255, 0, 0)},
    )

    boxes = np.array([[10, 10, 50, 50]], dtype=np.float32)
    labels = np.array([0], dtype=np.int64)

    output = OutputData(
        frame=fxt_large_frame,
        results=[{"pred_boxes": boxes, "pred_labels": labels}],
    )

    result = fxt_visualizer.visualize(output_data=output, visualization_info=vis_info)

    # Box edge should not be colored since boxes are disabled
    assert tuple(result[10, 20].tolist()) == (0, 0, 0)


def test_visualize_both_masks_and_boxes(fxt_visualizer_both: InferenceVisualizer, fxt_large_frame: np.ndarray) -> None:
    label_id = "00000000-0000-0000-0000-000000000001"
    vis_info = _make_vis_info(
        category_id_to_label_id={0: label_id},
        label_colors={label_id: (255, 0, 0)},
    )

    # Create a mask at position (50, 50)
    mask = np.zeros((1, 100, 100), dtype=np.float32)
    mask[0, 50, 50] = 1.0

    # Create a box around a different area
    boxes = np.array([[10, 10, 30, 30]], dtype=np.float32)
    labels = np.array([0], dtype=np.int64)

    output = OutputData(
        frame=fxt_large_frame,
        results=[{"pred_masks": mask, "pred_boxes": boxes, "pred_labels": labels}],
    )

    result = fxt_visualizer_both.visualize(output_data=output, visualization_info=vis_info)

    # Mask pixel should be colored
    assert tuple(result[50, 50].tolist()) == (255, 0, 0)
    # Box edge should also be colored
    assert tuple(result[10, 15].tolist()) == (255, 0, 0)
    # Area outside both should remain black
    assert tuple(result[80, 80].tolist()) == (0, 0, 0)


def test_box_renderer_draw(fxt_large_frame: np.ndarray) -> None:
    from runtime.webrtc.visualizer import BoxRenderer

    label_id = "00000000-0000-0000-0000-000000000001"
    vis_info = _make_vis_info(
        category_id_to_label_id={0: label_id},
        label_colors={label_id: (0, 255, 0)},
    )
    resolver = CategoryResolver(vis_info)
    renderer = BoxRenderer(box_thickness=2, visualize_labels=False, label_font_scale=0.5, resolver=resolver)

    boxes = np.array([[20, 20, 40, 40]], dtype=np.float32)
    labels = np.array([0], dtype=np.int64)
    prediction: dict[str, np.ndarray] = {"pred_boxes": boxes, "pred_labels": labels}

    result = renderer.draw(fxt_large_frame.copy(), prediction, labels)

    # Check box edge is drawn
    assert tuple(result[20, 30].tolist()) == (0, 255, 0)


def test_visualize_prediction_with_only_boxes_no_masks(
    fxt_visualizer_both: InferenceVisualizer, fxt_large_frame: np.ndarray
) -> None:
    label_id = "00000000-0000-0000-0000-000000000001"
    vis_info = _make_vis_info(
        category_id_to_label_id={0: label_id},
        label_colors={label_id: (0, 0, 255)},
    )

    boxes = np.array([[10, 10, 50, 50]], dtype=np.float32)
    labels = np.array([0], dtype=np.int64)

    output = OutputData(
        frame=fxt_large_frame,
        results=[{"pred_boxes": boxes, "pred_labels": labels}],  # No pred_masks key
    )

    result = fxt_visualizer_both.visualize(output_data=output, visualization_info=vis_info)

    # Box should be drawn
    assert tuple(result[10, 20].tolist()) == (0, 0, 255)


# --- Label caption tests ---


def test_visualize_draws_label_caption_with_name_and_score(
    fxt_visualizer_boxes_with_labels: InferenceVisualizer,
) -> None:
    """Label name and confidence score are rendered above the bounding box."""
    frame = np.zeros((200, 200, 3), dtype=np.uint8)
    label_id = "00000000-0000-0000-0000-000000000001"
    vis_info = _make_vis_info(
        category_id_to_label_id={0: label_id},
        label_colors={label_id: (255, 0, 0)},
        label_names={label_id: "Cat"},
    )

    boxes = np.array([[20, 40, 80, 100, 0.95]], dtype=np.float32)
    labels = np.array([0], dtype=np.int64)
    output = OutputData(frame=frame, results=[{"pred_boxes": boxes, "pred_labels": labels}])

    result = fxt_visualizer_boxes_with_labels.visualize(output_data=output, visualization_info=vis_info)

    # The region above the box (y < 40) should have non-zero pixels from the caption
    caption_region = result[0:40, 20:80]
    assert caption_region.any(), "Expected label caption pixels above the bounding box"


def test_visualize_draws_score_only_when_no_label_name(
    fxt_visualizer_boxes_with_labels: InferenceVisualizer,
) -> None:
    """When object_name is not set, only the confidence score is rendered."""
    frame = np.zeros((200, 200, 3), dtype=np.uint8)
    label_id = "00000000-0000-0000-0000-000000000001"
    vis_info = _make_vis_info(
        category_id_to_label_id={0: label_id},
        label_colors={label_id: (0, 255, 0)},
        # no label_names -> object_name is None
    )

    boxes = np.array([[20, 40, 80, 100, 0.88]], dtype=np.float32)
    labels = np.array([0], dtype=np.int64)
    output = OutputData(frame=frame, results=[{"pred_boxes": boxes, "pred_labels": labels}])

    result = fxt_visualizer_boxes_with_labels.visualize(output_data=output, visualization_info=vis_info)

    # Score-only caption should still appear above the box
    caption_region = result[0:40, 20:80]
    assert caption_region.any(), "Expected score caption pixels above the bounding box"


def test_visualize_no_caption_when_labels_disabled(
    fxt_visualizer_boxes_only: InferenceVisualizer, fxt_large_frame: np.ndarray
) -> None:
    """No caption is drawn when visualize_labels is False."""
    label_id = "00000000-0000-0000-0000-000000000001"
    vis_info = _make_vis_info(
        category_id_to_label_id={0: label_id},
        label_colors={label_id: (255, 0, 0)},
        label_names={label_id: "Dog"},
    )

    boxes = np.array([[20, 30, 80, 80, 0.90]], dtype=np.float32)
    labels = np.array([0], dtype=np.int64)
    output = OutputData(frame=fxt_large_frame, results=[{"pred_boxes": boxes, "pred_labels": labels}])

    result = fxt_visualizer_boxes_only.visualize(output_data=output, visualization_info=vis_info)

    # Area above the box should remain black (no caption)
    caption_region = result[0:28, 20:80]
    assert not caption_region.any(), "Expected no caption pixels when visualize_labels is disabled"


def test_visualize_no_caption_for_box_without_score_or_name(
    fxt_visualizer_boxes_with_labels: InferenceVisualizer,
) -> None:
    """No caption is drawn when there is no score column and no label name."""
    frame = np.zeros((200, 200, 3), dtype=np.uint8)
    vis_info = _make_vis_info(category_id_to_label_id={}, label_colors={})

    # Box with no score column (4 cols), unmapped category -> no name either
    boxes = np.array([[20, 40, 80, 100]], dtype=np.float32)
    labels = np.array([99], dtype=np.int64)
    output = OutputData(frame=frame, results=[{"pred_boxes": boxes, "pred_labels": labels}])

    result = fxt_visualizer_boxes_with_labels.visualize(output_data=output, visualization_info=vis_info)

    # Caption region should be empty (no name, no score)
    caption_region = result[0:38, 20:80]
    assert not caption_region.any(), "Expected no caption when neither name nor score is available"


def test_visualize_draws_label_name_only_without_score(
    fxt_visualizer_boxes_with_labels: InferenceVisualizer,
) -> None:
    """Label name is rendered even when there is no score column."""
    frame = np.zeros((200, 200, 3), dtype=np.uint8)
    label_id = "00000000-0000-0000-0000-000000000001"
    vis_info = _make_vis_info(
        category_id_to_label_id={0: label_id},
        label_colors={label_id: (0, 0, 255)},
        label_names={label_id: "Bike"},
    )

    # Box without score column (4 cols)
    boxes = np.array([[20, 40, 80, 100]], dtype=np.float32)
    labels = np.array([0], dtype=np.int64)
    output = OutputData(frame=frame, results=[{"pred_boxes": boxes, "pred_labels": labels}])

    result = fxt_visualizer_boxes_with_labels.visualize(output_data=output, visualization_info=vis_info)

    caption_region = result[0:40, 20:80]
    assert caption_region.any(), "Expected label-name-only caption above bounding box"


# --- CategoryResolver tests ---


class TestCategoryResolver:
    def test_resolve_mapped_category_returns_color_and_name(self) -> None:
        label_id = "00000000-0000-0000-0000-000000000001"
        vis_info = _make_vis_info(
            category_id_to_label_id={0: label_id},
            label_colors={label_id: (255, 0, 0)},
            label_names={label_id: "Cat"},
        )
        resolver = CategoryResolver(vis_info)

        info = resolver.resolve(0)

        assert info.color.to_tuple() == (255, 0, 0)
        assert info.object_name == "Cat"

    def test_resolve_mapped_category_without_name(self) -> None:
        label_id = "00000000-0000-0000-0000-000000000001"
        vis_info = _make_vis_info(
            category_id_to_label_id={0: label_id},
            label_colors={label_id: (0, 255, 0)},
        )
        resolver = CategoryResolver(vis_info)

        info = resolver.resolve(0)

        assert info.color.to_tuple() == (0, 255, 0)
        assert info.object_name is None

    def test_resolve_unmapped_category_returns_deterministic_color(self) -> None:
        resolver = CategoryResolver(None)

        info = resolver.resolve(7)

        assert info.color.to_tuple() == generate_deterministic_color(7)
        assert info.object_name is None

    def test_resolve_none_category_returns_fallback(self) -> None:
        resolver = CategoryResolver(None)

        info = resolver.resolve(None)

        assert info.color.to_tuple() == DEFAULT_FALLBACK_COLOR
        assert info.object_name is None

    def test_extract_category_id_valid(self) -> None:
        labels = np.array([3, 5], dtype=np.int64)
        assert CategoryResolver.extract_category_id(labels, 0) == 3
        assert CategoryResolver.extract_category_id(labels, 1) == 5

    def test_extract_category_id_out_of_bounds(self) -> None:
        labels = np.array([3], dtype=np.int64)
        assert CategoryResolver.extract_category_id(labels, 5) is None

    def test_extract_category_id_none_labels(self) -> None:
        assert CategoryResolver.extract_category_id(None, 0) is None
