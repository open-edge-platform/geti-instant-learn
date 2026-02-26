# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging

import cv2
import numpy as np

from domain.services.schemas.label import VisualizationInfo
from domain.services.schemas.processor import OutputData
from settings import get_settings

logger = logging.getLogger(__name__)

DEFAULT_FALLBACK_COLOR: tuple[int, int, int] = (128, 128, 128)


class InferenceVisualizer:
    """
    Overlay model predictions onto RGB frames for WebRTC streaming.

    The visualizer expects `OutputData.results` items to follow the model's prediction dict
    convention (e.g. `pred_masks`, `pred_labels`, optional `pred_boxes`).

    Color resolution uses:
      1) `category_id` from `pred_labels` (tensor/int) -> Python `int`
      2) `category_id_to_label_id[category_id]` -> label UUID `str`
      3) `label_colors[label_uuid]` -> RGB tuple

    If a category cannot be resolved to a configured label color, a deterministic per-category
    color is generated. If the category itself is missing, a neutral fallback is used.
    """

    def __init__(self, enable_visualization: bool = True) -> None:
        self._enabled = enable_visualization
        settings = get_settings()
        self._mask_alpha = settings.mask_alpha
        self._mask_outline_thickness = settings.mask_outline_thickness

    def visualize(self, output_data: OutputData, visualization_info: VisualizationInfo | None = None) -> np.ndarray:
        """Render model predictions onto the frame."""

        if not self._enabled or not output_data.results:
            return output_data.frame

        annotated = output_data.frame.copy()
        category_id_to_label_id: dict[int, str] = {}
        label_id_to_color: dict[str, tuple[int, int, int]] = {}

        if visualization_info is not None:
            category_id_to_label_id = visualization_info.category_mappings.category_id_to_label_id
            label_id_to_color = {str(item.id): item.color.to_tuple() for item in visualization_info.label_colors}

        logger.debug("Visualizing the output data: %s, categories=%s", output_data, category_id_to_label_id)
        for prediction in output_data.results:
            boxes = prediction.get("pred_boxes")
            if boxes is not None and boxes.size > 0:
                logger.warning("pred_boxes visualization is not supported and will be ignored")

            masks = prediction.get("pred_masks")
            labels = prediction.get("pred_labels")

            if masks is not None and masks.size > 0:
                annotated = self._draw_masks(annotated, masks, labels, label_id_to_color, category_id_to_label_id)
        return annotated

    def _draw_masks(
        self,
        frame: np.ndarray,
        masks: np.ndarray,
        labels: np.ndarray | None,
        label_colors: dict[str, tuple[int, int, int]],
        category_id_to_label_id: dict[int, str],
    ) -> np.ndarray:
        """
        Draw colored mask overlays and contours for a prediction.

        Args:
            frame: RGB frame to draw on.
            masks: Array of shape [N, H, W] with mask logits/probabilities.
            labels: Array of shape [N] with category IDs for each mask.
            label_colors: Mapping of label UUID (str) to RGB color tuple.
            category_id_to_label_id: Mapping of category ID (int) to label UUID (str).

        Returns:
            A new RGB frame with mask overlays applied.
        """
        labels_np: np.ndarray | None = None
        if labels is not None and labels.size > 0:
            labels_np = labels
        overlay = frame.copy()

        for mask_idx, mask in enumerate(masks):
            category_id = self._extract_category_id_from_array(labels_np, mask_idx)
            color = self._resolve_color_for_category(category_id, label_colors, category_id_to_label_id)

            mask_bool = mask > 0.5
            overlay = self._apply_mask_overlay(overlay, mask_bool, color)
            overlay = self._draw_mask_contours(overlay, mask_bool, color)

        return overlay

    @staticmethod
    def _extract_category_id_from_array(labels_np: np.ndarray | None, index: int) -> int | None:
        """
        Extract category ID from numpy array.

        Args:
            labels_np: Numpy array of predicted category IDs (already on CPU).
            index: Index of the mask/label to read.

        Returns:
            Category ID as int, or None if unavailable.
        """
        if labels_np is None or index >= len(labels_np):
            return None
        return int(labels_np[index])

    def _resolve_color_for_category(
        self,
        category_id: int | None,
        label_colors: dict[str, tuple[int, int, int]],
        category_id_to_label_id: dict[int, str],
    ) -> tuple[int, int, int]:
        """
        Resolve an RGB color for a predicted category.

        Args:
            category_id: Predicted category ID.
            label_colors: Mapping from label UUID to RGB.
            category_id_to_label_id: Mapping from category ID to label UUID.
        """
        if category_id is None:
            return DEFAULT_FALLBACK_COLOR

        label_id = category_id_to_label_id.get(category_id)

        if label_id is None:
            logger.warning("No label mapping found for category_id=%d", category_id)
            return self._generate_deterministic_color(category_id)

        color = label_colors.get(label_id)

        if color is None:
            logger.warning("No color found for label_id=%s (category_id=%d)", label_id, category_id)
            return self._generate_deterministic_color(category_id)

        logger.debug("Category %d -> label %s -> color %s", category_id, label_id, color)
        return color

    def _apply_mask_overlay(self, frame: np.ndarray, mask_bool: np.ndarray, color: tuple[int, int, int]) -> np.ndarray:
        """
        Alpha-blend a single mask into the frame.

        Args:
            frame: RGB frame to blend into.
            mask_bool: Boolean mask (H, W) selecting pixels to blend.
            color: RGB overlay color.
        """
        frame[mask_bool] = (
            frame[mask_bool] * (1 - self._mask_alpha) + np.array(color, dtype=np.float32) * self._mask_alpha
        ).astype(np.uint8)
        return frame

    def _draw_mask_contours(self, frame: np.ndarray, mask_bool: np.ndarray, color: tuple[int, int, int]) -> np.ndarray:
        """
        Draw mask contours on the frame.
        """
        mask_uint8 = (mask_bool * 255).astype(np.uint8)
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(frame, contours, -1, color, self._mask_outline_thickness)
        return frame

    @staticmethod
    def _generate_deterministic_color(index: int) -> tuple[int, int, int]:
        """
        Generate a stable RGB color for a numeric category ID.
        """
        hue = (index * 67) % 180
        hsv_color = np.array([[[hue, 255, 255]]], dtype=np.uint8)
        rgb_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2RGB)[0, 0]
        return int(rgb_color[0]), int(rgb_color[1]), int(rgb_color[2])
