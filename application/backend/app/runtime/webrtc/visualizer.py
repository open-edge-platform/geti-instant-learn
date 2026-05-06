# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from typing import Protocol
from uuid import UUID

import cv2
import numpy as np

from domain.services.schemas.label import RGBColor, VisualizationInfo, VisualizationLabel
from domain.services.schemas.processor import OutputData
from settings import get_settings

logger = logging.getLogger(__name__)

_FALLBACK_ID = UUID(int=0)
DEFAULT_FALLBACK_COLOR: tuple[int, int, int] = (128, 128, 128)

_REFERENCE_AREA: int = 1920 * 1080
_MIN_SCALE: float = 0.6
_MAX_SCALE: float = 2.0


class ResolutionScaler:
    """Scale visualization parameters relative to a 1920x1080 reference."""

    def __init__(self, multiplier: float = 1.0) -> None:
        self._multiplier = multiplier

    def factor(self, frame: np.ndarray) -> float:
        """Return the combined resolution and user scale factor."""
        h, w = frame.shape[:2]
        area_factor = (w * h / _REFERENCE_AREA) ** 0.5
        clamped = max(_MIN_SCALE, min(_MAX_SCALE, area_factor))
        return clamped * self._multiplier


class CategoryResolver:
    """Resolve category IDs to visualization labels.

    Encapsulates the category_id → label_id → VisualizationLabel lookup chain.
    Falls back to deterministic colors for unmapped categories and a neutral
    fallback when no category is available.
    """

    def __init__(self, visualization_info: VisualizationInfo | None = None) -> None:
        self._category_id_to_label_id: dict[int, str] = {}
        self._label_id_to_vis: dict[str, VisualizationLabel] = {}

        if visualization_info is not None:
            self._category_id_to_label_id = visualization_info.category_mappings.category_id_to_label_id
            self._label_id_to_vis = {str(item.id): item for item in visualization_info.label_colors}

    def resolve(self, category_id: int | None) -> VisualizationLabel:
        """Return visualization label for a predicted category."""
        if category_id is None:
            return VisualizationLabel(id=_FALLBACK_ID, color=RGBColor(*DEFAULT_FALLBACK_COLOR))

        label_id = self._category_id_to_label_id.get(category_id)
        if label_id is None:
            logger.warning("No label mapping found for category_id=%d", category_id)
            return VisualizationLabel(id=_FALLBACK_ID, color=RGBColor(*generate_deterministic_color(category_id)))

        info = self._label_id_to_vis.get(label_id)
        if info is None:
            logger.warning("No color found for label_id=%s (category_id=%d)", label_id, category_id)
            return VisualizationLabel(id=_FALLBACK_ID, color=RGBColor(*generate_deterministic_color(category_id)))

        logger.debug("Category %d -> label %s -> color %s", category_id, label_id, info.color)
        return info

    @staticmethod
    def extract_category_id(labels: np.ndarray | None, index: int) -> int | None:
        """Extract category ID from a numpy label array at the given index."""
        if labels is None or index >= len(labels):
            return None
        return int(labels[index])


class OverlayRenderer(Protocol):
    """Strategy interface for rendering a specific overlay type onto a frame."""

    def draw(
        self,
        frame: np.ndarray,
        prediction: dict[str, np.ndarray],
        labels: np.ndarray | None,
    ) -> np.ndarray: ...


class MaskRenderer:
    """Render colored mask overlays and contours onto a frame."""

    def __init__(
        self, mask_alpha: float, outline_thickness: int, resolver: CategoryResolver, scaler: ResolutionScaler
    ) -> None:
        self._alpha = mask_alpha
        self._outline_thickness = outline_thickness
        self._resolver = resolver
        self._scaler = scaler

    def draw(
        self,
        frame: np.ndarray,
        prediction: dict[str, np.ndarray],
        labels: np.ndarray | None,
    ) -> np.ndarray:
        masks = prediction.get("pred_masks")
        if masks is None or masks.size == 0:
            return frame

        labels_np = labels if labels is not None and labels.size > 0 else None
        overlay = frame.copy()
        scale = self._scaler.factor(frame)

        for mask_idx, mask in enumerate(masks):
            category_id = self._resolver.extract_category_id(labels_np, mask_idx)
            info = self._resolver.resolve(category_id)
            color = info.color.to_tuple()

            mask_bool = mask > 0.5
            overlay = self._apply_overlay(overlay, mask_bool, color)
            overlay = self._draw_contours(overlay, mask_bool, color, scale)

        return overlay

    def _apply_overlay(self, frame: np.ndarray, mask_bool: np.ndarray, color: tuple[int, int, int]) -> np.ndarray:
        frame[mask_bool] = (
            frame[mask_bool] * (1 - self._alpha) + np.array(color, dtype=np.float32) * self._alpha
        ).astype(np.uint8)
        return frame

    def _draw_contours(
        self, frame: np.ndarray, mask_bool: np.ndarray, color: tuple[int, int, int], scale: float
    ) -> np.ndarray:
        mask_uint8 = (mask_bool * 255).astype(np.uint8)
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        thickness = max(1, round(self._outline_thickness * scale))
        cv2.drawContours(frame, contours, -1, color, thickness)
        return frame


# Probe used to calibrate cv2 font scale to a target pixel height.
_FONT_PROBE_TEXT = "Ag"
_FONT_FACE = cv2.FONT_HERSHEY_SIMPLEX


def _font_scale_for_pixel_height(target_height_px: float) -> float:
    """Return the cv2 font scale that yields the requested pixel height."""
    (_, probe_h), _ = cv2.getTextSize(_FONT_PROBE_TEXT, _FONT_FACE, 1.0, 1)
    if probe_h <= 0:
        return 1.0
    return target_height_px / probe_h


class BoxRenderer:
    """Render bounding boxes with optional label captions onto a frame."""

    def __init__(
        self,
        box_thickness: int,
        visualize_labels: bool,
        label_text_height_px: int,
        resolver: CategoryResolver,
        scaler: ResolutionScaler,
    ) -> None:
        self._box_thickness = box_thickness
        self._visualize_labels = visualize_labels
        self._label_text_height_px = label_text_height_px
        self._resolver = resolver
        self._scaler = scaler

    def draw(
        self,
        frame: np.ndarray,
        prediction: dict[str, np.ndarray],
        labels: np.ndarray | None,
    ) -> np.ndarray:
        boxes = prediction.get("pred_boxes")
        if boxes is None or boxes.size == 0:
            return frame

        labels_np = labels if labels is not None and labels.size > 0 else None
        scale = self._scaler.factor(frame)

        for box_idx, box in enumerate(boxes):
            category_id = self._resolver.extract_category_id(labels_np, box_idx)
            info = self._resolver.resolve(category_id)
            color = info.color.to_tuple()

            x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            box_thick = max(1, round(self._box_thickness * scale))
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, box_thick)

            if self._visualize_labels:
                score = float(box[4]) if len(box) > 4 else None
                self._draw_caption(frame, x1, y1, info.object_name, score, color, scale)

        return frame

    def _draw_caption(
        self,
        frame: np.ndarray,
        x1: int,
        y1: int,
        label_name: str | None,
        score: float | None,
        color: tuple[int, int, int],
        scale: float,
    ) -> None:
        parts: list[str] = []
        if label_name:
            parts.append(label_name)
        if score is not None:
            parts.append(f"{100 * score:.0f}%")
        if not parts:
            return

        caption = " ".join(parts)
        target_h = self._label_text_height_px * scale
        font_scale = _font_scale_for_pixel_height(target_h)
        thickness = max(1, round(target_h / 14))
        (text_w, text_h), baseline = cv2.getTextSize(caption, _FONT_FACE, font_scale, thickness)

        pad = max(1, round(target_h * 0.25))
        gap = max(1, round(target_h * 0.25))
        text_y = max(y1 - gap, text_h + pad)
        text_x = x1

        cv2.rectangle(
            frame,
            (text_x, text_y - text_h - pad),
            (text_x + text_w + pad, text_y + baseline),
            color,
            cv2.FILLED,
        )
        cv2.putText(frame, caption, (text_x + 1, text_y), _FONT_FACE, font_scale, (255, 255, 255), thickness)


class InferenceVisualizer:
    """Overlay model predictions onto RGB frames for WebRTC streaming.

    Composes a list of ``OverlayRenderer`` strategies (masks, boxes) based on
    application settings.  The public API — ``visualize()`` — remains unchanged.
    """

    def __init__(self, enable_visualization: bool = True) -> None:
        self._enabled = enable_visualization
        settings = get_settings()
        self._visualize_masks = settings.visualize_masks
        self._visualize_boxes = settings.visualize_boxes
        self._mask_alpha = settings.mask_alpha
        self._mask_outline_thickness = settings.mask_outline_thickness
        self._box_thickness = settings.box_thickness
        self._visualize_labels = settings.visualize_labels
        self._label_text_height_px = settings.label_text_height_px
        self._scaler = ResolutionScaler(multiplier=settings.visualization_scale)

    def _build_renderers(self, resolver: CategoryResolver) -> list[OverlayRenderer]:
        """Create renderer instances for the current visualization pass."""
        renderers: list[OverlayRenderer] = []
        if self._visualize_masks:
            renderers.append(
                MaskRenderer(
                    mask_alpha=self._mask_alpha,
                    outline_thickness=self._mask_outline_thickness,
                    resolver=resolver,
                    scaler=self._scaler,
                )
            )
        if self._visualize_boxes:
            renderers.append(
                BoxRenderer(
                    box_thickness=self._box_thickness,
                    visualize_labels=self._visualize_labels,
                    label_text_height_px=self._label_text_height_px,
                    resolver=resolver,
                    scaler=self._scaler,
                )
            )
        return renderers

    def visualize(self, output_data: OutputData, visualization_info: VisualizationInfo | None = None) -> np.ndarray:
        """Render model predictions onto the frame."""
        if not self._enabled or not output_data.results:
            return output_data.frame

        annotated = output_data.frame.copy()
        resolver = CategoryResolver(visualization_info)
        renderers = self._build_renderers(resolver)

        logger.debug("Visualizing %d predictions", len(output_data.results))
        for prediction in output_data.results:
            labels = prediction.get("pred_labels")
            for renderer in renderers:
                annotated = renderer.draw(annotated, prediction, labels)

        return annotated


def generate_deterministic_color(index: int) -> tuple[int, int, int]:
    """Generate a stable RGB color for a numeric category ID."""
    hue = (index * 67) % 180
    hsv_color = np.array([[[hue, 255, 255]]], dtype=np.uint8)
    rgb_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2RGB)[0, 0]
    return int(rgb_color[0]), int(rgb_color[1]), int(rgb_color[2])
