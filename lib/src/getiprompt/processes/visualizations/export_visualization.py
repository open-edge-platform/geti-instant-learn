# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Export visualization to file."""

import colorsys
from pathlib import Path

import cv2
import numpy as np

from getiprompt.processes.visualizations.visualization_base import Visualization
from getiprompt.types import Annotations, Boxes, Image, Masks, Points
from getiprompt.utils import get_colors


class ExportMaskVisualization(Visualization):
    """The class exports the images for visualization.

    Examples:
        >>> import os
        >>> import numpy as np
        >>> from getiprompt.processes.visualizations import ExportMaskVisualization
        >>> from getiprompt.types import Image, Masks, Points
        >>>
        >>> visualizer = ExportMaskVisualization(output_folder="visualizations")
        >>> sample_image = Image(np.zeros((10, 10, 3), dtype=np.uint8))
        >>> visualizer(
        ...     images=[sample_image],
        ...     masks=[Masks()],
        ...     names=["test.png"],
        ...     points=[Points()],
        ... )
        >>> # Check if the visualization was saved
        >>> os.path.exists("visualizations/test.png")
        True
        >>> os.remove("visualizations/test.png")
    """

    def __init__(self, output_folder: str) -> None:
        super().__init__()
        self.output_folder = output_folder

    def __call__(
        self,
        images: list[Image] | None = None,
        masks: list[Masks] | None = None,
        names: list[str] | None = None,
        points: list[Points] | None = None,
        boxes: list[Boxes] | None = None,
        annotations: list[Annotations] | None = None,
        class_names: list[str] | None = None,
        show_legend: bool = False,
        legend_position: str = "top_right",
    ) -> None:
        """This method exports the visualization images.

        Args:
            images: List of images to visualize
            masks: List of masks to visualize
            names: List of names to visualize
            points: List of points to visualize
            boxes: List of boxes to visualize
            annotations: List of annotations to visualize
            class_names: List of class names to visualize
            show_legend: Whether to show the legend
            legend_position: Position of the legend
        """
        # Initialize defaults
        names = names or []
        masks = masks or []
        images = images or []

        # Setup colors and class info
        class_colors, num_classes = self._setup_colors(class_names)

        for i in range(len(images)):
            self._process_single_image(
                i,
                images,
                masks,
                names,
                points,
                boxes,
                annotations,
                class_colors,
                num_classes,
                show_legend,
                legend_position,
                class_names,
            )

    @staticmethod
    def add_legend(
        image: np.ndarray,
        class_names: list[str],
        class_colors: list[list[int]],
        legend_position: str = "top_right",
    ) -> np.ndarray:
        """Add a legend to the image showing class names and their colors.

        Args:
            image: The image to add the legend to
            class_names: List of class names
            class_colors: List of RGB colors corresponding to class names
            legend_position: Position of the legend ("top_right", "top_left", "bottom_right", "bottom_left")

        Returns:
            Image with legend added
        """
        if not class_names or not class_colors:
            return image

        # Calculate legend dimensions
        font_scale = max(0.5, image.shape[0] / 1000)  # Scale font with image size
        font_thickness = max(1, int(image.shape[0] / 500))
        line_height = int(30 * font_scale)
        legend_padding = int(10 * font_scale)
        color_box_size = int(20 * font_scale)

        # Calculate text dimensions for the longest class name
        max_text_width = 0
        for class_name in class_names:
            (text_width, _), _ = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
            max_text_width = max(max_text_width, text_width)

        legend_width = max_text_width + color_box_size + 3 * legend_padding
        legend_height = len(class_names) * line_height + 2 * legend_padding

        # Determine legend position
        img_height, img_width = image.shape[:2]
        if legend_position == "top_right":
            legend_x = img_width - legend_width - legend_padding
            legend_y = legend_padding
        elif legend_position == "top_left":
            legend_x = legend_padding
            legend_y = legend_padding
        elif legend_position == "bottom_right":
            legend_x = img_width - legend_width - legend_padding
            legend_y = img_height - legend_height - legend_padding
        elif legend_position == "bottom_left":
            legend_x = legend_padding
            legend_y = img_height - legend_height - legend_padding
        else:
            legend_x = img_width - legend_width - legend_padding
            legend_y = legend_padding

        # Draw legend background
        legend_bg = np.full((legend_height, legend_width, 3), 0, dtype=np.uint8)
        cv2.rectangle(legend_bg, (0, 0), (legend_width, legend_height), (255, 255, 255), -1)
        cv2.rectangle(legend_bg, (0, 0), (legend_width, legend_height), (0, 0, 0), 2)

        # Add legend items
        for i, (class_name, color) in enumerate(zip(class_names, class_colors, strict=False)):
            y_pos = legend_padding + i * line_height + line_height // 2

            # Draw color box
            color_x = legend_padding
            color_y = y_pos - color_box_size // 2
            cv2.rectangle(
                legend_bg,
                (color_x, color_y),
                (color_x + color_box_size, color_y + color_box_size),
                tuple(int(c) for c in color),
                -1,
            )
            cv2.rectangle(
                legend_bg,
                (color_x, color_y),
                (color_x + color_box_size, color_y + color_box_size),
                (0, 0, 0),
                1,
            )

            # Draw class name
            text_x = color_x + color_box_size + legend_padding
            cv2.putText(
                legend_bg,
                class_name,
                (text_x, y_pos + line_height // 3),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (0, 0, 0),
                font_thickness,
            )

        # Overlay legend on image
        image_with_legend = image.copy()
        image_with_legend[legend_y : legend_y + legend_height, legend_x : legend_x + legend_width] = legend_bg

        return image_with_legend

    @staticmethod
    def create_overlay(  # noqa: C901
        image: np.ndarray,
        masks: np.ndarray,
        mask_color: list[int],
        points: list[np.ndarray] | None = None,
        point_scores: list[float] | None = None,
        point_types: list[int] | None = None,
        boxes: list[np.ndarray] | None = None,
        box_scores: list[float] | None = None,
        box_types: list[int] | None = None,
        polygons: list[Points] | None = None,
        num_classes: int | None = None,
    ) -> np.ndarray:
        """Save a visualization of the segmentation mask overlaid on the image.

        Args:
            image: RGB image as numpy array
            masks: Segmentation mask object with containing instance masks
            mask_color: The RGB color for the masks.
            points: Optional points to visualize
            point_scores: Optional confidence scores for the points
            point_types: The type of point (usually for background, 1 for foreground)
            boxes: Optional boxes to visualize
            box_scores: Optional confidence scores for the boxes
            box_types: The type of box (class or label)
            polygons: Optional polygons to visualize
            num_classes: The number of classes for creating box colors per class
        """
        image_vis = image.copy()

        if masks is not None:
            # Draw each instance mask with the same class color and a border
            for instance in masks:
                # Apply mask with more transparency
                masked_img = np.where(instance[..., None], mask_color, image_vis)
                # Ensure both arrays have the same data type for cv2.addWeighted
                masked_img = masked_img.astype(np.uint8)
                image_vis = cv2.addWeighted(image_vis, 0.6, masked_img, 0.4, 0)

                # Add border to the mask
                instance_uint8 = instance.astype(np.uint8) * 255
                contours, _ = cv2.findContours(instance_uint8, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(image_vis, contours, -1, (255, 255, 255), 1)

            # Draw points and confidence scores if provided
            if points is not None and point_scores is not None and point_types is not None:
                for i, point in enumerate(points):
                    # Draw star marker
                    x, y = int(point[0]), int(point[1])
                    size = int(image.shape[0] / 100)  # Scale marker size with image
                    cv2.drawMarker(
                        image_vis,
                        (x, y),
                        (255, 255, 255),
                        cv2.MARKER_STAR if point_types[i] == 1.0 else cv2.MARKER_SQUARE,
                        size,
                    )

                    # Add confidence score text
                    confidence = float(point_scores[i])
                    cv2.putText(
                        image_vis,
                        f"{confidence:.2f}",
                        (x + 5, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        image.shape[0] / 1500,
                        (255, 255, 255),
                        1,
                    )

            # Draw boxes and confidence scores if provided
            if boxes is not None and box_scores is not None and box_types is not None:
                for i, box in enumerate(boxes):
                    # Draw star marker
                    x1, y1, x2, y2 = [int(box[0]), int(box[1]), int(box[2]), int(box[3])]
                    if num_classes is None or num_classes == 1:
                        rgb = (255, 64, 255)
                    else:
                        rgb = colorsys.hsv_to_rgb((box_types[i] / float(num_classes)), 1.0, 1.0)
                        rgb = [int(x * 255) for x in rgb]
                    cv2.rectangle(image_vis, (x1, y1), (x2, y2), color=rgb, thickness=2)

                    # Add confidence score text
                    confidence = float(box_scores[i])
                    cv2.putText(
                        image_vis,
                        f"{confidence:.2f}",
                        (x1 + 5, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        image.shape[0] / 1500,
                        (255, 255, 255),
                        1,
                    )

            # Draw the polygon and the vertices
            if polygons is not None:
                for polygon in polygons:
                    poly = np.array(polygon, np.int32)
                    poly = poly.reshape((-1, 1, 2))
                    cv2.polylines(image_vis, [poly], isClosed=True, color=(255, 0, 255), thickness=2)
                    for point in polygon:
                        x, y = int(point[0]), int(point[1])
                        size = int(image.shape[0] / 200)  # Scale marker size with image
                        cv2.drawMarker(
                            image_vis,
                            (x, y),
                            (0, 255, 0),
                            cv2.MARKER_SQUARE,
                            size,
                        )

        return image_vis

    @staticmethod
    def _setup_colors(class_names: list[str] | None) -> tuple[list[list[int]] | None, int]:
        """Setup colors and class information."""
        if class_names is not None:
            num_classes = len(class_names)
            class_colors = []
            for i in range(num_classes):
                if num_classes > 1:
                    rgb_float = colorsys.hsv_to_rgb(i / float(num_classes), 1.0, 1.0)
                    class_colors.append([int(x * 255) for x in rgb_float])
                else:
                    class_colors.append(get_colors(1)[0])
            return class_colors, num_classes
        return None, 1

    def _process_single_image(
        self,
        i: int,
        images: list[Image],
        masks: list[Masks],
        names: list[str],
        points: list[Points] | None,
        boxes: list[Boxes] | None,
        annotations: list[Annotations] | None,
        class_colors: list[list[int]] | None,
        num_classes: int,
        show_legend: bool,
        legend_position: str,
        class_names: list[str] | None = None,
    ) -> None:
        """Process a single image for visualization.

        Args:
            i: Index of the image
            images: List of images to visualize
            masks: List of masks to visualize
            names: List of names to visualize
            points: List of points to visualize
            boxes: List of boxes to visualize
            annotations: List of annotations to visualize
            class_colors: List of class colors to visualize
            num_classes: Number of classes to visualize
            show_legend: Whether to show the legend
            legend_position: Position of the legend
            class_names: List of class names to visualize
        """
        masks_per_class = masks[i]
        image_np = images[i].to_numpy()
        name = names[i]

        output_filename = Path(self.output_folder) / name
        Path.mkdir(Path(output_filename, parents=True).parent, exist_ok=True, parents=True)

        image_vis = image_np

        for class_id in masks_per_class.class_ids():
            image_vis = self._process_class_masks(
                image_vis, masks_per_class, class_id, points, boxes, annotations, i, class_colors, num_classes
            )

        # Add legend if requested
        if show_legend and class_colors is not None:
            # Use actual class names if provided, otherwise generate generic ones
            if class_names is not None:
                legend_class_names = class_names
            else:
                legend_class_names = [f"Class {i}" for i in range(len(class_colors))]
            image_vis = self.add_legend(image_vis, legend_class_names, class_colors, legend_position)

        # Save visualization
        cv2.imwrite(output_filename, cv2.cvtColor(image_vis, cv2.COLOR_RGB2BGR))

    def _process_class_masks(
        self,
        image_vis: np.ndarray,
        masks_per_class: Masks,
        class_id: int,
        points: list[Points] | None,
        boxes: list[Boxes] | None,
        annotations: list[Annotations] | None,
        i: int,
        class_colors: list[list[int]] | None,
        num_classes: int,
    ) -> np.ndarray:
        """Process masks for a specific class.

        Args:
            image_vis: Image to visualize
            masks_per_class: Masks to visualize
            class_id: Class ID
            points: Points to visualize
            boxes: Boxes to visualize
            annotations: Annotations to visualize
            i: Index of the image
            class_colors: List of class colors to visualize
            num_classes: Number of classes to visualize
        """
        mask_np = masks_per_class.to_numpy(class_id)

        # Extract points, boxes, and polygons
        point_data = self._extract_point_data(points, i, class_id)
        box_data = self._extract_box_data(boxes, i, class_id)
        polygons = self._get_polygons(annotations[i]) if annotations is not None else None

        # Get color for the class
        mask_color = self._get_class_color(class_id, class_colors, num_classes)

        return self.create_overlay(
            image=image_vis,
            masks=mask_np,
            mask_color=mask_color,
            **point_data,
            **box_data,
            polygons=polygons,
            num_classes=num_classes,
        )

    @staticmethod
    def _extract_point_data(points: list[Points] | None, i: int, class_id: int) -> dict:
        """Extract point data for visualization.

        Args:
            points: Points to visualize
            i: Index of the image
            class_id: Class ID
        """
        if points is not None and i < len(points) and points[i] is not None and not points[i].is_empty:
            current_points = points[i].data[class_id][0]
            return {
                "points": current_points.cpu().numpy()[:, :2],
                "point_scores": current_points.cpu().numpy()[:, 2],
                "point_types": current_points.cpu().numpy()[:, 3],
            }
        return {"points": None, "point_scores": None, "point_types": None}

    @staticmethod
    def _extract_box_data(boxes: list[Boxes] | None, i: int, class_id: int) -> dict:
        """Extract box data for visualization.

        Args:
            boxes: Boxes to visualize
            i: Index of the image
            class_id: Class ID
        """
        if boxes is not None and i < len(boxes) and boxes[i] is not None and not boxes[i].is_empty:
            current_boxes = boxes[i].data[class_id][0]
            return {
                "boxes": current_boxes.cpu().numpy()[:, :4],
                "box_scores": current_boxes.cpu().numpy()[:, 4],
                "box_types": current_boxes.cpu().numpy()[:, 5],
            }
        return {"boxes": None, "box_scores": None, "box_types": None}

    @staticmethod
    def _get_class_color(class_id: int, class_colors: list[list[int]] | None, num_classes: int) -> list[int]:
        """Get color for a specific class.

        Args:
            class_id: Class ID
            class_colors: List of class colors to visualize
            num_classes: Number of classes to visualize
        """
        if class_colors is not None and class_id < len(class_colors):
            return class_colors[class_id]
        if num_classes > 1:
            rgb_float = colorsys.hsv_to_rgb(class_id / float(num_classes), 1.0, 1.0)
            return [int(x * 255) for x in rgb_float]
        return get_colors(1)[0]

    @staticmethod
    def _get_polygons(annotations_per_class: Annotations) -> list[np.ndarray]:
        """Get polygons for a specific class.

        Args:
            annotations_per_class: Annotations to visualize
        """
        if not annotations_per_class.polygons.is_empy():
            msg = "Multiple class annotations not supported yet."
            raise RuntimeError(msg)
        return annotations_per_class.polygons[0]
