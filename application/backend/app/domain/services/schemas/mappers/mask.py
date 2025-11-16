import cv2
import numpy as np

from domain.services.schemas.annotation import PolygonAnnotation


def polygons_to_masks(polygons: list[PolygonAnnotation], image_height: int, image_width: int) -> np.ndarray:
    """
    Convert multiple polygon annotations to a stacked array of binary masks.

    Args:
        polygons: List of PolygonAnnotation objects with normalized coordinates (0.0-1.0)
        image_height: Height of the output masks in pixels
        image_width: Width of the output masks in pixels

    Returns:
        Binary masks as numpy array with shape (N, H, W) and dtype uint8.
        N is the number of polygons. Values are 1 inside polygons, 0 outside.

    Raises:
        RuntimeError: If polygons list is empty
    """
    if not polygons:
        raise RuntimeError("Cannot convert empty polygons list to masks")

    masks = np.zeros((len(polygons), image_height, image_width), dtype=np.uint8)

    for i, polygon in enumerate(polygons):
        pixel_points = np.array(
            [[int(pt.x * image_width), int(pt.y * image_height)] for pt in polygon.points],
            dtype=np.int32,
        )
        mask = np.zeros((image_height, image_width), dtype=np.uint8)
        cv2.fillPoly(mask, [pixel_points], (1,))
        masks[i] = mask

    return masks
