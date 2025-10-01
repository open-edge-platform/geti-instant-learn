# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Box aware mask filter."""

import torch

from getiprompt.filters.masks.mask_filter_base import MaskFilter
from getiprompt.types.boxes import Boxes
from getiprompt.types.masks import Masks


class BoxAwareMaskFilter(MaskFilter):
    """This filter removes masks and boxes when one mask is completely contained in another of the same class."""

    def __init__(self, overlap_threshold: float = 0.9) -> None:
        """Initialize the BoxAwareMaskFilter.

        Args:
            overlap_threshold: The threshold for overlap between masks.
        """
        self.overlap_threshold = overlap_threshold

    def __call__(
        self, masks_per_image: list[Masks], boxes_per_image: list[Boxes] | None = None
    ) -> tuple[list[Masks], list[Boxes] | None]:
        """Filter the masks and optionally the corresponding boxes.

        Args:
            masks_per_image: List of masks for each image
            boxes_per_image: Optional list of boxes for each image

        Returns:
            Tuple of filtered masks and boxes (or None if no boxes provided)
        """
        filtered_masks_per_image = []
        filtered_boxes_per_image = [] if boxes_per_image is not None else None

        for i, image_masks in enumerate(masks_per_image):
            image_boxes = boxes_per_image[i] if boxes_per_image is not None else None
            filtered_image_masks, filtered_image_boxes = self._filter_single_image(image_masks, image_boxes)

            filtered_masks_per_image.append(filtered_image_masks)
            if filtered_boxes_per_image is not None:
                filtered_boxes_per_image.append(filtered_image_boxes)

        return filtered_masks_per_image, filtered_boxes_per_image

    def _filter_single_image(self, image_masks: Masks, image_boxes: Boxes | None = None) -> tuple[Masks, Boxes | None]:
        """Filter masks and boxes for a single image.

        Args:
            image_masks: Masks for the image
            image_boxes: Optional boxes for the image

        Returns:
            Tuple of filtered masks and boxes
        """
        filtered_image_masks = Masks()
        filtered_image_boxes = Boxes() if image_boxes is not None else None

        for class_id in image_masks.class_ids():
            indices_to_keep = self._filter_masks_for_class(image_masks, class_id)

            if indices_to_keep:
                self._add_filtered_masks(filtered_image_masks, image_masks, class_id, indices_to_keep)

            if image_boxes is not None and filtered_image_boxes is not None and indices_to_keep:
                self._add_filtered_boxes(filtered_image_boxes, image_boxes, class_id, indices_to_keep)

        return filtered_image_masks, filtered_image_boxes

    def _filter_masks_for_class(self, image_masks: Masks, class_id: int) -> list[int]:
        """Filter masks for a single class.

        This method is optimized to reduce redundant comparisons. It sorts masks by area in descending order
        and then checks for containment. A smaller mask can be contained in a larger one, so we only
        need to check in one direction, effectively halving the number of comparisons in the inner loop.

        Args:
            image_masks: Masks for the image
            class_id: Class id to filter masks for

        Returns:
            List of indices to keep for the given class.
        """
        masks_for_class = image_masks.get(class_id)
        if len(masks_for_class) == 0:
            return []
        num_masks = masks_for_class.shape[0]

        areas = masks_for_class.sum(dim=(1, 2))
        sorted_indices = torch.argsort(areas, descending=True)

        is_removed = [False] * num_masks
        for i in range(num_masks):
            idx_i = sorted_indices[i].item()
            if is_removed[idx_i]:
                continue
            mask_i = masks_for_class[idx_i]

            for j in range(i + 1, num_masks):
                idx_j = sorted_indices[j].item()
                if is_removed[idx_j]:
                    continue
                mask_j = masks_for_class[idx_j]

                # Check if smaller mask mask_j is contained in larger mask mask_i
                if self._is_mask_contained(mask_j, mask_i, self.overlap_threshold):
                    is_removed[idx_j] = True

        return [i for i, removed in enumerate(is_removed) if not removed]

    @staticmethod
    def _is_mask_contained(
        mask: torch.Tensor,
        other_mask: torch.Tensor,
        overlap_threshold: float,
    ) -> bool:
        """Check if a mask is contained within any other mask in the list.

        Args:
            mask: The mask to check
            other_mask: The other mask to check
            overlap_threshold: The threshold for overlap

        Returns:
            True if the mask is contained within any other mask in the list
        """
        if mask.sum() == 0:
            return True
        overlap = BoxAwareMaskFilter._compute_overlap_percentage(mask, other_mask)
        return overlap >= overlap_threshold

    @staticmethod
    def _add_filtered_masks(
        filtered_masks: Masks, original_masks: Masks, class_id: int, indices_to_keep: list[int]
    ) -> None:
        """Add filtered masks to the result.

        Args:
            filtered_masks: The result masks object to add to
            original_masks: The original masks object
            class_id: The class ID
            indices_to_keep: List of indices to keep
        """
        class_masks = original_masks.data[class_id]
        individual_masks = [class_masks[mask_idx] for mask_idx in range(class_masks.shape[0])]
        kept_masks = [individual_masks[idx] for idx in indices_to_keep]
        stacked_masks = torch.stack(kept_masks, dim=0)
        filtered_masks.data[class_id] = stacked_masks

    @staticmethod
    def _add_filtered_boxes(
        filtered_boxes: Boxes, original_boxes: Boxes, class_id: int, indices_to_keep: list[int]
    ) -> None:
        """Add filtered boxes to the result.

        Args:
            filtered_boxes: The result boxes object to add to
            original_boxes: The original boxes object
            class_id: The class ID
            indices_to_keep: List of indices to keep
        """
        boxes_for_class = original_boxes.get(class_id)
        if not boxes_for_class:
            return

        for box_tensor in boxes_for_class:
            if len(box_tensor) > 0 and indices_to_keep:
                kept_box_indices = [idx for idx in indices_to_keep if idx < len(box_tensor)]
                if kept_box_indices:
                    kept_boxes = box_tensor[kept_box_indices]
                    if len(kept_boxes) > 0:
                        filtered_boxes.add(kept_boxes, class_id)

    @staticmethod
    def _compute_overlap_percentage(mask: torch.Tensor, other_mask: torch.Tensor) -> float:
        """Percentage of pixels of mask A that are also in mask B.

        Masks are boolean grids of shape (H, W).

        Args:
            mask: First mask
            other_mask: Second mask

        Returns:
            Percentage of pixels of mask A that are also in mask B
        """
        mask_sum = mask.sum().item()
        if mask_sum == 0:
            return 0.0
        return (mask & other_mask).sum().item() / mask_sum
