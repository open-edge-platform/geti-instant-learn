# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""ROI Cropper for handling small masks during feature extraction.

This module provides utilities for cropping regions of interest (ROI) around masks
to prevent small objects from disappearing during downsampling operations.

The key insight is that when you have a small object in a large image, after:
1. Resizing to encoder input_size (e.g., 512x512)
2. Applying pooling with patch_size (e.g., 16x16)

The small object may disappear entirely. The solution is to crop both the
reference IMAGE and MASK around the ROI before encoding, so the object
occupies a larger portion of the cropped region.

Example usage:
    >>> cropper = ROICropper(input_size=512, patch_size=16)
    >>> # Check if cropping is needed
    >>> if cropper.should_crop(mask, image_height=6440, image_width=12880):
    ...     crop_result = cropper.crop_image_and_mask(image, mask)
    ...     # Now encode the cropped image
    ...     embeddings = encoder(crop_result.cropped_image)
    ...     # Use cropped mask with embeddings
    ...     features = extractor(embeddings, crop_result.cropped_mask, ...)
"""

from dataclasses import dataclass

import torch


@dataclass
class CropRegion:
    """Represents a crop region with coordinates.

    Attributes:
        x_min: Left boundary of the crop region.
        y_min: Top boundary of the crop region.
        x_max: Right boundary of the crop region.
        y_max: Bottom boundary of the crop region.
    """

    x_min: int
    y_min: int
    x_max: int
    y_max: int

    @property
    def width(self) -> int:
        """Return the width of the crop region."""
        return self.x_max - self.x_min

    @property
    def height(self) -> int:
        """Return the height of the crop region."""
        return self.y_max - self.y_min


@dataclass
class CropResult:
    """Result of cropping an image and mask around ROI.

    Attributes:
        cropped_image: The image cropped to the ROI region.
        cropped_mask: The mask cropped to the ROI region.
        crop_region: The coordinates of the crop region in the original image.
    """

    cropped_image: torch.Tensor
    cropped_mask: torch.Tensor
    crop_region: CropRegion


def get_mask_bounding_box(mask: torch.Tensor) -> tuple[int, int, int, int]:
    """Compute the bounding box of non-zero regions in a binary mask.

    Args:
        mask: Binary mask tensor of shape (H, W) or (1, H, W) or (C, H, W).

    Returns:
        Tuple of (x_min, y_min, x_max, y_max) representing the bounding box.
        Returns (0, 0, W, H) if the mask is empty.
    """
    if mask.ndim == 3:
        mask = mask.squeeze(0) if mask.shape[0] == 1 else mask.sum(dim=0).bool()

    h, w = mask.shape
    nonzero = torch.nonzero(mask, as_tuple=True)

    if len(nonzero[0]) == 0:
        # Empty mask, return full image bounds
        return 0, 0, w, h

    y_min = int(nonzero[0].min().item())
    y_max = int(nonzero[0].max().item()) + 1
    x_min = int(nonzero[1].min().item())
    x_max = int(nonzero[1].max().item()) + 1

    return x_min, y_min, x_max, y_max


def compute_roi_crop_size(
    bbox_width: int,
    bbox_height: int,
    input_size: int,
    patch_size: int,
    min_object_ratio: float = 0.1,
    min_patches: int = 2,
) -> int:
    """Compute the optimal square crop size to preserve small objects after downsampling.

    The formula ensures that after resizing to `input_size` and applying
    pooling with `patch_size`, the object occupies at least `min_patches`
    patches in the feature map.

    For an object to survive pooling, after all transforms:
    - Object dimension in feature space = (obj_dim / crop_size) * (input_size / patch_size)
    - This must be >= min_patches

    Therefore: crop_size <= obj_dim * input_size / (patch_size * min_patches)

    We also enforce a minimum object-to-crop ratio to ensure the object
    is prominent in the crop.

    Args:
        bbox_width: Width of the object's bounding box in pixels.
        bbox_height: Height of the object's bounding box in pixels.
        input_size: The input size images are resized to before encoding.
        patch_size: The patch/kernel size used for pooling.
        min_object_ratio: Minimum ratio of object size to crop size (default: 0.1).
            Higher values create tighter crops around the object.
        min_patches: Minimum number of patches the object should occupy
            in the final feature map (default: 2).

    Returns:
        The computed square crop size in pixels.

    Examples:
        >>> # Object bbox is 50x30 pixels
        >>> crop_size = compute_roi_crop_size(
        ...     bbox_width=50,
        ...     bbox_height=30,
        ...     input_size=512,
        ...     patch_size=16,
        ... )
    """
    # Use the smaller dimension to ensure both sides survive pooling
    min_obj_dim = min(bbox_width, bbox_height)
    max_obj_dim = max(bbox_width, bbox_height)

    # Feature map size after resize and pool
    feature_size = input_size // patch_size

    # Calculate max crop size that ensures object survives
    # obj_dim_in_features = (min_obj_dim / crop_size) * feature_size >= min_patches
    # crop_size <= min_obj_dim * feature_size / min_patches
    max_crop_for_survival = (min_obj_dim * feature_size) // min_patches

    # Also enforce minimum object ratio in the crop
    # max_obj_dim / crop_size >= min_object_ratio
    # crop_size <= max_obj_dim / min_object_ratio
    max_crop_for_ratio = int(max_obj_dim / min_object_ratio)

    # Take the smaller of the two constraints
    crop_size = min(max_crop_for_survival, max_crop_for_ratio)

    # Ensure crop size is at least as large as the object bbox
    crop_size = max(crop_size, max_obj_dim)

    return crop_size


def compute_centered_crop_region(
    x_min: int,
    y_min: int,
    x_max: int,
    y_max: int,
    crop_size: int,
    image_height: int,
    image_width: int,
) -> CropRegion:
    """Compute a square crop region centered on the bounding box.

    Handles edge cases where the crop would extend beyond image boundaries
    by shifting the crop region while maintaining the crop size.

    Args:
        x_min: Left boundary of the object bounding box.
        y_min: Top boundary of the object bounding box.
        x_max: Right boundary of the object bounding box.
        y_max: Bottom boundary of the object bounding box.
        crop_size: Desired size of the square crop.
        image_height: Height of the source image.
        image_width: Width of the source image.

    Returns:
        CropRegion with (crop_x_min, crop_y_min, crop_x_max, crop_y_max).
    """
    # Center of the bounding box
    center_x = (x_min + x_max) // 2
    center_y = (y_min + y_max) // 2

    # Initial crop boundaries centered on bbox
    half_size = crop_size // 2
    crop_x_min = center_x - half_size
    crop_y_min = center_y - half_size
    crop_x_max = crop_x_min + crop_size
    crop_y_max = crop_y_min + crop_size

    # Adjust if crop extends beyond image boundaries
    if crop_x_min < 0:
        crop_x_max -= crop_x_min  # Shift right
        crop_x_min = 0
    if crop_y_min < 0:
        crop_y_max -= crop_y_min  # Shift down
        crop_y_min = 0
    if crop_x_max > image_width:
        crop_x_min -= crop_x_max - image_width  # Shift left
        crop_x_max = image_width
    if crop_y_max > image_height:
        crop_y_min -= crop_y_max - image_height  # Shift up
        crop_y_max = image_height

    # Final clamp to ensure valid boundaries
    crop_x_min = max(0, crop_x_min)
    crop_y_min = max(0, crop_y_min)
    crop_x_max = min(image_width, crop_x_max)
    crop_y_max = min(image_height, crop_y_max)

    return CropRegion(
        x_min=crop_x_min,
        y_min=crop_y_min,
        x_max=crop_x_max,
        y_max=crop_y_max,
    )


class ROICropper:
    """Crops regions of interest around masks to preserve small objects.

    This class computes optimal crop regions for images and masks to ensure
    that small objects are not lost during downsampling operations. The crop
    is centered on the mask's bounding box with a size calculated to maintain
    object visibility in the feature map after encoding.

    **Important**: This cropper should be used BEFORE encoding the image.
    The workflow is:
    1. Check if cropping is needed with `should_crop()`
    2. Crop both image and mask with `crop_image_and_mask()`
    3. Encode the CROPPED image to get embeddings
    4. Use the CROPPED mask with those embeddings in feature extraction

    Args:
        input_size: The input size images are resized to before encoding.
        patch_size: The patch/kernel size used for pooling.
        min_object_ratio: Minimum ratio of object size to crop size.
        min_patches: Minimum patches the object should occupy after pooling.

    Example:
        >>> from getiprompt.components.feature_extractors import ROICropper
        >>> cropper = ROICropper(input_size=512, patch_size=16)
        >>>
        >>> # Large image (6440x12880) with small object mask
        >>> image = load_image()  # shape: (3, 6440, 12880)
        >>> mask = load_mask()    # shape: (6440, 12880), small object
        >>>
        >>> if cropper.should_crop(mask):
        ...     crop_result = cropper.crop_image_and_mask(image, mask)
        ...     # Encode the CROPPED image
        ...     embeddings = encoder([crop_result.cropped_image])
        ...     # Use cropped mask with MaskedFeatureExtractor
        ...     features = extractor(embeddings, crop_result.cropped_mask.unsqueeze(0), ...)
        ... else:
        ...     # Original workflow
        ...     embeddings = encoder([image])
        ...     features = extractor(embeddings, mask.unsqueeze(0), ...)
    """

    def __init__(
        self,
        input_size: int,
        patch_size: int,
        min_object_ratio: float = 0.1,
        min_patches: int = 2,
    ) -> None:
        """Initialize the ROI cropper."""
        self.input_size = input_size
        self.patch_size = patch_size
        self.min_object_ratio = min_object_ratio
        self.min_patches = min_patches

    def should_crop(
        self,
        mask: torch.Tensor,
        image_height: int | None = None,
        image_width: int | None = None,
    ) -> bool:
        """Determine if an image/mask pair needs ROI cropping.

        A mask needs cropping if the object is small relative to the image
        and would be lost during downsampling.

        Args:
            mask: Binary mask tensor of shape (H, W) or (1, H, W).
            image_height: Height of the source image. Defaults to mask height.
            image_width: Width of the source image. Defaults to mask width.

        Returns:
            True if the image/mask would benefit from ROI cropping.
        """
        if mask.ndim == 3:
            mask_2d = mask.squeeze(0) if mask.shape[0] == 1 else mask.sum(dim=0).bool()
        else:
            mask_2d = mask

        h, w = mask_2d.shape
        image_height = image_height or h
        image_width = image_width or w

        x_min, y_min, x_max, y_max = get_mask_bounding_box(mask_2d)
        bbox_width = x_max - x_min
        bbox_height = y_max - y_min

        if bbox_width == 0 or bbox_height == 0:
            return False

        # Calculate object size in feature space WITHOUT cropping
        feature_size = self.input_size // self.patch_size

        # After resize to input_size, object becomes:
        # obj_w_resized = bbox_width * (input_size / image_width)
        # After pooling, in feature space:
        # obj_feat_w = obj_w_resized / patch_size = bbox_width * feature_size / image_width
        obj_feat_w = (bbox_width / image_width) * feature_size
        obj_feat_h = (bbox_height / image_height) * feature_size

        # If object would be smaller than min_patches, we should crop
        return min(obj_feat_w, obj_feat_h) < self.min_patches

    def should_crop_any(
        self,
        masks: list[torch.Tensor],
        image_height: int | None = None,
        image_width: int | None = None,
    ) -> bool:
        """Determine if any mask in a list needs ROI cropping.

        Args:
            masks: List of binary mask tensors, each of shape (H, W) or (1, H, W).
            image_height: Height of the source image. Defaults to first mask height.
            image_width: Width of the source image. Defaults to first mask width.

        Returns:
            True if any mask would benefit from ROI cropping.
        """
        for mask in masks:
            if self.should_crop(mask, image_height, image_width):
                return True
        return False

    def compute_crop_region(
        self,
        mask: torch.Tensor,
        image_height: int | None = None,
        image_width: int | None = None,
    ) -> CropRegion:
        """Compute the optimal crop region for a given mask.

        Args:
            mask: Binary mask tensor of shape (H, W) or (1, H, W).
            image_height: Height of the source image. Defaults to mask height.
            image_width: Width of the source image. Defaults to mask width.

        Returns:
            CropRegion with the crop boundaries.
        """
        if mask.ndim == 3:
            mask_2d = mask.squeeze(0) if mask.shape[0] == 1 else mask.sum(dim=0).bool()
        else:
            mask_2d = mask

        h, w = mask_2d.shape
        image_height = image_height or h
        image_width = image_width or w

        # Get bounding box of the mask
        x_min, y_min, x_max, y_max = get_mask_bounding_box(mask_2d)
        bbox_width = x_max - x_min
        bbox_height = y_max - y_min

        # Handle empty mask case
        if bbox_width == 0 or bbox_height == 0:
            return CropRegion(x_min=0, y_min=0, x_max=image_width, y_max=image_height)

        # Compute optimal crop size
        crop_size = compute_roi_crop_size(
            bbox_width=bbox_width,
            bbox_height=bbox_height,
            input_size=self.input_size,
            patch_size=self.patch_size,
            min_object_ratio=self.min_object_ratio,
            min_patches=self.min_patches,
        )

        # Clamp crop size to image dimensions
        crop_size = min(crop_size, image_height, image_width)

        # Compute centered crop region
        return compute_centered_crop_region(
            x_min=x_min,
            y_min=y_min,
            x_max=x_max,
            y_max=y_max,
            crop_size=crop_size,
            image_height=image_height,
            image_width=image_width,
        )

    def crop_image_and_mask(
        self,
        image: torch.Tensor,
        mask: torch.Tensor | list[torch.Tensor],
    ) -> CropResult:
        """Crop both image and mask(s) around the ROI.

        This is the main method to use. It crops both the reference image
        and mask(s) to a region centered on the object, sized appropriately
        to preserve the object through downsampling.

        When multiple masks are provided, the crop region is computed based on
        the smallest mask (the one that would need the most aggressive cropping)
        to ensure all objects are preserved. All masks are then cropped to the
        same region to maintain spatial alignment.

        Args:
            image: Image tensor of shape (C, H, W).
            mask: Binary mask tensor of shape (H, W) or (1, H, W), or a list
                of such tensors. When a list is provided, cropping is based
                on the smallest mask.

        Returns:
            CropResult containing cropped image, cropped mask(s), and crop region.
            When multiple masks are provided, cropped_mask will be a stacked
            tensor of shape (N, H, W) where N is the number of masks.

        Example:
            >>> cropper = ROICropper(input_size=512, patch_size=16)
            >>> # Single mask
            >>> crop_result = cropper.crop_image_and_mask(image, mask)
            >>> # Multiple masks - crops based on smallest
            >>> crop_result = cropper.crop_image_and_mask(image, [mask1, mask2, mask3])
        """
        _, image_height, image_width = image.shape

        # Handle single mask or list of masks
        if isinstance(mask, list):
            masks = mask
        else:
            masks = [mask]

        # Find the smallest mask (needs most aggressive cropping)
        # Smallest = smallest bounding box area
        smallest_mask = None
        smallest_area = float("inf")

        for m in masks:
            if m.ndim == 3:
                m_2d = m.squeeze(0) if m.shape[0] == 1 else m.sum(dim=0).bool()
            else:
                m_2d = m

            x_min, y_min, x_max, y_max = get_mask_bounding_box(m_2d)
            bbox_width = x_max - x_min
            bbox_height = y_max - y_min
            area = bbox_width * bbox_height

            if area > 0 and area < smallest_area:
                smallest_area = area
                smallest_mask = m

        # If no valid mask found, use first mask
        if smallest_mask is None:
            smallest_mask = masks[0]

        # Compute crop region based on smallest mask
        crop_region = self.compute_crop_region(
            mask=smallest_mask,
            image_height=image_height,
            image_width=image_width,
        )

        # Crop image (C, H, W) -> (C, crop_H, crop_W)
        cropped_image = image[
            :,
            crop_region.y_min : crop_region.y_max,
            crop_region.x_min : crop_region.x_max,
        ]

        # Crop all masks
        cropped_masks = []
        for m in masks:
            if m.ndim == 3:
                cropped_m = m[
                    :,
                    crop_region.y_min : crop_region.y_max,
                    crop_region.x_min : crop_region.x_max,
                ]
                # Squeeze to 2D for stacking
                cropped_m = cropped_m.squeeze(0)
            else:
                cropped_m = m[
                    crop_region.y_min : crop_region.y_max,
                    crop_region.x_min : crop_region.x_max,
                ]
            cropped_masks.append(cropped_m)

        cropped_mask = torch.stack(cropped_masks, dim=0)

        return CropResult(
            cropped_image=cropped_image,
            cropped_mask=cropped_mask,
            crop_region=crop_region,
        )
