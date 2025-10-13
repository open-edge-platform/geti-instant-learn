# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0


from copy import deepcopy

import numpy as np
import torch
from torch.nn import functional
from torchvision.transforms.functional import resize, to_pil_image

# TODO(Eugene): refactor ResizeLongestSide only keeping torch.Tensor implemenataion.
# https://github.com/open-edge-platform/geti-prompt/issues/174


class ResizeLongestSide:
    """Resizes images to the longest side 'target_length'.

    Also provides methods for resizing coordinates and boxes.
    Also provides methods for transforming both numpy array and batched torch tensors.
    """

    def __init__(self, target_length: int) -> None:
        self.target_length = target_length

    def apply_image(self, image: np.ndarray) -> np.ndarray:
        """Expects a numpy array with shape HxWxC in uint8 format."""
        target_size = self.get_preprocess_shape(image.shape[0], image.shape[1], self.target_length)
        return np.array(resize(to_pil_image(image), target_size))

    def apply_coords(
        self,
        coords: np.ndarray,
        original_size: tuple[int, ...],
    ) -> np.ndarray:
        """Resizes coordinates to the longest side 'target_length'.

        Args:
            coords: The coordinates to resize.
            original_size: The original size of the image.

        Returns:
            The resized coordinates.
        """
        old_h, old_w = original_size
        new_h, new_w = self.get_preprocess_shape(
            original_size[0],
            original_size[1],
            self.target_length,
        )
        coords = deepcopy(coords).astype(float)
        coords[..., 0] = coords[..., 0] * (new_w / old_w)
        coords[..., 1] = coords[..., 1] * (new_h / old_h)
        return coords

    def apply_boxes(
        self,
        boxes: np.ndarray,
        original_size: tuple[int, ...],
    ) -> np.ndarray:
        """Resizes boxes to the longest side 'target_length'.

        Args:
            boxes: The boxes to resize.
            original_size: The original size of the image.

        Returns:
            The resized boxes.
        """
        boxes = self.apply_coords(boxes.reshape(-1, 2, 2), original_size)
        return boxes.reshape(-1, 4)

    def apply_image_torch(
        self,
        image: torch.Tensor,
    ) -> torch.Tensor:
        """Expects batched images with shape BxCxHxW and float format.

        This transformation may not exactly match apply_image. apply_image is
        the transformation expected by the model.

        Args:
            image: The image to resize.

        Returns:
            Resized image.
        """
        # Expects an image in BCHW format. May not exactly match apply_image.
        target_size = self.get_preprocess_shape(image.shape[2], image.shape[3], self.target_length)
        return functional.interpolate(
            image,
            target_size,
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )

    def apply_coords_torch(
        self,
        coords: torch.Tensor,
        original_size: tuple[int, ...],
    ) -> torch.Tensor:
        """Resizes coordinates to the longest side 'target_length'.

        Args:
            coords: The coordinates to resize.
            original_size: The original size of the image.

        Returns:
            The resized coordinates.
        """
        old_h, old_w = original_size
        new_h, new_w = self.get_preprocess_shape(
            original_size[0],
            original_size[1],
            self.target_length,
        )
        coords = deepcopy(coords).to(torch.float)
        coords[..., 0] *= new_w / old_w
        coords[..., 1] *= new_h / old_h
        return coords

    def apply_boxes_torch(
        self,
        boxes: torch.Tensor,
        original_size: tuple[int, ...],
    ) -> torch.Tensor:
        """Resizes boxes to the longest side 'target_length'.

        Args:
            boxes: The boxes to resize.
            original_size: The original size of the image.

        Returns:
            The resized boxes.
        """
        boxes = self.apply_coords_torch(boxes.reshape(-1, 2, 2), original_size)
        return boxes.reshape(-1, 4)

    def apply_inverse_coords_torch(
        self,
        coords: torch.Tensor,
        original_size: tuple[int, ...],
    ) -> torch.Tensor:
        """Inverts the coordinate transformation back to the original image size."""
        old_h, old_w = original_size
        new_h, new_w = self.get_preprocess_shape(
            original_size[0],
            original_size[1],
            self.target_length,
        )
        coords = torch.clone(coords).to(torch.float)
        coords[..., 0] = coords[..., 0] * (old_w / new_w)
        coords[..., 1] = coords[..., 1] * (old_h / new_h)
        return coords

    @staticmethod
    def get_preprocess_shape(
        oldh: int,
        oldw: int,
        long_side_length: int,
    ) -> tuple[int, int]:
        """Compute the output size given input size and target long side length."""
        scale = long_side_length * 1.0 / max(oldh, oldw)
        newh, neww = oldh * scale, oldw * scale
        neww = int(neww + 0.5)
        newh = int(newh + 0.5)
        return (newh, neww)
