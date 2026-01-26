# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# Original code Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under Apache License 2.0.

"""SAM2 transforms for EfficientSAM3 preprocessing and postprocessing."""

import warnings
from typing import Any

import torch
import torch.nn.functional
from torch import nn
from torchvision.transforms import Normalize, Resize, ToTensor


# Adapted from https://github.com/facebookresearch/sam2/blob/main/sam2/utils/transforms.py
class SAM2Transforms(nn.Module):
    """Transforms for SAM2-style image processing."""

    def __init__(
        self,
        resolution: int,
        mask_threshold: float,
        max_hole_area: float = 0.0,
        max_sprinkle_area: float = 0.0,
    ) -> None:
        """Initialize transforms.

        Args:
            resolution: Target resolution for image resize
            mask_threshold: Threshold for mask binarization
            max_hole_area: Maximum area for hole filling
            max_sprinkle_area: Maximum area for sprinkle removal
        """
        super().__init__()
        self.resolution = resolution
        self.mask_threshold = mask_threshold
        self.max_hole_area = max_hole_area
        self.max_sprinkle_area = max_sprinkle_area
        self.mean = [0.5, 0.5, 0.5]
        self.std = [0.5, 0.5, 0.5]
        self.to_tensor = ToTensor()
        self.transforms = torch.jit.script(
            nn.Sequential(
                Resize((self.resolution, self.resolution)),
                Normalize(self.mean, self.std),
            ),
        )
        self.device = None  # Will be set when used with a model

    def set_device(self, device: torch.device) -> None:
        """Set device for transforms."""
        self.device = device

    def __call__(self, x: torch.Tensor | Any) -> torch.Tensor:  # noqa: ANN401
        """Transform a single image."""
        x = self.to_tensor(x)
        x = self.transforms(x)
        if self.device is not None:
            x = x.to(self.device)
        return x

    def forward_batch(self, img_list: list) -> torch.Tensor:
        """Transform a batch of images."""
        img_batch = [self.transforms(self.to_tensor(img)) for img in img_list]
        img_batch = torch.stack(img_batch, dim=0)
        if self.device is not None:
            img_batch = img_batch.to(self.device)
        return img_batch

    def transform_coords(
        self,
        coords: torch.Tensor,
        normalize: bool = False,
        orig_hw: tuple[int, int] | None = None,
    ) -> torch.Tensor:
        """Transform coordinates to model input space.

        Args:
            coords: Coordinates tensor with length 2 in last dimension
            normalize: Whether coords are in absolute image coordinates
            orig_hw: Original image (height, width) if normalize=True

        Returns:
            Coordinates in range [0, resolution] expected by model
        """
        if normalize:
            assert orig_hw is not None
            h, w = orig_hw
            coords = coords.clone()
            coords[..., 0] /= w
            coords[..., 1] /= h

        return coords * self.resolution  # unnormalize coords

    def transform_boxes(
        self,
        boxes: torch.Tensor,
        normalize: bool = False,
        orig_hw: tuple[int, int] | None = None,
    ) -> torch.Tensor:
        """Transform boxes to model input space.

        Args:
            boxes: Boxes tensor of shape Bx4
            normalize: Whether boxes are in absolute image coordinates
            orig_hw: Original image (height, width) if normalize=True

        Returns:
            Transformed boxes
        """
        return self.transform_coords(boxes.reshape(-1, 2, 2), normalize, orig_hw)

    def postprocess_masks(self, masks: torch.Tensor, orig_hw: tuple[int, int]) -> torch.Tensor:
        """Post-process output masks.

        Args:
            masks: Predicted masks
            orig_hw: Original image (height, width)

        Returns:
            Post-processed masks at original resolution
        """
        masks = masks.float()
        input_masks = masks
        mask_flat = masks.flatten(0, 1).unsqueeze(1)  # flatten as 1-channel image

        try:
            from getiprompt.models.foundation.sam3.perflib.connected_components import (  # noqa: PLC0415
                connected_components,
            )

            if self.max_hole_area > 0:
                # Holes are those connected components in background with area <= self.fill_hole_area
                labels, areas = connected_components(
                    (mask_flat <= self.mask_threshold).to(torch.uint8),
                )
                is_hole = (labels > 0) & (areas <= self.max_hole_area)
                is_hole = is_hole.reshape_as(masks)
                masks = torch.where(is_hole, self.mask_threshold + 10.0, masks)

            if self.max_sprinkle_area > 0:
                labels, areas = connected_components(
                    (mask_flat > self.mask_threshold).to(torch.uint8),
                )
                is_hole = (labels > 0) & (areas <= self.max_sprinkle_area)
                is_hole = is_hole.reshape_as(masks)
                masks = torch.where(is_hole, self.mask_threshold - 10.0, masks)
        except Exception as e:  # noqa: BLE001
            # Skip post-processing if CUDA kernel fails
            warnings.warn(
                f"{e}\n\nSkipping post-processing step. SAM3 will still work, "
                "but some post-processing functionality may be limited.",
                category=UserWarning,
                stacklevel=2,
            )
            masks = input_masks

        return torch.nn.functional.interpolate(masks, orig_hw, mode="bilinear", align_corners=False)
