# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Sample classes for InstantLearn datasets using simple dataclasses.

This module defines the sample structure for few-shot segmentation tasks
using Python's built-in @dataclass for simplicity.
"""

from dataclasses import dataclass, field

import numpy as np
import torch
from torchvision import tv_tensors

from instantlearn.data.utils.image import read_image, read_mask


@dataclass
class Sample:
    """Sample class for InstantLearn few-shot segmentation datasets.

    Supports both single-instance (N=1, PerSeg) and multi-instance (N>1, LVIS/COCO) scenarios.
    One sample = one image with N instances.

    Attributes:
        image: Input image. numpy (H, W, C) or torch (C, H, W).
        image_path: Path to the source image file. Auto-loads if image not provided.
        masks: N masks with shape (N, H, W). Auto-loads from mask_paths if not provided.
        bboxes: Bounding boxes with shape (N, 4).
        points: Point coordinates with shape (N, 2).
        categories: List of N category names. Defaults to ["object"].
        category_ids: List of N category IDs. Auto-generated from categories if not provided.
        mask_paths: Path(s) to mask files. Accepts single string or list of strings.
        is_reference: Reference flag(s) for each instance. Defaults to [False].
        n_shot: Shot number(s) for each instance. Defaults to [-1].

    Note:
        If `image` is None but `image_path` is provided, the image is auto-loaded.
        If `masks` is None but `mask_paths` is provided, masks are auto-loaded.
        If `category_ids` is None, it is auto-generated as [0, 1, ..., len(categories)-1].

    Examples:
        Visual-only models (PerDINO, Matcher) - minimal usage:

        >>> sample = Sample(image=image, masks=mask)

        With path-based loading:

        >>> sample = Sample(
        ...     image_path="path/to/image.jpg",
        ...     mask_paths="path/to/mask.png",
        ... )

        Multiple masks with categories:

        >>> sample = Sample(
        ...     image_path="path/to/image.jpg",
        ...     mask_paths=["mask1.png", "mask2.png"],
        ...     categories=["cat", "dog"],
        ...     category_ids=[0, 1],
        ... )
    """

    # Required fields
    image: np.ndarray | tv_tensors.Image | None = None
    image_path: str | None = None

    # Optional annotation fields (defaults to None)
    masks: np.ndarray | torch.Tensor | None = None
    bboxes: np.ndarray | torch.Tensor | None = None
    points: np.ndarray | torch.Tensor | None = None
    scores: np.ndarray | torch.Tensor | None = None

    # Metadata fields
    categories: list[str] = field(default_factory=lambda: ["object"])
    category_ids: list[int] | np.ndarray | torch.Tensor | None = None
    mask_paths: str | list[str] | None = None

    # Optional task-specific fields (with sensible defaults)
    # Always lists to maintain consistency between single and multi-instance
    is_reference: list[bool] = field(default_factory=lambda: [False])
    n_shot: list[int] = field(default_factory=lambda: [-1])

    def __post_init__(self) -> None:
        """Auto-load images/masks from paths and generate category_ids if needed."""
        # Normalize mask_paths to list
        if isinstance(self.mask_paths, str):
            self.mask_paths = [self.mask_paths]

        if self.image is None and self.image_path is not None:
            self.image = read_image(self.image_path, as_tensor=True)  # CHW tensor

        if self.masks is None and self.mask_paths is not None:
            masks = [read_mask(p, as_tensor=True) for p in self.mask_paths]
            self.masks = torch.stack(masks, dim=0)  # (N, H, W) tensor

        # Auto-generate category_ids from categories if not provided
        if self.category_ids is None:
            self.category_ids = list(range(len(self.categories)))
