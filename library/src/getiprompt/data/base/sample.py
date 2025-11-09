# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Sample classes for GetiPrompt datasets using simple dataclasses.

This module defines the sample structure for few-shot segmentation tasks
using Python's built-in @dataclass for simplicity.
"""

from dataclasses import dataclass, field

import numpy as np
import torch
from torchvision import tv_tensors


@dataclass
class Sample:
    """Sample class for GetiPrompt few-shot segmentation datasets.

    Supports both single-instance (N=1, PerSeg) and multi-instance (N>1, LVIS/COCO) scenarios.
    One sample = one image with N instances.

    Attributes:
        image (np.ndarray | torch.Tensor): Input image with shape:
            - numpy: (H, W, C) - Channel-last format for model preprocessors
            - torch: (C, H, W) - Channel-first format
            Required.
        image_path (str): Path to the source image file. Required.
        masks (np.ndarray | torch.Tensor | None): N masks with shape (N, H, W) - all same HxW. Defaults to None.
        bboxes (np.ndarray | torch.Tensor | None): Bounding boxes with shape (N, 4). Defaults to None.
        points (np.ndarray | torch.Tensor | None): Point coordinates with shape (N, 2). Defaults to None.
        categories (list[str] | None): List of N category names. Defaults to None.
        category_ids (np.ndarray | torch.Tensor | None): Array of N category IDs with shape (N,). Defaults to None.
        mask_paths (list[str] | None): List of N paths to mask files. Defaults to None.
        is_reference (list[bool]): Reference flag(s) for each instance. Defaults to [False].
        n_shot (list[int]): Shot number(s) for each instance. Defaults to [-1].

    Note:
        Images are stored in HWC format (numpy) for compatibility with model preprocessors
        (HuggingFace, SAM transforms). Future refactoring may move preprocessing to dataset.

    Note:
        - For single-instance (PerSeg): N=1
        - For multi-instance (LVIS): N>1
        - All masks (if provided) must have the same HxW (typically the image size)
        - At least one of masks, bboxes, or points should be provided for meaningful segmentation tasks
        - If masks not provided, you can use bboxes or points to generate masks later (e.g., with SAM)

    Examples:
        Single instance (PerSeg):
        >>> import numpy as np
        >>> from getiprompt.data.sample import Sample

        >>> sample = Sample(
        ...     image=np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8),  # HWC format
        ...     image_path="path/to/image.jpg",
        ...     masks=np.random.randint(0, 2, (1, 224, 224), dtype=np.uint8),
        ...     bboxes=np.array([[10, 20, 100, 120]], dtype=np.float32),
        ...     points=np.array([[50, 60]], dtype=np.float32),
        ...     categories=["cat"],
        ...     category_ids=np.array([0], dtype=np.int32),
        ...     is_reference=[True],
        ...     n_shot=[0],
        ...     mask_paths=["path/to/mask.png"]
        ... )

        >>> sample.image.shape
        (224, 224, 3)  # HWC format
        >>> sample.masks.shape
        (1, 224, 224)
        >>> sample.is_reference
        [True]
        >>> sample.n_shot
        [0]

        Only bboxes (no masks - generate masks later with SAM):
        >>> sample = Sample(
        ...     image=np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8),  # HWC format
        ...     image_path="path/to/image.jpg",
        ...     bboxes=np.array([[10, 20, 110, 120], [200, 150, 350, 270]], dtype=np.float32),
        ...     categories=["cat", "dog"],
        ...     category_ids=np.array([0, 1], dtype=np.int32),
        ...     is_reference=[True, True],
        ...     n_shot=[0, 0]
        ... )

        >>> sample.masks is None
        True
        >>> sample.bboxes.shape
        (2, 4)
        >>> sample.is_reference
        [True, True]

        Only points (no masks or bboxes):
        >>> sample = Sample(
        ...     image=np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8),  # HWC format
        ...     image_path="path/to/target.jpg",
        ...     points=np.array([[100, 150], [300, 400]], dtype=np.float32),
        ...     categories=["person", "person"],
        ...     category_ids=np.array([2, 2], dtype=np.int32),
        ...     is_reference=[False, False],
        ...     n_shot=[-1, -1]
        ... )

        >>> sample.points.shape
        (2, 2)
        >>> sample.masks is None
        True
        >>> sample.bboxes is None
        True
        >>> sample.is_reference
        [False, False]
        >>> sample.n_shot
        [-1, -1]
    """

    # Required fields
    image: np.ndarray | tv_tensors.Image | None = None
    image_path: str | None = None

    # Optional annotation fields (defaults to None)
    masks: np.ndarray | torch.Tensor | None = None
    bboxes: np.ndarray | torch.Tensor | None = None
    points: np.ndarray | torch.Tensor | None = None
    scores: np.ndarray | torch.Tensor | None = None

    # Optional metadata fields (defaults to None)
    categories: list[str] | None = None
    category_ids: np.ndarray | torch.Tensor | None = None
    mask_paths: list[str] | None = None

    # Optional task-specific fields (with sensible defaults)
    # Always lists to maintain consistency between single and multi-instance
    is_reference: list[bool] = field(default_factory=lambda: [False])
    n_shot: list[int] = field(default_factory=lambda: [-1])
