# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Sample and TensorSample dataclasses for InstantLearn.

This module defines the sample structure for few-shot segmentation tasks.
"""

from __future__ import annotations  # so the class definition itself does not require torch at import time

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from instantlearn.data.utils.image import read_image, read_mask

if TYPE_CHECKING:
    import torch


@dataclass
class Sample:
    """Sample class for InstantLearn few-shot segmentation datasets.

    Supports both single-instance (N=1, PerSeg) and multi-instance (N>1, LVIS/COCO) scenarios.
    One sample = one image with N instances.

    Attributes:
        image: Input image. HWC uint8/float32 numpy array or ``None``.
        image_path: Path to the source image file. Auto-loads if image not provided.
        masks: ``(N, H, W)`` bool/uint8 numpy array or ``None``.
        bboxes: ``(N, 4)`` float32 xyxy numpy array or ``None``.
        points: ``(N, K, 2)`` float32 numpy array or ``None``.
        scores: ``(N,)`` float32 numpy array or ``None``.
        categories: List of N category name strings. Defaults to
            ``["object"]``.
        category_ids: List of N integer category IDs.  Auto-generated from
            ``categories`` if not provided.
        mask_paths: Path(s) to mask files.  Auto-loaded to ``masks`` if
            ``masks`` is ``None``.

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
    image: np.ndarray | None = None
    image_path: str | None = None

    # Optional annotation fields (defaults to None)
    masks: np.ndarray | None = None
    bboxes: np.ndarray | None = None
    points: np.ndarray | None = None
    scores: np.ndarray | None = None

    # Metadata fields
    categories: list[str] = field(default_factory=lambda: ["object"])
    category_ids: list[int] | None = None
    mask_paths: str | list[str] | None = None

    def __post_init__(self) -> None:
        """Auto-load images/masks from paths and generate category_ids if needed."""
        # Normalize mask_paths to list
        if isinstance(self.mask_paths, str):
            self.mask_paths = [self.mask_paths]

        if self.image is None and self.image_path is not None:
            # Load to HWC uint8 numpy
            self.image = read_image(self.image_path, as_tensor=False)

        if self.masks is None and self.mask_paths is not None:
            # Load each mask as 2-D numpy array (H, W) then stack to (N, H, W)
            mask_arrays = [read_mask(p, as_tensor=False) for p in self.mask_paths]
            self.masks = np.stack(mask_arrays, axis=0)

        if self.category_ids is None:
            self.category_ids = list(range(len(self.categories)))

    def filter_by_category(self, category_name: str) -> Sample | None:
        """Return a new Sample containing only instances matching *category_name*.

        Filters ``categories``, ``category_ids``, ``masks``, ``bboxes``, ``points``, and ``scores``.
        ``image`` / ``image_path`` are shared (no copy).

        Args:
            category_name: The category name to keep.

        Returns:
            A new :class:`Sample` with only matching instances, or ``None`` if no instances match.

        Examples:
            >>> sample = Sample(
            ...     image=img,
            ...     categories=["cat", "dog", "cat"],
            ...     category_ids=[0, 1, 0],
            ...     masks=masks_3hw,
            ... )
            >>> filtered = sample.filter_by_category("cat")
            >>> len(filtered.categories)
            2
        """
        if self.categories is None:
            return None

        indices = [i for i, cat in enumerate(self.categories) if cat == category_name]
        if not indices:
            return None

        def _select(arr: np.ndarray | None) -> np.ndarray | None:
            return arr[indices] if arr is not None else None

        return Sample(
            image=self.image,
            image_path=self.image_path,
            categories=[self.categories[i] for i in indices],
            category_ids=[self.category_ids[i] for i in indices] if self.category_ids is not None else None,
            masks=_select(self.masks),
            bboxes=_select(self.bboxes),
            points=_select(self.points),
            scores=_select(self.scores),
        )

    def to_tensors(self, device: str = "cpu") -> TensorSample:
        """Convert numpy arrays to torch tensors.

        Lazy-imports torch — callers without torch cannot use this method.
        ``image`` is permuted from HWC to CHW and cast to float32.

        Args:
            device: Target device string, e.g. ``"cpu"`` or ``"cuda"``.

        Returns:
            A :class:`TensorSample` with all non-``None`` fields as tensors.
        """
        import torch  # noqa: PLC0415

        image_t = None
        if self.image is not None:
            arr = self.image
            if arr.ndim == 3:
                arr = arr.transpose(2, 0, 1)  # HWC -> CHW
            image_t = torch.from_numpy(np.ascontiguousarray(arr)).float().to(device)

        return TensorSample(
            image=image_t,
            masks=torch.from_numpy(self.masks).to(device) if self.masks is not None else None,
            bboxes=torch.from_numpy(self.bboxes).float().to(device) if self.bboxes is not None else None,
            points=torch.from_numpy(self.points).float().to(device) if self.points is not None else None,
            scores=torch.from_numpy(self.scores).float().to(device) if self.scores is not None else None,
            categories=self.categories,
            category_ids=torch.tensor(self.category_ids, dtype=torch.int32, device=device)
            if self.category_ids is not None
            else None,
        )


@dataclass
class TensorSample:
    """Torch-native counterpart of :class:`Sample`.

    Used internally byb:class:`~instantlearn.models.torch_base.TorchModel` subclasses.

    Attributes:
        image: ``(C, H, W)`` float32 tensor or ``None``.
        masks: ``(N, H, W)`` tensor or ``None``.
        bboxes: ``(N, 4)`` float32 tensor or ``None``.
        points: ``(N, K, 2)`` float32 tensor or ``None``.
        scores: ``(N,)`` float32 tensor or ``None``.
        categories: List of category name strings.
        category_ids: ``(N,)`` int32 tensor or ``None``.
    """

    image: torch.Tensor | None = None
    masks: torch.Tensor | None = None
    bboxes: torch.Tensor | None = None
    points: torch.Tensor | None = None
    scores: torch.Tensor | None = None
    categories: list[str] | None = None
    category_ids: torch.Tensor | None = None
