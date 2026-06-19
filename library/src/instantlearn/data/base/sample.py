# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Sample and TensorSample dataclasses for instantlearn."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from instantlearn.data.utils.image import read_image

if TYPE_CHECKING:
    import torch


@dataclass
class Sample:
    """A single image with its annotations, used as input to all models.

    Supports both single-instance (N=1) and multi-instance (N>1) scenarios.
    All array fields are numpy — models that run on PyTorch call
    ``to_tensors()`` internally.

    If ``image`` is ``None`` but ``image_path`` is provided, the image is
    loaded automatically on construction. If ``category_ids`` is ``None``,
    it is auto-generated as ``[0, 1, ..., len(categories) - 1]``.

    Attributes:
        image: Input image as an ``(H, W, C)`` uint8 or float32 array.
        image_path: Path to the source image file.
        masks: Instance masks as an ``(N, H, W)`` bool or uint8 array.
        bboxes: Bounding boxes as an ``(N, 4)`` float32 array in xyxy format.
        points: Prompt points as an ``(N, K, 2)`` float32 array.
        scores: Per-instance confidence scores as an ``(N,)`` float32 array.
        categories: List of N category name strings.
        category_ids: List of N integer category IDs.

    Example:
        >>> sample = Sample(image=image, masks=masks, categories=["cat"])
        >>> sample = Sample(image_path="image.jpg")
        >>> sample = Sample(
        ...     image_path="image.jpg",
        ...     masks=masks,
        ...     categories=["cat", "dog"],
        ...     category_ids=[0, 1],
        ... )
    """

    image: np.ndarray | None = None
    image_path: str | None = None

    masks: np.ndarray | None = None
    bboxes: np.ndarray | None = None
    points: np.ndarray | None = None
    scores: np.ndarray | None = None

    categories: list[str] = field(default_factory=lambda: ["object"])
    category_ids: list[int] | None = None

    def __post_init__(self) -> None:
        """Auto-load image from path and generate category_ids if needed."""
        if self.image is None and self.image_path is not None:
            self.image = read_image(self.image_path, as_tensor=False)

        if self.category_ids is None:
            self.category_ids = list(range(len(self.categories)))

    def filter_by_category(self, category_name: str) -> Sample | None:
        """Return a new Sample containing only instances matching ``category_name``.

        Filters ``categories``, ``category_ids``, ``masks``, ``bboxes``,
        ``points``, and ``scores``. ``image`` and ``image_path`` are shared
        (not copied).

        Args:
            category_name: The category name to keep.

        Returns:
            A new ``Sample`` with only matching instances, or ``None`` if no
            instances match.

        Example:
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

        Lazily imports torch — only usable in environments where torch is
        installed. ``image`` is permuted from HWC to CHW and cast to float32.

        Args:
            device: Target device string, e.g. ``"cpu"`` or ``"cuda"``.

        Returns:
            A ``TensorSample`` with all non-``None`` fields converted to
            tensors on *device*.
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
    """Torch-native counterpart of ``Sample``.

    Produced by ``Sample.to_tensors()`` and consumed internally by
    ``TorchModel`` subclasses. All array fields are tensors; ``categories``
    stays as a plain list.

    Attributes:
        image: Image tensor of shape ``(C, H, W)`` float32.
        masks: Instance masks of shape ``(N, H, W)``.
        bboxes: Bounding boxes of shape ``(N, 4)`` float32 in xyxy format.
        points: Prompt points of shape ``(N, K, 2)`` float32.
        scores: Per-instance scores of shape ``(N,)`` float32.
        categories: List of category name strings.
        category_ids: Category IDs of shape ``(N,)`` int32.
    """

    image: torch.Tensor | None = None
    masks: torch.Tensor | None = None
    bboxes: torch.Tensor | None = None
    points: torch.Tensor | None = None
    scores: torch.Tensor | None = None
    categories: list[str] | None = None
    category_ids: torch.Tensor | None = None
