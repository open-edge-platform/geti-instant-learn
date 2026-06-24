# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Sample and Category dataclasses for instantlearn.

``Sample`` is backend-neutral: every array field is numpy and the module
imports zero torch. Torch models convert a ``Sample`` to a ``TensorSample``
through :func:`instantlearn.models.torch_adapter.sample_to_tensors`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from instantlearn.data.utils.image import read_image

if TYPE_CHECKING:
    import numpy as np


@dataclass(frozen=True)
class Category:
    """An instance category: an integer id paired with a string label.

    Frozen so it behaves as an immutable, hashable value object.

    Attributes:
        id: Integer category id.
        label: Human-readable category name.
    """

    id: int
    label: str


@dataclass
class Sample:
    """A single image with its annotations, used as input to all models.

    Supports both single-instance (N=1) and multi-instance (N>1) scenarios.
    All array fields are numpy — torch models convert internally via
    :func:`instantlearn.models.torch_adapter.sample_to_tensors`.

    If ``image`` is ``None`` but ``image_path`` is provided, the image is
    loaded automatically on construction. ``categories`` must be a list of
    :class:`Category` objects (id + label).

    Attributes:
        image: Input image as an ``(H, W, C)`` uint8 or float32 array.
        image_path: Path to the source image file.
        masks: Instance masks as an ``(N, H, W)`` bool or uint8 array.
        bboxes: Bounding boxes as an ``(N, 4)`` float32 array in xyxy format.
        points: Prompt points as an ``(N, K, 2)`` float32 array.
        scores: Per-instance confidence scores as an ``(N,)`` float32 array.
        categories: List of :class:`Category` (id + label) objects.
        is_reference: Per-instance flags marking reference (support) instances.
        n_shot: Per-instance shot index for n-shot references.
        mask_paths: Optional per-instance source mask file paths.

    Example:
        >>> sample = Sample(image=image, masks=masks, categories=[Category(0, "cat")])
        >>> sample = Sample(image_path="image.jpg")
        >>> sample = Sample(
        ...     image_path="image.jpg",
        ...     masks=masks,
        ...     categories=[Category(0, "cat"), Category(1, "dog")],
        ... )
    """

    image: np.ndarray | None = None
    image_path: str | None = None

    masks: np.ndarray | None = None
    bboxes: np.ndarray | None = None
    points: np.ndarray | None = None
    scores: np.ndarray | None = None

    categories: list[Category] = field(default_factory=lambda: [Category(id=0, label="object")])

    is_reference: list[bool] | None = field(default_factory=lambda: [False])
    n_shot: list[int] | None = field(default_factory=lambda: [-1])
    mask_paths: list[str] | None = None

    def __post_init__(self) -> None:
        """Auto-load the image from ``image_path`` when ``image`` is unset."""
        if self.image is None and self.image_path is not None:
            self.image = read_image(self.image_path, as_tensor=False)

    @property
    def category_labels(self) -> list[str]:
        """Category label strings, one per instance."""
        return [c.label for c in self.categories]

    @property
    def label_ids(self) -> list[int]:
        """Integer category ids, one per instance."""
        return [c.id for c in self.categories]

    def filter_by_category(self, category_name: str) -> Sample | None:
        """Return a new Sample containing only instances matching ``category_name``.

        Filters ``categories``, ``masks``, ``bboxes``, ``points``, ``scores``,
        ``is_reference``, ``n_shot``, and ``mask_paths``. ``image`` and
        ``image_path`` are shared (not copied).

        Args:
            category_name: The category label to keep.

        Returns:
            A new ``Sample`` with only matching instances, or ``None`` if no
            instances match.

        Example:
            >>> sample = Sample(
            ...     image=img,
            ...     categories=[Category(0, "cat"), Category(1, "dog"), Category(0, "cat")],
            ...     masks=masks_3hw,
            ... )
            >>> filtered = sample.filter_by_category("cat")
            >>> len(filtered.categories)
            2
        """
        indices = [i for i, cat in enumerate(self.categories) if cat.label == category_name]
        if not indices:
            return None

        def _select(arr: np.ndarray | None) -> np.ndarray | None:
            return arr[indices] if arr is not None else None

        def _select_list(values: list | None) -> list | None:
            return [values[i] for i in indices] if values is not None else None

        return Sample(
            image=self.image,
            image_path=self.image_path,
            categories=[self.categories[i] for i in indices],
            is_reference=_select_list(self.is_reference),
            n_shot=_select_list(self.n_shot),
            mask_paths=_select_list(self.mask_paths),
            masks=_select(self.masks),
            bboxes=_select(self.bboxes),
            points=_select(self.points),
            scores=_select(self.scores),
        )
