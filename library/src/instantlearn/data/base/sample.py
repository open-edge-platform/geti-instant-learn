# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Sample and Category dataclasses for instantlearn.

``Sample`` is backend-neutral: every array field is numpy and the module
imports zero torch. Torch models convert a ``Sample`` to a ``TensorSample``
through :func:`instantlearn.models.torch_adapter.sample_to_tensors`.
"""

from __future__ import annotations

from dataclasses import InitVar, dataclass, field
from typing import TYPE_CHECKING

from instantlearn.data.utils.image import read_image

if TYPE_CHECKING:
    import numpy as np


@dataclass(frozen=True)
class Category:
    """A category that keeps its integer id and string label together.

    Pairing id and label in one object removes the fragile "match two parallel
    lists by index" pattern used previously (``categories`` + ``category_ids``).
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
    loaded automatically on construction. ``categories`` accepts either
    :class:`Category` objects or plain label strings (ids auto-assigned by
    position). The legacy ``category_ids`` argument is still accepted and is
    zipped with the labels into :class:`Category` objects.

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

    categories: list[Category] = field(default_factory=lambda: [Category(id=0, label="object")])

    is_reference: list[bool] | None = field(default_factory=lambda: [False])
    n_shot: list[int] | None = field(default_factory=lambda: [-1])
    mask_paths: list[str] | None = None

    #: Legacy constructor-only alias. When given, ids are zipped with the
    #: labels in ``categories`` to build :class:`Category` objects.
    category_ids: InitVar[list[int] | np.ndarray | None] = None

    def __setattr__(self, name: str, value: object) -> None:
        """Normalise ``categories`` (strings -> :class:`Category`) on every set."""
        if name == "categories" and value is not None:
            value = self._normalize_categories(value)
        super().__setattr__(name, value)

    def __post_init__(self, category_ids: list[int] | np.ndarray | None) -> None:
        """Auto-load image from path and merge legacy ``category_ids``."""
        if self.image is None and self.image_path is not None:
            self.image = read_image(self.image_path, as_tensor=False)

        if category_ids is not None:
            ids = category_ids.tolist() if hasattr(category_ids, "tolist") else list(category_ids)
            # Apply ids positionally; labels without a matching id keep their
            # auto-assigned id (no length validation — same as the legacy API).
            self.categories = [
                Category(id=int(ids[i]) if i < len(ids) else cat.id, label=cat.label)
                for i, cat in enumerate(self.categories)
            ]

    @staticmethod
    def _normalize_categories(categories: list) -> list[Category]:
        """Coerce a list of ``Category`` or label strings into ``Category`` objects."""
        return [c if isinstance(c, Category) else Category(id=i, label=str(c)) for i, c in enumerate(categories)]

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
            ...     categories=["cat", "dog", "cat"],
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
