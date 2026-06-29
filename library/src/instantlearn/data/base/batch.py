# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Batch handling for instantlearn datasets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Union

from instantlearn.data.base.sample import Category, Sample

if TYPE_CHECKING:
    from collections.abc import Iterator

    import numpy as np

#: Union type for all inputs accepted by :meth:`Batch.collate`.
Collatable = Union[Sample, list[Sample], "Batch", str, Path, list[str], list[Path]]


@dataclass
class Batch:
    """A list of ``Sample`` objects with convenience batch-level accessors.

    The core data structure is a plain ``list[Sample]``. Properties give
    efficient batch-level access to images, masks, bboxes, points, and
    metadata as numpy arrays.

    Attributes:
        samples: Ordered list of samples in this batch.

    Example:
        >>> batch = Batch.collate([sample1, sample2, sample3])
        >>> len(batch)
        3
        >>> for sample in batch:
        ...     process(sample.image, sample.masks)
        >>> images = batch.images   # list[np.ndarray], HWC
        >>> masks = batch.masks     # list[np.ndarray | None], NHW
    """

    samples: list[Sample]

    def __len__(self) -> int:
        """Return the number of samples in the batch."""
        return len(self.samples)

    def __getitem__(self, index: int) -> Sample:
        """Return the sample at *index*.

        Args:
            index: Zero-based sample index.

        Returns:
            The ``Sample`` at the given index.
        """
        return self.samples[index]

    def __iter__(self) -> Iterator[Sample]:
        """Iterate over samples."""
        return iter(self.samples)

    @property
    def images(self) -> list[np.ndarray | None]:
        """All images as a list of HWC numpy arrays.

        Returns:
            One ``np.ndarray`` per sample, or ``None`` where the sample has no
            image loaded.
        """
        return [s.image for s in self.samples]

    @property
    def masks(self) -> list[np.ndarray | None]:
        """All instance masks as a list of ``(N, H, W)`` numpy arrays.

        Returns:
            One array per sample, or ``None`` where the sample has no masks.
        """
        return [s.masks for s in self.samples]

    @property
    def bboxes(self) -> list[np.ndarray | None]:
        """All bounding boxes as a list of ``(N, 4)`` xyxy numpy arrays.

        Returns:
            One array per sample, or ``None`` where the sample has no boxes.
        """
        return [s.bboxes for s in self.samples]

    @property
    def points(self) -> list[np.ndarray | None]:
        """All prompt points as a list of ``(N, K, 2)`` numpy arrays.

        Returns:
            One array per sample, or ``None`` where the sample has no points.
        """
        return [s.points for s in self.samples]

    @property
    def categories(self) -> list[list[Category]]:
        """Get all categories as a list of per-sample :class:`Category` lists.

        Preserves multi-instance structure:
        - Single-instance: [[Category(0, 'cat')], [Category(0, 'dog')], ...]
        - Multi-instance: [[Category(0, 'person'), ...], [Category(0, 'dog')], ...]

        Returns:
            list[list[Category]]: List of category lists.
        """
        return [s.categories for s in self.samples]

    @property
    def category_labels(self) -> list[list[str]]:
        """All category label strings, grouped per sample.

        Returns:
            One list of label strings per sample.
        """
        return [s.category_labels for s in self.samples]

    @property
    def label_ids(self) -> list[list[int]]:
        """All integer category ids, grouped per sample.

        Returns:
            One list of integer ids per sample.
        """
        return [s.label_ids for s in self.samples]

    @property
    def is_reference(self) -> list[list[bool] | None]:
        """Per-instance reference flags, one list per sample.

        Returns:
            One list of bool flags per sample, or ``None`` where not set.
        """
        return [s.is_reference for s in self.samples]

    @property
    def n_shot(self) -> list[list[int] | None]:
        """Per-instance shot indices, one list per sample.

        Returns:
            One list of shot indices per sample, or ``None`` where not set.
        """
        return [s.n_shot for s in self.samples]

    @property
    def mask_paths(self) -> list[list[str] | None]:
        """Source mask file paths, one list per sample.

        Returns:
            One list of mask path strings per sample, or ``None`` where not set.
        """
        return [s.mask_paths for s in self.samples]

    @property
    def image_paths(self) -> list[str | None]:
        """Source image path for each sample.

        Returns:
            A list of path strings, or ``None`` where not set.
        """
        return [s.image_path for s in self.samples]

    @classmethod
    def collate(cls, samples: Collatable) -> Batch:
        """Wrap one or more samples into a ``Batch``.

        Converts image paths to ``Sample`` objects automatically. The method
        is idempotent — passing a ``Batch`` returns it unchanged.

        Args:
            samples: Input to collate. Accepted types:
                - ``Sample``: wrapped in a single-element batch.
                - ``list[Sample]``: used directly.
                - ``Batch``: returned unchanged.
                - ``str | Path``: treated as an image path; creates one
                  ``Sample`` with ``image_path`` set.
                - ``list[str] | list[Path]``: each element treated as an
                  image path.

        Returns:
            A ``Batch`` containing the provided samples.

        Raises:
            TypeError: If *samples* is not a supported type.
            ValueError: If the sample list is empty.

        Example:
            >>> batch = Batch.collate(sample)
            >>> batch = Batch.collate([sample1, sample2])
            >>> batch = Batch.collate("image.jpg")
            >>> batch = Batch.collate(["img1.jpg", "img2.jpg"])
            >>> Batch.collate(batch) is batch  # idempotent
            True
        """
        # Return Batch unchanged (idempotent)
        if isinstance(samples, Batch):
            return samples

        # Convert single sample to list
        if isinstance(samples, Sample):
            samples = [samples]

        # Convert a single path to a one-element list
        elif isinstance(samples, (str, Path)):
            samples = [Sample(image_path=str(samples))]

        # Convert list of paths to list of Samples
        elif isinstance(samples, list) and samples and isinstance(samples[0], (str, Path)):
            samples = [Sample(image_path=str(p)) for p in samples]

        if not isinstance(samples, list):
            msg = f"Unsupported input type for collate: {type(samples)}"
            raise TypeError(msg)

        if not samples:
            msg = "Cannot collate empty list of samples"
            raise ValueError(msg)

        return cls(samples=samples)
