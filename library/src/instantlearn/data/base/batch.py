# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Batch handling for InstantLearn datasets.

This module provides batch collation functionality for Geti Instant Learn `Sample` object.
The batch is a thin wrapper around list[Sample] with convenient
properties for batch-level access to tensors.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field

import numpy as np
import torch
from torchvision import tv_tensors

from instantlearn.data.base.sample import Sample


@dataclass
class Batch:
    """Batch of InstantLearn samples.

    A thin wrapper around `list[Sample]` with convenience properties
    for batch-level access to images, masks, bboxes, points, and metadata.

    The core data structure is simply a list of samples, preserving full
    multi-instance information. Properties provide easy batch-level access
    with lazy tensor conversion and caching for performance.

    Args:
        samples (list[Sample]): List of samples in this batch.

    Examples:
        Creating a batch:
        >>> samples = [sample1, sample2, sample3]
        >>> batch = Batch.collate(samples)
        >>> len(batch)
        3

        Direct constructor usage:
        >>> batch = Batch(samples=[sample1, sample2, sample3])
        >>> len(batch)
        3

        Accessing individual samples:
        >>> first_sample = batch[0]  # Sample
        >>> for sample in batch:
        ...     process(sample.image, sample.masks)

        Batch-level access (returns lists):
        >>> images = batch.images  # list[tv_tensors.Image]
        >>> masks = batch.masks    # list[torch.Tensor | None]
        >>> categories = batch.categories  # list[list[str]]

        Multi-instance example:
        >>> # Sample 0: 1 instance (PerSeg)
        >>> batch[0].categories  # ['backpack']
        >>> batch.categories[0]  # ['backpack']

        >>> # Sample 1: 3 instances (LVIS)
        >>> batch[1].categories  # ['person', 'person', 'car']
        >>> batch.categories[1]  # ['person', 'person', 'car']
    """

    samples: list[Sample]

    # Cached tensors for performance (lazy conversion)
    _images: list[tv_tensors.Image] | None = field(default=None, init=False, repr=False)
    _masks: list[torch.Tensor | None] | None = field(default=None, init=False, repr=False)

    def __len__(self) -> int:
        """Get the batch size (number of samples)."""
        return len(self.samples)

    def __getitem__(self, index: int) -> Sample:
        """Get a sample by index.

        Args:
            index (int): Sample index.

        Returns:
            Sample: The sample at the given index.
        """
        return self.samples[index]

    def __iter__(self) -> Iterator[Sample]:
        """Iterate over samples in the batch."""
        return iter(self.samples)

    @property
    def images(self) -> list[tv_tensors.Image]:
        """Get all images as list of tensors.

        Converts numpy arrays to tensors and caches the result.
        Each tensor has shape (C, H, W).

        Returns:
            list[tv_tensors.Image]: List of image tensors.
        """
        if self._images is None:
            self._images = [
                tv_tensors.Image(torch.from_numpy(s.image.copy())) if isinstance(s.image, np.ndarray) else s.image
                for s in self.samples
            ]
        return self._images

    @property
    def masks(self) -> list[torch.Tensor | None]:
        """Get all masks as list of tensors.

        Converts numpy arrays to tensors and caches the result.
        Each tensor has shape (N, H, W) where N is the number of instances.

        Returns:
            list[torch.Tensor | None]: List of mask tensors or None.
        """
        if self._masks is None:
            self._masks = []
            for s in self.samples:
                if s.masks is not None:
                    mask = torch.from_numpy(s.masks.copy()) if isinstance(s.masks, np.ndarray) else s.masks
                else:
                    mask = None
                self._masks.append(mask)
        return self._masks

    @property
    def bboxes(self) -> list[torch.Tensor | None]:
        """Get all bboxes as list of tensors.

        Each tensor has shape (N, 4) where N is the number of instances.
        Bounding boxes are in [x, y, w, h] format.

        Returns:
            list[torch.Tensor | None]: List of bbox tensors or None.
        """
        result = []
        for s in self.samples:
            if s.bboxes is not None:
                bbox = torch.from_numpy(s.bboxes.copy()) if isinstance(s.bboxes, np.ndarray) else s.bboxes
            else:
                bbox = None
            result.append(bbox)
        return result

    @property
    def points(self) -> list[torch.Tensor | None]:
        """Get all points as list of tensors.

        Each tensor has shape (N, 2) where N is the number of instances.
        Points are in [x, y] format.

        Returns:
            list[torch.Tensor | None]: List of point tensors or None.
        """
        result = []
        for s in self.samples:
            if s.points is not None:
                pts = torch.from_numpy(s.points.copy()) if isinstance(s.points, np.ndarray) else s.points
            else:
                pts = None
            result.append(pts)
        return result

    @property
    def categories(self) -> list[list[str]]:
        """Get all categories as list of lists.

        Preserves multi-instance structure:
        - Single-instance: [['cat'], ['dog'], ...]
        - Multi-instance: [['person', 'person', 'car'], ['dog'], ...]

        Returns:
            list[list[str]]: List of category lists.
        """
        return [s.categories for s in self.samples]

    @property
    def category_ids(self) -> list[torch.Tensor]:
        """Get all category IDs as list of tensors.

        Each tensor has shape (N,) where N is the number of instances.

        Returns:
            list[torch.Tensor]: List of category ID tensors.
        """
        result = []
        for s in self.samples:
            if s.category_ids is not None:
                ids = (
                    torch.from_numpy(s.category_ids.copy())
                    if isinstance(s.category_ids, np.ndarray)
                    else s.category_ids
                )
            else:
                ids = torch.tensor([], dtype=torch.int32)
            result.append(ids)
        return result

    @property
    def is_reference(self) -> list[list[bool]]:
        """Get reference flags for all samples.

        Each entry is a list of bools (one per instance in the sample):
        - Single-instance: [[True], [False], [True]]
        - Multi-instance: [[True, False, True], [False, False]]

        Returns:
            list[list[bool]]: List of reference flag lists.
        """
        return [s.is_reference for s in self.samples]

    @property
    def n_shot(self) -> list[list[int]]:
        """Get shot numbers for all samples.

        Each entry is a list of ints (one per instance in the sample):
        - Single-instance: [[0], [-1], [1]]
        - Multi-instance: [[0, -1, 1], [-1, -1]]

        Returns:
            list[list[int]]: List of shot number lists.
        """
        return [s.n_shot for s in self.samples]

    @property
    def image_paths(self) -> list[str]:
        """Get all image paths.

        Returns:
            list[str]: List of image file paths.
        """
        return [s.image_path for s in self.samples]

    @property
    def mask_paths(self) -> list[list[str] | None]:
        """Get all mask paths as list of lists.

        Each entry can be:
        - List of paths: ['mask1.png', 'mask2.png']
        - None: No mask paths

        Returns:
            list[list[str] | None]: List of mask path lists.
        """
        return [s.mask_paths for s in self.samples]

    @classmethod
    def collate(cls, samples: Sample | list[Sample] | Batch) -> Batch:
        """Collate sample(s) into a batch.

        Simply wraps the sample(s) in a Batch.
        No data transformation is performed - tensor conversion happens
        lazily when properties are accessed.

        This method is idempotent - passing a Batch returns it unchanged.

        Args:
            samples: Single sample, Batch, or list of samples to batch.

        Returns:
            Batch: The batched samples.

        Raises:
            ValueError: If the sample list is empty.

        Examples:
            Collate a list of samples:
            >>> samples = [sample1, sample2, sample3]
            >>> batch = Batch.collate(samples)
            >>> len(batch)
            3

            Collate a single sample:
            >>> batch = Batch.collate(sample1)
            >>> len(batch)
            1
            >>> images = batch.images  # Lazy conversion to tensors

            Idempotent behavior (returns Batch unchanged):
            >>> batch = Batch.collate(samples)
            >>> same_batch = Batch.collate(batch)
            >>> batch is same_batch
            True
        """
        # Return Batch unchanged (idempotent)
        if isinstance(samples, Batch):
            return samples

        # Convert single sample to list
        if isinstance(samples, Sample):
            samples = [samples]

        if not samples:
            msg = "Cannot collate empty list of samples"
            raise ValueError(msg)

        return cls(samples=samples)
