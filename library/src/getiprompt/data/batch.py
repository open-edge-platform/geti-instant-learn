# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Batch handling for GetiPrompt datasets.

This module provides batch collation functionality for GetiPrompt samples.
The batch is a thin wrapper around list[GetiPromptSample] with convenient
properties for batch-level access to tensors.
"""

from dataclasses import dataclass, field

import numpy as np
import torch

from .sample import GetiPromptSample


@dataclass
class GetiPromptBatch:
    """Batch of GetiPrompt samples.

    A thin wrapper around `list[GetiPromptSample]` with convenience properties
    for batch-level access to images, masks, bboxes, points, and metadata.

    The core data structure is simply a list of samples, preserving full
    multi-instance information. Properties provide easy batch-level access
    with lazy tensor conversion and caching for performance.

    Args:
        samples (list[GetiPromptSample]): List of samples in this batch.

    Examples:
        Creating a batch:
        >>> samples = [sample1, sample2, sample3]
        >>> batch = GetiPromptBatch.collate(samples)
        >>> len(batch)
        3

        Accessing individual samples:
        >>> first_sample = batch[0]  # GetiPromptSample
        >>> for sample in batch:
        ...     process(sample.image, sample.masks)

        Batch-level access (returns lists):
        >>> images = batch.images  # list[torch.Tensor]
        >>> masks = batch.masks    # list[torch.Tensor | None]
        >>> categories = batch.categories  # list[list[str]]

        Multi-instance example:
        >>> # Sample 0: 1 instance (PerSeg)
        >>> batch[0].categories  # ['backpack']
        >>> batch.categories[0]  # ['backpack']
        >>>
        >>> # Sample 1: 3 instances (LVIS)
        >>> batch[1].categories  # ['person', 'person', 'car']
        >>> batch.categories[1]  # ['person', 'person', 'car']
    """

    samples: list[GetiPromptSample]

    # Cached tensors for performance (lazy conversion)
    _images: list[torch.Tensor] | None = field(default=None, init=False, repr=False)
    _masks: list[torch.Tensor | None] | None = field(default=None, init=False, repr=False)

    def __len__(self) -> int:
        """Get the batch size (number of samples)."""
        return len(self.samples)

    def __getitem__(self, index: int) -> GetiPromptSample:
        """Get a sample by index.

        Args:
            index (int): Sample index.

        Returns:
            GetiPromptSample: The sample at the given index.
        """
        return self.samples[index]

    def __iter__(self):
        """Iterate over samples in the batch."""
        return iter(self.samples)

    @property
    def images(self) -> list[torch.Tensor]:
        """Get all images as list of tensors.

        Converts numpy arrays to tensors and caches the result.
        Each tensor has shape (C, H, W).

        Returns:
            list[torch.Tensor]: List of image tensors.
        """
        if self._images is None:
            self._images = [
                torch.from_numpy(s.image.copy()) if isinstance(s.image, np.ndarray) else s.image for s in self.samples
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

    # === NUMPY PROPERTIES (no caching, usually already numpy from dataset) ===

    @property
    def images_np(self) -> list[np.ndarray]:
        """Get all images as numpy arrays.

        No conversion overhead if samples already contain numpy arrays.
        Each array has shape (C, H, W).

        Returns:
            list[np.ndarray]: List of image arrays.
        """
        result = []
        for s in self.samples:
            if isinstance(s.image, np.ndarray):
                result.append(s.image)
            elif isinstance(s.image, torch.Tensor):
                result.append(s.image.cpu().numpy())
            else:
                result.append(np.array(s.image))
        return result

    @property
    def masks_np(self) -> list[np.ndarray | None]:
        """Get all masks as numpy arrays.

        No conversion overhead if samples already contain numpy arrays.
        Each array has shape (N, H, W) where N is the number of instances.

        Returns:
            list[np.ndarray | None]: List of mask arrays or None.
        """
        result = []
        for s in self.samples:
            if s.masks is None:
                result.append(None)
            elif isinstance(s.masks, np.ndarray):
                result.append(s.masks)
            elif isinstance(s.masks, torch.Tensor):
                result.append(s.masks.cpu().numpy())
            else:
                result.append(np.array(s.masks))
        return result

    @property
    def bboxes_np(self) -> list[np.ndarray | None]:
        """Get all bboxes as numpy arrays.

        Each array has shape (N, 4) where N is the number of instances.

        Returns:
            list[np.ndarray | None]: List of bbox arrays or None.
        """
        result = []
        for s in self.samples:
            if s.bboxes is None:
                result.append(None)
            elif isinstance(s.bboxes, np.ndarray):
                result.append(s.bboxes)
            elif isinstance(s.bboxes, torch.Tensor):
                result.append(s.bboxes.cpu().numpy())
            else:
                result.append(np.array(s.bboxes))
        return result

    @property
    def points_np(self) -> list[np.ndarray | None]:
        """Get all points as numpy arrays.

        Each array has shape (N, 2) where N is the number of instances.

        Returns:
            list[np.ndarray | None]: List of point arrays or None.
        """
        result = []
        for s in self.samples:
            if s.points is None:
                result.append(None)
            elif isinstance(s.points, np.ndarray):
                result.append(s.points)
            elif isinstance(s.points, torch.Tensor):
                result.append(s.points.cpu().numpy())
            else:
                result.append(np.array(s.points))
        return result

    @property
    def category_ids_np(self) -> list[np.ndarray]:
        """Get all category IDs as numpy arrays.

        Each array has shape (N,) where N is the number of instances.

        Returns:
            list[np.ndarray]: List of category ID arrays.
        """
        result = []
        for s in self.samples:
            if s.category_ids is None:
                result.append(np.array([], dtype=np.int32))
            elif isinstance(s.category_ids, np.ndarray):
                result.append(s.category_ids)
            elif isinstance(s.category_ids, torch.Tensor):
                result.append(s.category_ids.cpu().numpy())
            else:
                result.append(np.array(s.category_ids, dtype=np.int32))
        return result

    @classmethod
    def collate(cls, samples: list[GetiPromptSample]) -> "GetiPromptBatch":
        """Collate a list of samples into a batch.

        Simply wraps the list of samples in a GetiPromptBatch.
        No data transformation is performed - tensor conversion happens
        lazily when properties are accessed.

        Args:
            samples (list[GetiPromptSample]): List of samples to batch.

        Returns:
            GetiPromptBatch: The batched samples.

        Raises:
            ValueError: If the sample list is empty.

        Example:
            >>> samples = [sample1, sample2, sample3]
            >>> batch = GetiPromptBatch.collate(samples)
            >>> len(batch)
            3
            >>> images = batch.images  # Lazy conversion to tensors
        """
        if not samples:
            raise ValueError("Cannot collate empty list of samples")

        return cls(samples=samples)
