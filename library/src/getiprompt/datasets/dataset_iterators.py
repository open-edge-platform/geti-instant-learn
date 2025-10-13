# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Dataset iterators."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from getiprompt.datasets.dataset_iterator_base import DatasetIter

if TYPE_CHECKING:
    from collections.abc import Iterator

    import numpy as np

    from getiprompt.datasets import Dataset


class IndexIter(DatasetIter):
    """Standard PyTorch style iterator producing batches of images and masks.

    Examples:
        >>> from getiprompt.datasets.dataset_iterators import IndexIter
        >>> from getiprompt.datasets.dataset_base import Dataset
        >>> import numpy as np
        >>>
        >>> class MyDataset(Dataset):
        ...     def __len__(self) -> int:
        ...         return 1
        ...     def get_image_by_index(self, index: int) -> np.ndarray:
        ...         return np.zeros((10, 10, 3), dtype=np.uint8)
        ...     def get_masks_by_index(self, index: int) -> dict:
        ...         return {}
        ...
        >>> dataset = MyDataset(iterator_type=IndexIter)
        >>> iterator = IndexIter(parent=dataset)
        >>> item = iterator[0]
        >>> isinstance(item, tuple)
        True
    """

    def __init__(self, parent: Dataset) -> None:
        super().__init__(parent)
        self.index = 0

    def __getitem__(self, index: int) -> tuple[np.ndarray, dict[int, np.ndarray]]:
        """Get an item from the dataset.

        Args:
            index: Index of the item

        Returns:
            Tuple of image and masks
        """
        return self._parent.get_image_by_index(index), self._parent.get_masks_by_index(index)

    def __iter__(self) -> Iterator[tuple[np.ndarray, dict[int, np.ndarray]]]:
        """Iterate over the dataset.

        Returns:
            Iterator over the dataset
        """
        self.index = 1
        return self

    def __len__(self) -> int:
        """Get the length of the dataset.

        Returns:
            Length of the dataset
        """
        return self._parent.get_image_count()

    def __next__(self) -> tuple[np.ndarray, dict[int, np.ndarray]]:
        """Get the next item from the dataset.

        Returns:
            Next item from the dataset
        """
        if self.index < len(self):
            item = self.__getitem__(self.index)
            self.index += 1
            return item
        raise StopIteration

    def get_image_filename(self, *indices: int) -> str:
        """Get the image filename.

        Args:
            indices: Indices of the item

        Returns:
            Image filename
        """
        return self._parent.get_image_filename(indices[0])


class BatchedSingleCategoryIter(DatasetIter):
    """This class iterates over batches of images and masks of a given category.

    Examples:
        >>> from getiprompt.datasets.dataset_iterators import BatchedSingleCategoryIter
        >>> from getiprompt.datasets import Dataset
        >>> import numpy as np
        >>>
        >>> class MyDataset(Dataset):
        ...     def get_images_by_category(self, *args, **kwargs) -> list[np.ndarray]:
        ...         return [np.zeros((10, 10, 3))]
        ...
        ...     def get_masks_by_category(self, *args, **kwargs) -> list[np.ndarray]:
        ...         return [np.zeros((10, 10))]
        ...
        ...     def get_image_count_per_category(self, *args, **kwargs) -> int:
        ...         return 1
        >>>
        >>> dataset = MyDataset(iterator_type=BatchedSingleCategoryIter,
        ...     iterator_kwargs={"batch_size": 1, "category_index": 0})
        >>> iterator = BatchedSingleCategoryIter(parent=dataset, batch_size=1, category_index=0)
        >>> images, masks = iterator[0]
    """

    def __init__(self, parent: Dataset, batch_size: int, category_index: int) -> None:
        super().__init__(parent)
        self._batch_size = batch_size
        self._category_index = category_index
        self._batch_index = 0

    def __getitem__(
        self,
        batch_index: int,
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """Get an item from the dataset.

        Args:
            batch_index: Index of the batch

        Returns:
            Tuple of images and masks
        """
        images = self._parent.get_images_by_category(
            self._category_index,
            start=self._batch_index * self._batch_size,
            end=(batch_index + 1) * self._batch_size,
        )
        masks = self._parent.get_masks_by_category(
            self._category_index,
            start=self._batch_index * self._batch_size,
            end=(batch_index + 1) * self._batch_size,
        )
        return images, masks

    def __iter__(self) -> Iterator[tuple[list[np.ndarray], list[np.ndarray]]]:
        """Iterate over the dataset.

        Returns:
            Iterator over the dataset
        """
        self.batch_index = 1
        return self

    def __len__(self) -> int:
        """Get the length of the dataset.

        Returns:
            Length of the dataset
        """
        return math.ceil(
            self._parent.get_image_count_per_category(self._category_index) / self._batch_size,
        )

    def __next__(self) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """Get the next item from the dataset.

        Returns:
            Next item from the dataset
        """
        if self._batch_index < len(self):
            item = self.__getitem__(self._batch_index)
            self._batch_index += 1
            return item
        raise StopIteration

    def reset(self) -> None:
        """Reset the iterator."""
        self._batch_index = 0

    def get_image_filename(self, *indices: int) -> str:
        """Get the image filename.

        Args:
            indices: Indices of the item

        Returns:
            Image filename
        """
        return self._parent.get_image_filename_in_category(
            self._category_index,
            indices[0] * self._batch_size + indices[1],
        )


class BatchedCategoryIter(DatasetIter):
    """This class iterates over categories and returns a new iterator for creating batches per category.

    Examples:
        >>> from getiprompt.datasets.dataset_iterators import BatchedCategoryIter
        >>> from getiprompt.datasets import Dataset
        >>>
        >>> class MyDataset(Dataset):
        ...     def get_category_count(self) -> int:
        ...         return 1
        >>>
        >>> dataset = MyDataset(iterator_type=BatchedCategoryIter, iterator_kwargs={"batch_size": 1})
        >>> iterator = BatchedCategoryIter(parent=dataset, batch_size=1)
        >>> batch_iterator = iterator[0]
    """

    def __init__(self, parent: Dataset, batch_size: int) -> None:
        super().__init__(parent)
        self._batch_size = batch_size
        self._category_index = 0

    def __getitem__(self, category_index: int) -> BatchedSingleCategoryIter:
        """Get an item from the dataset.

        Args:
            category_index: Index of the category

        Returns:
            BatchedSingleCategoryIter
        """
        return BatchedSingleCategoryIter(
            self._parent,
            self._batch_size,
            self._category_index,
        )

    def __iter__(self) -> Iterator[BatchedSingleCategoryIter]:
        """Iterate over the dataset.

        Returns:
            Iterator over the dataset
        """
        self._category_index = 1
        return self

    def __len__(self) -> int:
        """Get the length of the dataset.

        Returns:
            Length of the dataset
        """
        return self._parent.get_category_count()

    def __next__(self) -> BatchedSingleCategoryIter:
        """Get the next item from the dataset.

        Returns:
            Next item from the dataset
        """
        if self._category_index < len(self):
            item = self.__getitem__(self._category_index)
            self._category_index += 1
            return item
        raise StopIteration

    def get_image_filename(self, *indices: int) -> str:
        """Get the image filename.

        Args:
            indices: Indices of the item

        Returns:
            Image filename
        """
        return self._parent.get_image_filename_in_category(
            indices[0],
            indices[1] * self._batch_size + indices[2],
        )
