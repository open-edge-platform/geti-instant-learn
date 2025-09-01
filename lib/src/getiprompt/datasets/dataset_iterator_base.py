# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Base class for dataset iterators."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sized
from typing import TYPE_CHECKING, NoReturn

if TYPE_CHECKING:
    import numpy as np

    from getiprompt.datasets import Dataset


class DatasetIter(Sized, Iterable):
    """Base class for dataset iterators.

    Args:
        parent: Parent dataset

    Examples:
        >>> from getiprompt.datasets import Dataset
        >>> from getiprompt.datasets.dataset_iterator_base import DatasetIter
        >>> import numpy as np
        >>>
        >>> class MyDatasetIter(DatasetIter):
        ...     def __len__(self) -> int:
        ...         return 1
        ...
        ...     def __getitem__(self, index: int) -> tuple[np.ndarray, dict[int, np.ndarray]]:
        ...         return (np.zeros((10, 10, 3)), {})
        >>>
        >>> class MyDataset(Dataset):
        ...     def __len__(self) -> int:
        ...         return 1
        ...
        ...     def __getitem__(self, index: int) -> tuple:
        ...         return (np.zeros((10, 10, 3)), {})
        >>>
        >>> dataset = MyDataset(iterator_type=MyDatasetIter)
        >>> iterator = MyDatasetIter(parent=dataset)
        >>> item = iterator[0]
    """

    def __init__(self, parent: Dataset, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._parent = parent

    def __len__(self) -> int:
        """Get the length of the dataset.

        Returns:
            Length of the dataset
        """
        raise NotImplementedError

    def __getitem__(self, index: int) -> tuple[np.ndarray, dict[int, np.ndarray]]:
        """Get an item from the dataset.

        Args:
            index: Index of the item

        Returns:
            Tuple of image and masks
        """
        raise NotImplementedError

    def __iter__(self) -> Iterator[tuple[np.ndarray, dict[int, np.ndarray]]]:
        """Iterate over the dataset.

        Returns:
            Iterator over the dataset
        """
        raise NotImplementedError

    def __next__(self) -> tuple[np.ndarray, dict[int, np.ndarray]]:
        """Get the next item from the dataset.

        Returns:
            Next item from the dataset
        """
        raise NotImplementedError

    def get_image_filename(self, *indices: int) -> NoReturn:
        """Returns the filename of the original image given a list of indices.

        Args:
            *indices: Depending on the iterator type this can be a certain number of indices
                (e.g. category #, batch #, image #)

        Returns:
            The filename of the original image

        """
        raise NotImplementedError

    def reset(self) -> NoReturn:
        """Reset the iterator."""
        raise NotImplementedError
