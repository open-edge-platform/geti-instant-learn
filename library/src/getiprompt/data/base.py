# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Base class for datasets."""

import logging
import zipfile
from abc import ABC
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import NoReturn

import numpy as np
import requests
import torch

from getiprompt.data.dataset_iterators import DatasetIter


class Image:
    """Image class.

    Args:
        height: Height of the image
        width: Width of the image

    Examples:
        >>> from getiprompt.datasets.dataset_base import Image
        >>> import numpy as np
        >>>
        >>> class MyImage(Image):
        ...     def get_image(self) -> np.ndarray:
        ...         return np.zeros((self.height, self.width, 3))
        >>>
        >>> image = MyImage(height=10, width=10)
        >>> img_array = image.get_image()
    """

    def __init__(self, height: int, width: int) -> None:
        self.height = height
        self.width = width

    def get_image(self) -> np.ndarray:
        """Get the image.

        Returns:
            Image
        """
        raise NotImplementedError


class Dataset(torch.utils.data.Dataset, Iterable, ABC):  # noqa: PLR0904
    """Dataset class.

    Args:
        iterator_type: Iterator type
        iterator_kwargs: Iterator kwargs

    Examples:
        >>> from getiprompt.datasets.dataset_base import Dataset
        >>> from getiprompt.datasets.dataset_iterators import DatasetIter
        >>>
        >>> class MyDataset(Dataset):
        ...     def __len__(self) -> int:
        ...         return 1
        ...
        ...     def __getitem__(self, index: int) -> tuple:
        ...         return (np.zeros((10, 10, 3)), {})
        >>>
        >>> dataset = MyDataset(iterator_type=DatasetIter)
        >>> item = dataset[0]
    """

    def __init__(self, iterator_type: type[DatasetIter], iterator_kwargs: dict | None = None) -> None:
        if iterator_kwargs is None:
            iterator_kwargs = {}
        self._iterator_type = iterator_type
        self._iterator_kwargs = iterator_kwargs
        self.index = 0

    def get_categories(self) -> NoReturn:
        """Get the categories.

        Returns:
            Categories
        """
        raise NotImplementedError

    def get_image_by_index(self, index: int) -> np.ndarray:
        """This method returns an image based on its index.

        Args:
            index: The index of the image

        Returns:
            A numpy array containing an image

        """
        raise NotImplementedError

    def get_masks_by_index(self, index: int) -> dict[int, np.ndarray]:
        """This method returns a set of masks based on the image index.

        This method returns one mask per category where each individual instance
        has a unique pixel value.

        Args:
            index: the image index for which to return masks

        Returns:
            A dict of masks per category.
        """
        raise NotImplementedError

    def get_categories_of_image(self, index: int) -> list[str]:
        """This method gets all categories presents on an image.

        Args:
            index: The image index

        """
        raise NotImplementedError

    def get_images_by_category(
        self,
        category_index_or_name: int | str,
        start: int | None = None,
        end: int | None = None,
    ) -> list[np.ndarray]:
        """This method returns a list of images of a certain category.

        The parameters start and end are passed through Python's slice() function.

        Args:
            category_index_or_name: The category name or category index
            start: The first index to return
            end: end-1 is the last index to return

        Returns:
            A list of numpy arrays

        """
        raise NotImplementedError

    def get_masks_by_category(
        self,
        category_index_or_name: int | str,
        start: int | None = None,
        end: int | None = None,
    ) -> list[np.ndarray]:
        """This method returns a list of masks of a certain category.

        Each individual instance of the category has a unique pixel value.
        The parameters start and end are passed through Python's slice() function.

        Args:
            category_index_or_name: The category name or category index
            start: The first index to return
            end: end-1 is the last index to return

        Returns:
            A dict of masks per category.
        """
        raise NotImplementedError

    def get_image_count(self) -> int:
        """This method returns the number of images in the dataset."""
        raise NotImplementedError

    def get_image_count_per_category(self, category_index_or_name: int | str) -> int:
        """This method returns the number of images per category.

        Args:
            category_index_or_name: The category name or category index

        Returns:
            The number of images in a certain category
        """
        raise NotImplementedError

    def get_instance_count_per_category(self, category_index_or_name: int | str) -> int:
        """This method returns the number of instances per category.

        Args:
            category_index_or_name: The category name or category index

        Returns:
            The number of instances in a certain category
        """
        raise NotImplementedError

    def get_category_count(self) -> int:
        """This method returns the number of categories."""
        raise NotImplementedError

    def get_image_filename(self, index: int) -> str:
        """Gives the source filename of the image.

        Args:
            index: The index for which to return the filename

        Returns:
            The filename of the image.

        """
        raise NotImplementedError

    def get_image_filename_in_category(
        self,
        category_index_or_name: int | str,
        index: int,
    ) -> str:
        """Gives the source filename of the image.

        Args:
            category_index_or_name: The category name or category index
            index: The index for which to return the filename (index within the category)

        Returns:
            The filename of the image.

        """
        raise NotImplementedError

    def category_index_to_id(self, index: int) -> int:
        """Return the category id given the category index."""
        raise NotImplementedError

    def category_id_to_name(self, cat_id: int) -> str:
        """Return the category name given a category id."""
        raise NotImplementedError

    def category_name_to_id(self, name: str) -> int:
        """Return the category id given a category name."""
        raise NotImplementedError

    def category_name_to_index(self, name: str) -> int:
        """Return the category index given a category name."""
        raise NotImplementedError

    def category_index_to_name(self, index: int) -> str:
        """Return the category index given a category name."""
        return self.category_id_to_name(self.category_index_to_id(index))

    @staticmethod
    def _download(source: str, destination: str) -> None:
        """Helper function to download data with caching."""
        if Path(destination).is_file():
            logging.debug(f"Using cached downloaded file {destination}")
            return
        logging.info(f"Downloading data from {source} to {destination}...")

        response = requests.get(source, timeout=60)
        if response.status_code == 200:
            with Path(destination).open("wb") as file:
                file.write(response.content)
            logging.info(f"Downloaded {source} to {destination}")
        else:
            logging.info(f"Failed to download {source}")

    @staticmethod
    def _unzip(source: str, destination: str) -> None:
        """Helper function to unzip data with caching."""
        if Path(destination).exists():
            logging.debug(f"Using cached unzipped file or folder {destination}")
            return

        with zipfile.ZipFile(source, "r") as zf:
            zf.extractall(Path(source).parent)

        logging.info(f"Unzipped {source} to {destination}")

    def __len__(self) -> int:
        """Get the number of items in the dataset.

        What an item exactly entails is determined by the iterator.

        Returns:
            The number of items in the dataset
        """
        return len(self._iterator_type(parent=self, **self._iterator_kwargs))

    def __iter__(self) -> Iterator[tuple[np.ndarray, dict[int, np.ndarray]]]:
        """Iterate over the dataset.

        Returns:
            Iterator over the dataset
        """
        return self._iterator_type(parent=self, **self._iterator_kwargs)

    def __getitem__(self, index: int) -> tuple:
        """get_item is implemented for compatibility with torch's Dataset.

        What an item exactly entails is determined by the iterator.

        Args:
            index: The index to retrieve.

        Returns:
            A new item from the dataset iterator
        """
        return self._iterator_type(parent=self, **self._iterator_kwargs)[index]
