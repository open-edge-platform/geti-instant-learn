# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from logging import getLogger
from pathlib import Path

import numpy as np

from getiprompt.datasets.dataset_base import Dataset
from getiprompt.datasets.dataset_iterators import BatchedCategoryIter
from getiprompt.datasets.lvis.lvis_dataset import LVISDataset
from getiprompt.datasets.perseg.perseg_dataset import PerSegDataset

logger = getLogger("Geti Prompt")


def load_dataset(dataset_name: str, whitelist: list[str] | None = None, batch_size: int = 5) -> Dataset:
    """Load a dataset.

    Args:
        batch_size: The batch size used during inference
        dataset_name: Name of the dataset
        whitelist: Whitelist of categories

    Returns:
        Dataset

    Raises:
        ValueError: If the dataset name is not recognized
    """
    # add logging that we are loading the dataset
    logger.info(f"Loading dataset: {dataset_name}")
    if dataset_name == "PerSeg":
        return PerSegDataset(
            whitelist=whitelist,
            iterator_type=BatchedCategoryIter,
            iterator_kwargs={"batch_size": batch_size},
        )
    if dataset_name == "lvis":
        whitelist = whitelist if whitelist is not None else ("cupcake", "sheep", "pastry", "doughnut")
        return LVISDataset(
            whitelist=whitelist,
            iterator_type=BatchedCategoryIter,
            iterator_kwargs={"batch_size": batch_size},
        )
    if dataset_name == "lvis_validation":
        whitelist = whitelist if whitelist is not None else ("cupcake", "sheep", "pastry", "doughnut")
        return LVISDataset(
            whitelist=whitelist,
            iterator_type=BatchedCategoryIter,
            iterator_kwargs={"batch_size": batch_size},
            name="validation",
        )
    msg = f"Unknown dataset name {dataset_name}"
    raise ValueError(msg)


def get_image_and_mask_from_filename(
    filename: str, dataset: Dataset, category_name: str
) -> tuple[np.ndarray, np.ndarray]:
    """Get the image and mask from using a base filename and dataset.

    Args:
        filename: The base filename.
        dataset: The dataset to search.
        category_name: The category that the image belongs to.

    Returns:
        The image array and the mask and if the category corresponds
    """
    # match as much from the path as the user hase given us.
    f = Path(filename).parts
    filenames = {
        "".join(Path(dataset.get_image_filename(i)).parts[-len(f) :]): i for i in range(dataset.get_image_count())
    }
    idx = filenames["".join(f)]

    cat_id = dataset.category_name_to_id(category_name)
    image = dataset.get_image_by_index(idx)
    masks = dataset.get_masks_by_index(idx)

    mask = masks[cat_id] if cat_id in masks else np.zeros_like(image)[:, :, 0]
    return image, mask


def get_filename_categories(filenames: list[str], dataset: Dataset) -> list[str] | None:
    """This method gets all the categories associated with the filenames.

    Args:
        filenames: A list of filenames to get the categories from.
        dataset: The dataset of the image

    Returns:
        a list of category ids
    """
    if filenames is None:
        return None

    categories = []
    for filename in filenames:
        f = Path(filename).parts
        fns = {
            "".join(Path(dataset.get_image_filename(i)).parts[-len(f) :]): i for i in range(dataset.get_image_count())
        }
        image_idx = fns["".join(f)]
        categories.extend(dataset.get_categories_of_image(image_idx))
    return categories
