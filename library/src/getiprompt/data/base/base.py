# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Base dataset classes using custom Sample dataclass for GetiPrompt.

This module provides dataset implementations that leverage a custom
Sample dataclass with manual DataFrame management.
"""

import copy
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence

import numpy as np
import polars as pl
import torch
from torch.utils.data import Dataset as TorchDataset

from getiprompt.data.base.batch import Batch
from getiprompt.data.base.sample import Sample
from getiprompt.data.utils.image import read_image


class Dataset(TorchDataset, ABC):
    """Base class for datasets using Sample.

    This class provides a PyTorch-compatible interface while leveraging
    Sample class for type-safe data structures. Supports both single-instance
    (PerSeg) and multi-instance (LVIS) datasets.

    Subclasses should implement `_load_dataframe()` to return a Polars DataFrame
    with required columns:
    - image_id: str/int - Unique image identifier
    - image_path: str - Path to image file
    - categories: list[str] - List of category names (per instance)
    - category_ids: list[int] - List of category IDs (per instance)
    - is_reference: list[bool] - Reference flags (per instance)
    - n_shot: list[int] - Shot numbers (per instance)

    Optional columns:
    - annotation_ids: list[int] - Annotation IDs
    - mask_paths: list[str] - Paths to mask files (PerSeg)
    - bboxes: list[list[float]] - Bounding boxes [[x,y,w,h], ...]
    - segmentations: list[dict] - COCO RLE/polygon (LVIS)

    Args:
        n_shots (int, optional): Number of reference shots per category. Defaults to 1.

    Example:
        >>> dataset = PerSegDataset(n_shots=1)
        >>> len(dataset)  # Get dataset length
        100
        >>> sample = dataset[0]  # Get first sample (Sample)
        >>> sample.image.shape
        (224, 224, 3)  # HWC format for model preprocessors
    """

    def __init__(self, n_shots: int = 1) -> None:
        super().__init__()
        self.n_shots = n_shots
        self._df: pl.DataFrame | None = None

    @property
    def name(self) -> str:
        """Get the name of the dataset."""
        class_name = self.__class__.__name__
        class_name = class_name.removesuffix("Dataset")
        return class_name

    @property
    def df(self) -> pl.DataFrame:
        """Get the underlying Polars DataFrame."""
        if self._df is None:
            raise RuntimeError(
                "Dataset not initialized. Ensure that DataFrame has been loaded.",
            )
        return self._df

    @df.setter
    def df(self, dataframe: pl.DataFrame) -> None:
        """Set the underlying Polars DataFrame.

        Required columns for multi-instance support:
        - image_id: str/int - Unique image identifier
        - image_path: str - Path to image file
        - categories: list[str] - List of category names (for all instances in image)
        - category_ids: list[int] - List of category IDs
        - is_reference: list[bool] - Reference flags (per instance)
        - n_shot: list[int] - Shot numbers (per instance)

        Optional columns:
        - annotation_ids: list[int] - Annotation IDs
        - mask_paths: list[str] - Paths to mask files (PerSeg)
        - bboxes: list[list[float]] - Bounding boxes
        - segmentations: list[dict] - COCO RLE/polygon (LVIS)
        """
        # Validate required columns (updated for multi-instance)
        required_columns = {"image_path", "categories", "category_ids", "is_reference", "n_shot"}
        if not required_columns.issubset(set(dataframe.columns)):
            missing = required_columns - set(dataframe.columns)
            raise ValueError(f"DataFrame missing required columns: {missing}")
        self._df = dataframe

    def __len__(self) -> int:
        """Get the number of samples in the dataset."""
        return len(self.df)

    def __getitem__(self, index: int) -> Sample:
        """Get a sample by index.

        Loads the image and masks from disk and returns a Sample.
        Supports both single-instance (PerSeg) and multi-instance (LVIS) datasets.

        TODO: Move image preprocessing (resize, normalize) to dataset level
              Currently models handle preprocessing (HuggingFace, SAM transforms).
              Future refactoring should:
              1. Add optional transform parameter to __init__
              2. Apply transforms here in __getitem__
              3. Return images in CHW format consistently
              4. Remove preprocessing logic from model code
              This would improve:
              - Consistency across models
              - Testability (can test transforms separately)
              - Performance (preprocessing once vs. per-batch)
              - Clarity (separation of concerns)
        """
        # Get raw sample from DataFrame
        raw_sample = self.df.row(index, named=True)

        # Load image (once per sample!)
        # Returns HWC format for model preprocessors (HuggingFace, SAM)
        image = read_image(raw_sample["image_path"], as_tensor=False)  # numpy array, (H, W, C)

        # Load masks using dataset-specific implementation
        masks = self._load_masks(raw_sample)  # (N, H, W) or None

        # Load bboxes if available
        bboxes = None
        if "bboxes" in raw_sample and raw_sample["bboxes"] is not None:
            bboxes = np.array(raw_sample["bboxes"], dtype=np.float32)  # (N, 4)

        # Get is_reference and n_shot (always lists now)
        is_reference = raw_sample["is_reference"]
        n_shot = raw_sample["n_shot"]

        # Create and return Sample
        return Sample(
            image=image,
            masks=masks,  # (N, H, W) or None
            bboxes=bboxes,  # (N, 4) or None
            categories=raw_sample["categories"],  # list[str]
            category_ids=np.array(raw_sample["category_ids"], dtype=np.int32),  # (N,)
            is_reference=is_reference,  # list[bool]
            n_shot=n_shot,  # list[int]
            image_path=raw_sample["image_path"],
            mask_paths=raw_sample.get("mask_paths"),  # list[str] or None
        )

    @property
    def categories(self) -> list[str]:
        """Get all unique categories in the dataset."""
        # Explode list column to get all unique categories
        categories = self.df.select(pl.col("categories").explode()).unique()["categories"].to_list()
        return sorted(categories)

    @property
    def category_ids(self) -> list[int]:
        """Get all unique category IDs in the dataset."""
        # Explode list column to get all unique category IDs
        category_ids = self.df.select(pl.col("category_ids").explode()).unique()["category_ids"].to_list()
        return sorted(category_ids)

    @property
    def num_categories(self) -> int:
        """Get the number of categories in the dataset."""
        return len(self.categories)

    def get_category_name(self, category_id: int) -> str:
        """Get category name from category ID.

        Searches through all rows and instances to find the matching category ID.
        """
        for row in self.df.iter_rows(named=True):
            for i, cid in enumerate(row["category_ids"]):
                if cid == category_id:
                    return row["categories"][i]
        raise ValueError(f"Category ID {category_id} not found in dataset")

    def get_category_id(self, category_name: str) -> int:
        """Get category ID from category name.

        Searches through all rows and instances to find the matching category name.
        """
        for row in self.df.iter_rows(named=True):
            for i, cat in enumerate(row["categories"]):
                if cat == category_name:
                    return row["category_ids"][i]
        raise ValueError(f"Category '{category_name}' not found in dataset")

    def get_reference_samples_df(self, category: str | None = None) -> pl.DataFrame:
        """Get reference samples as Polars DataFrame.

        Filters to rows that contain at least one reference instance.
        For multi-instance images with mixed reference/target, consider using
        split_by_reference_instances() for cleaner separation.
        """
        # Filter rows with at least one True in is_reference list
        reference_df = self.df.filter(pl.col("is_reference").list.contains(True))

        if category is not None:
            # Further filter to rows containing the category
            reference_df = reference_df.filter(pl.col("categories").list.contains(category))

        return reference_df

    def get_target_samples_df(self, category: str | None = None) -> pl.DataFrame:
        """Get target samples as Polars DataFrame.

        Filters to rows where ALL instances are targets (no reference instances).
        For multi-instance images with mixed reference/target, consider using
        split_by_reference_instances() for cleaner separation.
        """
        # Filter rows where is_reference does NOT contain True (all are False)
        target_df = self.df.filter(~pl.col("is_reference").list.contains(True))

        if category is not None:
            # Further filter to rows containing the category
            target_df = target_df.filter(pl.col("categories").list.contains(category))

        return target_df

    def get_reference_dataset(self, category: str | None = None) -> "Dataset":
        """Create a new dataset containing only reference samples."""
        reference_df = self.get_reference_samples_df(category)

        # Create new dataset with filtered DataFrame
        new_dataset = copy.deepcopy(self)
        new_dataset._df = reference_df
        return new_dataset

    def get_target_dataset(self, category: str | None = None) -> "Dataset":
        """Create a new dataset containing only target samples."""
        target_df = self.get_target_samples_df(category)

        # Create new dataset with filtered DataFrame
        new_dataset = copy.deepcopy(self)
        new_dataset._df = target_df
        return new_dataset

    def subsample(self, indices: Sequence[int], inplace: bool = False) -> "Dataset":
        """Create a subset of the dataset using the provided indices."""
        if len(set(indices)) != len(indices):
            raise ValueError("No duplicates allowed in indices.")

        # Get subset of the DataFrame
        subset_df = self.df[list(indices)]

        dataset = self if inplace else copy.deepcopy(self)
        dataset._df = subset_df
        return dataset

    def __add__(self, other_dataset: "Dataset") -> "Dataset":
        """Concatenate this dataset with another dataset."""
        if not isinstance(other_dataset, self.__class__):
            raise TypeError("Cannot concatenate datasets that are not of the same type.")

        # Concatenate the DataFrames
        combined_df = pl.concat([self.df, other_dataset.df])

        dataset = copy.deepcopy(self)
        dataset._df = combined_df
        return dataset

    @property
    def collate_fn(self) -> Callable:
        """Get the collate function for batching dataset items."""
        return Batch.collate

    @abstractmethod
    def _load_masks(self, raw_sample: dict) -> torch.Tensor | None:
        """Load masks for a sample.

        This method should be implemented by subclasses to load masks in their
        specific format (e.g., from files for PerSeg, from RLE for LVIS).

        Args:
            raw_sample: Dictionary from DataFrame row containing sample metadata.

        Returns:
            torch.Tensor with shape (N, H, W) where N is the number of instances,
            and dtype torch.bool, or None if no masks are available.
        """

    @abstractmethod
    def _load_dataframe(self) -> pl.DataFrame:
        """Load samples and return a Polars DataFrame.

        This method should be implemented by subclasses to load their specific
        dataset format and return a Polars DataFrame with required columns:
        - image_path: str - Path to image file
        - categories: list[str] - List of category names
        - category_ids: list[int] - List of category IDs
        - is_reference: list[bool] - Reference flags (per instance)
        - n_shot: list[int] - Shot numbers (per instance)

        Optional columns:
        - image_id: str/int - Unique image identifier
        - annotation_ids: list[int] - Annotation IDs
        - mask_paths: list[str] - Paths to mask files (PerSeg)
        - bboxes: list[list[float]] - Bounding boxes [[x,y,w,h], ...]
        - segmentations: list[dict] - COCO RLE/polygon (LVIS)

        Returns:
            pl.DataFrame: DataFrame containing sample metadata.
        """
