# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Folder dataset implementation for custom folder structures.

This module provides a dataset class for loading images and masks from a custom
folder structure with separate images/ and masks/ directories.
"""

from collections.abc import Sequence
from logging import getLogger
from pathlib import Path

import polars as pl
import torch

from getiprompt.data.base import Dataset
from getiprompt.data.utils.image import read_mask

# File extensions
IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif")
MASK_EXTENSIONS = (".png", ".bmp", ".tiff", ".tif")

logger = getLogger("Geti Prompt")


class FolderDataset(Dataset):
    """Folder dataset class for few-shot segmentation with custom folder structure.

    Dataset class for loading images and masks from a folder structure where:
    - Images are stored in: root/images/{category}/*.jpg
    - Masks are stored in: root/masks/{category}/*.png

    This is a single-instance dataset (one object per image), similar to PerSeg.

    Args:
        root (Path | str): Path to root directory containing the dataset.
            Should contain 'images' and optionally 'masks' subdirectories.
        images_dir (str, optional): Name of images subdirectory. Defaults to "images".
        masks_dir (str, optional): Name of masks subdirectory. Defaults to "masks".
        categories (Sequence[str] | None, optional): List of category names to include.
            If None, includes all available categories. Defaults to None.
        n_shots (int, optional): Number of reference shots per category. Defaults to 1.
        img_extensions (tuple[str, ...], optional): Valid image file extensions.
            Defaults to (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif").
        mask_extensions (tuple[str, ...], optional): Valid mask file extensions.
            Defaults to (".png", ".bmp", ".tiff", ".tif").
        masks_required (bool, optional): If True, masks directory is required and only
            image-mask pairs are included. If False, masks are optional and all images
            are included (with or without masks). Defaults to True.

    Example:
        >>> from pathlib import Path
        >>> from getiprompt.data.folder import FolderDataset

        >>> dataset = FolderDataset(
        ...     root=Path("./datasets/fss-1000"),
        ...     categories=["apple", "basketball"],
        ...     n_shots=1
        ... )

        >>> sample = dataset[0]
        >>> sample.image.shape
        torch.Size([3, 256, 256])  # CHW format

        >>> sample.masks.shape
        torch.Size([1, 256, 256])  # Single instance

        >>> sample.categories
        ['apple']  # List with one element
    """

    def __init__(
        self,
        root: Path | str,
        images_dir: str = "images",
        masks_dir: str = "masks",
        categories: Sequence[str] | None = None,
        n_shots: int = 1,
        img_extensions: tuple[str, ...] = IMG_EXTENSIONS,
        mask_extensions: tuple[str, ...] = MASK_EXTENSIONS,
        masks_required: bool = True,
    ) -> None:
        """Initialize the FolderDataset."""
        super().__init__(n_shots=n_shots)

        self.root = Path(root).expanduser().resolve()
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.categories_filter = categories
        self.img_extensions = img_extensions
        self.mask_extensions = mask_extensions
        self.masks_required = masks_required

        # Load the DataFrame
        self.df = self._load_dataframe()

    def _load_masks(self, raw_sample: dict) -> torch.Tensor | None:
        """Load single mask from file path.

        Args:
            raw_sample: Dictionary from DataFrame row.

        Returns:
            torch.Tensor with shape (1, H, W) for single-instance mask,
            and dtype torch.bool, or None if no mask path is available.
        """
        mask_paths = raw_sample.get("mask_paths")
        if not mask_paths or mask_paths[0] is None:
            return None

        # Load single mask
        mask = read_mask(mask_paths[0], as_tensor=True)  # (H, W)
        # Add instance dimension: (1, H, W) for consistency
        return mask[None, ...].to(torch.bool)

    def _load_dataframe(self) -> pl.DataFrame:
        """Load folder samples into Polars DataFrame."""
        return make_folder_dataframe(
            root=self.root,
            images_dir=self.images_dir,
            masks_dir=self.masks_dir,
            categories=self.categories_filter,
            n_shots=self.n_shots,
            img_extensions=self.img_extensions,
            mask_extensions=self.mask_extensions,
            masks_required=self.masks_required,
        )


def make_folder_dataframe(
    root: Path,
    images_dir: str = "images",
    masks_dir: str = "masks",
    categories: Sequence[str] | None = None,
    n_shots: int = 1,
    img_extensions: tuple[str, ...] = IMG_EXTENSIONS,
    mask_extensions: tuple[str, ...] = MASK_EXTENSIONS,
    masks_required: bool = True,
) -> pl.DataFrame:
    """Create a Polars DataFrame for folder dataset.

    Args:
        root (Path): Root directory of the dataset.
        images_dir (str): Name of images subdirectory. Defaults to "images".
        masks_dir (str): Name of masks subdirectory. Defaults to "masks".
        categories (Sequence[str] | None, optional): Categories to include.
            If None, includes all. Defaults to None.
        n_shots (int, optional): Number of reference shots per category. Defaults to 1.
        img_extensions (tuple[str, ...], optional): Valid image extensions.
        mask_extensions (tuple[str, ...], optional): Valid mask extensions.
        masks_required (bool, optional): If True, masks directory is required and only
            image-mask pairs are included. If False, masks are optional and all images
            are included (with or without masks). Defaults to True.

    Returns:
        pl.DataFrame: DataFrame containing sample metadata.

    Raises:
        FileNotFoundError: If required directories don't exist.
        ValueError: If no matching image-mask pairs are found.
    """
    images_root = root / images_dir
    masks_root = root / masks_dir

    if not images_root.exists():
        msg = f"Images directory not found: {images_root}"
        raise FileNotFoundError(msg)
    if masks_required and not masks_root.exists():
        msg = f"Masks directory not found: {masks_root}"
        raise FileNotFoundError(msg)

    # Get all available categories
    available_categories = []
    for category_dir in images_root.iterdir():
        if category_dir.is_dir():
            category_name = category_dir.name
            # If masks are required, category must exist in both images and masks
            if masks_required:
                mask_category_dir = masks_root / category_name
                if mask_category_dir.exists() and mask_category_dir.is_dir():
                    available_categories.append(category_name)
            else:
                # If masks are optional, just check images directory
                available_categories.append(category_name)

    # Filter categories if specified
    if categories is not None:
        available_categories = [cat for cat in available_categories if cat in categories]

    if not available_categories:
        msg = "No valid categories found"
        raise ValueError(msg)

    # Create category to ID mapping
    category_to_id = {cat: idx for idx, cat in enumerate(sorted(available_categories))}

    samples_data = []

    for category in available_categories:
        category_id = category_to_id[category]
        img_dir = images_root / category
        mask_dir = masks_root / category

        # Get all image files
        image_files = []
        for ext in img_extensions:
            image_files.extend(img_dir.glob(f"*{ext}"))

        # Sort by filename (assuming numeric naming like 1.jpg, 2.jpg, etc.)
        image_files = sorted(image_files, key=lambda x: (len(x.stem), x.stem))

        # Find corresponding mask files (optional if masks_required is False)
        valid_pairs = []
        for img_file in image_files:
            # Try different mask extensions
            mask_file = None
            if masks_root.exists() and mask_dir.exists():
                for ext in mask_extensions:
                    potential_mask = mask_dir / f"{img_file.stem}{ext}"
                    if potential_mask.exists():
                        mask_file = potential_mask
                        break

            # If masks are required, only include pairs with masks
            # If masks are optional, include all images (with or without masks)
            if mask_file is not None or not masks_required:
                valid_pairs.append((img_file, mask_file))

        if len(valid_pairs) < n_shots:
            msg = (
                f"Category '{category}' has only {len(valid_pairs)} samples, but {n_shots} reference shots requested",
            )
            logger.warning(msg)

        # Assign reference and target samples
        for i, (img_file, mask_file) in enumerate(valid_pairs):
            is_reference = i < n_shots
            n_shot = i if is_reference else -1

            # Wrap single values in lists for multi-instance compatibility
            samples_data.append({
                "image_id": img_file.stem,  # e.g., "1", "2", etc.
                "image_path": str(img_file),
                "categories": [category],  # List with single element
                "category_ids": [category_id],  # List with single element
                "mask_paths": [str(mask_file)] if mask_file is not None else [None],  # List with single element
                "is_reference": [is_reference],  # List with single element
                "n_shot": [n_shot],  # List with single element
            })

    if not samples_data:
        if masks_required:
            msg = "No valid image-mask pairs found"
        else:
            msg = "No valid images found"
        raise ValueError(msg)

    # Create DataFrame
    df = pl.DataFrame(samples_data)

    # Sort by category_id, then by first element of is_reference and n_shot lists
    # This keeps the same ordering as before (reference samples first within each category)
    df = df.with_columns([
        pl.col("is_reference").list.first().alias("_is_ref_sort"),
        pl.col("n_shot").list.first().alias("_n_shot_sort"),
    ])
    df = df.sort(["category_ids", "_is_ref_sort", "_n_shot_sort"], descending=[False, True, False])
    df = df.drop(["_is_ref_sort", "_n_shot_sort"])

    return df
