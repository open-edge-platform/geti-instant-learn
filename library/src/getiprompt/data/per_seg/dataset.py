# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""PerSeg dataset implementation."""

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

__all__ = ["PerSegDataset", "make_perseg_dataframe"]


logger = getLogger("Geti Prompt")


class PerSegDataset(Dataset):
    """PerSeg dataset class for few-shot segmentation.

    Dataset class for loading and processing PerSeg dataset images for few-shot
    segmentation tasks. PerSeg is a single-instance dataset (one object per image).

    Args:
        root (Path | str): Path to root directory containing the dataset.
            Defaults to "./datasets/PerSeg".
        categories (Sequence[str] | None, optional): List of category names to include.
            If None, includes all available categories. Defaults to None.
        n_shots (int, optional): Number of reference shots per category. Defaults to 1.

    Example:
        >>> from pathlib import Path
        >>> from getiprompt.data.datasets import PerSegDataset

        >>> dataset = PerSegDataset(
        ...     root=Path("./datasets/PerSeg"),
        ...     categories=["backpack", "dog"],
        ...     n_shots=1
        ... )

        >>> sample = dataset[0]
        >>> sample.image.shape
        (256, 256, 3)  # HWC format for model preprocessors

        >>> sample.masks.shape
        (1, 256, 256)  # Single instance

        >>> sample.categories
        ['backpack']  # List with one element
    """

    def __init__(
        self,
        root: Path | str = "./datasets/PerSeg",
        categories: Sequence[str] | None = None,
        n_shots: int = 1,
    ) -> None:
        super().__init__(n_shots=n_shots)

        self.root = Path(root)
        self.categories_filter = categories

        # Load the DataFrame
        self.df = self._load_dataframe()

    def _load_masks(self, raw_sample: dict) -> torch.Tensor | None:
        """Load single mask from file path.

        Args:
            raw_sample: Dictionary from DataFrame row.

        Returns:
            torch.Tensor with shape (1, H, W) for single-instance PerSeg mask,
            and dtype torch.bool, or None if no mask path is available.
        """
        mask_paths = raw_sample.get("mask_paths")
        if not mask_paths or mask_paths[0] is None:
            return None

        # Load single mask for PerSeg
        mask = read_mask(mask_paths[0], as_tensor=True)  # (H, W)
        # Add instance dimension: (1, H, W) for consistency
        return mask[None, ...].to(torch.bool)

    def _load_dataframe(self) -> pl.DataFrame:
        """Load PerSeg samples into Polars DataFrame."""
        return make_perseg_dataframe(
            self.root,
            categories=self.categories_filter,
            n_shots=self.n_shots,
            img_extensions=IMG_EXTENSIONS,
            mask_extensions=MASK_EXTENSIONS,
        )


def make_perseg_dataframe(
    root: Path,
    categories: Sequence[str] | None = None,
    n_shots: int = 1,
    img_extensions: tuple[str, ...] = IMG_EXTENSIONS,
    mask_extensions: tuple[str, ...] = MASK_EXTENSIONS,
) -> pl.DataFrame:
    """Create a Polars DataFrame for PerSeg dataset.

    Args:
        root (Path): Root directory of the PerSeg dataset.
        categories (Sequence[str] | None, optional): Categories to include.
            If None, includes all. Defaults to None.
        n_shots (int, optional): Number of reference shots per category. Defaults to 1.
        img_extensions (tuple[str, ...], optional): Valid image extensions.
        mask_extensions (tuple[str, ...], optional): Valid mask extensions.

    Returns:
        pl.DataFrame: DataFrame containing sample metadata.

    Raises:
        FileNotFoundError: If required directories don't exist.
        ValueError: If no matching image-mask pairs are found.
    """
    images_dir = root / "Images"
    annotations_dir = root / "Annotations"

    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    if not annotations_dir.exists():
        raise FileNotFoundError(f"Annotations directory not found: {annotations_dir}")

    # Get all available categories
    available_categories = [d.name for d in images_dir.iterdir() if d.is_dir() and (annotations_dir / d.name).exists()]

    # Filter categories if specified
    if categories is not None:
        available_categories = [cat for cat in available_categories if cat in categories]

    if not available_categories:
        raise ValueError("No valid categories found")

    # Create category to ID mapping
    category_to_id = {cat: idx for idx, cat in enumerate(sorted(available_categories))}

    samples_data = []

    for category in available_categories:
        category_id = category_to_id[category]
        img_dir = images_dir / category
        mask_dir = annotations_dir / category

        # Get all image files
        image_files = []
        for ext in img_extensions:
            image_files.extend(img_dir.glob(f"*{ext}"))

        # Sort by filename (assuming numeric naming like 00.jpg, 01.jpg, etc.)
        image_files = sorted(image_files, key=lambda x: x.stem)

        # Find corresponding mask files
        valid_pairs = []
        for img_file in image_files:
            # Try different mask extensions
            mask_file = None
            for ext in mask_extensions:
                potential_mask = mask_dir / f"{img_file.stem}{ext}"
                if potential_mask.exists():
                    mask_file = potential_mask
                    break

            if mask_file is not None:
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
                "image_id": img_file.stem,  # e.g., "00", "01", etc.
                "image_path": str(img_file),
                "categories": [category],  # List with single element
                "category_ids": [category_id],  # List with single element
                "mask_paths": [str(mask_file)],  # List with single element
                "is_reference": [is_reference],  # List with single element
                "n_shot": [n_shot],  # List with single element
            })

    if not samples_data:
        raise ValueError("No valid image-mask pairs found")

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
