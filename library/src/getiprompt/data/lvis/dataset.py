# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""LVIS dataset implementation for multi-instance few-shot segmentation.

This module provides the LVIS dataset implementation that supports
multi-instance images (multiple objects per image with different categories).
"""

from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

import polars as pl
import torch
from lvis import LVIS
from pycocotools import mask as mask_utils

from getiprompt.data.base import Dataset


class LVISDataset(Dataset):
    """LVIS dataset class for semantic few-shot segmentation.

    Dataset class for loading and processing LVIS dataset images for semantic
    few-shot segmentation tasks. Unlike traditional instance segmentation,
    this class merges instances of the same category into semantic masks.

    Args:
        root (Path | str): Path to root directory containing the dataset.
        split (str, optional): Dataset split ('train', 'val'). Defaults to 'val'.
        categories (Sequence[str] | None, optional): List of category names to include.
            If None, includes all available categories. Defaults to None.
        n_shots (int, optional): Number of reference shots per category. Defaults to 1.

    Example:
        >>> from pathlib import Path
        >>> from getiprompt.data.datasets import LVISDataset

        >>> dataset = LVISDataset(
        ...     root=Path("./datasets/LVIS"),
        ...     split="val",
        ...     categories=["person", "car", "dog"],
        ...     n_shots=1
        ... )

        >>> sample = dataset[0]

        >>> sample.image.shape
        (480, 640, 3)  # HWC format for model preprocessors

        >>> sample.masks.shape
        (2, 480, 640)  # 2 semantic masks (person and car categories)

        >>> sample.categories
        ['car', 'person']  # Unique categories only

        >>> sample.bboxes is None
        True  # No bboxes for semantic masks
    """

    def __init__(
        self,
        root: Path | str = "./datasets/LVIS",
        split: str = "val",
        categories: Sequence[str] | None = None,
        n_shots: int = 1,
    ) -> None:
        """Initialize the LVISDataset."""
        super().__init__(n_shots=n_shots)

        self.root = Path(root).expanduser()
        self.split = split
        self.categories_filter = categories

        # Load the DataFrame
        self.df = self._load_dataframe()

    def _load_masks(self, raw_sample: dict) -> torch.Tensor | None:
        """Decode and merge masks from COCO RLE format into semantic masks.

        Since the DataFrame is exploded (one row per image-category combination),
        segmentations is a single list containing all segmentations for one category.

        Args:
            raw_sample: Dictionary from DataFrame row.

        Returns:
            torch.Tensor with shape (1, H, W) where 1 represents the single category
            in this row, and dtype torch.bool, or None if no segmentations are available.

        Raises:
            TypeError: If unknown segmentation format is encountered.
        """
        segmentations = raw_sample.get("segmentations")
        if not segmentations:
            return None

        # Get image dimensions
        h, w = raw_sample.get("img_dim")

        # After explode, segmentations is a single list of segmentations for one category
        # Decode all masks for this category and merge them
        category_mask = torch.zeros((h, w), dtype=torch.bool)

        for segmentation in segmentations:
            if isinstance(segmentation, dict):  # RLE format
                mask = mask_utils.decode(segmentation)  # (H, W)
            elif isinstance(segmentation, list):  # Polygon format
                # Convert polygon to RLE then decode
                rles = mask_utils.frPyObjects(segmentation, h, w)
                mask = mask_utils.decode(rles)
            else:
                msg = f"Unknown segmentation format: {type(segmentation)}"
                raise TypeError(msg)

            # Handle potential 3D masks from polygon conversion
            mask = torch.from_numpy(mask)
            if mask.ndim > 2:
                mask = torch.max(mask, dim=-1).values
            # Merge with category mask using logical OR
            category_mask = category_mask | mask

        # Return as (1, H, W) to maintain consistency with Sample structure
        return category_mask.unsqueeze(0)  # (1, H, W)

    def _load_dataframe(self) -> pl.DataFrame:
        """Load LVIS samples into Polars DataFrame with semantic mask structure."""
        images_dir = self.root / self.split
        annotations_file = self.root / f"lvis_v1_{self.split}.json"
        _lvis_api = LVIS(str(annotations_file))
        return make_lvis_dataframe(
            lvis_api=_lvis_api,
            images_dir=images_dir,
            categories=self.categories_filter,
            n_shots=self.n_shots,
        )


def make_lvis_dataframe(
    lvis_api: LVIS,
    images_dir: Path,
    categories: Sequence[str] | None = None,
    n_shots: int = 1,
) -> pl.DataFrame:
    """Create a Polars DataFrame for LVIS dataset with semantic masks.

    This function creates a DataFrame where instances of the same category
    are merged into semantic masks. Each row represents one image-category combination
    (exploded format: one row per instance).

    Args:
        lvis_api (LVIS): LVIS API instance.
        images_dir (Path): Directory containing the images.
        categories (Sequence[str] | None, optional): Categories to include.
            If None, includes all. Defaults to None.
        n_shots (int, optional): Number of reference shots per category. Defaults to 1.

    Returns:
        pl.DataFrame: DataFrame containing sample metadata with one row per instance.
            Each row represents one image-category combination.

    Raises:
        FileNotFoundError: If image file not found.
        ValueError: If no matching annotations are found.
    """
    # Get category filtering
    if categories is not None:
        all_cats = lvis_api.load_cats(lvis_api.get_cat_ids())
        category_name_to_id = {cat["name"]: cat["id"] for cat in all_cats}
        valid_category_ids = [category_name_to_id[cat] for cat in categories if cat in category_name_to_id]
    else:
        valid_category_ids = None

    # Get filtered annotation IDs using LVIS API
    ann_ids = lvis_api.get_ann_ids(cat_ids=valid_category_ids)
    annotations = lvis_api.load_anns(ann_ids)

    # Group annotations by image_id
    image_annotations = defaultdict(list)
    for ann in annotations:
        image_annotations[ann["image_id"]].append(ann)

    # Build samples (one row per image with semantic structure)
    samples_data = []
    category_shot_counts = defaultdict(int)  # Track n_shots per category

    for image_id, anns in image_annotations.items():
        if not anns:  # Skip images with no annotations after filtering
            continue

        img_info = lvis_api.imgs[image_id]
        # Extract the COCO subfolder (train2017/val2017) from the coco_url
        # LVIS val set can contain images from both COCO train2017 and val2017
        coco_url_path = Path(img_info["coco_url"])
        coco_subset = coco_url_path.parent.name  # e.g., "train2017" or "val2017"
        image_filename = coco_url_path.name
        # Build path using the actual COCO subfolder
        image_path = images_dir.parent / coco_subset / image_filename
        if not image_path.exists():
            msg = f"Image file not found: {image_path}"
            raise FileNotFoundError(msg)

        img_h, img_w = img_info["height"], img_info["width"]

        # Group annotations by category
        category_annotations = defaultdict(list)
        for ann in anns:
            cat_id = ann["category_id"]
            cat_name = lvis_api.cats[cat_id]["name"]
            category_annotations[cat_name].append(ann)

        # Build semantic structure (one entry per unique category)
        categories_list = []
        category_ids_list = []
        segmentations_list = []
        is_reference_list = []
        n_shot_list = []

        for cat_name in sorted(category_annotations.keys()):
            cat_anns = category_annotations[cat_name]

            # Use the first annotation's category_id (all should be the same)
            cat_id = cat_anns[0]["category_id"]

            # Determine reference status: category is reference if ANY instance is reference
            current_shot_count = category_shot_counts[cat_name]
            is_ref = current_shot_count < n_shots
            shot_num = current_shot_count if is_ref else -1

            if is_ref:
                category_shot_counts[cat_name] += 1

            # Collect all segmentations for this category
            cat_segmentations = [ann["segmentation"] for ann in cat_anns]

            categories_list.append(cat_name)
            category_ids_list.append(cat_id)
            segmentations_list.append(cat_segmentations)  # List of segmentations for this category
            is_reference_list.append(is_ref)
            n_shot_list.append(shot_num)

        samples_data.append({
            "image_id": image_id,
            "image_path": image_path,
            "categories": categories_list,
            "category_ids": category_ids_list,
            "segmentations": segmentations_list,  # List of lists: one list per category
            "is_reference": is_reference_list,
            "n_shot": n_shot_list,
            "img_dim": (img_h, img_w),
        })

    if not samples_data:
        msg = "No valid annotations found"
        raise ValueError(msg)

    # Create DataFrame
    df = pl.DataFrame(samples_data)

    # Explode to split multi-instance rows into single-instance rows
    # This creates one row per image-category combination
    explode_columns = ["categories", "category_ids", "segmentations", "is_reference", "n_shot"]
    df = df.explode(explode_columns)

    # Convert exploded scalar columns to single-element lists for consistency
    # This ensures the format matches what Sample expects (lists)
    # Note: segmentations is already a list after explode (list of segmentations for one category)
    df = df.with_columns([
        pl.col("categories").map_elements(lambda x: [x], return_dtype=pl.List(pl.String)),
        pl.col("category_ids").map_elements(lambda x: [x], return_dtype=pl.List(pl.Int64)),
        pl.col("is_reference").map_elements(lambda x: [x], return_dtype=pl.List(pl.Boolean)),
        pl.col("n_shot").map_elements(lambda x: [x], return_dtype=pl.List(pl.Int64)),
        # segmentations is already a list, no conversion needed
    ])

    # Sort by image_id for consistency
    df = df.sort("image_id")

    return df
