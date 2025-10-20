# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""LVIS dataset implementation for multi-instance few-shot segmentation.

This module provides the LVIS dataset implementation that supports
multi-instance images (multiple objects per image with different categories).
"""

import json
from collections import defaultdict
from pathlib import Path
from typing import Sequence

import numpy as np
import polars as pl

from getiprompt.data.datasets.base import GetiPromptDataset

__all__ = ["LVISDataset", "make_lvis_dataframe"]


class LVISDataset(GetiPromptDataset):
    """LVIS dataset class for few-shot segmentation.
    
    Dataset class for loading and processing LVIS dataset images for few-shot
    segmentation tasks. LVIS uses COCO format and supports multi-instance images
    (multiple objects with potentially different categories in one image).
    
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
        (3, 480, 640)
        >>> sample.masks.shape
        (3, 480, 640)  # 3 instances in this image
        >>> sample.categories
        ['person', 'person', 'car']  # 2 persons and 1 car
    """
    
    def __init__(
        self,
        root: Path | str = "./datasets/LVIS",
        split: str = "val",
        categories: Sequence[str] | None = None,
        n_shots: int = 1,
    ) -> None:
        super().__init__(n_shots=n_shots)
        
        self.root = Path(root)
        self.split = split
        self.categories_filter = categories
        
        # Load the DataFrame
        self.df = self._load_dataframe()
    
    def _load_masks(self, raw_sample: dict) -> np.ndarray | None:
        """Decode masks from COCO RLE format.
        
        Args:
            raw_sample: Dictionary from DataFrame row.
        
        Returns:
            np.ndarray with shape (N, H, W) where N is the number of instances,
            or None if no segmentations are available.
        """
        segmentations = raw_sample.get("segmentations")
        if not segmentations:
            return None
        
        try:
            from pycocotools import mask as mask_utils
        except ImportError:
            raise ImportError(
                "pycocotools is required for LVIS dataset. "
                "Install it with: pip install pycocotools"
            )
        
        # Get image dimensions from first segmentation (they're all the same size)
        if isinstance(segmentations[0], dict) and "size" in segmentations[0]:
            h, w = segmentations[0]["size"]
        else:
            # If size not available, we need to infer from bboxes or load the image
            # For now, raise an error
            raise ValueError("Cannot determine image size from segmentations")
        
        masks = []
        for seg in segmentations:
            if isinstance(seg, dict):  # RLE format
                mask = mask_utils.decode(seg)  # (H, W)
            elif isinstance(seg, list):  # Polygon format
                # Convert polygon to RLE then decode
                rle = mask_utils.frPyObjects([seg], h, w)
                mask = mask_utils.decode(rle[0]) if isinstance(rle, list) else mask_utils.decode(rle)
            else:
                raise ValueError(f"Unknown segmentation format: {type(seg)}")
            masks.append(mask)
        
        return np.stack(masks, axis=0)  # (N, H, W)
    
    def _load_dataframe(self) -> pl.DataFrame:
        """Load LVIS samples into Polars DataFrame."""
        annotations_file = self.root / f"lvis_v1_{self.split}.json"
        images_dir = self.root / self.split  # Assuming images are in split subdirectory
        
        return make_lvis_dataframe(
            annotations_file=annotations_file,
            images_dir=images_dir,
            categories=self.categories_filter,
            n_shots=self.n_shots,
        )


def make_lvis_dataframe(
    annotations_file: Path,
    images_dir: Path,
    categories: Sequence[str] | None = None,
    n_shots: int = 1,
) -> pl.DataFrame:
    """Create a Polars DataFrame for LVIS dataset.
    
    Args:
        annotations_file (Path): Path to LVIS annotations JSON file (COCO format).
        images_dir (Path): Directory containing the images.
        categories (Sequence[str] | None, optional): Categories to include. 
            If None, includes all. Defaults to None.
        n_shots (int, optional): Number of reference shots per category. Defaults to 1.
        
    Returns:
        pl.DataFrame: DataFrame containing sample metadata with multi-instance support.
        
    Raises:
        FileNotFoundError: If annotations file doesn't exist.
        ValueError: If no matching annotations are found.
    """
    if not annotations_file.exists():
        raise FileNotFoundError(f"Annotations file not found: {annotations_file}")
    
    # Load COCO format annotations
    with open(annotations_file) as f:
        coco_data = json.load(f)
    
    # Build mappings
    images_dict = {img["id"]: img for img in coco_data["images"]}
    categories_dict = {cat["id"]: cat["name"] for cat in coco_data["categories"]}
    
    # Filter categories if specified
    if categories is not None:
        category_name_to_id = {cat["name"]: cat["id"] for cat in coco_data["categories"]}
        valid_category_ids = {category_name_to_id[cat] for cat in categories if cat in category_name_to_id}
    else:
        valid_category_ids = None
    
    # Group annotations by image_id
    image_annotations = defaultdict(list)
    for ann in coco_data["annotations"]:
        # Skip if category filtering is enabled and this category is not in the filter
        if valid_category_ids is not None and ann["category_id"] not in valid_category_ids:
            continue
        image_annotations[ann["image_id"]].append(ann)
    
    # Build samples (one row per image with all instances)
    samples_data = []
    category_shot_counts = defaultdict(int)  # Track n_shots per category
    
    for image_id, anns in image_annotations.items():
        if not anns:  # Skip images with no annotations after filtering
            continue
        
        img_info = images_dict[image_id]
        image_path = str(images_dir / img_info["file_name"])
        
        # Collect all instances in this image
        categories_list = []
        category_ids_list = []
        annotation_ids_list = []
        bboxes_list = []
        segmentations_list = []
        is_reference_list = []
        n_shot_list = []
        
        for ann in anns:
            cat_id = ann["category_id"]
            cat_name = categories_dict[cat_id]
            
            # Determine if this is a reference sample
            current_shot_count = category_shot_counts[cat_name]
            is_ref = current_shot_count < n_shots
            shot_num = current_shot_count if is_ref else -1
            
            if is_ref:
                category_shot_counts[cat_name] += 1
            
            categories_list.append(cat_name)
            category_ids_list.append(cat_id)
            annotation_ids_list.append(ann["id"])
            bboxes_list.append(ann["bbox"])  # [x, y, w, h]
            segmentations_list.append(ann["segmentation"])  # RLE or polygon
            is_reference_list.append(is_ref)
            n_shot_list.append(shot_num)
        
        samples_data.append({
            "image_id": image_id,
            "image_path": image_path,
            "categories": categories_list,
            "category_ids": category_ids_list,
            "annotation_ids": annotation_ids_list,
            "mask_paths": None,  # LVIS doesn't use mask files
            "bboxes": bboxes_list,
            "segmentations": segmentations_list,
            "is_reference": is_reference_list,
            "n_shot": n_shot_list,
        })
    
    if not samples_data:
        raise ValueError("No valid annotations found")
    
    # Create DataFrame
    df = pl.DataFrame(samples_data)
    
    # Sort by image_id for consistency
    df = df.sort("image_id")
    
    return df
