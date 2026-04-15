# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""COCO-format dataset implementation for few-shot segmentation benchmarking.

Loads datasets stored in standard COCO format (annotations/instances_*.json + images/).
Supports both SEMANTIC (merged masks per category) and INSTANCE (per-instance masks + boxes)
annotation modes, following the same conventions as LVISDataset.
"""

from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

import polars as pl
import torch
from pycocotools import mask as mask_utils
from pycocotools.coco import COCO

from instantlearn.data.base import Dataset
from instantlearn.data.lvis import LVISAnnotationMode


class COCODataset(Dataset):
    """COCO-format dataset for few-shot segmentation.

    Loads annotations from a standard COCO JSON file and images from a companion
    images directory. The first ``n_shots`` images per category are used as
    reference (few-shot support), the rest as targets.

    Supports two annotation modes:

    - **SEMANTIC**: Instances of the same category are merged into a single
      semantic mask per category per image.
    - **INSTANCE**: Each instance keeps its own mask and bounding box.

    Args:
        root: Path to dataset root containing ``annotations/`` and ``images/`` dirs.
        annotation_file: Name of the annotation JSON inside ``annotations/``.
            Defaults to ``"instances_default.json"``.
        categories: Category names to include. None means all.
        n_shots: Number of reference shots per category.
        annotation_mode: How to structure annotations.

    Examples:
        >>> dataset = COCODataset(root="./datasets/LED_COCO")
        >>> len(dataset)
        15
        >>> dataset.categories
        ['LED']
    """

    def __init__(
        self,
        root: Path | str,
        annotation_file: str = "instances_default.json",
        categories: Sequence[str] | None = None,
        n_shots: int = 1,
        annotation_mode: LVISAnnotationMode | str = LVISAnnotationMode.SEMANTIC,
    ) -> None:
        """Initialize the COCODataset."""
        super().__init__(n_shots=n_shots)

        self.root = Path(root).expanduser()
        self.annotation_file = annotation_file
        self.categories_filter = categories
        self.annotation_mode = LVISAnnotationMode(annotation_mode)

        self.df = self._load_dataframe()

    @staticmethod
    def _decode_single(segmentation: list | dict, h: int, w: int) -> torch.Tensor:
        """Decode a single COCO segmentation (RLE or polygon) to a binary mask.

        Raises:
            TypeError: If segmentation format is unknown.
        """
        if isinstance(segmentation, dict):  # RLE format
            mask = mask_utils.decode(segmentation)
        elif isinstance(segmentation, list):  # Polygon format
            rles = mask_utils.frPyObjects(segmentation, h, w)
            mask = mask_utils.decode(rles)
        else:
            msg = f"Unknown segmentation format: {type(segmentation)}"
            raise TypeError(msg)

        mask = torch.from_numpy(mask)
        if mask.ndim > 2:
            mask = torch.max(mask, dim=-1).values
        return mask.bool()

    def _load_masks(self, raw_sample: dict) -> torch.Tensor | None:
        """Decode masks from COCO segmentation format.

        Args:
            raw_sample: Dictionary from DataFrame row.

        Returns:
            Tensor with shape (N, H, W) or None.
        """
        segmentations = raw_sample.get("segmentations")
        if not segmentations:
            return None

        h, w = raw_sample.get("img_dim")

        if self.annotation_mode == LVISAnnotationMode.SEMANTIC:
            category_mask = torch.zeros((h, w), dtype=torch.bool)
            for segmentation in segmentations:
                mask = self._decode_single(segmentation, h, w)
                category_mask = category_mask | mask  # noqa: PLR6104
            return category_mask.unsqueeze(0)  # (1, H, W)

        # INSTANCE mode
        category_masks = torch.zeros((len(segmentations), h, w), dtype=torch.bool)
        for idx, segmentation in enumerate(segmentations):
            category_masks[idx] = self._decode_single(segmentation, h, w)
        return category_masks  # (num_instances, H, W)

    def _find_images_dir(self) -> Path:
        """Locate the images directory, handling nested 'default' subfolders."""
        images_dir = self.root / "images"
        # Check for a single subfolder (e.g., images/default/)
        subdirs = [d for d in images_dir.iterdir() if d.is_dir()]
        if len(subdirs) == 1:
            return subdirs[0]
        return images_dir

    def _load_dataframe(self) -> pl.DataFrame:  # noqa: C901, PLR0915
        """Load COCO annotations into a Polars DataFrame.

        Raises:
            FileNotFoundError: If an image file is not found.
            ValueError: If no annotations are found.
        """
        annotations_path = self.root / "annotations" / self.annotation_file
        coco_api = COCO(str(annotations_path))
        images_dir = self._find_images_dir()

        # Get category filtering
        all_cats = coco_api.loadCats(coco_api.getCatIds())
        cat_name_to_id = {cat["name"]: cat["id"] for cat in all_cats}

        if self.categories_filter is not None:
            valid_cat_ids = [cat_name_to_id[c] for c in self.categories_filter if c in cat_name_to_id]
        else:
            valid_cat_ids = list(cat_name_to_id.values())

        # Get annotations
        ann_ids = coco_api.getAnnIds(catIds=valid_cat_ids)
        annotations = coco_api.loadAnns(ann_ids)

        # Group by image
        image_annotations: dict[int, list] = defaultdict(list)
        for ann in annotations:
            image_annotations[ann["image_id"]].append(ann)

        # Build DataFrame rows
        samples_data = []
        category_shot_counts: dict[str, int] = defaultdict(int)
        cat_id_to_name = {cat["id"]: cat["name"] for cat in all_cats}

        for image_id, anns in image_annotations.items():
            if not anns:
                continue

            img_info = coco_api.imgs[image_id]
            file_name = img_info["file_name"]
            image_path = images_dir / file_name
            if not image_path.exists():
                # Try root-relative path
                image_path = self.root / "images" / file_name
                if not image_path.exists():
                    msg = f"Image file not found: {image_path}"
                    raise FileNotFoundError(msg)

            img_h, img_w = img_info["height"], img_info["width"]

            # Group by category
            cat_annotations: dict[str, list] = defaultdict(list)
            for ann in anns:
                cat_name = cat_id_to_name[ann["category_id"]]
                cat_annotations[cat_name].append(ann)

            categories_list = []
            category_ids_list = []
            segmentations_list = []
            bbox_list = []
            is_reference_list = []
            n_shot_list = []

            for cat_name in sorted(cat_annotations.keys()):
                cat_anns = cat_annotations[cat_name]
                cat_id = cat_anns[0]["category_id"]

                current_shot_count = category_shot_counts[cat_name]
                is_ref = current_shot_count < self.n_shots
                shot_num = current_shot_count if is_ref else -1
                if is_ref:
                    category_shot_counts[cat_name] += 1

                cat_segmentations = [ann["segmentation"] for ann in cat_anns]
                cat_bboxes = []
                for ann in cat_anns:
                    x, y, w, h = ann["bbox"]
                    cat_bboxes.append([x, y, x + w, y + h])

                if self.annotation_mode == LVISAnnotationMode.SEMANTIC:
                    categories_list.append(cat_name)
                    category_ids_list.append(cat_id)
                    segmentations_list.append(cat_segmentations)
                else:
                    categories_list.extend([cat_name] * len(cat_anns))
                    category_ids_list.extend([cat_id] * len(cat_anns))
                    segmentations_list.extend(cat_segmentations)
                    bbox_list.extend(cat_bboxes)
                is_reference_list.append(is_ref)
                n_shot_list.append(shot_num)

            samples_data.append({
                "image_id": image_id,
                "image_path": str(image_path),
                "categories": categories_list,
                "category_ids": category_ids_list,
                "segmentations": segmentations_list,
                "bboxes": bbox_list or None,
                "is_reference": is_reference_list,
                "n_shot": n_shot_list,
                "img_dim": (img_h, img_w),
            })

        if not samples_data:
            msg = f"No annotations found in {annotations_path}"
            raise ValueError(msg)

        data_frame = pl.DataFrame(samples_data)

        if self.annotation_mode == LVISAnnotationMode.SEMANTIC:
            explode_columns = ["categories", "category_ids", "segmentations", "is_reference", "n_shot"]
            data_frame = data_frame.explode(explode_columns)

            data_frame = data_frame.with_columns([
                pl.col("categories").map_elements(lambda x: [x], return_dtype=pl.List(pl.String)),
                pl.col("category_ids").map_elements(lambda x: [x], return_dtype=pl.List(pl.Int64)),
                pl.col("is_reference").map_elements(lambda x: [x], return_dtype=pl.List(pl.Boolean)),
                pl.col("n_shot").map_elements(lambda x: [x], return_dtype=pl.List(pl.Int64)),
                pl.lit(None).alias("bboxes"),
            ])

        return data_frame.sort("image_id")
