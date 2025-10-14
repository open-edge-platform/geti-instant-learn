# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Clean LVIS Category Dataset using LVIS API."""

from enum import Enum
from logging import getLogger
from pathlib import Path

import numpy as np
import pycocotools.mask as mask_utils
import requests
import torch.utils.data
from lvis import LVIS
from PIL import Image as PILImage

from getiprompt.types import Image, Masks, Priors, Text

logger = getLogger("Geti Prompt")


class Subset(Enum):
    """Enum for subset of the dataset."""

    TRAIN = "train"
    VAL = "val"


class LVISCategoryDataset(torch.utils.data.Dataset):
    """Clean LVIS dataset for a single category using the LVIS API.

    This dataset:
    - Inherits from torch.utils.data.Dataset for PyTorch compatibility
    - Uses the official LVIS API for annotation handling
    - Focuses on a single category at a time
    - Provides clean separation between priors and prediction data
    - Works seamlessly with torch.utils.data.DataLoader

    Args:
        label: The category label to load (e.g., "cat", "dog", "cupcake")
        n_shots: Number of shots (prior examples) to use for learning
        root_path: Path to the LVIS dataset root directory
        subset: Which subset to use (TRAIN or VAL)
        max_samples: Maximum number of samples to load (-1 for all)

    Example:
        >>> # Get priors for learning
        >>> dataset = LVISCategoryDataset(label="cat", n_shots=5)
        >>> for prior_image, prior in dataset.priors:
        >>>     model.learn(reference_images=[prior_image], reference_priors=[prior])
        >>> 
        >>> # Use DataLoader for prediction batches
        >>> from torch.utils.data import DataLoader
        >>> dataloader = DataLoader(dataset, batch_size=8, shuffle=False)
        >>> for batch in dataloader:
        >>>     images, masks = batch
        >>>     predictions = model.infer(target_images=images)
    """

    def __init__(
        self,
        label: str,
        n_shots: int = 1,
        root_path: str | Path = "~/data/lvis",
        subset: Subset = Subset.TRAIN,
        max_samples: int = -1,
    ) -> None:
        """Initialize the LVIS category dataset."""
        super().__init__()
        self.label = label
        self.n_shots = n_shots
        self.subset = subset
        self._root_path = Path(root_path).expanduser()
        self._root_path.mkdir(parents=True, exist_ok=True)

        # Setup subset-specific paths
        self._setup_subset_paths()

        # Load LVIS API
        logger.info(f"Loading LVIS dataset for category '{label}' from {self._annotation_file}")
        self.lvis_api = LVIS(str(self._annotation_file))

        # Get category ID
        self.category_id = self._get_category_id()

        # Load all image IDs for this category
        img_ids = self.lvis_api.get_img_ids_from_cat_ids([self.category_id])

        # Limit samples if specified
        if max_samples > 0:
            img_ids = img_ids[:max_samples]

        # Split into priors and prediction sets
        self.prior_img_ids = img_ids[: self.n_shots]
        self.pred_img_ids = img_ids[self.n_shots :]

        logger.info(
            f"Loaded {len(img_ids)} images for '{label}' "
            f"({len(self.prior_img_ids)} priors, {len(self.pred_img_ids)} for prediction)"
        )

    def _setup_subset_paths(self) -> None:
        """Setup paths for the selected subset."""
        subset_configs = {
            Subset.TRAIN: {
                "annotation_file": self._root_path / "lvis_v1_train.json",
                "annotation_url": "https://dl.fbaipublicfiles.com/LVIS/lvis_v1_train.json.zip",
                "images_dir": self._root_path / "train2017",
                "images_url": "http://images.cocodataset.org/zips/train2017.zip",
            },
            Subset.VAL: {
                "annotation_file": self._root_path / "lvis_v1_val.json",
                "annotation_url": "https://dl.fbaipublicfiles.com/LVIS/lvis_v1_val.json.zip",
                "images_dir": self._root_path / "val2017",
                "images_url": "http://images.cocodataset.org/zips/val2017.zip",
            },
        }

        config = subset_configs[self.subset]
        self._annotation_file = config["annotation_file"]
        self._annotation_url = config["annotation_url"]
        self._images_dir = config["images_dir"]
        self._images_url = config["images_url"]

    def _get_category_id(self) -> int:
        """Get the category ID for the label.

        Returns:
            Category ID

        Raises:
            ValueError: If the category label is not found
        """
        cats = self.lvis_api.load_cats(self.lvis_api.get_cat_ids())
        label_to_id = {cat["name"]: cat["id"] for cat in cats}

        if self.label not in label_to_id:
            available = ", ".join(sorted(label_to_id.keys())[:10])
            msg = (
                f"Category '{self.label}' not found in LVIS dataset. "
                f"Available categories include: {available}..."
            )
            raise ValueError(msg)

        return label_to_id[self.label]

    def _load_image(self, img_id: int) -> np.ndarray:
        """Load an image from disk or download if needed.

        Args:
            img_id: LVIS image ID

        Returns:
            Image as numpy array (H, W, 3)
        """
        img_info = self.lvis_api.load_imgs([img_id])[0]
        img_filename = img_info["coco_url"].split("/")[-1]
        img_path = self._images_dir / img_filename

        # Download if not exists
        if not img_path.exists():
            logger.info(f"Downloading image: {img_filename}")
            self._images_dir.mkdir(parents=True, exist_ok=True)
            response = requests.get(img_info["coco_url"], timeout=60)
            response.raise_for_status()
            img_path.write_bytes(response.content)

        # Load and convert to RGB
        pil_image = PILImage.open(img_path).convert("RGB")
        return np.array(pil_image)

    def _load_mask(self, img_id: int) -> np.ndarray:
        """Load segmentation mask for this category from an image.

        Args:
            img_id: LVIS image ID

        Returns:
            Mask as numpy array (H, W) with instance IDs as values
        """
        # Get all annotations for this image and category
        ann_ids = self.lvis_api.get_ann_ids(img_ids=[img_id], cat_ids=[self.category_id])
        anns = self.lvis_api.load_anns(ann_ids)

        if not anns:
            img_info = self.lvis_api.load_imgs([img_id])[0]
            return np.zeros((img_info["height"], img_info["width"]), dtype=np.uint8)

        # Decode masks and combine with instance IDs
        img_info = self.lvis_api.load_imgs([img_id])[0]
        masks = []
        for instance_id, ann in enumerate(anns, start=1):
            # LVIS uses COCO-style RLE format
            if isinstance(ann["segmentation"], dict):
                # Already in RLE format
                rle = ann["segmentation"]
            else:
                # Polygon format, convert to RLE
                rle = mask_utils.frPyObjects(
                    ann["segmentation"], img_info["height"], img_info["width"]
                )
                rle = mask_utils.merge(rle)

            mask = mask_utils.decode(rle).astype(np.uint8)
            masks.append(mask * instance_id)

        # Combine all instance masks
        combined_mask = np.max(masks, axis=0) if masks else np.zeros_like(masks[0])
        return combined_mask

    @property
    def priors(self) -> list[tuple[Image, Priors]]:
        """Get the prior examples for learning.

        Returns:
            List of tuples (Image, Priors), one per prior image
        """
        priors_list = []

        for img_id in self.prior_img_ids:
            # Load image
            image_array = self._load_image(img_id)
            image = Image(image_array)

            # Load mask
            mask_array = self._load_mask(img_id)

            # Convert to binary mask (all instances merged)
            binary_mask = (mask_array > 0).astype(np.uint8)[:, :, None]  # (H, W, 1)

            # Create Masks object
            masks = Masks()
            masks.add(binary_mask)

            # Create Text object with label
            text = Text()
            text.add(self.label, class_id=0)

            # Create Priors object
            prior = Priors(masks=masks, text=text)
            priors_list.append((image, prior))

        return priors_list

    def __len__(self) -> int:
        """Get the number of prediction samples.

        Returns:
            Number of prediction samples (not batches)
        """
        return len(self.pred_img_ids)

    def __getitem__(self, index: int) -> tuple[Image, np.ndarray]:
        """Get a single sample by index.

        This follows the torch.utils.data.Dataset interface, returning a single
        sample. Use torch.utils.data.DataLoader for batching.

        Args:
            index: Sample index

        Returns:
            Tuple of (Image, mask) where:
                - Image: Image object containing the image data
                - mask: numpy array with ground truth mask (H, W)
        """
        if index >= len(self):
            msg = f"Index {index} out of range for dataset with {len(self)} samples"
            raise IndexError(msg)

        img_id = self.pred_img_ids[index]
        image_array = self._load_image(img_id)
        mask_array = self._load_mask(img_id)

        return Image(image_array), mask_array


def collate_fn(batch: list[tuple[Image, np.ndarray]]) -> tuple[list[Image], list[np.ndarray]]:
    """Custom collate function for DataLoader.

    Since we return custom Image objects, we need a custom collate function
    that doesn't try to stack them into tensors.

    Args:
        batch: List of (Image, mask) tuples from __getitem__

    Returns:
        Tuple of (images, masks) where:
            - images: List of Image objects
            - masks: List of numpy arrays
    """
    images = [item[0] for item in batch]
    masks = [item[1] for item in batch]
    return images, masks

