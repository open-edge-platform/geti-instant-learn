# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""PerSeg dataset."""

from logging import getLogger
from pathlib import Path

import cv2
import numpy as np
from PIL import Image as PILImage

from getiprompt.datasets.dataset_base import Annotation, Dataset, DatasetIter, Image
from getiprompt.datasets.dataset_iterators import CategoryIter, IndexIter
from getiprompt.utils import color_overlay

logger = getLogger("Vision Prompt")


class PerSegAnnotation(Annotation):
    """PerSeg annotation class.

    Args:
        filename: The filename of the annotation
        category_id: The category id of the annotation

    Examples:
        >>> from getiprompt.datasets.perseg import PerSegAnnotation
        >>>
        >>> annotation = PerSegAnnotation(filename="test.png", category_id=0)
        >>> # mask = annotation.get_mask() # This would read the file
    """

    def __init__(self, filename: str, category_id: int) -> None:
        super().__init__(0, 0)
        self.category_id = category_id
        self.filename = filename

    def get_mask(self) -> np.ndarray:
        """Get the mask.

        Returns:
            Mask
        """
        pil_image = PILImage.open(self.filename).convert("RGB")
        arr = np.array(pil_image)
        return (arr[:, :, 0] > 0).astype(np.uint8)


class PerSegImage(Image):
    """PerSeg image class.

    Args:
        filename: The filename of the image

    Examples:
        >>> from getiprompt.datasets.perseg import PerSegImage
        >>>
        >>> image = PerSegImage(filename="test.jpg")
        >>> # img_array = image.get_image() # This would read the file
    """

    def __init__(self, filename: str) -> None:
        """Initializes the Image object.

        If the filename is not found it is either taken from source_filename or
        downloaded from the url (in that order).

        Args:
            filename: The filename of the image
        """
        super().__init__(0, 0)
        self.filename = filename

    def get_image(self) -> np.ndarray:
        """Get the image.

        Returns:
            Image
        """
        pil_image = PILImage.open(self.filename).convert("RGB")
        return np.array(pil_image)


class PerSegDataset(Dataset):
    """PerSeg dataset class.

    See https://arxiv.org/abs/2305.03048 for more information.

    Args:
        root_path: The path to the root directory of the perseg dataset.
        iterator_type: The iterator type
        iterator_kwargs: Keyword arguments passed to the iterator_type
        whitelist: Optional list of category names to load. If None, load all.

    Examples:
        >>> from getiprompt.datasets.perseg import PerSegDataset
        >>> dataset = PerSegDataset()
        >>> item = dataset[0]
    """

    def __init__(
        self,
        root_path: str | Path = "~/data/perseg",
        iterator_type: type[DatasetIter] = IndexIter,
        iterator_kwargs: dict | None = None,
        whitelist: list[str] | None = None,
    ) -> None:
        """This method initializes the PerSeg dataset.

        Args:
            root_path: The path to the root directory of the perseg dataset.
            iterator_type: The iterator type
            iterator_kwargs: Keyword arguments passed to the iterator_type
            whitelist: Optional list of category names to load. If None, load all.
        """
        if iterator_kwargs is None:
            iterator_kwargs = {}
        super().__init__(iterator_type=iterator_type, iterator_kwargs=iterator_kwargs)
        self._root_path = Path(root_path).expanduser()
        self._whitelist = whitelist
        Path(self._root_path).mkdir(parents=True, exist_ok=True)
        self._files = {
            "downloads_source": "https://drive.usercontent.google.com/download?id=18TbrwhZtAPY5dlaoEqkPa5h08G9Rjcio&confirm=t&export=download",
            "downloads_destination": Path(self._root_path) / "downloads" / "PerSeg.zip",
            "unzipped_destination": Path(self._root_path) / "downloads" / "data 3",
        }

        self.instance_count: dict[str, int] = {}  # name: count
        self.image_count: dict[str, int] = {}  # name: count
        self.instances_per_image: dict[str, float] = {}  # name: count per image

        # Category information
        self._category_index_to_name: list[str] = []
        self._category_name_to_index: dict[str, int] = {}

        # Images and Annotations
        self._images: list[PerSegImage] = []
        self._annotations: list[PerSegAnnotation] = []

        self._annotation_to_category: dict[
            int,
            int,
        ] = {}  # [annotation_id: category_id]
        self._category_to_annotations: dict[
            int,
            list[int],
        ] = {}  # [category_id: [annotation_id]]
        self._image_to_annotations: dict[int, int] = {}  # [image_id: annotation_id]

        # Download metadata (these are automatically cached)
        self._download_dataset()
        self._load_data()

    def _load_data(self) -> None:
        # For this dataset the image_id and annotation_id are equal to the indices
        images_folder = Path(self._files["unzipped_destination"]) / "Images"
        annotations_folder = Path(self._files["unzipped_destination"]) / "Annotations"

        all_category_paths = [
            path for path in images_folder.iterdir() if path.is_dir() and not path.name.startswith(".")
        ]
        if self._whitelist:
            logger.info(f"Applying whitelist: {self._whitelist}")
            allowed_category_paths = [path for path in all_category_paths if path.name in self._whitelist]
            if not allowed_category_paths:
                logger.warning(f"Warning: Whitelist {self._whitelist} resulted in zero categories being loaded.")
        else:
            allowed_category_paths = all_category_paths

        self._category_index_to_name = [path.name for path in allowed_category_paths]
        if not self._category_index_to_name:
            logger.warning("Warning: No categories found or remaining after whitelist filtering.")
            return  # Stop loading if no categories are left

        self._category_name_to_index = {name: index for index, name in enumerate(self._category_index_to_name)}
        self._category_to_annotations = {i: [] for i, _ in enumerate(self._category_index_to_name)}

        # Loop through only the *allowed* categories
        for cat_index, category in enumerate(self._category_index_to_name):
            images_sub_folder = images_folder / category
            if not images_sub_folder.is_dir():
                logger.warning(f"Warning: Category folder not found: {images_sub_folder}, skipping.")
                continue

            images_filenames = [name for name in images_sub_folder.iterdir() if not name.name.startswith(".")]
            if not images_filenames:
                logger.warning(f"Warning: No images found in category folder: {images_sub_folder}")
                # Initialize counts to 0 if no images are found for this whitelisted category
                self.image_count[category] = 0
                self.instance_count[category] = 0
                self.instances_per_image[category] = 0.0
                continue  # Skip to next category if no images

            for image_filename in images_filenames:
                # Fill statistics (only for categories being processed)
                self.image_count[category] = len(images_filenames)
                self.instance_count[category] = len(images_filenames)  # Assuming 1 instance per image for PerSeg
                self.instances_per_image[category] = 1.0

                # Get all paths
                image_full_path = images_sub_folder / image_filename
                # Construct mask path relative to the base annotations folder
                annotation_full_path = annotations_folder / category / (image_filename.stem + ".png")

                # Check if annotation file exists before adding
                if not annotation_full_path.exists():
                    logger.warning(f"Warning: Annotation file not found, skipping image: {annotation_full_path}")
                    # Decrement counts if skipping due to missing annotation
                    self.image_count[category] -= 1
                    self.instance_count[category] -= 1
                    continue  # Skip this image

                # Fill objects
                self._images.append(PerSegImage(str(image_full_path)))  # Pass paths as strings
                self._annotations.append(
                    PerSegAnnotation(str(annotation_full_path), cat_index),  # Pass paths as strings
                )

                # fill references for easy access
                annot_id = len(self._annotations) - 1
                image_id = len(self._images) - 1
                self._category_to_annotations[cat_index].append(annot_id)
                self._image_to_annotations[image_id] = annot_id  # Assumes one annotation per image

            # Update instances_per_image after processing all images for the category
            if self.image_count.get(category, 0) > 0:
                self.instances_per_image[category] = self.instance_count[category] / self.image_count[category]
            else:
                self.instances_per_image[category] = 0.0

    def get_root_path(self) -> str:
        """Get the root path.

        Returns:
            The root path
        """
        return self._root_path

    def get_categories(self) -> list[str]:
        """Get the categories.

        Returns:
            The categories
        """
        return self._category_index_to_name

    def category_index_to_id(self, index: int) -> int:  # noqa: PLR6301
        """Get the category id.

        Args:
            index: The index

        Returns:
            The category id
        """
        return index

    def category_id_to_name(self, cat_id: int) -> str:
        """Get the category name.

        Args:
            cat_id: The category id

        Returns:
            The category name
        """
        return self._category_index_to_name[cat_id]

    def category_name_to_id(self, name: str) -> int:
        """Get the category id.

        Args:
            name: The category name

        Returns:
            The category id
        """
        return self._category_name_to_index[name]

    def category_name_to_index(self, name: str) -> int:
        """Get the category index.

        Args:
            name: The category name

        Returns:
            The category index
        """
        return self._category_name_to_index[name]

    def get_image_filename(self, index: int) -> str:
        """Get the image filename.

        Args:
            index: The index

        Returns:
            The image filename
        """
        return self._images[index].filename

    def get_image_filename_in_category(
        self,
        category_index_or_name: int | str,
        index: int,
    ) -> str:
        """Get the image filename in a category.

        Args:
            category_index_or_name: The category index or name
            index: The index
        """
        if isinstance(category_index_or_name, int):
            category_id = category_index_or_name
        elif isinstance(category_index_or_name, str):
            category_id = self._category_name_to_index[category_index_or_name]

        image_ids = self._category_to_annotations[category_id]
        return self._images[image_ids[index]].filename

    def get_image_by_index(self, index: int) -> np.ndarray:
        """Get the image by index.

        Args:
            index: The index

        Returns:
            The image
        """
        return self._images[index].get_image()

    def get_masks_by_index(self, index: int) -> dict[int, np.ndarray]:
        """Get the masks by index.

        Args:
            index: The index

        Returns:
            The masks
        """
        return {self._annotations[index].category_id: self._annotations[index].get_mask()}

    def get_categories_of_image(self, index: int) -> list[str]:
        """This method gets all categories presents on an image.

        Args:
            index: The image index

        """
        return [self._category_index_to_name[self._annotations[index].category_id]]

    def get_images_by_category(
        self,
        category_index_or_name: int | str,
        start: int | None = None,
        end: int | None = None,
    ) -> list[np.ndarray]:
        """Get the images by category.

        Args:
            category_index_or_name: The category index or name
            start: The start index
            end: The end index
        """
        if isinstance(category_index_or_name, int):
            category_id = category_index_or_name
        elif isinstance(category_index_or_name, str):
            category_id = self._category_name_to_index[category_index_or_name]

        image_ids = self._category_to_annotations[category_id]
        image_ids = image_ids[slice(start, end)]

        return [self._images[i].get_image() for i in image_ids]

    def get_masks_by_category(
        self,
        category_index_or_name: int | str,
        start: int | None = None,
        end: int | None = None,
    ) -> list[np.ndarray]:
        """Get the masks by category.

        Args:
            category_index_or_name: The category index or name
            start: The start index
            end: The end index
        """
        if isinstance(category_index_or_name, int):
            category_id = category_index_or_name
        elif isinstance(category_index_or_name, str):
            category_id = self._category_name_to_index[category_index_or_name]

        annotation_ids = self._category_to_annotations[category_id]
        annotation_ids = annotation_ids[slice(start, end)]

        return [self._annotations[i].get_mask() for i in annotation_ids]

    def get_category_count(self) -> int:
        """Get the category count.

        Returns:
            The category count
        """
        return len(self._category_index_to_name)

    def get_image_count(self) -> int:
        """Get the image count.

        Returns:
            The image count
        """
        return len(self._images)

    def get_image_count_per_category(self, category_index_or_name: int | str) -> int:
        """Get the image count per category.

        Args:
            category_index_or_name: The category index or name

        Returns:
            The image count per category
        """
        if isinstance(category_index_or_name, int):
            category_id = category_index_or_name
        elif isinstance(category_index_or_name, str):
            category_id = self._category_name_to_index[category_index_or_name]
        else:
            msg = f"Invalid category_index_or_name: {category_index_or_name}"
            raise TypeError(msg)

        image_ids = self._category_to_annotations[category_id]
        return len(image_ids)

    def get_instance_count_per_category(self, category_index_or_name: int | str) -> int:
        """Get the instance count per category.

        Args:
            category_index_or_name: The category index or name

        Returns:
            The instance count per category
        """
        if isinstance(category_index_or_name, int):
            category_id = category_index_or_name
        elif isinstance(category_index_or_name, str):
            category_id = self._category_name_to_index[category_index_or_name]
        else:
            msg = f"Invalid category_index_or_name: {category_index_or_name}"
            raise TypeError(msg)

        cat_name = self._category_index_to_name[category_id]
        return self.instance_count[cat_name]

    def _download_dataset(self) -> None:
        """Downloads the dataset."""
        Path(self._root_path / "downloads").mkdir(parents=True, exist_ok=True)
        download_src = self._files["downloads_source"]
        download_dest = self._files["downloads_destination"]
        unzip_dest = self._files["unzipped_destination"]
        self._download(download_src, str(download_dest))
        self._unzip(str(download_dest), str(unzip_dest))
        if not unzip_dest.exists():
            msg = f"Failed to produce {unzip_dest}"
            raise RuntimeError(msg)


def test_index_iter() -> None:
    """Test the index iterator."""
    # Use default index iterator (PyTorch style)
    dataset = PerSegDataset()

    for image_index, (image, masks) in enumerate(dataset):
        # Generate and save overlays
        for category_id, mask in masks.items():
            overlay = color_overlay(image, mask)
            cat = dataset.get_categories()[category_id]
            output_folder = Path(dataset.get_root_path()) / "overlays" / cat
            orig_filename = Path(Path(dataset.get_image_filename(image_index)).name).stem
            Path.mkdir(output_folder, parents=True)
            cv2.imwrite(str(Path(output_folder / f"{orig_filename}_{cat}.jpg")), overlay)


def test_category_iter() -> None:
    """Test the category iterator."""
    # Use category iterator
    dataset = PerSegDataset(iterator_type=CategoryIter)

    for category_index, (images, masks) in enumerate(dataset):
        for image_index, (image, mask) in enumerate(zip(images, masks, strict=False)):
            # Generate and save overlays
            overlay = color_overlay(image, mask)
            cat = dataset.get_categories()[category_index]
            output_folder = Path(dataset.get_root_path()) / "overlays" / cat
            orig_filename = Path(
                Path(
                    dataset.get_image_filename_in_category(category_index, image_index),
                ).name
            ).stem
            Path.mkdir(output_folder, parents=True)
            cv2.imwrite(
                str(Path(output_folder) / f"{orig_filename}_{cat}.jpg"),
                overlay,
            )


if __name__ == "__main__":
    test_index_iter()
    test_category_iter()
