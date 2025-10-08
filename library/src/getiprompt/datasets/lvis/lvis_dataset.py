# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""LVIS dataset."""

import json
import pickle  # noqa: S403
import shutil
from collections import OrderedDict
from logging import getLogger
from pathlib import Path

import numpy as np
import pycocotools.mask as mask_utils
import requests
from PIL import Image as PILImage

from getiprompt.datasets.dataset_base import Dataset, DatasetIter, Image
from getiprompt.datasets.dataset_iterators import IndexIter

logger = getLogger("Geti Prompt")


def segment_to_mask(segment: list[float], height: int, width: int) -> np.ndarray:
    """This method converts a segment to a mask using RLE.

    Args:
        segment: The segment to be converted
        height: height of the image
        width: width of the image

    Returns:
        Decoded mask
    """
    if isinstance(segment, list):
        # Merge all parts
        rle_segments = mask_utils.frPyObjects(segment, height, width)
        rle_segment = mask_utils.merge(rle_segments)
    elif isinstance(segment["counts"], list):
        # uncompressed
        rle_segment = mask_utils.frPyObjects(segment, height, width)
    else:
        # regular RLE
        rle_segment = segment
    return mask_utils.decode(rle_segment)


class LVISAnnotation:
    """This class represents an annotation for the LVIS dataset.

    Args:
        height: The height of the image
        width: The width of the image
        segments: The segments of the annotation
        category_id: The category id of the annotation

    Examples:
        >>> from getiprompt.datasets.lvis import LVISAnnotation
        >>>
        >>> annotation = LVISAnnotation(height=10, width=10, segments=[], category_id=0)
        >>> # mask = annotation.get_mask()
    """

    def __init__(
        self,
        segments: list[float],
        category_id: int,
    ) -> None:
        self.segments = segments
        self.category_id = category_id

    def get_mask(self) -> np.ndarray:
        """Get the mask for the annotation.

        Returns:
            The mask for the annotation
        """
        return segment_to_mask(self.segments, self.height, self.width)


class LVISImage(Image):
    """This class represents an image for the LVIS dataset.

    Args:
        filename: The filename of the image
        height: The height of the image
        width: The width of the image

    Examples:
        >>> from getiprompt.datasets.lvis import LVISImage
        >>>
        >>> image = LVISImage(filename="test.jpg", height=10, width=10)
        >>> # img_array = image.get_image() # This would read the file
    """

    def __init__(
        self,
        filename: str,
        height: int,
        width: int,
        source_url: str | None = None,
        source_filename: str | None = None,
        copy_file: bool = False,
    ) -> None:
        """Initializes the Image object.

        If the filename is not found it is either taken from source_filename or
        downloaded from the url (in that order).

        Args:
            filename: The filename of the image
            height: The height of the image
            width: The width of the image
            source_url: The url of the image from which the image can be downloaded
            source_filename: The filename where the image can be found
            copy_file: If True this will copy the file from source_filename to filename
        """
        super().__init__(height, width)
        self.source_url = source_url
        self.source_filename = source_filename
        self.filename = filename
        self.copy_file = copy_file

    def get_image(self) -> np.ndarray:
        """Get the image.

        Returns:
            The image
        """
        folder = Path(self.filename).parent
        folder.mkdir(parents=True, exist_ok=True)

        image_filename = self.filename

        if self.source_filename is not None:
            if self.copy_file:
                if Path(self.source_filename).exists() and not Path(self.filename).exists():
                    logger.info(f"Copy {self.source_filename} to {self.filename}")
                    shutil.copyfile(self.source_filename, self.filename)
            else:
                image_filename = self.source_filename

        if not Path(image_filename).exists():
            logger.info(f"Downloading {self.source_url}")
            img_data = requests.get(self.source_url, timeout=60).content
            Path(image_filename).parent.mkdir(parents=True, exist_ok=True)
            with Path(image_filename).open("wb") as handler:
                handler.write(img_data)

        pil_image = PILImage.open(image_filename).convert("RGB")
        return np.array(pil_image)


class LVISDataset(Dataset):
    """This class represents the LVIS dataset.

    Args:
        root_path: The path to the root directory of the LVIS dataset.
        whitelist: The classes that are selected, if empty, all categories are used
        download_full_dataset: If True download the full dataset otherwise,
                               each image is downloaded on demand.
        copy_files: If the full dataset is download then copy_files will copy files from the
                    COCO dataset to the LVIS dataset folders.
                    If copy_files is True, then after copying, download_full_dataset can be set to false.
        iterator_kwargs: Keyword arguments passed to the iterator_type
    """

    def __init__(
        self,
        root_path: str | Path = "~/data/lvis",
        whitelist: str | list[str] | None = None,
        name: str = "training",
        iterator_type: type[DatasetIter] = IndexIter,
        download_full_dataset: bool = True,
        copy_files: bool = False,
        iterator_kwargs: dict | None = None,
    ) -> None:
        if iterator_kwargs is None:
            iterator_kwargs = {}
        super().__init__(iterator_type=iterator_type, iterator_kwargs=iterator_kwargs)
        self._root_path = Path(root_path).expanduser()
        Path(self._root_path).mkdir(parents=True, exist_ok=True)

        self._subset_files = {
            "sources": {  # original sources of the json annotations.
                "training": "https://dl.fbaipublicfiles.com/LVIS/lvis_v1_train.json.zip",
                "validation": "https://dl.fbaipublicfiles.com/LVIS/lvis_v1_val.json.zip",
            },
            "files": {  # extracted file location of the annotations.
                "training": Path(self._root_path) / "lvis_v1_train.json",
                "validation": Path(self._root_path) / "lvis_v1_val.json",
            },
            "downloads": {  # downloads of the full dataset.
                "training": "http://images.cocodataset.org/zips/train2017.zip",
                "validation": "http://images.cocodataset.org/zips/val2017.zip",
            },
            "source_folders": {  # folders where the downloaded files are extracted.
                "training": Path(self._root_path) / "downloads" / "train2017",
                "validation": Path(self._root_path) / "downloads" / "val2017",
            },
        }

        if whitelist is None:
            self._whitelist = []
        elif isinstance(whitelist, str):
            self._whitelist = [whitelist]
        else:
            self._whitelist = whitelist
        self._name = name  # training or validation

        # Category information
        self._category_index_to_id: list[int] = []  # only white listed categories appear here
        self._category_id_to_name: dict[int, str] = {}
        self._category_name_to_id: dict[str, int] = {}

        self.instance_count: dict[str, int] = {}  # name: count
        self.image_count: dict[str, int] = {}  # name: count
        self.instances_per_image: dict[str, float] = {}  # name: count per image

        # Images and Annotations
        self._image_index_to_id: dict[str, list[int]] = {}  # subset_name: [image_id]
        self._images: dict[str, dict[int, LVISImage]] = {}  # subset_name: [image_id: image]
        self._annotations: dict[str, dict[int, LVISAnnotation]] = {}  # subset_name: [annotation_id, image]
        self._annotation_to_image: dict[str, dict[int, int]] = {}  # subset_name: [annotation_id: image_id]
        self._annotation_to_category: dict[str, dict[int, int]] = {}  # subset_name: [annotation_id: category_id]
        self._category_to_annotations: dict[
            str, dict[int, list[int]]
        ] = {}  # subset_name: [category_id: [annotation_id]]
        self._image_to_annotations: dict[str, dict[int, list[int]]] = {}  # subset_name: [image_id: [annotation_id]]

        # Download metadata (these are automatically cached)
        self._download_metadata()
        if download_full_dataset:
            self._download_images()
        self._copy_files = copy_files

        # Check if cache needs to be invalidated (delete any of these files to invalidate cache)
        self._cache_check_file = Path(self._root_path) / "cache_check.bin"
        self._cached_metadata = Path(self._root_path) / "metadata.bin"
        self._cached_data = Path(self._root_path) / "data.bin"
        valid = self._check_cache()

        # Load metadata and data
        if valid:
            logger.debug(
                f"Using cached {self._cached_metadata} and {self._cached_data}",
            )
            self._load_metadata(self._cached_metadata)
            self._load_data(self._cached_data)
        else:
            logger.info("Cache files have been invalidated, data is reloaded")
            categories_info, images_info, annotations_info = self._get_metadata()
            self._set_metadata(categories_info)
            self._set_data(images_info, annotations_info)
            self._save_metadata(self._cached_metadata)
            self._save_data(self._cached_data)
        if len(self._whitelist) == 0:  # Populate whitelist from all categories
            self._whitelist = list(self._category_name_to_id.keys())

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
        return [self._category_id_to_name[cid] for cid in self._category_index_to_id]

    def category_index_to_id(self, index: int) -> int:
        """Get the category id.

        Args:
            index: The index

        Returns:
            The category id
        """
        return self._category_index_to_id[index]

    def category_id_to_name(self, cat_id: int) -> str:
        """Get the category name.

        Args:
            cat_id: The category id

        Returns:
            The category name
        """
        return self._category_id_to_name[cat_id]

    def category_name_to_id(self, name: str) -> int:
        """Get the category id.

        Args:
            name: The category name

        Returns:
            The category id
        """
        return self._category_name_to_id[name]

    def category_name_to_index(self, name: str) -> int:
        """Get the category index.

        Args:
            name: The category name

        Returns:
            The category index
        """
        cat_id = self._category_name_to_id[name]
        return self._category_index_to_id.index(cat_id)

    def get_image_filename(self, index: int) -> str:
        """Get the image filename.

        Args:
            index: The index

        Returns:
            The image filename
        """
        return self._images[self._name][self._image_index_to_id[self._name][index]].filename

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
            category_id = self._category_index_to_id[category_index_or_name]
        elif isinstance(category_index_or_name, str):
            category_id = self._category_name_to_id[category_index_or_name]

        image_ids = []
        annotations = self._category_to_annotations[self._name][category_id]
        for annotation_id in annotations:
            image_id = self._annotation_to_image[self._name][annotation_id]
            image_ids.append(image_id)

        image_ids = list(OrderedDict.fromkeys(image_ids))  # preserves order
        return self._images[self._name][image_ids[index]].filename

    def set_name(self, name: str) -> None:
        """Set the name.

        Args:
            name: The name
        """
        self._name = name

    def get_image_by_index(self, index: int) -> np.ndarray:
        """Get the image by index.

        Args:
            index: The index

        Returns:
            The image
        """
        image_id = self._image_index_to_id[self._name][index]
        return self._images[self._name][image_id].get_image()

    def get_masks_by_index(self, index: int) -> dict[int, np.ndarray]:
        """Get the masks by index.

        Args:
            index: The index

        Returns:
            The masks
        """
        image_id = self._image_index_to_id[self._name][index]
        annotation_ids = self._image_to_annotations[self._name][image_id]
        masks = {}

        # Merge all masks from the same class and set each pixel value to the instance_id
        for annotation_id in annotation_ids:
            annot = self._annotations[self._name][annotation_id]
            if annot.category_id not in masks:
                masks[annot.category_id] = [annot.get_mask().astype(int)]
            else:
                instance_id = len(masks[annot.category_id]) + 1
                masks[annot.category_id].append(
                    annot.get_mask().astype(int) * instance_id,
                )

        for category_id in masks:
            # Merge all instances into one mask
            masks[category_id] = np.max(masks[category_id], axis=0)

        return masks

    def get_categories_of_image(self, index: int) -> list[str]:
        """This method gets all categories presents on an image.

        Args:
            index: The image index

        """
        cats = []
        image_id = self._image_index_to_id[self._name][index]
        annotation_ids = self._image_to_annotations[self._name][image_id]
        for annotation_id in annotation_ids:
            annot = self._annotations[self._name][annotation_id]
            name = self._category_id_to_name[annot.category_id]
            cats.append(name)
        return cats

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
            category_id = self._category_index_to_id[category_index_or_name]
        elif isinstance(category_index_or_name, str):
            category_id = self._category_name_to_id[category_index_or_name]

        image_ids = []
        annotations = self._category_to_annotations[self._name][category_id]
        image_ids = [self._annotation_to_image[self._name][annotation_id] for annotation_id in annotations]
        image_ids = list(
            OrderedDict.fromkeys(image_ids),
        )  # remove redundant, preserves same order as get_masks_by_category
        image_ids = image_ids[slice(start, end)]

        return [self._images[self._name][i].get_image() for i in image_ids]

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
            category_id = self._category_index_to_id[category_index_or_name]
        elif isinstance(category_index_or_name, str):
            category_id = self._category_name_to_id[category_index_or_name]

        annotations = self._category_to_annotations[self._name][category_id]
        image_ids = [self._annotation_to_image[self._name][annotation_id] for annotation_id in annotations]
        image_ids = list(
            dict.fromkeys(image_ids),
        )  # remove redundant, preserves same order as get_images_by_category
        image_ids = image_ids[slice(start, end)]

        all_masks = []
        for image_id in image_ids:
            annotation_ids = self._image_to_annotations[self._name][image_id]
            category_ids = [self._annotation_to_category[self._name][a_id] for a_id in annotation_ids]
            # Only keep image annotations if the category matches
            annotation_ids = [
                a_id for (a_id, c_id) in zip(annotation_ids, category_ids, strict=False) if c_id == category_id
            ]
            masks = []
            for instance_id, annotation_id in enumerate(annotation_ids):
                annot = self._annotations[self._name][annotation_id]
                masks.append(annot.get_mask().astype(int) * (instance_id + 1))
            all_masks.append(np.max(masks, axis=0))

        return all_masks

    def get_category_count(self) -> int:
        """Get the category count.

        Returns:
            The category count
        """
        return len(self._category_index_to_id)

    def get_image_count(self) -> int:
        """Get the image count.

        Returns:
            The image count
        """
        return len(self._image_index_to_id[self._name])

    def get_image_count_per_category(self, category_index_or_name: int | str) -> int:
        """Get the image count per category.

        Args:
            category_index_or_name: The category index or name

        Returns:
            The image count per category
        """
        if isinstance(category_index_or_name, int):
            category_id = self._category_index_to_id[category_index_or_name]
        elif isinstance(category_index_or_name, str):
            category_id = self._category_name_to_id[category_index_or_name]

        annotations = self._category_to_annotations[self._name][category_id]
        image_ids = [self._annotation_to_image[self._name][annotation_id] for annotation_id in annotations]
        image_ids = list(dict.fromkeys(image_ids))  # remove redundant
        return len(image_ids)

    def get_instance_count_per_category(self, category_index_or_name: int | str) -> int:
        """Get the instance count per category.

        Args:
            category_index_or_name: The category index or name

        Returns:
            The instance count per category
        """
        if isinstance(category_index_or_name, int):
            category_id = self._category_index_to_id[category_index_or_name]
        elif isinstance(category_index_or_name, str):
            category_id = self._category_name_to_id[category_index_or_name]

        cat_name = self._category_id_to_name[category_id]
        return self.instance_count[cat_name]

    def _set_metadata(self, categories_info: list[dict]) -> None:
        """Creates statistics about the dataset.

        Args:
            categories_info: The categories info
        """
        self._category_id_to_name = {d["id"]: d["name"] for d in categories_info}
        self._category_name_to_id = {d["name"]: d["id"] for d in categories_info}
        self.image_count = {d["name"]: d["image_count"] for d in categories_info}
        self.instance_count = {d["name"]: d["instance_count"] for d in categories_info}
        self.instances_per_image = {
            c["name"]: c["instance_count"] / c["image_count"] for c in categories_info if c["image_count"] > 0
        }

        if len(self._whitelist) == 0:
            self._whitelist = list(self._category_name_to_id.keys())
        self._category_index_to_id = [self._category_name_to_id[cn] for cn in self._whitelist]

    def _set_data(self, images_info: dict[str, list[dict]], annotations_info: dict[str, list[dict]]) -> None:
        """Reads through the dictionaries of the LVIS dataset and creates data containers.

        Args:
            images_info: The images info
            annotations_info: The annotations info
        """
        for name in images_info:
            self._annotations[name] = {}
            self._annotation_to_image[name] = {}
            self._annotation_to_category[name] = {}
            self._image_to_annotations[name] = {}
            self._category_to_annotations[name] = {}
            for annotation_info in annotations_info[name]:
                if self._category_id_to_name[annotation_info["category_id"]] in self._whitelist:
                    image_id, annotation_id, category_id = (
                        annotation_info["image_id"],
                        annotation_info["id"],
                        annotation_info["category_id"],
                    )
                    a = LVISAnnotation(
                        segments=annotation_info["segmentation"],
                        category_id=category_id,
                    )
                    self._annotations[name][annotation_info["id"]] = a

                    # Create mapping between images <-> annotations <-> categories
                    self._annotation_to_image[name][annotation_id] = image_id
                    self._annotation_to_category[name][annotation_id] = category_id

                    if category_id not in self._category_to_annotations[name]:
                        self._category_to_annotations[name][category_id] = [
                            annotation_id,
                        ]
                    else:
                        self._category_to_annotations[name][category_id].append(
                            annotation_id,
                        )

                    if image_id not in self._image_to_annotations[name]:
                        self._image_to_annotations[name][image_id] = [annotation_id]
                    else:
                        self._image_to_annotations[name][image_id].append(annotation_id)
            self._images[name] = {}
            self._image_index_to_id[name] = []
            for image_info in images_info[name]:
                image_id = image_info["id"]
                if image_id in self._image_to_annotations[name]:
                    coco_subset = Path(image_info["coco_url"]).parent.name
                    base_name = Path(image_info["coco_url"]).name
                    output_filename = Path(self._root_path, name, base_name)
                    parent_folder = self._subset_files["source_folders"][name].parent
                    base_folder = parent_folder / coco_subset
                    source_filename = base_folder / base_name
                    i = LVISImage(
                        str(output_filename),
                        image_info["height"],
                        image_info["width"],
                        source_url=image_info["coco_url"],
                        source_filename=str(source_filename),
                        copy_file=self._copy_files,
                    )
                    for annotation_id in self._image_to_annotations[name][image_id]:
                        annotation = self._annotations[name][annotation_id]
                        annotation.height = image_info["height"]
                        annotation.width = image_info["width"]
                    self._images[name][image_info["id"]] = i

                    # this is used to iterate through the images in order
                    self._image_index_to_id[name].append(image_id)

    def _download_images(self) -> None:
        """Downloads the COCO datasets."""
        Path(self._root_path / "downloads").mkdir(parents=True, exist_ok=True)

        for name, source in self._subset_files["downloads"].items():
            destination = self._subset_files["source_folders"][name]

            dst = Path(self._root_path / "downloads" / Path(source).name)
            if not destination.exists():
                self._download(source, dst)
            if dst.suffix == ".zip":
                self._unzip(dst, destination)
            else:
                destination = dst
            if not destination.exists():
                msg = f"Failed to produce {destination}"
                raise RuntimeError(msg)

    def _download_metadata(self) -> None:
        """Downloads the LVIS dataset metadata."""
        for name, source in self._subset_files["sources"].items():
            destination = Path(self._root_path, self._subset_files["files"][name])
            dst = Path(self._root_path, Path(source).name)
            self._download(source, dst)
            if dst.suffix == ".zip":
                self._unzip(dst, destination)
            else:
                destination = dst
            if not destination.exists():
                msg = f"Failed to produce {destination}"
                raise RuntimeError(msg)

    def _get_metadata(self) -> tuple[list[dict], dict[str, dict], dict[str, dict]]:
        """Extract the relevant metadata from the LVIS metadata."""
        images_info: dict[str, dict] = {}
        annotations_info: dict[str, dict] = {}
        categories_info = None
        for name, filename in self._subset_files["files"].items():
            with Path(filename).open(encoding="utf-8") as f:
                data = json.load(f)
                if categories_info is None:
                    categories_info = data["categories"]
                images_info[name] = data["images"]
                annotations_info[name] = data["annotations"]
        return categories_info, images_info, annotations_info

    def _save_metadata(self, filename: Path) -> None:
        """Saves the metadata in a cache file.

        Args:
            filename: The filename of the metadata
        """
        with Path(filename).open("wb") as f:
            data = {
                "category_id_to_name": self._category_id_to_name,
                "category_name_to_id": self._category_name_to_id,
                "instance_count": self.instance_count,
                "image_count": self.image_count,
                "instances_per_image": self.instances_per_image,
                "category_index_to_id": self._category_index_to_id,
            }
            pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

    def _load_metadata(self, filename: Path) -> None:
        """Load the metadata from a cache file.

        Args:
            filename: The filename of the metadata
        """
        try:
            with Path(filename).open("rb") as f:
                data = pickle.load(f)  # noqa: S301
                self._category_id_to_name = data["category_id_to_name"]
                self._category_name_to_id = data["category_name_to_id"]
                self.instance_count = data["instance_count"]
                self.image_count = data["image_count"]
                self.instances_per_image = data["instances_per_image"]
                self._category_index_to_id = data["category_index_to_id"]
        except (ModuleNotFoundError, ImportError, KeyError, EOFError):
            # Cache file is corrupted or contains old module references
            # Delete the cache file and let it be regenerated
            if Path(filename).exists():
                Path(filename).unlink()
            raise

    def _load_data(self, filename: Path) -> None:
        """Load the data from a cache file.

        Args:
            filename: The filename of the data
        """
        try:
            with Path(filename).open("rb") as f:
                data = pickle.load(f)  # noqa: S301
                self._image_index_to_id = data["image_index_to_id"]
                self._images = data["images"]
                self._annotations = data["annotations"]
                self._annotation_to_image = data["annotation_to_image"]
                self._annotation_to_category = data["annotation_to_category"]
                self._image_to_annotations = data["image_to_annotations"]
                self._category_to_annotations = data["category_to_annotations"]
        except (ModuleNotFoundError, ImportError, KeyError, EOFError):
            # Cache file is corrupted or contains old module references
            # Delete the cache file and let it be regenerated
            if Path(filename).exists():
                Path(filename).unlink()
            raise

    def _save_data(self, filename: Path) -> None:
        """Saves the data in a cache file.

        Args:
            filename: The filename of the data
        """
        with Path(filename).open("wb") as f:
            data = {
                "image_index_to_id": self._image_index_to_id,
                "images": self._images,
                "annotations": self._annotations,
                "annotation_to_image": self._annotation_to_image,
                "annotation_to_category": self._annotation_to_category,
                "image_to_annotations": self._image_to_annotations,
                "category_to_annotations": self._category_to_annotations,
            }

            pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

    def _check_cache(self) -> bool:
        """Check if a list has been changed to determine when to invalidate caches.

        Also checks if the cached files exist.

        Returns:
            True if the cache is valid, False otherwise
        """
        valid = True

        # Check if whitelist is the same
        if Path(self._cache_check_file).exists():
            identical = True
            with Path(self._cache_check_file).open("rb") as f:
                data = pickle.load(f)  # noqa: S301
                identical = identical and data["whitelist"] == self._whitelist
        else:
            identical = False

        with Path(self._cache_check_file).open("wb") as f:
            data = {"whitelist": self._whitelist}
            pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

        valid = valid and identical

        # Check if cached files exist
        valid = valid and Path(self._cached_metadata).exists() and Path(self._cached_data).exists()

        if not valid:
            if Path(self._cached_metadata).exists():
                Path(self._cached_metadata).unlink()
            if Path(self._cached_data).exists():
                Path(self._cached_data).unlink()

        return valid
