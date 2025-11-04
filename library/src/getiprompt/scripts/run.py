# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""This module contains functionality for running the model on custom data."""

import ast
import pathlib
import re
from logging import getLogger
from pathlib import Path

import numpy as np
from PIL import Image as PILImage
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn
from torchvision import tv_tensors

from getiprompt.data.base.batch import Batch
from getiprompt.data.base.sample import Sample
from getiprompt.data.utils.image import read_image
from getiprompt.models import Model
from getiprompt.types import Masks, Points
from getiprompt.utils.constants import IMAGE_EXTENSIONS
from getiprompt.utils.utils import setup_logger
from getiprompt.visualize import ExportMaskVisualization

logger = getLogger("Geti Prompt")
setup_logger()


def run_model(
    model: Model,
    target_images: str,
    reference_images: str | None = None,
    reference_prompts: str | None = None,
    reference_points_str: str | None = None,
    reference_text_prompt: str | None = None,
    output_location: str | None = None,
    batch_size: int = 5,
) -> None:
    """Loads reference data (images and prompts) and target images and runs the model.

    User can provide a reference image directory with or without a reference prompt directory.
    Reference prompt directory can contain multiple directories, each containing the prompts for a single class.
    Prompt files are expected to have the same filename as that of the reference image file.
    Prompts can also be passed in the form of a point coordinate list passed as a string.
        in the form of a list of coordinates for each image:
            Image1, Image2, Image3, ... with ImageX: [class_id:[coord_x, coord_y], ...]
            class_id: -1 is for background, 0 and up are for foreground classes.
            coord_x and coord_y are the x and y coordinates of the point.
            the order of the images is the order of the images in the reference image directory.
        For coordinates of two classes for multiple images:
            "[-1:[x1,y1], 0:[x1, y1], 0:[x2, y2], 1:[x3, y3], ...], [0:[x1, y1], 1:[x3, y3], ...], ..."
        For coordinates of one class for multiple images:
            "[-1:[x1,y1], 0:[x1, y1], 0:[x2, y2],...], [-1:[x1,y1], 0:[x1, y1], 0:[x2, y2],...], ..."

    Example:
        reference_images: "~/data/reference"
        reference_prompts: "~/data/reference/prompts"
            containing class directories with prompt files for a mask e.g. "class_1/<reference_image_filename>.png"
            or point files e.g. "class_1/<reference_image_filename>.txt"

    Example:
        reference_images: "~/data/reference"
        reference_points_str: "[-1:[50, 50], 0:[200, 320], 0:[100, 100], ...], [-1:[100, 100], 1:[100, 100], ...], ..."

    Args:
        model: The model to run.
        target_images: The directory containing all target images.
        reference_images: The directory containing all reference images.
        reference_prompts: The directory containing all reference prompt such as mask files or point files.
        reference_points_str: The string containing all reference points.
        reference_text_prompt: The string containing the text prompt.
        output_location: Custom location for the output data. If not provided, the output will be saved in the
            root of the target image directory.
        batch_size: The number of images to process in each batch.
    """
    reference_images = Path(reference_images).expanduser() if reference_images else None
    reference_prompts = Path(reference_prompts).expanduser() if reference_prompts else None
    target_images = Path(target_images).expanduser()
    output_location = (
        Path(output_location).expanduser() if output_location else pathlib.Path(target_images).parent.parent / "output"
    )

    reference_samples, class_strings = parse_reference_data(
        reference_images,
        reference_prompts,
        reference_points_str,
        reference_text_prompt,
    )
    target_images_list, _, _ = parse_image_files(target_images)
    # Convert target images to samples
    # Target samples don't need categories - they're unlabeled images for inference
    target_samples = [
        Sample(
            image=img,
            image_path=str(img.image_path) if hasattr(img, "image_path") else None,
            is_reference=[False],
            categories=None,
        )
        for img in target_images_list
    ]

    if reference_samples:
        reference_batch = Batch.collate(reference_samples)
        model.learn(reference_batch)

        # Visualize reference samples
        for sample in reference_samples:
            if sample.image_path:
                image_path = Path(sample.image_path) if isinstance(sample.image_path, str) else sample.image_path
                output_dir = str(output_location / "reference" / image_path.parent.name)
                image_name = image_path.name
            else:
                output_dir = str(output_location / "reference")
                image_name = "reference"

            # Convert masks to Masks objects
            masks_obj = None
            if sample.masks is not None:
                masks_obj = Masks()
                for instance_idx in range(sample.masks.shape[0]):
                    class_id = (
                        sample.category_ids[instance_idx]
                        if sample.category_ids is not None and instance_idx < len(sample.category_ids)
                        else 0
                    )
                    masks_obj.add(sample.masks[instance_idx], class_id=int(class_id))

            # Convert points to Points objects
            points_obj = None
            if sample.points is not None and len(sample.points) > 0:
                points_obj = Points()
                # Points are (N, 2) format, need to add with class_id
                # Convert to (N, 4) format: [x, y, score, label]
                points_array = np.array(sample.points)
                if points_array.ndim == 1:
                    points_array = points_array.reshape(1, -1)
                if points_array.shape[1] == 2:
                    # Add score and label columns
                    scores = np.ones((points_array.shape[0], 1))
                    labels = np.ones((points_array.shape[0], 1))
                    points_with_labels = np.hstack([points_array, scores, labels])
                else:
                    points_with_labels = points_array

                # Group by category_id if available
                if sample.category_ids is not None and len(sample.category_ids) == len(points_with_labels):
                    # Points might have multiple instances per class
                    for point_idx, point in enumerate(points_with_labels):
                        class_id = int(sample.category_ids[point_idx]) if point_idx < len(sample.category_ids) else 0
                        points_obj.add(point.reshape(1, -1), class_id=class_id)
                else:
                    # If no category_ids, use class_id 0 for all points
                    points_obj.add(points_with_labels, class_id=0)

            ExportMaskVisualization(output_dir)(
                images=[sample.image] if sample.image is not None else [],
                masks=[masks_obj] if masks_obj is not None else None,
                file_names=[image_name],
                points=[points_obj] if points_obj is not None else None,
                class_names=class_strings,
                show_legend=True,
            )
    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeRemainingColumn(),
        TimeElapsedColumn(),
    )
    with progress:
        task = progress.add_task("[cyan]Inference", total=len(target_samples))
        for i in range(0, len(target_samples), batch_size):
            chunk_samples = target_samples[i : i + batch_size]
            target_batch = Batch.collate(chunk_samples)
            results = model.infer(target_batch)

            chunk_images = [sample.image for sample in chunk_samples if sample.image is not None]
            chunk_names = []
            for idx, sample in enumerate(chunk_samples):
                if sample.image_path:
                    image_path = Path(sample.image_path) if isinstance(sample.image_path, str) else sample.image_path
                    chunk_names.append(image_path.name)
                else:
                    chunk_names.append(f"image_{idx}")

            ExportMaskVisualization(str(output_location / "target"))(
                images=chunk_images,
                masks=results.masks,
                file_names=chunk_names,
                points=results.used_points,
                boxes=results.used_boxes,
                class_names=class_strings,
                show_legend=True,
            )
            progress.update(task, advance=len(chunk_samples))
    msg = f"Ouput saved in {output_location}"
    logger.info(msg)


def parse_reference_prompt_from_directory(reference_prompts: Path) -> dict[str, dict[int, np.ndarray]]:
    """Parse the reference prompt from a directory.

    This function can parse a directory with class sub-directories, or a single directory with masks.
    The function returns a dictionary of image filenames to mask dictionaries.

    Args:
        reference_prompts: The root directory containing all
            reference prompt such as mask files or point files.

    Returns:
        A dictionary of image filenames to mask dictionaries (class_id -> mask array).
    """
    masks_map: dict[str, dict[int, np.ndarray]] = {}
    image_suffixes = {f".{ext.strip('*.')}" for ext in IMAGE_EXTENSIONS}
    prompt_globs = [*IMAGE_EXTENSIONS, "*.txt", "*.json"]

    dirs_to_scan = [d for d in reference_prompts.iterdir() if d.is_dir()]
    if not dirs_to_scan:
        # No subdirectories found, so scan the root directory.
        dirs_to_scan = [reference_prompts]

    for class_id, directory in enumerate(dirs_to_scan):
        prompt_files = [p for ext in prompt_globs for p in directory.glob(ext)]

        for prompt_file in prompt_files:
            image_filename = str(class_id) + "_" + prompt_file.stem
            if image_filename not in masks_map:
                masks_map[image_filename] = {}

            if prompt_file.suffix.lower() in image_suffixes:
                pil_image = PILImage.open(prompt_file).convert("L")
                mask_data = np.array(pil_image)
                masks_map[image_filename][class_id] = mask_data

    return masks_map


def parse_reference_str_prompt(reference_points_str: str) -> list[dict[int, np.ndarray]]:
    """Parse the reference prompt string into a list of point dictionaries.

    This function can parse a string with multiple images and multiple classes.
    class_id -1 is for background, 0 and up are for foreground classes.
    coord_x and coord_y are the x and y coordinates of the point.
    the order of the images is the order of the images in the reference image directory.

    The string is expected to be in the form of:
        "[-1:[x1,y1], 0:[x1, y1], 0:[x2, y2], 1:[x3, y3], ...], [0:[x1, y1], 1:[x3, y3], ...], ..."

    Args:
        reference_points_str: The string containing all reference points.

    Returns:
        A list of point dictionaries, where each dict maps class_id to points array.
    """
    if not reference_points_str:
        return []

    points_list = []
    wrapped_str = f"[{reference_points_str}]"
    tuple_str = re.sub(r"(-?\d+):(\[[^\]]+\])", r"(\1, \2)", wrapped_str)
    image_prompts = ast.literal_eval(tuple_str)

    if image_prompts and isinstance(image_prompts[0], tuple):
        image_prompts = [image_prompts]

    for image_prompt_list in image_prompts:
        points_by_class: dict[int, list[list[float]]] = {}
        for class_id, coord in image_prompt_list:
            points_by_class.setdefault(class_id, []).append(coord)

        background_points = (
            np.array([[p[0], p[1], 0, 0] for p in points_by_class.get(-1, [])])
            if points_by_class.get(-1)
            else np.array([]).reshape(0, 4)
        )

        foreground_classes = sorted([k for k in points_by_class if k >= 0])
        points_dict: dict[int, np.ndarray] = {}

        for class_id in foreground_classes:
            class_points = np.array([[p[0], p[1], 1, 1] for p in points_by_class[class_id]])
            # Add background points to each foreground class
            if len(background_points) > 0:
                class_points = (
                    np.vstack([class_points, background_points]) if len(class_points) > 0 else background_points
                )
            points_dict[class_id] = class_points

        points_list.append(points_dict)
    return points_list


def parse_reference_data(
    reference_image_root: str | None = None,
    reference_prompt_root: str | None = None,
    reference_points_str: str | None = None,
    reference_text_prompt: str | None = None,
) -> tuple[list[Sample], list[str]]:
    """Parse the reference data into Sample objects.

    Args:
        reference_image_root: The root directory of the reference data.
        reference_prompt_root: The root directory containing all reference prompt such as mask files or point files.
        reference_points_str: The string containing all reference points.
        reference_text_prompt: The string containing the text prompt (comma or dot separated categories).

    Raises:
        ValueError: reference_images must be provided with reference_prompts

    Returns:
        A tuple of list of Sample objects and class strings.
    """
    class_strings = [""]
    reference_samples: list[Sample] = []

    if reference_image_root:
        reference_images, class_strings, class_ids = parse_image_files(reference_image_root)

        # Convert images to samples
        masks_map: dict[str, dict[int, np.ndarray]] = {}
        if reference_prompt_root is not None:
            if not reference_image_root:
                msg = "reference_images must be provided with reference_prompts"
                raise ValueError(msg)
            masks_map = parse_reference_prompt_from_directory(reference_prompt_root)

        for class_id, image in zip(class_ids, reference_images, strict=False):
            image_filename = (
                str(class_id) + "_" + image.image_path.stem if hasattr(image, "image_path") else f"image_{class_id}"
            )
            masks_for_image = masks_map.get(image_filename, {})

            # Combine all masks for this image
            mask_list = []
            category_list = []
            category_id_list = []
            is_reference_list = []

            for mask_class_id, mask_data in masks_for_image.items():
                mask_list.append(mask_data)
                category_list.append(class_strings[mask_class_id] if mask_class_id < len(class_strings) else "")
                category_id_list.append(mask_class_id)
                is_reference_list.append(True)

            # If no masks, create a sample with just the image and category info
            if not mask_list:
                mask_list = None
                category_list = [class_strings[class_id]] if class_id < len(class_strings) else [""]
                category_id_list = [class_id]
                is_reference_list = [True]
            else:
                # Stack masks into (N, H, W) format
                mask_list = np.stack(mask_list, axis=0)

            sample = Sample(
                image=image,
                image_path=str(image.image_path) if hasattr(image, "image_path") else None,
                masks=mask_list,
                categories=category_list,
                category_ids=np.array(category_id_list, dtype=np.int32),
                is_reference=is_reference_list,
                n_shot=[0] * len(is_reference_list),
            )
            reference_samples.append(sample)

    # Handle point prompts from string
    if reference_points_str is not None:
        points_list = parse_reference_str_prompt(reference_points_str)
        # Create samples for point-based prompts (requires images)
        if reference_image_root:
            for i, points_dict in enumerate(points_list):
                if i < len(reference_samples):
                    # Add points to existing sample
                    # Combine all points from all classes
                    all_points = []
                    all_categories = []
                    all_category_ids = []
                    all_is_ref = []
                    for class_id, points_array in points_dict.items():
                        if len(points_array) > 0:
                            all_points.append(points_array[:, :2])  # Just x, y coordinates
                            all_categories.append(class_strings[class_id] if class_id < len(class_strings) else "")
                            all_category_ids.append(class_id)
                            all_is_ref.append(True)
                    if all_points:
                        # Stack points: (N, 2)
                        points_array = np.vstack(all_points)
                        reference_samples[i].points = points_array
                        if not reference_samples[i].categories:
                            reference_samples[i].categories = all_categories
                            reference_samples[i].category_ids = np.array(all_category_ids, dtype=np.int32)
                            reference_samples[i].is_reference = all_is_ref

    # Handle text prompt
    if reference_text_prompt is not None:
        # Split text based on dots and commas
        split_text = [t.strip() for t in re.split(r"[.,]", reference_text_prompt) if t.strip()]
        class_strings = split_text

        # Create a sample with text categories (for DinoTxt model)
        # If we have images, use the first one; otherwise create a dummy sample
        if reference_samples:
            # Update categories in existing samples
            for sample in reference_samples:
                if sample.categories is None:
                    sample.categories = split_text
                    sample.category_ids = np.array(list(range(len(split_text))), dtype=np.int32)
                    sample.is_reference = [True] * len(split_text)
        else:
            # Create a minimal sample for text-only prompts
            sample = Sample(
                image=None,
                image_path=None,
                categories=split_text,
                category_ids=np.array(list(range(len(split_text))), dtype=np.int32),
                is_reference=[True] * len(split_text),
                n_shot=[0] * len(split_text),
            )
            reference_samples.append(sample)

    return reference_samples, class_strings


def parse_image_files(root_dir: str) -> tuple[list[tv_tensors.Image], list[str], list[int]]:
    """Parse the image files from a directory.

    Args:
        root_dir: The root directory of the image files.

    Returns:
        A tuple of:
        - A list of images (tv_tensors.Image objects with image_path attribute).
        - A list of class strings.
        - A list of class IDs.
    """
    root_dir = pathlib.Path(root_dir)
    class_dirs = [d for d in root_dir.iterdir() if d.is_dir()]
    class_ids = []

    image_files = []
    if class_dirs:
        # Root directory contains class directories
        for class_id, class_dir in enumerate(class_dirs):
            for ext in IMAGE_EXTENSIONS:
                list_of_images = list(class_dir.glob(ext))
                if list_of_images:
                    for image_file in list_of_images:
                        image_files.append(image_file)
                        class_ids.append(class_id)
        class_names = [class_dir.name for class_dir in class_dirs]
    else:
        # Root directory contains images
        for ext in IMAGE_EXTENSIONS:
            image_files.extend(root_dir.glob(ext))
        class_names = [""]
        class_ids = [0] * len(image_files)

    # Read images and attach image_path
    images = []
    for image_file in image_files:
        img = read_image(image_file)
        # Attach image_path to the image object
        if not hasattr(img, "image_path"):
            img.image_path = image_file
        images.append(img)

    return images, class_names, class_ids
