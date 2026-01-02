# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""This module contains functionality for running the model on custom data."""

import re
from logging import getLogger
from pathlib import Path

import numpy as np
import polars as pl
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from getiprompt.data.base.batch import Batch
from getiprompt.data.base.sample import Sample
from getiprompt.data.folder.dataset import FolderDataset
from getiprompt.models import GroundedSAM, Model
from getiprompt.utils.utils import setup_logger
from getiprompt.visualizer import Visualizer

logger = getLogger("Geti Prompt")
setup_logger()


def run_model(
    model: Model,
    data_root: str | None = None,
    text_prompt: str | None = None,
    output_location: str | None = None,
    batch_size: int = 5,
    n_shots: int = 1,
) -> None:
    """Loads data and runs the model.

    For Grounding models (e.g., GroundedSAM):
    - Only requires images from data_root (no masks needed)
    - Requires text_prompt with comma/dot-separated categories
    - Folder structure: data_root/images/{category}/*.jpg (masks/ directory not needed)

    For other models (e.g., Matcher):
    - Requires FolderDataset structure with both images and masks
    - Folder structure: data_root/images/{category}/*.jpg and data_root/masks/{category}/*.png
    - Each category must have at least (n_shots + 1) images and masks
    - First n_shots images per category are used as reference
    - Remaining images are used as target

    Args:
        model: The model to run (must be explicitly set).
        data_root: The root directory containing the dataset.
            For Grounding models: Only images/ directory needed.
            For other models: Both images/ and masks/ directories required.
        text_prompt: The string containing the text prompt (comma/dot-separated categories).
            Required for Grounding models. Optional for other models.
        output_location: Custom location for the output data. If not provided, saved in data_root/output.
        batch_size: The number of images to process in each batch.
        n_shots: Number of reference shots per category. Defaults to 1.
            Only used for non-Grounding models. Each category must have at least (n_shots + 1) images and masks.

    Raises:
        ValueError: If the dataset is not found or invalid, or if required parameters are missing.
        FileNotFoundError: If the dataset is not found.
    """
    # Check if model is a Grounding model
    is_grounding_model = isinstance(model, GroundedSAM)

    if is_grounding_model:
        # Grounding models: only need images and text prompt (no masks)
        if not data_root:
            msg = "data_root is required for Grounding models"
            raise ValueError(msg)
        if not text_prompt:
            msg = "text_prompt is required for Grounding models"
            raise ValueError(msg)

        data_root = Path(data_root).expanduser()
        output_location = Path(output_location).expanduser() if output_location else data_root / "output"

        # Parse text prompt
        split_text = [t.strip() for t in re.split(r"[.,]", text_prompt) if t.strip()]
        class_strings = split_text

        # Create a minimal sample for category mapping (GroundedSAM only needs this for learn())
        sample = Sample(
            image=None,
            image_path=None,
            categories=split_text,
            category_ids=np.array(list(range(len(split_text))), dtype=np.int32),
            is_reference=[True] * len(split_text),
            n_shot=[0] * len(split_text),
        )
        reference_samples = [sample]

        # Load target images (masks not required for Grounding models)
        try:
            target_dataset = FolderDataset(
                root=data_root,
                images_dir="images",
                masks_dir="masks",
                n_shots=0,  # All images are targets for Grounding models
                masks_required=False,  # Masks not needed for Grounding models
            )
            target_samples = [target_dataset[i] for i in range(len(target_dataset))]
        except (FileNotFoundError, ValueError) as e:
            msg = (
                f"Failed to load images for Grounding model:\n"
                f"  Expected: {data_root}/images/{{category}}/*.jpg\n"
                f"  Masks directory not required for Grounding models\n"
                f"Error: {e}"
            )
            raise ValueError(msg) from e
    else:
        # Other models: require FolderDataset structure with masks
        if not data_root:
            msg = "data_root is required"
            raise ValueError(msg)

        data_root = Path(data_root).expanduser()
        output_location = Path(output_location).expanduser() if output_location else data_root / "output"

        try:
            dataset = FolderDataset(
                root=data_root,
                images_dir="images",
                masks_dir="masks",
                n_shots=n_shots,
            )

            # Validate that each category has enough samples
            for category in dataset.categories:
                category_df = dataset.df.filter(
                    pl.col("categories").list.contains(category),
                )
                num_samples = len(category_df)
                min_required = n_shots + 1
                if num_samples < min_required:
                    msg = (
                        f"Category '{category}' has only {num_samples} samples, "
                        f"but at least {min_required} are required (n_shots={n_shots} + 1 target)."
                    )
                    raise ValueError(msg)

            # Split into reference and target datasets
            reference_dataset = dataset.get_reference_dataset()
            target_dataset = dataset.get_target_dataset()

            reference_samples = [reference_dataset[i] for i in range(len(reference_dataset))]
            target_samples = [target_dataset[i] for i in range(len(target_dataset))]

        except (FileNotFoundError, ValueError) as e:
            msg = (
                f"Failed to load dataset with FolderDataset structure:\n"
                f"  Expected: {data_root}/images/{{category}}/*.jpg\n"
                f"  Expected: {data_root}/masks/{{category}}/*.png\n"
                f"  Each category must have at least {n_shots + 1} images and masks\n"
                f"Error: {e}"
            )
            raise ValueError(msg) from e

    if reference_samples:
        reference_batch = Batch.collate(reference_samples)
        model.fit(reference_batch)
    if not target_samples:
        logger.warning("No target samples found. Only reference learning was performed.")
        return

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
            predictions = model.predict(target_batch)

            chunk_images = [sample.image for sample in chunk_samples if sample.image is not None]
            chunk_names = []
            for idx, sample in enumerate(chunk_samples):
                if sample.image_path:
                    image_path = Path(sample.image_path) if isinstance(sample.image_path, str) else sample.image_path
                    chunk_names.append(image_path.name)
                else:
                    chunk_names.append(f"image_{idx}")

            class_strings = dataset.categories
            class_ids = [dataset.get_category_id(category) for category in class_strings]
            class_map = {category: class_id for category, class_id in zip(class_strings, class_ids, strict=True)}
            Visualizer(str(output_location / "target"), class_map=class_map).visualize(
                images=chunk_images,
                predictions=predictions,
                file_names=chunk_names,
            )
            progress.update(task, advance=len(chunk_samples))
    msg = f"Ouput saved in {output_location}"
    logger.info(msg)
