# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Geti Prompt Benchmark Script."""

from __future__ import annotations

import itertools
from logging import getLogger
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import polars as pl
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn
from torch.utils.data import DataLoader

from getiprompt.data import Dataset, LVISDataset, PerSegDataset
from getiprompt.metrics import SegmentationMetrics
from getiprompt.models import Model, load_model
from getiprompt.types import Masks, Priors, Text
from getiprompt.utils import setup_logger
from getiprompt.utils.args import get_arguments, parse_experiment_args
from getiprompt.utils.benchmark import (
    _get_output_path_for_experiment,
    _save_results,
    prepare_output_directory,
)
from getiprompt.utils.constants import get_category_presets
from getiprompt.utils.utils import masks_to_custom_masks
from getiprompt.visualize import ExportMaskVisualization

logger = getLogger("Geti Prompt")


if TYPE_CHECKING:
    import argparse

    from torchvision import tv_tensors

    from getiprompt.data.base import Sample


def extract_priors(
    sample: Sample,
    category_name: str,
) -> Priors:
    """Extract priors from a Sample.

    Args:
        sample: Sample - The sample to extract priors from.
        category_name: str - The category name for the text prior.

    Returns:
        Priors - The priors extracted from the sample.
    """
    # Convert masks to Masks type
    masks_obj = Masks()
    if sample.masks is not None:
        mask_np = sample.masks if isinstance(sample.masks, np.ndarray) else sample.masks.cpu().numpy()
        category_ids = (
            sample.category_ids if isinstance(sample.category_ids, np.ndarray) else sample.category_ids.cpu().numpy()
        )

        for mask, category_id in zip(mask_np, category_ids, strict=True):
            masks_obj.add(mask, class_id=int(category_id))

    # Create text prior
    text_prior = Text()
    text_prior.add(category_name, class_id=int(category_id))

    # Create priors object
    return Priors(masks=masks_obj, text=text_prior)


def infer_on_category(
    dataset: Dataset,
    model: Model,
    category_name: str,
    priors_batch_index: int,
    visualizer: ExportMaskVisualization,
    metrics_calculators: dict[int, SegmentationMetrics],
    progress: Progress,
    batch_size: int = 4,
    visualize: bool = True,
) -> tuple[int, int]:
    """Perform inference on all samples of a category.

    Args:
        dataset: The dataset containing target samples
        model: The model to run
        category_name: The current category
        priors_batch_index: The current prior batch
        visualizer: The visualizer for exporting
        metrics_calculators: The calculator for the metrics
        progress: The progress bar
        batch_size: Batch size for DataLoader
        visualize: Whether to visualize the results

    Returns:
        The number of samples that were processed and the total time it took.

    Raises:
        ValueError: If no target samples are found for the category.
    """
    # Get target samples for this category
    target_dataset = dataset.get_target_dataset(category=category_name)

    if len(target_dataset) == 0:
        msg = f"No target samples found for category: {category_name}"
        logger.warning(msg)
        raise ValueError(msg)

    category_id = target_dataset.get_category_id(category_name)

    # Create DataLoader
    dataloader = DataLoader(
        target_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=target_dataset.collate_fn,
    )

    # Task for batches for the current category and prior (transient)
    batches_task = progress.add_task(
        f"[magenta]Infer step: {category_name}",
        total=len(dataloader),
        transient=True,
    )

    total_time = 0
    n_samples = 0
    for batch in dataloader:
        # Run inference
        target_images = batch.images
        results = model.infer(target_images=target_images)
        total_time += results.duration
        n_samples += len(batch)

        # Convert ground truth masks to Masks objects
        gt_masks = masks_to_custom_masks(
            batch.masks,
            class_id=category_id,
        )

        # Calculate metrics
        metrics_calculators[priors_batch_index](
            predictions=results.masks,
            ground_truths=gt_masks,
            mapping={category_id: category_name},
        )

        if visualize:
            # Generate export paths
            file_names = [
                str(
                    Path("predictions") / f"priors_batch_{priors_batch_index}" / category_name / Path(img_path).name,
                )
                for img_path in batch.image_paths
            ]
            file_names_debug = [
                str(
                    Path("predictions_debug")
                    / f"priors_batch_{priors_batch_index}"
                    / category_name
                    / Path(img_path).name,
                )
                for img_path in batch.image_paths
            ]
            file_names_gt = [
                str(
                    Path("ground_truth") / f"priors_batch_{priors_batch_index}" / category_name / Path(img_path).name,
                )
                for img_path in batch.image_paths
            ]

            # Visualize predictions and ground truth
            visualizer(
                images=target_images,
                masks=results.masks,
                file_names=file_names,
                points=results.used_points,
                boxes=visualizer.boxes_from_priors(results.priors),
            )
            visualizer(
                images=target_images,
                masks=results.masks,
                file_names=file_names_debug,
                points=visualizer.points_from_priors(results.priors),
                boxes=visualizer.boxes_from_priors(results.priors),
            )
            visualizer(
                images=target_images,
                masks=gt_masks,
                file_names=file_names_gt,
            )

        progress.update(batches_task, advance=1)

    progress.remove_task(batches_task)
    return total_time, n_samples


def learn_from_category(
    dataset: Dataset,
    model: Model,
    category_name: str,
    n_shot: int,
    visualizer: ExportMaskVisualization,
    priors_batch_index: int,
) -> tuple[list[tv_tensors.Image], list[Priors]]:
    """Learn from reference samples of a category.

    Args:
        dataset: The dataset containing reference samples
        model: The model to train
        category_name: The category to learn from
        n_shot: Number of reference shots to use
        visualizer: The visualizer for exporting
        priors_batch_index: The current prior batch index

    Returns:
        list[tv_tensors.Image] - The reference images used for learning.
        list[Priors] - The reference priors used for learning.

    Raises:
        ValueError: If no reference samples are found for the category.
    """
    # Get reference samples for this category
    reference_dataset = dataset.get_reference_dataset(category=category_name)
    if len(reference_dataset) == 0:
        msg = f"No reference samples found for category: {category_name}"
        raise ValueError(msg)

    # Limit to n_shot samples
    n_samples = min(n_shot, len(reference_dataset))

    # Convert samples to legacy format (Image, Priors)
    reference_images = []
    reference_priors = []

    for i in range(n_samples):
        sample = reference_dataset[i]
        priors = extract_priors(sample, category_name)
        reference_images.append(sample.image)
        reference_priors.append(priors)

    # Learn
    model.learn(
        reference_images=reference_images,
        reference_priors=reference_priors,
    )

    # Save priors visualization
    priors_file_names = [
        str(
            Path("priors") / f"priors_batch_{priors_batch_index}" / category_name / f"prior_{image_index}.png",
        )
        for image_index in range(len(reference_images))
    ]
    masks_priors = visualizer.masks_from_priors(reference_priors)
    visualizer(
        images=reference_images,
        masks=masks_priors,
        file_names=priors_file_names,
    )

    return reference_images, reference_priors


def predict_on_dataset(
    args: argparse.Namespace,
    model: Model,
    dataset: Dataset,
    output_path: Path,
    dataset_name: str,
    model_name: str,
    backbone_name: str,
    number_of_priors_tests: int,
) -> pl.DataFrame:
    """Run predictions on the dataset and evaluate them.

    Args:
        args: Args from the argparser.
        model: The model to use.
        dataset: The dataset (contains both reference and target samples)
        output_path: Output path
        dataset_name: The dataset name
        model_name: The algorithm name
        backbone_name: The model name
        number_of_priors_tests: The number of priors to try

    Returns:
        The timing DataFrame
    """
    output_path = prepare_output_directory(output_path, args.overwrite)
    msg = f"Output path: {output_path}"
    logger.info(msg)

    visualizer = ExportMaskVisualization(
        output_folder=str(output_path),
    )
    metrics_calculators: dict[int, SegmentationMetrics] = {}  # keep metrics per prior

    time_sum = 0
    time_count = 0

    # Setup Rich Progress
    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeRemainingColumn(),
        TimeElapsedColumn(),
    )

    # Get all unique categories
    categories = dataset.categories

    with progress:
        # Main task for categories (persistent)
        categories_task = progress.add_task(f"[cyan]Processing {dataset_name}", total=len(categories))

        # Iterate over all categories in the dataset
        for category_name in categories:
            # Task for priors for the current category (transient)
            priors_task = progress.add_task(f"[green]Learn step: {category_name}", total=1, transient=True)

            # For now, only use 1 prior batch (can be extended for multiple prior batches)
            for priors_batch_index in range(number_of_priors_tests):
                # Add a new metrics calculator if needed
                if priors_batch_index not in metrics_calculators:
                    metrics_calculators[priors_batch_index] = SegmentationMetrics(categories=categories)

                # Learn from reference samples
                learn_from_category(
                    dataset=dataset,
                    model=model,
                    category_name=category_name,
                    n_shot=args.n_shot,
                    visualizer=visualizer,
                    priors_batch_index=priors_batch_index,
                )
                progress.update(priors_task, advance=1)

                # Infer on target samples
                total_time, n_samples = infer_on_category(
                    dataset=dataset,
                    model=model,
                    category_name=category_name,
                    priors_batch_index=priors_batch_index,
                    visualizer=visualizer,
                    metrics_calculators=metrics_calculators,
                    progress=progress,
                    batch_size=args.batch_size,
                )

                time_sum += total_time
                time_count += n_samples

            progress.remove_task(priors_task)
            progress.update(categories_task, advance=1)

    # Construct the output metrics file from the calculated metrics
    all_metrics = None
    for prior_index, calculator in metrics_calculators.items():
        metrics = calculator.get_metrics()
        ln = len(metrics["category"])
        metrics["prior_index"] = [prior_index] * ln
        metrics["inference_time"] = [time_sum / time_count if time_count > 0 else 0] * ln
        metrics["dataset_name"] = [dataset_name] * ln
        metrics["model_name"] = [model_name] * ln
        metrics["backbone_name"] = [backbone_name] * ln
        if all_metrics is None:
            all_metrics = metrics
        else:
            for key in all_metrics:
                all_metrics[key].extend(metrics[key])

    return pl.DataFrame(all_metrics)


def load_dataset_by_name(
    dataset_name: str,
    categories: list[str] | str | None = None,
    n_shots: int = 1,
    dataset_root: str | Path | None = None,
) -> Dataset:
    """Load a dataset by name.

    Args:
        dataset_name: Name of the dataset (e.g., "PerSeg", "LVIS")
        categories: Categories to filter. Can be:
            - None: uses "default" preset if available, else all categories
            - "all": uses all available categories
            - "default": uses default preset (quick testing)
            - "benchmark": uses benchmark preset (comprehensive testing)
            - list[str]: explicit list of category names
        n_shots: Number of reference shots per category
        dataset_root: Root directory where datasets are stored. If None, uses defaults.

    Raises:
        ValueError: If the dataset name is unknown or preset is invalid.

    Returns:
        GetiPromptDataset instance

    Example:
        Load dataset with different category configurations:

        >>> # Use default preset (quick testing, 4 categories for LVIS)
        >>> dataset = load_dataset_by_name("lvis")
        >>> # Or explicitly:
        >>> dataset = load_dataset_by_name("lvis", categories="default")
        >>>
        >>> # Use benchmark preset (comprehensive testing, 92 categories on LVIS-92 fold 0)
        >>> dataset = load_dataset_by_name("lvis", categories="benchmark")
        >>>
        >>> # Use all available categories in the dataset
        >>> dataset = load_dataset_by_name("lvis", categories="all")
        >>>
        >>> # Use explicit category list
        >>> dataset = load_dataset_by_name("lvis", categories=["cat", "dog", "bird"])
        >>>
        >>> # Extend a preset by importing constants
        >>> from getiprompt.utils.constants import LVIS_DEFAULT_CATEGORIES
        >>> custom_categories = list(LVIS_DEFAULT_CATEGORIES) + ["tiger", "lion"]
        >>> dataset = load_dataset_by_name("lvis", categories=custom_categories)
        >>>
        >>> # Use benchmark categories from a specific fold
        >>> from getiprompt.utils.constants import LVIS_BENCHMARK_CATEGORIES
        >>> dataset = load_dataset_by_name("lvis", categories=LVIS_BENCHMARK_CATEGORIES["fold_1"])
        >>>
        >>> # Configure n_shots and custom dataset root
        >>> dataset = load_dataset_by_name(
        ...     "lvis",
        ...     categories="benchmark",
        ...     n_shots=3,
        ...     dataset_root="~/my_datasets"
        ... )
    """
    # Resolve category presets
    if categories is None:
        categories = "default"

    if isinstance(categories, str):
        preset_key = categories.lower()
        if preset_key == "all":
            resolved_categories = None  # Dataset will use all available categories
        else:
            category_presets = get_category_presets()
            if dataset_name.lower() in category_presets:
                if preset_key in category_presets[dataset_name.lower()]:
                    resolved_categories = category_presets[dataset_name.lower()][preset_key]
                else:
                    available_presets = list(category_presets[dataset_name.lower()].keys())
                    msg = f"Unknown preset '{categories}' for dataset '{dataset_name}'. Available: {available_presets}"
                    raise ValueError(msg)
            else:
                msg = f"No presets defined for dataset '{dataset_name}'"
                raise ValueError(msg)
    else:
        resolved_categories = categories

    if dataset_name.lower() == "perseg":
        root = (
            Path(dataset_root).expanduser() / "PerSeg"
            if dataset_root is not None
            else Path("~/datasets/PerSeg").expanduser()
        )
        return PerSegDataset(
            root=root,
            categories=resolved_categories,
            n_shots=n_shots,
        )
    if dataset_name.lower() == "lvis":
        root = (
            Path(dataset_root).expanduser() / "lvis"
            if dataset_root is not None
            else Path("~/datasets/lvis").expanduser()
        )
        return LVISDataset(
            root=root,
            categories=resolved_categories,
            n_shots=n_shots,
        )
    msg = f"Unknown dataset: {dataset_name}"
    raise ValueError(msg)


def perform_benchmark_experiment(args: argparse.Namespace | None = None) -> None:
    """Main function to run the experiments.

    This function initializes the arguments, determines which models, datasets, and models to process,
    and then iterates over all combinations to run the predictions and evaluate them.

    Args:
        args: The arguments to use.
    """
    # Initialization
    if args is None:
        args = get_arguments()

    base_output_path = Path("~").expanduser() / "outputs"
    # The final results path will include the experiment name if provided.
    final_results_path = base_output_path / args.experiment_name if args.experiment_name else base_output_path

    setup_logger(final_results_path, args.log_level)
    final_results_path.mkdir(parents=True, exist_ok=True)

    # Get experiment lists and generate a plan
    datasets_to_run, models_to_run, backbones_to_run = parse_experiment_args(args)
    experiments = list(itertools.product(datasets_to_run, models_to_run, backbones_to_run))

    # Execute experiments
    all_results = []
    for dataset_enum, model_enum, backbone_enum in experiments:
        msg = (
            f"Starting experiment with Dataset={dataset_enum.value}, "
            f"Model={model_enum.value}, Backbone={backbone_enum.value}",
        )
        logger.info(msg)

        # Parse categories from CLI argument
        # Support both presets (e.g., "default", "benchmark", "all") and explicit lists (e.g., "cat,dog,bird")
        if args.class_name:
            # Check if it's a preset or a comma-separated list
            if "," not in args.class_name and args.class_name.lower() in {"default", "benchmark", "all"}:
                categories_arg = args.class_name  # Pass preset string directly
            else:
                categories_arg = [c.strip() for c in args.class_name.split(",")]  # Split comma-separated list
        else:
            categories_arg = None  # Will default to "default" preset

        # Load dataset using new API
        dataset = load_dataset_by_name(
            dataset_enum.value,
            categories=categories_arg,
            n_shots=args.n_shot,
            dataset_root=args.dataset_root,
        )

        model = load_model(sam=backbone_enum, model_name=model_enum, args=args)

        # Individual experiment artifacts are saved in a path derived from the base path.
        output_path = _get_output_path_for_experiment(
            base_output_path,
            args.experiment_name,
            dataset_enum,
            model_enum,
            backbone_enum,
        )

        all_metrics_df = predict_on_dataset(
            args,
            model,
            dataset=dataset,
            output_path=output_path,
            dataset_name=dataset_enum.value,
            model_name=model_enum.value,
            backbone_name=backbone_enum.value,
            number_of_priors_tests=args.num_priors,
        )
        all_results.append(all_metrics_df)

    # Save aggregated results to the final results path
    _save_results(all_results, final_results_path)
