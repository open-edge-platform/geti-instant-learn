# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E402

"""Refactored Geti Prompt Benchmark Script using new dataset design.

This benchmark script uses:
- GetiPromptDataset (PerSegDataset/LVISDataset)
- GetiPromptSample
- GetiPromptBatch
- PyTorch DataLoader
"""

import argparse
import shutil
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import itertools
from logging import getLogger

import cv2
import numpy as np
import pandas as pd
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn
from torch.utils.data import DataLoader, Subset

from getiprompt.data import GetiPromptDataset, PerSegDataset
from getiprompt.data.batch import GetiPromptBatch
from getiprompt.data.sample import GetiPromptSample
from getiprompt.metrics import SegmentationMetrics
from getiprompt.models import Model, load_model
from getiprompt.types import Image, Masks, Priors, Text
from getiprompt.utils import setup_logger
from getiprompt.utils.args import get_arguments, parse_experiment_args
from getiprompt.utils.constants import DatasetName, ModelName, SAMModelName
from getiprompt.visualize import ExportMaskVisualization

logger = getLogger("Geti Prompt")


def handle_output_path(output_path: str, overwrite: bool) -> Path:
    """Handle output path to avoid overwriting existing data.

    Args:
        output_path: The path to the output data
        overwrite: Whether to overwrite existing data

    Raises:
        ValueError: If the output path already exists and overwrite is False

    Returns:
        The path to the output data
    """
    output_path_obj = Path(output_path)
    if output_path_obj.exists():
        if overwrite:
            shutil.rmtree(output_path_obj)
        else:
            msg = (
                f"Output path {output_path_obj} already exists. "
                "Set overwrite=True to overwrite it or change the output path."
            )
            raise ValueError(msg)

    output_path_obj.mkdir(parents=True, exist_ok=True)
    return output_path_obj


def sample_to_image_and_priors(
    sample: GetiPromptSample,
    category_name: str,
) -> tuple[Image, Priors]:
    """Convert a GetiPromptSample to legacy Image and Priors format.

    This centralizes the conversion logic from the new sample format to the
    legacy format used by models.

    Args:
        sample: GetiPromptSample with image and masks
        category_name: Category name for the text prior

    Returns:
        Tuple of (Image, Priors) ready for model.learn()
    """
    # Convert image to Image type
    image = Image(sample.image if isinstance(sample.image, np.ndarray) else sample.image.cpu().numpy())
    
    # Convert masks to Masks type
    masks_obj = Masks()
    if sample.masks is not None:
        mask_np = sample.masks if isinstance(sample.masks, np.ndarray) else sample.masks.cpu().numpy()
        category_ids = sample.category_ids if isinstance(sample.category_ids, np.ndarray) else sample.category_ids.cpu().numpy()
        
        for mask, category_id in zip(mask_np, category_ids, strict=True):
            masks_obj.add(mask, class_id=int(category_id))
    
    # Create text prior
    text_prior = Text()
    text_prior.add(category_name, class_id=0)
    
    # Create priors object
    priors = Priors(masks=masks_obj, text=text_prior)
    
    return image, priors


def save_priors(prior_images: list[Image], prior_masks: list[Masks], output_path: str) -> None:
    """This method saves the priors to disk.

    Args:
        prior_images: The list of prior images
        prior_masks: The list of prior masks
        output_path: The path to save the priors
    """
    output_path_obj = Path(output_path)
    output_path_obj.mkdir(parents=True, exist_ok=True)
    for i, (image, mask) in enumerate(zip(prior_images, prior_masks, strict=False)):
        mask_np = mask.to_numpy()[0]  # Mask is CHW
        image_np = image.to_numpy()
        mask_np[mask_np > 0] = 255
        mask_np = cv2.cvtColor(mask_np.astype(np.uint8), cv2.COLOR_GRAY2BGR)
        overlay = cv2.addWeighted(image_np, 0.7, mask_np, 0.3, 0)
        overlay = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(output_path_obj / f"prior_{i}.png"), overlay)


def get_category_dataset(
    dataset: GetiPromptDataset, 
    category: str, 
    is_reference: bool = False,
) -> GetiPromptDataset:
    """Get a filtered dataset for a specific category.

    This is a convenience wrapper around the base class methods.

    Args:
        dataset: The GetiPromptDataset
        category: Category name to filter
        is_reference: If True, get only reference samples; if False, get only target samples

    Returns:
        Filtered GetiPromptDataset for the category
    """
    if is_reference:
        return dataset.get_reference_dataset(category=category)
    return dataset.get_target_dataset(category=category)


def infer_on_category(
    dataset: GetiPromptDataset,
    model: Model,
    category_name: str,
    priors_batch_index: int,
    visualizer: ExportMaskVisualization,
    metrics_calculators: dict[int, SegmentationMetrics],
    progress: Progress,
    batch_size: int = 4,
    number_of_batches: int | None = None,
    image_size: tuple[int, int] | None = None,
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
        number_of_batches: The number of batches to process (None = all)
        image_size: The size of the images to resize to

    Returns:
        The number of samples that were processed and the total time it took.
    """
    # Get target samples for this category
    target_dataset = get_category_dataset(dataset, category_name, is_reference=False)
    
    if len(target_dataset) == 0:
        logger.warning(f"No target samples found for category: {category_name}")
        return 0, 0
    
    # Create DataLoader
    dataloader = DataLoader(
        target_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=target_dataset.collate_fn
    )
    
    # Task for batches for the current category and prior (transient)
    num_batches_to_process = len(dataloader) if number_of_batches is None else min(len(dataloader), number_of_batches)
    batches_task = progress.add_task(
        f"[magenta]Infer step: {category_name}",
        total=num_batches_to_process,
        transient=True,
    )
    
    time_sum = 0
    time_count = 0
    
    for batch_index, batch in enumerate(dataloader):
        if number_of_batches is not None and batch_index >= number_of_batches:
            break
        
        # Convert batch to Image objects
        target_images = [Image(img_np) for img_np in batch.images_np]
        
        # Run inference
        results = model.infer(target_images=target_images)
        time_sum += results.duration
        time_count += len(batch)
        
        # Generate export paths
        export_paths = [
            str(
                Path("predictions")
                / f"priors_batch_{priors_batch_index}"
                / category_name
                / Path(img_path).name,
            )
            for img_path in batch.image_paths
        ]
        export_paths_debug = [
            str(
                Path("predictions_debug")
                / f"priors_batch_{priors_batch_index}"
                / category_name
                / Path(img_path).name,
            )
            for img_path in batch.image_paths
        ]
        export_paths_gt = [
            str(
                Path("ground_truth")
                / f"priors_batch_{priors_batch_index}"
                / category_name
                / Path(img_path).name,
            )
            for img_path in batch.image_paths
        ]
        
        # Visualize predictions
        visualizer(
            images=target_images,
            masks=results.masks,
            names=export_paths,
            points=results.used_points,
            boxes=visualizer.boxes_from_priors(results.priors),
        )
        visualizer(
            images=target_images,
            masks=results.masks,
            names=export_paths_debug,
            points=visualizer.points_from_priors(results.priors),
            boxes=visualizer.boxes_from_priors(results.priors),
        )
        
        # Visualize ground truth
        gt_masks = visualizer.arrays_to_masks(batch.masks_np)
        if image_size is not None:
            gt_masks = [mask.resize(image_size) for mask in gt_masks]
        
        visualizer(
            images=target_images,
            masks=gt_masks,
            names=export_paths_gt,
        )
        
        # Calculate metrics
        metrics_calculators[priors_batch_index](
            predictions=results.masks,
            references=gt_masks,
            mapping={0: category_name},
        )
        
        progress.update(batches_task, advance=1)
    
    progress.remove_task(batches_task)
    return time_sum, time_count


def learn_from_category(
    dataset: GetiPromptDataset,
    model: Model,
    category_name: str,
    n_shot: int,
    visualizer: ExportMaskVisualization,
    priors_batch_index: int,
) -> tuple[list[Image], list[Priors]]:
    """Learn from reference samples of a category.

    Args:
        dataset: The dataset containing reference samples
        model: The model to train
        category_name: The category to learn from
        n_shot: Number of reference shots to use
        visualizer: The visualizer for exporting
        priors_batch_index: The current prior batch index

    Returns:
        Tuple of (reference_images, reference_priors) used for learning
    """
    # Get reference samples for this category
    reference_dataset = get_category_dataset(
        dataset, 
        category_name, 
        is_reference=True
    )
    
    if len(reference_dataset) == 0:
        raise ValueError(f"No reference samples found for category: {category_name}")
    
    # Limit to n_shot samples
    n_samples = min(n_shot, len(reference_dataset))
    
    # Convert samples to legacy format (Image, Priors)
    reference_images = []
    reference_priors = []
    
    for i in range(n_samples):
        sample = reference_dataset[i]
        image, priors = sample_to_image_and_priors(sample, category_name)
        reference_images.append(image)
        reference_priors.append(priors)
    
    # Learn
    model.learn(
        reference_images=reference_images,
        reference_priors=reference_priors,
    )
    
    # Save priors visualization
    priors_export_paths = [
        str(
            Path("priors")
            / f"priors_batch_{priors_batch_index}"
            / category_name
            / f"prior_{image_index}.png",
        )
        for image_index in range(len(reference_images))
    ]
    masks_priors = visualizer.masks_from_priors(reference_priors)
    visualizer(
        images=reference_images,
        masks=masks_priors,
        names=priors_export_paths,
    )
    
    return reference_images, reference_priors


def predict_on_dataset(  # noqa: C901, PLR0915
    args: argparse.Namespace,
    model: Model,
    dataset: GetiPromptDataset,
    unique_output: Path,
    dataset_name: str,
    model_name: str,
    backbone_name: str,
    number_of_priors_tests: int,
    number_of_batches: int | None,
    image_size: tuple[int, int] | None = None,
) -> pd.DataFrame:
    """Run predictions on the dataset and evaluate them.

    Args:
        args: Args from the argparser.
        model: The model to use.
        dataset: The dataset (contains both reference and target samples)
        unique_output: Unique output name
        dataset_name: The dataset name
        model_name: The algorithm name
        backbone_name: The model name
        number_of_priors_tests: The number of priors to try
        number_of_batches: The number of batches per class to process (limited testing)
            pass None to process all data
        image_size: The size of the images to resize to

    Returns:
        The timing DataFrame
    """
    unique_output_path = handle_output_path(unique_output, args.overwrite)
    msg = f"Output path: {unique_output_path}"
    logger.info(msg)

    visualizer = ExportMaskVisualization(
        output_folder=str(unique_output_path),
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

                try:
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
                except (ValueError, RuntimeError) as e:
                    logger.exception(f"Failed to learn from priors for category {category_name}: {e}")
                    progress.update(priors_task, advance=1)
                    continue

                # Infer on target samples
                ts, tc = infer_on_category(
                    dataset=dataset,
                    model=model,
                    category_name=category_name,
                    priors_batch_index=priors_batch_index,
                    visualizer=visualizer,
                    metrics_calculators=metrics_calculators,
                    progress=progress,
                    batch_size=args.batch_size,
                    number_of_batches=number_of_batches,
                    image_size=image_size,
                )

                time_sum += ts
                time_count += tc

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

    return pd.DataFrame.from_dict(all_metrics)


def _generate_experiment_plan(
    datasets: list[DatasetName],
    models: list[ModelName],
    backbones: list[SAMModelName],
) -> list[tuple[DatasetName, ModelName, SAMModelName]]:
    """Generate a list of valid experiment configurations to run.

    Args:
        datasets: The datasets to run
        models: The models to run
        backbones: The backbones to run

    Returns:
        A list of valid experiment configurations
    """
    all_combinations = list(itertools.product(datasets, models, backbones))
    return all_combinations


def _get_output_path_for_experiment(
    output_path: Path,
    experiment_name: str | None,
    dataset: DatasetName,
    model: ModelName,
    backbone: SAMModelName,
) -> Path:
    """Construct a unique output path for an experiment.

    Args:
        output_path: The path to save the results
        experiment_name: The name of the experiment
        dataset: The dataset to run
        model: The model to run
        backbone: The backbone to run

    Returns:
        The path to save the results
    """
    combo_str = f"{dataset.value}_{backbone.value}_{model.value}"

    if experiment_name:
        return output_path / experiment_name / combo_str

    return output_path / combo_str


def _save_results(all_results: list[pd.DataFrame], output_path: Path) -> None:
    """Concatenate and save all experiment results.

    Args:
        all_results: The results to save
        output_path: The path to save the results
    """
    if not all_results:
        logger.warning("No experiments were run. Check your arguments.")
        return

    all_result_dataframe = pd.concat(all_results, ignore_index=True)
    all_results_dataframe_filename = output_path / "all_results.csv"
    all_results_dataframe_filename.parent.mkdir(parents=True, exist_ok=True)
    all_result_dataframe.to_csv(str(all_results_dataframe_filename))
    msg = f"Saved all results to: {all_results_dataframe_filename}"
    logger.info(msg)

    avg_results_dataframe_filename = output_path / "avg_results.csv"
    avg_results_dataframe_filename.parent.mkdir(parents=True, exist_ok=True)
    avg_result_dataframe = all_result_dataframe.groupby(
        ["dataset_name", "model_name", "backbone_name"],
    ).mean(numeric_only=True)
    avg_result_dataframe.to_csv(str(avg_results_dataframe_filename))
    msg = f"Saved average results to: {avg_results_dataframe_filename}"
    logger.info(msg)
    msg = f"\n\n Final Average Results:\n {avg_result_dataframe}"
    logger.info(msg)


def load_dataset_by_name(dataset_name: str, categories: list[str] | None = None, n_shots: int = 1) -> GetiPromptDataset:
    """Load a dataset by name.

    Args:
        dataset_name: Name of the dataset (e.g., "PerSeg", "LVIS")
        categories: Optional list of categories to filter
        n_shots: Number of reference shots per category

    Returns:
        GetiPromptDataset instance
    """
    # Map dataset names to dataset classes
    # You'll need to implement this mapping based on your dataset structure
    if dataset_name.lower() == "perseg":
        return PerSegDataset(
            root=Path("~/datasets/PerSeg").expanduser(),
            categories=categories,
            n_shots=n_shots
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


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
    experiment_plan = _generate_experiment_plan(datasets_to_run, models_to_run, backbones_to_run)

    # Execute experiments
    all_results = []
    for dataset_enum, model_enum, backbone_enum in experiment_plan:
        msg = (
            f"Starting experiment with Dataset={dataset_enum.value}, "
            f"Model={model_enum.value}, Backbone={backbone_enum.value}",
        )
        logger.info(msg)

        # Load dataset using new API
        dataset = load_dataset_by_name(
            dataset_enum.value,
            categories=args.class_name.split(",") if args.class_name else None,
            n_shots=args.n_shot
        )
        
        model = load_model(sam=backbone_enum, model_name=model_enum, args=args)

        # Individual experiment artifacts are saved in a path derived from the base path.
        unique_output_path = _get_output_path_for_experiment(
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
            unique_output=unique_output_path,
            dataset_name=dataset_enum.value,
            model_name=model_enum.value,
            backbone_name=backbone_enum.value,
            number_of_priors_tests=args.num_priors,
            number_of_batches=args.num_batches,
            image_size=args.image_size,
        )
        all_results.append(all_metrics_df)

    # Save aggregated results to the final results path
    _save_results(all_results, final_results_path)


if __name__ == "__main__":
    perform_benchmark_experiment()

