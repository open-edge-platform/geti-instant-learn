# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E402

"""Geti Prompt Benchmark Script."""

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

from getiprompt.data import BatchedSingleCategoryIter, Dataset
from getiprompt.metrics import SegmentationMetrics
from getiprompt.models import Model, load_model
from getiprompt.types import Image, Masks, Priors, Text
from getiprompt.utils import setup_logger
from getiprompt.utils.args import get_arguments, parse_experiment_args
from getiprompt.utils.constants import DatasetName, ModelName, SAMModelName
from getiprompt.utils.data import get_filename_categories, get_image_and_mask_from_filename, load_dataset
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


def get_all_instances(images: list[np.ndarray], masks: list[np.ndarray], count: int) -> tuple[list[Image], list[Masks]]:
    """This method returns priors including masks.

    Args:
        images: The list of images of a certain category
        masks: The list of masks of a certain category
        count: The number of image masks to return

    Returns:
        List of images and masks
    """
    # Load all prior images and masks
    prior_images = []
    prior_masks = []
    for i, (image, mask) in enumerate(zip(images, masks, strict=False)):
        prior_images.append(Image(image))
        mask2 = mask
        mask2[mask2 > 1] = 1  # Keep all instances
        mask2 = mask2[:, :, None]
        masks = Masks()
        masks.add(mask2)
        prior_masks.append(masks)
        if i >= count - 1:
            break
    return prior_images, prior_masks


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


def infer_all_batches(
    batches: BatchedSingleCategoryIter,
    model: Model,
    category_name: str,
    priors_batch_index: int,
    visualizer: ExportMaskVisualization,
    metrics_calculators: dict[int, SegmentationMetrics],
    number_of_batches: int,
    progress: Progress,
    image_size: tuple[int, int] | None = None,
) -> tuple[int, int]:
    """This performs an inference run on all batches.

    Args:
        batches: An iterable of batches that return numpy images and masks
        model: The model to run
        category_name: The current category
        priors_batch_index: The current prior batch
        visualizer: The visualizer for exporting
        metrics_calculators: The calculator for the metrics
        number_of_batches: The number of batches.
        progress: The progress bar
        image_size: The size of the images to resize to

    Returns:
        The number of samples that were processed and the total time it took.
    """
    # Task for batches for the current category and prior (transient)
    batches.reset()  # reset batch iterator because it was consumed
    num_batches_to_process = len(batches) if number_of_batches is None else min(len(batches), number_of_batches + 1)
    batches_task = progress.add_task(
        f"[magenta]Infer step: {category_name}",
        total=num_batches_to_process,
        transient=True,
    )
    time_sum = 0
    time_count = 0
    for batch_index, (images, masks) in enumerate(batches):
        target_images = [Image(image) for image in images]
        results = model.infer(target_images=target_images)
        time_sum += results.duration
        time_count += len(images)

        # Generate names for exported files and export them
        export_paths = [
            str(
                Path("predictions")
                / f"priors_batch_{priors_batch_index}"
                / category_name
                / Path(batches.get_image_filename(batch_index, image_index)).name,
            )
            for image_index in range(len(images))
        ]
        export_paths_all_points = [
            str(
                Path("predictions_debug")
                / f"priors_batch_{priors_batch_index}"
                / category_name
                / Path(batches.get_image_filename(batch_index, image_index)).name,
            )
            for image_index in range(len(images))
        ]
        export_paths_gt = [
            str(
                Path("ground_truth")
                / f"priors_batch_{priors_batch_index}"
                / category_name
                / Path(batches.get_image_filename(batch_index, image_index)).name,
            )
            for image_index in range(len(images))
        ]
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
            names=export_paths_all_points,
            points=visualizer.points_from_priors(results.priors),
            boxes=visualizer.boxes_from_priors(results.priors),
        )
        gt_masks = visualizer.arrays_to_masks(masks)
        # resize masks to target image size
        gt_masks = [mask.resize(image_size) for mask in gt_masks]
        visualizer(
            images=target_images,
            masks=gt_masks,
            names=export_paths_gt,
        )
        metrics_calculators[priors_batch_index](
            predictions=results.masks,
            references=gt_masks,
            mapping={0: category_name},
        )
        progress.update(batches_task, advance=1)
        if number_of_batches is not None and batch_index >= number_of_batches:  # Adjusted condition
            break
    progress.remove_task(batches_task)
    return time_sum, time_count


def infer_all_images(
    filenames: list[str],
    dataset: Dataset,
    model: Model,
    category_name: str,
    priors_batch_index: int,
    visualizer: ExportMaskVisualization,
    metrics_calculators: dict[int, SegmentationMetrics],
    progress: Progress,
) -> tuple[int, int]:
    """This performs an inference run on all batches.

    Args:
        filenames: A list of filenames
        dataset: The dataset containing the images
        model: The model to run
        category_name: The current category
        priors_batch_index: The current prior batch
        visualizer: The visualizer for exporting
        metrics_calculators: The calculator for the metrics
        progress: The progress bar

    Returns:
        The number of samples that were processed and the total time it took.
    """
    # Task for batches for the current category and prior (transient)
    batches_task = progress.add_task(f"[magenta]Infer step: {category_name}", total=1, transient=True)
    time_sum = 0
    time_count = 0

    images = []
    masks = []
    for filename in filenames:
        image, mask = get_image_and_mask_from_filename(filename, dataset, category_name)
        images.append(image)
        masks.append(mask)

    target_images = [Image(image) for image in images]
    results = model.infer(target_images=target_images)
    time_sum += results.duration
    time_count += len(images)

    # Generate names for exported files and export them
    export_paths = [
        str(
            Path("predictions")
            / f"priors_batch_{priors_batch_index}"
            / category_name
            / Path(filenames[image_index]).name,
        )
        for image_index in range(len(images))
    ]
    export_paths_all_points = [
        str(
            Path("predictions_debug")
            / f"priors_batch_{priors_batch_index}"
            / category_name
            / Path(filenames[image_index]).name,
        )
        for image_index in range(len(images))
    ]
    export_paths_gt = [
        str(
            Path("ground_truth")
            / f"priors_batch_{priors_batch_index}"
            / category_name
            / Path(filenames[image_index]).name,
        )
        for image_index in range(len(images))
    ]
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
        names=export_paths_all_points,
        points=visualizer.points_from_priors(results.priors),
        annotations=results.annotations,
        boxes=visualizer.boxes_from_priors(results.priors),
    )
    gt_masks = visualizer.arrays_to_masks(masks)
    visualizer(
        images=target_images,
        masks=gt_masks,
        names=export_paths_gt,
    )
    metrics_calculators[priors_batch_index](
        predictions=results.masks,
        references=gt_masks,
        mapping={0: category_name},
    )
    progress.update(batches_task, advance=1)
    progress.remove_task(batches_task)
    return time_sum, time_count


def predict_on_dataset(  # noqa: C901, PLR0915
    args: argparse.Namespace,
    model: Model,
    priors_dataset: Dataset,
    dataset: Dataset,
    unique_output: Path,
    dataset_name: str,
    model_name: str,
    backbone_name: str,
    number_of_priors_tests: int,
    number_of_batches: int | None,
    dataset_filenames: list[str] | None,
    image_size: tuple[int, int] | None = None,
) -> pd.DataFrame:
    """This runs predictions on the dataset and evaluates them.

    Args:
        args: Args from the argparser.
        model: The model to use.
        priors_dataset: The training set that is used for priors
        dataset: The validation set that is processed
        unique_output: Unique output name
        dataset_name: The dataset name
        model_name: The algorithm name
        backbone_name: The model name
        number_of_priors_tests: The number of priors to try
        number_of_batches: The number of batches per class to process (limited testing)
            pass None to process all data
        dataset_filenames: Only do inference on these images
        image_size: The size of the images to resize to
    Returns: The timing

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
    cat_filter = get_filename_categories(dataset_filenames, dataset)

    with progress:
        # Main task for categories (persistent)
        categories_task = progress.add_task(f"[cyan]Processing {dataset_name}", total=dataset.get_category_count())

        # Iterate over all categories in the dataset
        for category_index, batches in enumerate(dataset):
            category_name = dataset.category_index_to_name(category_index)
            if cat_filter is not None and category_name not in cat_filter:
                continue  # skip this category because it is filtered

            priors_cat_index = priors_dataset.category_name_to_index(category_name)

            # Task for priors for the current category (transient)
            priors_iter = BatchedSingleCategoryIter(priors_dataset, args.n_shot, priors_cat_index)
            priors_task = progress.add_task(f"[green]Learn step: {category_name}", total=1, transient=True)

            # Iterate over all priors in the batch (break after number_of_prior_test iterations)
            for priors_batch_index, (priors_images, priors_masks) in enumerate(priors_iter):
                # Add a new metrics calculator if needed
                if priors_batch_index not in metrics_calculators:
                    metrics_calculators[priors_batch_index] = SegmentationMetrics(categories=dataset.get_categories())

                # Select priors
                priors_images2, priors_masks2 = get_all_instances(
                    priors_images,
                    priors_masks,
                    args.n_shot,
                )

                # Learn using the priors
                text_prior = Text()
                text_prior.add(category_name, class_id=0)
                reference_priors = [Priors(masks=priors_masks2[i], text=text_prior) for i in range(len(priors_masks2))]
                try:
                    model.learn(
                        reference_images=priors_images2,
                        reference_priors=reference_priors,
                    )
                    progress.update(priors_task, advance=1)
                except ValueError:
                    logger.exception("Failed to learn from priors.")
                    progress.update(priors_task, advance=1)
                    continue

                # Save priors
                priors_export_paths = [
                    str(
                        Path("priors")
                        / f"priors_batch_{priors_batch_index}"
                        / category_name
                        / f"prior_{image_index}.png",
                    )
                    for image_index in range(len(priors_images2))
                ]
                masks_priors = visualizer.masks_from_priors(
                    reference_priors,
                )
                visualizer(
                    images=priors_images2,
                    masks=masks_priors,
                    names=priors_export_paths,
                )
                if dataset_filenames is None:
                    # Iterate over all batches
                    ts, tc = infer_all_batches(
                        batches=batches,
                        model=model,
                        category_name=category_name,
                        priors_batch_index=priors_batch_index,
                        visualizer=visualizer,
                        metrics_calculators=metrics_calculators,
                        number_of_batches=number_of_batches,
                        progress=progress,
                        image_size=image_size,
                    )
                else:
                    # Infer on a single image
                    ts, tc = infer_all_images(
                        filenames=dataset_filenames,
                        dataset=dataset,
                        model=model,
                        category_name=category_name,
                        priors_batch_index=priors_batch_index,
                        visualizer=visualizer,
                        metrics_calculators=metrics_calculators,
                        progress=progress,
                    )

                time_sum += ts
                time_count += tc

                if priors_batch_index >= number_of_priors_tests - 1:
                    break  # Do not proceed with the next batch of priors
            progress.remove_task(priors_task)
            progress.update(categories_task, advance=1)

    # Construct the output metrics file from the calculated metrics
    all_metrics = None
    for prior_index, calculator in metrics_calculators.items():
        metrics = calculator.get_metrics()
        ln = len(metrics["category"])
        metrics["prior_index"] = [prior_index] * ln
        metrics["inference_time"] = [time_sum / time_count] * ln
        metrics["images_per_category"] = [
            dataset.get_image_count_per_category(cat_name) for cat_name in metrics["category"]
        ]
        metrics["instances_per_category"] = [
            dataset.get_instance_count_per_category(cat_name) for cat_name in metrics["category"]
        ]
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

    Returns: A list of valid experiment configurations
    """
    all_combinations = list(itertools.product(datasets, models, backbones))
    valid_configs = []

    for dataset, pipeline, backbone in all_combinations:
        valid_configs.append((dataset, pipeline, backbone))

    return valid_configs


def _get_output_path_for_experiment(
    output_path: Path,
    experiment_name: str | None,
    dataset: DatasetName,
    model: ModelName,
    backbone: SAMModelName,
    dataset_filenames: str | None,
) -> Path:
    """Construct a unique output path for an experiment.

    Args:
        output_path: The path to save the results
        experiment_name: The name of the experiment
        dataset: The dataset to run
        model: The model to run
        backbone: The backbone to run
        dataset_filenames: The filenames to run

    Returns: The path to save the results
    """
    combo_str = f"{dataset.value}_{backbone.value}_{model.value}"

    if experiment_name:
        return output_path / experiment_name / combo_str

    if dataset_filenames:
        fn_str = dataset_filenames.replace("/", "_")
        return output_path / f"{combo_str}_{fn_str}"

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
    logger.info()

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

        dataset = load_dataset(dataset_enum.value, whitelist=args.class_name, batch_size=args.batch_size)
        model = load_model(sam=backbone_enum, model_name=model_enum, args=args)

        # Individual experiment artifacts are saved in a path derived from the base path.
        unique_output_path = _get_output_path_for_experiment(
            base_output_path,
            args.experiment_name,
            dataset_enum,
            model_enum,
            backbone_enum,
            args.dataset_filenames,
        )

        all_metrics_df = predict_on_dataset(
            args,
            model,
            priors_dataset=dataset,
            dataset=dataset,
            unique_output=unique_output_path,
            dataset_name=dataset_enum.value,
            model_name=model_enum.value,
            backbone_name=backbone_enum.value,
            number_of_priors_tests=args.num_priors,
            number_of_batches=args.num_batches,
            dataset_filenames=args.dataset_filenames.split(",") if args.dataset_filenames else None,
            image_size=args.image_size,
        )
        all_results.append(all_metrics_df)

    # Save aggregated results to the final results path
    _save_results(all_results, final_results_path)


if __name__ == "__main__":
    perform_benchmark_experiment()
