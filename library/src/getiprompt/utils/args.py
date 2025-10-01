# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import argparse
from enum import Enum
from typing import TypeVar

from getiprompt.processes.prompt_generators import GroundingModel
from getiprompt.utils.constants import DatasetName, PipelineName, SAMModelName

# Generate help strings with choices
AVAILABLE_MODELS = ", ".join([model.value for model in SAMModelName])
AVAILABLE_PIPELINES = ", ".join([p.value for p in PipelineName])
AVAILABLE_DATASETS = ", ".join([d.value for d in DatasetName])

HELP_SAM_NAME = (
    f"Backbone segmentation model name or comma-separated list. Use 'all' to run all. Available: [{AVAILABLE_MODELS}]"
)
HELP_PIPELINE = f"Pipeline name or comma-separated list. Use 'all' to run all. Available: [{AVAILABLE_PIPELINES}]"
HELP_DATASET_NAME = f"Dataset name or comma-separated list. Use 'all' to run all. Available: [{AVAILABLE_DATASETS}]"


def populate_benchmark_parser(parser: argparse.ArgumentParser) -> None:
    """Populate the argument parser with benchmark arguments."""
    parser.add_argument(
        "--sam",
        type=str,
        default="SAM-HQ-tiny",
        choices=["all"] + [model.value for model in SAMModelName],
        help=HELP_SAM_NAME,
    )
    parser.add_argument(
        "--pipeline",
        type=str,
        default="Matcher",
        choices=["all"] + [p.value for p in PipelineName],
        help=HELP_PIPELINE,
    )
    parser.add_argument(
        "--n_shot",
        type=int,
        default=1,
        help="Number of prior images to use as references",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="lvis",
        choices=["all"] + [d.value for d in DatasetName],
        help=HELP_DATASET_NAME,
    )
    parser.add_argument(
        "--dataset_filenames",
        type=str,
        nargs="+",
        help="Only perform inference on these files from the dataset. "
        "Filename ambiguity can be solved by including subfolders. "
        "For example: can/01.jpg instead of 01.jpg",
    )
    parser.add_argument("--save", action="store_true", help="Save results to disk")
    parser.add_argument(
        "--apply_mask_refinement",
        action="store_true",
        help="Apply mask refinement",
    )
    parser.add_argument(
        "--class_name",
        type=str,
        default=None,
        help="Filter on class name",
    )
    parser.add_argument(
        "--num_grid_cells",
        type=int,
        default=16,
        help="Number of grid cells to use for the grid prompt generator",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=None,
        help="Size of the image to use for inference. If not provided, the original size will be used. "
        "If provided, the image will be resized to the given size, maintaining aspect ratio. "
        "Note: images are always resized to 1024x1024 for SAM and to 518x518 for DINO. "
        "This will mainly influence the UI rendering.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output data",
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        default=None,
        help="If passed, will save all",
    )
    parser.add_argument(
        "--num_clusters",
        type=int,
        default=3,
        help="Number of clusters of features to create, if using the ClusterFeatures module",
    )
    parser.add_argument(
        "--similarity_threshold",
        type=float,
        default=0.65,
        help="Threshold for segmenting the image",
    )
    parser.add_argument(
        "--mask_similarity_threshold",
        type=float,
        default=0.42,
        help="Threshold for filtering masks based on average similarity",
    )
    parser.add_argument(
        "--skip_points_in_existing_masks",
        type=bool,
        default=True,
        help="Skip foreground points that fall within already generated masks for the same class",
    )
    parser.add_argument(
        "--num_foreground_points",
        type=int,
        default=40,
        help="Maximum number of foreground points to sample, if using the MaxPointFilter module",
    )
    parser.add_argument(
        "--num_background_points",
        type=int,
        default=2,
        help="Number of background points to sample",
    )
    parser.add_argument(
        "--use_sampling",
        action="store_true",
        help="Use sampling",
    )
    parser.add_argument(
        "--use_spatial_sampling",
        action="store_true",
        help="Use spatial sampling",
    )
    parser.add_argument(
        "--approximate_matching",
        action="store_true",
        help="Use approximate matching",
    )
    parser.add_argument(
        "--softmatching_score_threshold",
        type=float,
        default=0.4,
        help="The score threshold for the soft matching",
    )
    parser.add_argument(
        "--softmatching_bidirectional",
        action="store_true",
        help="Use bidirectional soft matching",
    )
    parser.add_argument(
        "--num_priors",
        type=int,
        default=1,
        help="Number of runs to perform, each time using the next image in the dataset as a prior",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=5,
        help="The maximum batch size used during inference.",
    )
    parser.add_argument(
        "--num_batches",
        type=int,
        help="The maximum number of batches per class to process. "
        "This can be used to limit the amount images that are processed. "
        "The number of processed images will not exceed num_classes * num_batches * batch_size",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="bf16",
        choices=["fp32", "fp16", "bf16"],
        help="The precision to use for the models. Maps to torch.float32, torch.float16, or torch.bfloat16",
    )
    parser.add_argument(
        "--compile_models",
        type=bool,
        default=False,
        help="Whether to compile the models",
    )
    parser.add_argument(
        "--benchmark_inference_speed",
        action="store_true",
        help="Whether to show the inference time of the optimized models",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="The device to use for the models",
    )
    parser.add_argument(
        "--grounding_model",
        type=str,
        default=GroundingModel.LLMDET_TINY.value,
        choices=[g.value for g in GroundingModel],
        help="The grounding model to use",
    )
    parser.add_argument(
        "--box_threshold",
        type=float,
        default=0.4,
        help="The box threshold for the grounding model",
    )
    parser.add_argument(
        "--text_threshold",
        type=float,
        default=0.3,
        help="The text threshold for the grounding model",
    )


def get_arguments(arg_list: list[str] | None = None) -> argparse.Namespace:
    """Get arguments.

    Args:
        arg_list: List of arguments

    Returns:
        Arguments
    """
    parser = argparse.ArgumentParser()
    populate_benchmark_parser(parser)

    return parser.parse_args(arg_list)


TEnum = TypeVar("TEnum", bound=Enum)


def _parse_enum_list(arg_str: str, enum_cls: type[TEnum], arg_name: str) -> list[TEnum]:
    """Parses a comma-separated string of enum values, returning a list of valid enum members."""
    if arg_str == "all":
        return list(enum_cls)

    items_to_run = [p.strip() for p in arg_str.split(",")]
    valid_enum_values = {e.value for e in enum_cls}

    invalid_items = [item for item in items_to_run if item not in valid_enum_values]
    if invalid_items:
        msg = f"Invalid {arg_name}(s): {invalid_items}. Available {arg_name}s: {[e.value for e in enum_cls]}"
        raise ValueError(msg)

    return [enum_cls(item) for item in items_to_run]


def _parse_str_list(arg_str: str, all_values: list[str], arg_name: str) -> list[str]:
    """Parses a comma-separated string of values, returning a list of valid values."""
    if arg_str == "all":
        return [v for v in all_values if v != "all"]

    items_to_run = [p.strip() for p in arg_str.split(",")]
    valid_items = [item for item in items_to_run if item in all_values]
    invalid_items = [item for item in items_to_run if item not in all_values]

    if invalid_items:
        msg = f"Invalid {arg_name}(s): {invalid_items}. Available {arg_name}s: {[v for v in all_values if v != 'all']}"
        raise ValueError(msg)

    return valid_items


def parse_experiment_args(args: argparse.Namespace) -> tuple[list[DatasetName], list[PipelineName], list[SAMModelName]]:
    """Parse experiment arguments.

    Args:
        args: Arguments

    Returns:
        tuple containing:
            - datasets_to_run: List of dataset enums to run
            - pipelines_to_run: List of pipeline enums to run
            - backbones_to_run: List of SAM model enums to run

    Raises:
        ValueError: If any invalid arguments are provided or if no valid arguments remain after filtering
    """
    valid_datasets = _parse_enum_list(args.dataset_name, DatasetName, "dataset")
    valid_pipelines = _parse_enum_list(args.pipeline, PipelineName, "pipeline")
    valid_backbones = _parse_enum_list(args.sam, SAMModelName, "SAM model")

    if not valid_datasets:
        msg = f"No valid datasets found from '{args.dataset_name}'. Available: {[d.value for d in DatasetName]}"
        raise ValueError(msg)
    if not valid_pipelines:
        msg = f"No valid pipelines found from '{args.pipeline}'. Available: {[p.value for p in PipelineName]}"
        raise ValueError(msg)
    if not valid_backbones:
        msg = f"No valid SAM models found from '{args.sam}'. Available: {[m.value for m in SAMModelName]}"
        raise ValueError(msg)

    return valid_datasets, valid_pipelines, valid_backbones
