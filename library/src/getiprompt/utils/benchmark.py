# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Benchmark utilities."""

import shutil
from argparse import Namespace
from logging import getLogger
from pathlib import Path

import polars as pl
import torch

from getiprompt.components.prompt_generators import GroundingModel
from getiprompt.data.base import Batch
from getiprompt.models import GroundedSAM, InferenceMatcher, Matcher, Model, PerDino, SoftMatcher
from getiprompt.utils.constants import DatasetName, ModelName, SAMModelName

logger = getLogger("Geti Prompt")


def prepare_output_directory(output_path: str, overwrite: bool) -> Path:
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


def _save_results(all_results: list[pl.DataFrame], output_path: Path) -> None:
    """Concatenate and save all experiment results.

    Args:
        all_results: The results to save
        output_path: The path to save the results
    """
    if not all_results:
        logger.warning("No experiments were run. Check your arguments.")
        return

    all_result_dataframe = pl.concat(all_results)
    all_results_dataframe_filename = output_path / "all_results.csv"
    all_results_dataframe_filename.parent.mkdir(parents=True, exist_ok=True)
    all_result_dataframe.write_csv(str(all_results_dataframe_filename))
    msg = f"Saved all results to: {all_results_dataframe_filename}"
    logger.info(msg)

    avg_results_dataframe_filename = output_path / "avg_results.csv"
    avg_results_dataframe_filename.parent.mkdir(parents=True, exist_ok=True)
    avg_result_dataframe = all_result_dataframe.group_by(
        ["dataset_name", "model_name", "backbone_name"],
    ).mean()
    avg_result_dataframe.write_csv(str(avg_results_dataframe_filename))
    msg = f"Saved average results to: {avg_results_dataframe_filename}"
    logger.info(msg)
    msg = f"\n\n Final Average Results:\n {avg_result_dataframe}"
    logger.info(msg)


def convert_masks_to_one_hot_tensor(
    predictions: list[dict[str, torch.Tensor | None]],
    ground_truths: Batch,
    num_classes: int,
    category_id_to_index: dict[int, int],
    device: torch.device,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Convert predictions and ground truths to one-hot boolean tensors for torchmetrics.

    Args:
        predictions: List of prediction dictionaries containing 'pred_masks' and 'pred_labels'
        ground_truths: Batch of ground truth samples
        num_classes: Total number of classes in the dataset
        category_id_to_index: Mapping from category ID to class index (0-based)
        device: Device to place tensors on

    Returns:
        Tuple of (pred_tensors, gt_tensors) where each is a list of tensors in one-hot format:
        - Each tensor has shape (C, H, W) where:
          - C is number of classes
          - H, W are height and width (can vary per image)
    """
    batch_pred_tensors: list[torch.Tensor] = []
    batch_gt_tensors: list[torch.Tensor] = []

    for prediction, gt_sample in zip(predictions, ground_truths.samples, strict=True):
        # Get image dimensions from this sample
        h, w = gt_sample.masks.shape[-2:]

        # Initialize tensors (C, H, W) for this image
        pred_tensor = torch.zeros(num_classes, h, w, dtype=torch.bool, device=device)
        gt_tensor = torch.zeros(num_classes, h, w, dtype=torch.bool, device=device)

        # Process ground truth masks
        for gt_mask, cat_id in zip(gt_sample.masks, gt_sample.category_ids, strict=True):
            if cat_id in category_id_to_index:
                class_idx = category_id_to_index[cat_id]
                # Apply logical OR to handle multiple instances of same class
                gt_tensor[class_idx] = gt_tensor[class_idx] | gt_mask.to(device)  # noqa: PLR6104

        # Process prediction masks
        pred_masks = prediction["pred_masks"]
        pred_labels = prediction["pred_labels"]
        for pred_mask, pred_label in zip(pred_masks, pred_labels, strict=True):
            pred_label_id = pred_label.item()
            if pred_label_id in category_id_to_index:
                class_idx = category_id_to_index[pred_label_id]
                # Apply logical OR to handle multiple instances of same class
                pred_tensor[class_idx] = pred_tensor[class_idx] | pred_mask.to(device)  # noqa: PLR6104

        batch_pred_tensors.append(pred_tensor.unsqueeze(0))
        batch_gt_tensors.append(gt_tensor.unsqueeze(0))

    return batch_pred_tensors, batch_gt_tensors


def load_model(sam: SAMModelName, model_name: ModelName, args: Namespace) -> Model:
    """Instantiate and return the requested model.

    Args:
        sam: The name of the SAM model.
        model_name: The name of the model.
        args: The arguments to the model.

    Returns:
        The instantiated model.
    """
    # Check if OpenVINO backend is requested
    backend = getattr(args, "backend", "pytorch").lower()
    if backend == "openvino":
        logger.info("Constructing model with OpenVINO backend: %s", model_name.value)
        # Currently only Matcher supports OpenVINO backend
        if model_name != ModelName.MATCHER:
            msg = f"OpenVINO backend is currently only supported for Matcher model, not {model_name.value}"
            raise NotImplementedError(msg)

        # Get export directory
        export_dir = getattr(args, "export_dir", None)
        if export_dir is None:
            export_dir = Path("./exports") / "matcher"
        else:
            export_dir = Path(export_dir)

        return InferenceMatcher(
            model_folder=export_dir,
            sam=sam,
            num_foreground_points=args.num_foreground_points,
            num_background_points=args.num_background_points,
            mask_similarity_threshold=args.mask_similarity_threshold,
            device=args.device.lower(),
            precision=args.precision.lower(),
        )

    msg = f"Constructing model: {model_name.value}"
    logger.info(msg)

    match model_name:
        case ModelName.PER_DINO:
            return PerDino(
                sam=sam,
                encoder_model=args.encoder_model,
                num_foreground_points=args.num_foreground_points,
                num_background_points=args.num_background_points,
                num_grid_cells=args.num_grid_cells,
                similarity_threshold=args.similarity_threshold,
                mask_similarity_threshold=args.mask_similarity_threshold,
                precision=args.precision,
                compile_models=args.compile_models,
                device=args.device,
            )
        case ModelName.MATCHER:
            return Matcher(
                sam=sam,
                encoder_model=args.encoder_model,
                num_foreground_points=args.num_foreground_points,
                num_background_points=args.num_background_points,
                mask_similarity_threshold=args.mask_similarity_threshold,
                precision=args.precision,
                compile_models=args.compile_models,
                device=args.device,
            )
        case ModelName.SOFT_MATCHER:
            return SoftMatcher(
                sam=sam,
                encoder_model=args.encoder_model,
                num_foreground_points=args.num_foreground_points,
                num_background_points=args.num_background_points,
                mask_similarity_threshold=args.mask_similarity_threshold,
                use_sampling=args.use_sampling,
                use_spatial_sampling=args.use_spatial_sampling,
                approximate_matching=args.approximate_matching,
                softmatching_score_threshold=args.softmatching_score_threshold,
                softmatching_bidirectional=args.softmatching_bidirectional,
                precision=args.precision,
                compile_models=args.compile_models,
                device=args.device,
            )
        case ModelName.GROUNDED_SAM:
            return GroundedSAM(
                sam=sam,
                grounding_model=GroundingModel(args.grounding_model),
                box_threshold=args.box_threshold,
                text_threshold=args.text_threshold,
                precision=args.precision,
                compile_models=args.compile_models,
                device=args.device,
            )
        case _:
            msg = f"Algorithm {model_name.value} not implemented yet"
            raise NotImplementedError(msg)
