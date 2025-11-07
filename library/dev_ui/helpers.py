# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""This file contains helper functions for the Development UI.

It includes functions for image preparation, data processing, and pipeline management.
"""

import argparse
import base64
import json
import logging
import random
from collections.abc import Generator
from typing import Any

import cv2
import numpy as np
import torch
from getiprompt.utils.data import load_dataset

from getiprompt.data.base import Dataset
from getiprompt.models import Model, load_model
from getiprompt.types import Image, Masks, Points, Priors, Similarities
from getiprompt.utils.constants import ModelName, SAMModelName

logger = logging.getLogger(__name__)


def prepare_image_for_web(image_np: np.ndarray) -> str:
    """Encodes an image (assumed RGB) as Base64 PNG data URI.

    Args:
        image_np: RGB image array of shape (H, W, 3).

    Returns:
        str: Base64-encoded PNG data URI string.

    Raises:
        ValueError: If the image cannot be encoded to PNG.
    """
    # Assuming input is a valid RGB image (H, W, 3) from a getiprompt.Image object
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    if image_bgr.dtype != np.uint8:
        image_bgr = np.clip(image_bgr, 0, 255).astype(np.uint8)

    is_success, buffer = cv2.imencode(".png", image_bgr)
    if not is_success:
        msg = f"Could not encode image to PNG: {image_bgr.shape}"
        raise ValueError(msg)

    png_as_text = base64.b64encode(buffer).decode("utf-8")
    return f"data:image/png;base64,{png_as_text}"


def prepare_mask_for_web(mask_np: np.ndarray, color: str = "red") -> str:
    """Encodes a single-channel mask as a transparent Base64 PNG data URI.

    Args:
        mask_np: Binary mask array of shape (H, W).
        color: Color to use for the mask overlay. Either "red" or "green".

    Returns:
        str: Base64-encoded PNG data URI string with transparency.

    Raises:
        ValueError: If the color is invalid or the mask cannot be encoded.
    """
    if mask_np.dtype != np.uint8:
        mask_np = mask_np.astype(np.uint8)

    h, w = mask_np.shape
    bgra_mask = np.zeros((h, w, 4), dtype=np.uint8)

    if color == "red":
        bgra_mask[mask_np > 0, 2] = 255  # Red
    elif color == "green":
        bgra_mask[mask_np > 0, 1] = 255  # Green
    else:
        msg = f"Invalid color for mask: {color}"
        raise ValueError(msg)

    bgra_mask[mask_np > 0, 3] = 255  # Alpha channel

    is_success, buffer = cv2.imencode(".png", bgra_mask)
    if not is_success:
        msg = "Could not encode mask image to PNG"
        raise ValueError(msg)

    png_as_text = base64.b64encode(buffer).decode("utf-8")
    return f"data:image/png;base64,{png_as_text}"


def process_points_for_web(points_obj: Points) -> list[dict[str, Any]]:
    """Converts Points object to a JSON-serializable list.

    Args:
        points_obj: Points object containing point data.

    Returns:
        list[dict[str, Any]]: List of dictionaries with point information.
    """
    processed_points = []
    if not points_obj or not hasattr(points_obj, "data"):
        return processed_points

    for class_id, list_of_tensors in points_obj.data.items():
        for tensor in list_of_tensors:
            points_list = tensor.cpu().tolist()
            for x, y, score, label in points_list:
                processed_points.append(
                    {
                        "class_id": class_id,
                        "x": x,
                        "y": y,
                        "score": score,
                        "label": int(label),
                    },
                )
    return processed_points


def process_similarity_maps_for_web(
    similarities_obj: Similarities,
) -> list[dict[str, Any]]:
    """Converts Similarity object maps to JSON-serializable list of data URIs.

    Args:
        similarities_obj: Similarities object containing similarity maps.

    Returns:
        list[dict[str, Any]]: List of dictionaries with similarity map data URIs.
    """
    processed_maps = []
    if not similarities_obj or not hasattr(similarities_obj, "data") or not similarities_obj.data:
        return processed_maps

    for class_id, sim_map_tensor in similarities_obj.data.items():
        try:
            sim_map_tensor_cpu = sim_map_tensor.cpu()
            squeezed_tensor = sim_map_tensor_cpu.squeeze()

            if squeezed_tensor.ndim == 3:  # Shape [N, H, W]
                tensor_to_process = squeezed_tensor
            elif squeezed_tensor.ndim == 2:  # Shape [H, W]
                tensor_to_process = squeezed_tensor.unsqueeze(0)
            else:
                logger.warning(
                    f"Unexpected sim map shape {sim_map_tensor_cpu.shape} after squeeze "
                    f"for class {class_id}, skipping.",
                )
                continue

            for idx, sim_map_single_tensor in enumerate(tensor_to_process):
                sim_map_np = sim_map_single_tensor.float().cpu().numpy()
                if sim_map_np.size == 0:
                    logger.warning(
                        f"Sim map for class {class_id}, instance {idx} is empty, skipping.",
                    )
                    continue

                normalized_map = (sim_map_np * 255).astype(np.uint8)
                colored_map_bgr = cv2.applyColorMap(
                    normalized_map,
                    cv2.COLORMAP_JET,
                )
                # prepare_image_for_web expects RGB, so convert from BGR
                colored_map_rgb = cv2.cvtColor(colored_map_bgr, cv2.COLOR_BGR2RGB)
                map_uri = prepare_image_for_web(colored_map_rgb)

                processed_maps.append(
                    {
                        "point_index": idx,
                        "map_data_uri": map_uri,
                    },
                )
        except Exception as e:
            logger.error(
                f"Error processing similarity map for class {class_id}, instance {idx}: {e}",
                exc_info=True,
            )

    return processed_maps


def prepare_reference_data(
    reference_images: list[Image],
    reference_priors: list[Priors],
) -> list[dict[str, str]]:
    """Prepares reference images and masks for frontend display.

    Args:
        reference_images: List of reference images.
        reference_priors: List of reference priors containing masks.

    Returns:
        list[dict[str, str]]: List of dictionaries with image and mask data URIs.
    """
    prepared_data = []
    for ref_img, ref_prior in zip(reference_images, reference_priors, strict=False):
        ref_img_uri = prepare_image_for_web(ref_img.data)
        mask_tensor = next(iter(ref_prior.masks.data.values()), [torch.empty(0)])[0]
        mask_np = mask_tensor.cpu().numpy()

        if mask_np.dtype != np.uint8:
            mask_np = (mask_np > 0).astype(np.uint8)

        ref_mask_uri = prepare_mask_for_web(mask_np, color="green")

        prepared_data.append({
            "image_data_uri": ref_img_uri,
            "mask_data_uri": ref_mask_uri,
        })
    return prepared_data


def parse_request_and_check_reload(
    request_data: dict,
    current_pipeline_name: str,
    current_args: argparse.Namespace,
) -> tuple[bool, dict[str, Any], argparse.Namespace]:
    """Parses request data and checks if a pipeline reload is needed.

    Args:
        request_data: Dictionary containing request parameters.
        current_pipeline_name: Name of the currently loaded pipeline.
        current_args: Current argument namespace.

    Returns:
        tuple[bool, dict[str, Any], argparse.Namespace]: A tuple containing:
            - reload_needed: Whether pipeline reload is required.
            - requested_values: Dictionary of changed parameter values.
            - new_args: Updated argument namespace.
    """
    reload_needed = False
    requested_values = {}
    new_args = argparse.Namespace(**vars(current_args))

    # Pipeline
    if (new_pipeline_name := request_data.get("pipeline", new_args.model)) != current_pipeline_name:
        reload_needed = True
        requested_values["pipeline"] = new_pipeline_name

    # SAM Model
    current_sam = getattr(new_args, "sam", "SAM-HQ-tiny")
    if (new_sam_name := request_data.get("sam", current_sam)) != current_sam:
        reload_needed = True
        requested_values["sam"] = new_sam_name
        new_args.sam = new_sam_name

    # ImageEncoder Model
    current_encoder = getattr(new_args, "encoder_model", "dinov3_large")
    if (new_encoder_model := request_data.get("encoder_model", current_encoder)) != current_encoder:
        reload_needed = True
        requested_values["encoder_model"] = new_encoder_model
        new_args.encoder_model = new_encoder_model

    # Precision
    current_precision = getattr(new_args, "precision", "bf16")
    new_precision_str = request_data.get("precision", current_precision)
    if new_precision_str != current_precision:
        reload_needed = True
        requested_values["precision"] = new_precision_str
        new_args.precision = new_precision_str

    # Compile Models
    current_compile = getattr(new_args, "compile_models", False)
    new_compile_models = request_data.get("compile_models", current_compile)
    if isinstance(new_compile_models, str):
        new_compile_models = new_compile_models.lower() == "true"
    if new_compile_models != current_compile:
        reload_needed = True
        requested_values["compile_models"] = new_compile_models
        new_args.compile_models = new_compile_models

    # Other parameters - only those actually sent by the UI
    params_to_update = {
        "n_shot": int,
        "num_background_points": int,
        "similarity_threshold": float,
        "mask_similarity_threshold": float,
    }
    for key, cast_type in params_to_update.items():
        if key in request_data:
            new_val = cast_type(request_data[key])
            # Use getattr with None default to handle missing attributes
            old_val = getattr(new_args, key, None)
            if old_val != new_val:
                setattr(new_args, key, new_val)
                # Some parameter changes might not require a full pipeline reload
                # For now, we assume these do, but this can be refined.
                reload_needed = True
                requested_values[key] = new_val

    current_dataset = getattr(new_args, "dataset_name", "lvis")
    new_args.dataset_name = request_data.get("dataset_name", current_dataset)
    return reload_needed, requested_values, new_args


def reload_model_if_needed(
    reload_needed: bool,
    requested_values: dict[str, Any],
    current_args: argparse.Namespace,
    current_pipeline_instance: Model,
) -> tuple[Model, str, argparse.Namespace]:
    """Reloads the pipeline if necessary based on changed critical parameters.

    Args:
        reload_needed: Whether a reload is required.
        requested_values: Dictionary of changed parameter values.
        current_args: Current argument namespace.
        current_pipeline_instance: Current pipeline model instance.

    Returns:
        tuple[Model, str, argparse.Namespace]: A tuple containing:
            - model_instance: The loaded or reloaded model instance.
            - model_name: The name of the loaded model.
            - current_args: Updated argument namespace.

    Raises:
        ValueError: If model reloading fails.
    """
    model_instance = current_pipeline_instance
    model_name = current_args.model

    if model_instance is None:
        reload_needed = True
        logger.info("Model not loaded yet, triggering initial load.")
        if "sam" not in requested_values:
            requested_values["sam"] = current_args.sam
        if "model" not in requested_values:
            requested_values["model"] = current_args.model
        if "encoder_model" not in requested_values:
            requested_values["encoder_model"] = current_args.encoder_model

    if reload_needed:
        msg = f"Reloading model due to changes in: {list(requested_values.keys())}"
        logger.info(msg)
        try:
            # Use the current model name if no model change was requested
            target_model_name = requested_values.get("model", model_name)
            model_instance = load_model(
                sam=SAMModelName(current_args.sam),
                model_name=ModelName(target_model_name),
                args=current_args,
            )
            model_name = target_model_name
            logger.info("Model reloaded successfully.")
        except Exception as e:
            msg = f"Error reloading model: {e}"
            logger.error(msg, exc_info=True)
            raise ValueError(msg) from e

    return model_instance, model_name, current_args


def load_and_prepare_data(
    dataset_name: str,
    class_name_filter: str,
    n_shot: int,
    num_target_images: int | None = None,
    random_prior: bool = False,
) -> tuple[list[Image], list[Priors], list[int], Dataset]:
    """Loads dataset, validates parameters, and prepares reference/target data.

    Args:
        dataset_name: Name of the dataset to load.
        class_name_filter: Class name to filter for.
        n_shot: Number of reference shots to use.
        num_target_images: Maximum number of target images to use, or None for all.
        random_prior: Whether to randomly sample reference images.

    Returns:
        tuple[list[Image], list[Priors], list[int], Dataset]: A tuple containing:
            - reference_images: List of reference images.
            - reference_priors: List of reference priors with masks.
            - target_indices: List of target image indices.
            - full_dataset: The loaded dataset.

    Raises:
        ValueError: If dataset validation fails or insufficient samples exist.
    """
    full_dataset = load_dataset(dataset_name)
    image_count_for_category = full_dataset.get_image_count_per_category(class_name_filter)

    if image_count_for_category == 0:
        msg = f"No data for class '{class_name_filter}' in '{dataset_name}'"
        raise ValueError(msg)

    if image_count_for_category <= n_shot:
        msg = (
            f"Not enough samples ({image_count_for_category}) for class '{class_name_filter}' "
            f"to provide {n_shot} reference shot(s) and at least one target image."
        )
        raise ValueError(msg)

    all_indices = list(range(image_count_for_category))
    reference_indices = sorted(random.sample(all_indices, n_shot)) if random_prior else all_indices[:n_shot]  # nosec B311

    target_indices = [i for i in all_indices if i not in reference_indices]
    if not target_indices:
        msg = f"No target images left for class '{class_name_filter}' after selecting {n_shot} priors."
        raise ValueError(msg)

    # Limit target images if requested
    if num_target_images is not None and num_target_images >= 1:
        target_indices = target_indices[: int(num_target_images)]
        logger.info(f"Limiting to {len(target_indices)} target images.")

    reference_images, reference_priors = [], []
    for i in reference_indices:
        image_np = full_dataset.get_images_by_category(class_name_filter, start=i, end=i + 1)[0]
        mask_np = full_dataset.get_masks_by_category(class_name_filter, start=i, end=i + 1)[0]

        if mask_np.dtype != np.uint8:
            mask_np = (mask_np > 0).astype(np.uint8)

        masks = Masks()
        masks.add(mask_np, class_id=0)
        reference_images.append(Image(image_np))
        reference_priors.append(Priors(masks=masks))

    return reference_images, reference_priors, target_indices, full_dataset


def _normalize_mask(
    mask_np: np.ndarray,
    target_shape: tuple[int, int],
) -> np.ndarray:
    """Normalizes a mask tensor to a uint8 numpy array of the target shape.

    Args:
        mask_np: Input mask array.
        target_shape: Target shape (H, W) for the mask.

    Returns:
        np.ndarray: Normalized uint8 mask array.
    """
    if mask_np.ndim > 2:
        mask_np = np.squeeze(mask_np)

    if mask_np.dtype in {bool, np.bool_}:
        mask_np = mask_np.astype(np.uint8)
    elif mask_np.dtype in {np.float32, np.float16}:
        mask_np = (mask_np > 0.5).astype(np.uint8)
    elif mask_np.dtype != np.uint8:
        mask_np = (mask_np > 0).astype(np.uint8)

    if mask_np.shape != target_shape:
        mask_np = cv2.resize(
            mask_np,
            (target_shape[1], target_shape[0]),  # cv2 wants (W, H)
            interpolation=cv2.INTER_NEAREST,
        )
    return mask_np


def process_inference_chunk(
    pipeline: Model,
    full_dataset: Dataset,
    chunk_indices: list[int],
    class_name_filter: str,
) -> list[dict[str, Any]]:
    """Processes a single chunk of target images for inference and returns serializable results.

    Args:
        pipeline: Model pipeline for inference.
        full_dataset: Dataset containing images and masks.
        chunk_indices: List of image indices to process.
        class_name_filter: Class name to filter for.

    Returns:
        list[dict[str, Any]]: List of dictionaries containing inference results.

    Raises:
        ValueError: If pipeline returns incomplete results.
    """
    if not chunk_indices:
        return []

    chunk_target_image_objects = [
        Image(full_dataset.get_images_by_category(class_name_filter, start=i, end=i + 1)[0]) for i in chunk_indices
    ]

    if not chunk_target_image_objects:
        return []

    inference_results = pipeline.infer(chunk_target_image_objects)

    num_results_in_chunk = len(chunk_target_image_objects)
    if (
        len(inference_results.masks) < num_results_in_chunk
        or len(inference_results.used_points) < num_results_in_chunk
        or len(inference_results.priors) < num_results_in_chunk
    ):
        msg = "Pipeline returned incomplete results for the chunk."
        raise ValueError(msg)

    results_chunk = []
    for j in range(num_results_in_chunk):
        target_img_obj = chunk_target_image_objects[j]
        masks_obj = inference_results.masks[j]
        used_points_obj = inference_results.used_points[j]
        prior_obj = inference_results.priors[j]

        img_data_uri = prepare_image_for_web(target_img_obj.data)

        processed_mask_data_uris = []
        for class_id, list_of_tensors in masks_obj.data.items():
            for instance_counter, mask_tensor in enumerate(list_of_tensors):
                mask_np = mask_tensor.cpu().numpy()
                normalized_mask = _normalize_mask(mask_np, target_img_obj.data.shape[:2])
                mask_data_uri = prepare_mask_for_web(normalized_mask, color="red")
                processed_mask_data_uris.append(
                    {
                        "class_id": class_id,
                        "instance_id": f"mask_{instance_counter}",
                        "mask_data_uri": mask_data_uri,
                    },
                )

        web_used_points = process_points_for_web(used_points_obj)
        web_prior_points = process_points_for_web(prior_obj.points)

        web_similarity_maps = []
        if hasattr(inference_results, "similarities") and len(inference_results.similarities) > j:
            web_similarity_maps = process_similarity_maps_for_web(
                inference_results.similarities[j],
            )

        gt_masks_list = full_dataset.get_masks_by_category(
            class_name_filter,
            start=chunk_indices[j],
            end=chunk_indices[j] + 1,
        )
        gt_mask_uri = None
        if gt_masks_list:
            gt_mask_np = gt_masks_list[0]
            if gt_mask_np.dtype != np.uint8:
                gt_mask_np = (gt_mask_np > 0).astype(np.uint8)
            gt_mask_uri = prepare_mask_for_web(gt_mask_np, color="green")

        results_chunk.append(
            {
                "image_data_uri": img_data_uri,
                "masks": processed_mask_data_uris,
                "used_points": web_used_points,
                "prior_points": web_prior_points,
                "similarity_maps": web_similarity_maps,
                "gt_mask_uri": gt_mask_uri,
            },
        )

    return results_chunk


def stream_inference(
    pipeline: Model,
    full_dataset: Dataset,
    target_indices: list[int],
    class_name_filter: str,
    prepared_reference_data: list[dict[str, str]],
    batch_size: int = 5,
) -> Generator[str, None, None]:
    """Generator function to process targets in chunks and yield results as JSON strings.

    Args:
        pipeline: Model pipeline for inference.
        full_dataset: Dataset containing images and masks.
        target_indices: List of target image indices to process.
        class_name_filter: Class name to filter for.
        prepared_reference_data: List of prepared reference data dictionaries.
        batch_size: Number of images to process per batch.

    Yields:
        str: JSON-encoded strings containing inference results or metadata.
            First yield contains total_targets and reference_data.
            Subsequent yields contain target_results or error messages.
    """
    total_targets = len(target_indices)

    initial_message = {
        "total_targets": total_targets,
        "reference_data": prepared_reference_data,
    }
    yield json.dumps(initial_message) + "\n"

    for chunk_start_idx in range(0, total_targets, batch_size):
        chunk_end_idx = min(chunk_start_idx + batch_size, total_targets)
        current_chunk_indices = target_indices[chunk_start_idx:chunk_end_idx]

        try:
            results_chunk = process_inference_chunk(
                pipeline,
                full_dataset,
                current_chunk_indices,
                class_name_filter,
            )
            if results_chunk:
                yield json.dumps({"target_results": results_chunk}) + "\n"
        except Exception as e:
            msg = f"Error processing chunk: {e}"
            logger.error(msg, exc_info=True)
            yield json.dumps({"error": msg}) + "\n"
