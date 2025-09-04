# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Models and pipelines can be constructed using the methods in this file."""

from logging import getLogger
from typing import TYPE_CHECKING

from efficientvit.models.efficientvit import EfficientViTSamPredictor
from efficientvit.sam_model_zoo import create_efficientvit_sam_model
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from segment_anything_fast import sam_model_fast_registry
from segment_anything_fast.predictor import SamPredictor as SamFastPredictor
from segment_anything_hq import sam_model_registry as sam_hq_model_registry
from segment_anything_hq.predictor import SamPredictor as SamHQPredictor

from getiprompt.models.model_optimizer import optimize_model
from getiprompt.models.per_segment_anything import SamPredictor, sam_model_registry
from getiprompt.utils import download_file, precision_to_torch_dtype
from getiprompt.utils.constants import DATA_PATH, MODEL_MAP, SAMModelName

if TYPE_CHECKING:
    from segment_anything_hq.modeling.sam import Sam as SamHQ

    from getiprompt.models.per_segment_anything.modeling.sam import Sam

logger = getLogger("Geti Prompt")


def load_sam_model(
    sam: SAMModelName,
    device: str = "cuda",
    precision: str = "bf16",
    compile_models: bool = False,
    benchmark_inference_speed: bool = False,
) -> SamPredictor | SamHQPredictor | SamFastPredictor | EfficientViTSamPredictor | SAM2ImagePredictor:
    """Load and optimize a SAM model.

    Args:
        sam: The name of the SAM model.
        device: The device to use for the model.
        precision: The precision of the model.
        compile_models: Whether to compile the model.
        benchmark_inference_speed: Whether to benchmark the inference speed.

    Returns:
        The loaded model.
    """
    if sam not in MODEL_MAP:
        msg = f"Invalid model type: {sam}"
        raise ValueError(msg)

    model_info = MODEL_MAP[sam]
    check_model_weights(sam)

    registry_name = model_info["registry_name"]
    local_filename = model_info["local_filename"]
    checkpoint_path = DATA_PATH.joinpath(local_filename)

    logger.info(f"Loading segmentation model: {sam} from {checkpoint_path}")

    if sam in {SAMModelName.SAM, SAMModelName.MOBILE_SAM}:
        model: Sam = sam_model_registry[registry_name](checkpoint=str(checkpoint_path)).to(device).eval()
        predictor = SamPredictor(model)
    elif sam in {SAMModelName.SAM2_TINY, SAMModelName.SAM2_SMALL, SAMModelName.SAM2_BASE, SAMModelName.SAM2_LARGE}:
        config_path = "configs/sam2.1/" + model_info["config_filename"]
        sam_model = build_sam2(config_path, str(checkpoint_path))
        predictor = SAM2ImagePredictor(sam_model)
    elif sam in {SAMModelName.SAM_HQ, SAMModelName.SAM_HQ_TINY}:
        model: SamHQ = sam_hq_model_registry[registry_name](checkpoint=str(checkpoint_path)).to(device).eval()
        predictor = SamHQPredictor(model)
    elif sam == SAMModelName.SAM_FAST:
        model = sam_model_fast_registry[registry_name](checkpoint=str(checkpoint_path)).to(device).eval()
        predictor = SamFastPredictor(model)
    elif sam == SAMModelName.EFFICIENT_VIT_SAM:
        model = (
            create_efficientvit_sam_model(
                name=registry_name,
                weight_url=str(checkpoint_path),
            )
            .to(device)
            .eval()
        )
        predictor = EfficientViTSamPredictor(model)
    else:
        msg = f"Model {sam} not implemented yet"
        raise NotImplementedError(msg)

    return optimize_model(
        model=predictor,
        device=device,
        precision=precision_to_torch_dtype(precision),
        compile_models=compile_models,
        benchmark_inference_speed=benchmark_inference_speed,
    )


def check_model_weights(model_name: SAMModelName) -> None:
    """Check if model weights exist locally, download if necessary.

    Args:
        model_name: The name of the model.
    """
    if model_name not in MODEL_MAP:
        msg = f"Model '{model_name.value}' not found in MODEL_MAP for weight checking."
        raise ValueError(msg)

    model_info = MODEL_MAP[model_name]
    local_filename = model_info["local_filename"]
    download_url = model_info["download_url"]
    sha_sum = model_info["sha_sum"]

    if not local_filename or not download_url:
        msg = f"Missing 'local_filename' or 'download_url' for {model_name.value} in MODEL_MAP."
        raise ValueError(msg)

    target_path = DATA_PATH.joinpath(local_filename)

    if not target_path.exists():
        logger.info(f"Model weights for {model_name.value} not found at {target_path}, downloading...")
        download_file(download_url, target_path, sha_sum)
