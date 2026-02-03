# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""This file contains several methods to optimize a model for inference."""

import time
from logging import getLogger

import numpy as np
import torch
from sam2.sam2_image_predictor import SAM2ImagePredictor
from segment_anything_hq.predictor import SamPredictor
from transformers import AutoModel

logger = getLogger("Geti Instant Learn")


def get_dummy_input(model: AutoModel, precision: torch.dtype, device: str) -> torch.Tensor:
    """Gets or creates a dummy input tensor for the model."""
    if hasattr(model, "dummy_inputs") and "pixel_values" in model.dummy_inputs:
        return model.dummy_inputs["pixel_values"].to(precision).to(device)

    # Fallback if dummy_inputs is not available or doesn't have pixel_values
    image_size = model.config.image_size
    num_channels = getattr(model.config, "num_channels", 3)
    return torch.randn(1, num_channels, image_size, image_size, device=device, dtype=precision)


def optimize_model(
    model: AutoModel | SamPredictor | SAM2ImagePredictor,
    device: str,
    precision: torch.dtype,
    compile_models: bool,
    compile_backend: str = "inductor",
) -> AutoModel | SamPredictor | SAM2ImagePredictor:
    """This method optimizes a model by quantizing it and compiling it.

    Args:
        model: The model to optimize.
        device: The device to use for the model.
        precision: The precision to use for the model.
        compile_models: Whether to compile the model.
        compile_backend: The backend to use for the model.

    Returns:
        The optimized model.
    """
    if precision != torch.float32:
        if isinstance(model, SamPredictor | SAM2ImagePredictor):
            model.model.to(precision)
        else:
            model = model.to(dtype=precision)

    # Compile
    if compile_models:
        logger.debug("Compiling model, this can take a while...")
        if torch.cuda.is_available() and torch.cuda.get_device_capability() in {(7, 0), (8, 0), (9, 0)}:
            if isinstance(model, SamPredictor | SAM2ImagePredictor):
                model.model.image_encoder.forward = torch.compile(
                    model.model.image_encoder.forward,
                    mode="max-autotune",
                    fullgraph=True,
                    dynamic=False,
                    backend=compile_backend,
                )
                model.set_image(np.ones(shape=(1024, 1024, 3), dtype=np.uint8))
                model.predict(point_coords=np.array([[512, 512]]), point_labels=np.array([1]))
            else:
                model = torch.compile(model, backend=compile_backend)
                model(pixel_values=get_dummy_input(model, precision, device))  # Run one inference to compile
        else:
            logger.warning("GPU is not NVIDIA V100, A100, or H100. Compilation will be skipped.")

    return model


@torch.inference_mode()
def benchmark_inference(
    model: SamPredictor | SAM2ImagePredictor | AutoModel,
    precision: torch.dtype,
    repeat: int = 10,
) -> float:
    """Test the inference time of the model.

    Args:
        model: The SAM or huggingface model to benchmark.
        precision: The precision to use for the inference.
        repeat: The number of times to repeat the inference.

    Returns:
        The average inference time.
    """
    # Create a dummy image with two squares.
    image = np.zeros((1024, 1024, 3), dtype=np.uint8)
    image[256:768, 256:768, :] = 255
    image[512:1024, 512:1024, :] = 255

    # Create points in the middle of the two squares.
    points = np.array([[512, 512], [768, 768]])

    if not isinstance(model, SamPredictor | SAM2ImagePredictor):
        image = get_dummy_input(model, precision, model.device)
    dtype = (
        next(model.model.parameters())
        if isinstance(model, SamPredictor | SAM2ImagePredictor)
        else next(model.parameters())
    ).dtype

    with torch.autocast(model.device.type, dtype=dtype):
        start = time.time()
        for _ in range(repeat):
            torch.cuda.synchronize()
            if isinstance(model, SamPredictor | SAM2ImagePredictor):
                model.set_image(image)
                for p in points:
                    model.predict(point_coords=np.array([p]), point_labels=np.array([1]))
            else:
                _ = model(pixel_values=image)

        torch.cuda.synchronize()
        avg_duration = (time.time() - start) / repeat
    if model.device.type == "cuda":
        model_size = torch.cuda.memory_allocated() / 1e6
    elif model.device.type == "xpu":
        model_size = torch.xpu.memory_allocated() / 1e6
    else:
        model_size = torch.cpu.memory_allocated() / 1e6
    msg = (
        f"Inference time: {avg_duration:.2f} seconds, FPS: {1 / avg_duration:.2f}, "
        f"Memory allocated: {model_size:.2f} MB",
    )
    logger.debug(msg)
    return avg_duration
