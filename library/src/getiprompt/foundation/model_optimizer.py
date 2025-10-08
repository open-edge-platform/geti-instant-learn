# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""This file contains several methods to optimize a model for inference."""

import time
from collections.abc import Callable
from logging import getLogger
from typing import Any

import numpy as np
import torch
from efficientvit.models.efficientvit import EfficientViTSamPredictor
from efficientvit.models.nn import UpSampleLayer
from sam2.sam2_image_predictor import SAM2ImagePredictor
from segment_anything_hq.predictor import SamPredictor as SamHQPredictor
from torch import nn
from transformers import AutoModel

from getiprompt.foundation.per_segment_anything import SamPredictor
from getiprompt.foundation.per_segment_anything.modeling.tiny_vit_sam import Attention, TinyViT

logger = getLogger("Geti Prompt")


def get_dummy_input(model: AutoModel, precision: torch.dtype, device: str) -> torch.Tensor:
    """Gets or creates a dummy input tensor for the model."""
    if hasattr(model, "dummy_inputs") and "pixel_values" in model.dummy_inputs:
        return model.dummy_inputs["pixel_values"].to(precision).to(device)

    # Fallback if dummy_inputs is not available or doesn't have pixel_values
    image_size = model.config.image_size
    num_channels = getattr(model.config, "num_channels", 3)
    return torch.randn(1, num_channels, image_size, image_size, device=device, dtype=precision)


def optimize_model(
    model: AutoModel | SamPredictor | SamHQPredictor | EfficientViTSamPredictor | SAM2ImagePredictor,
    device: str,
    precision: torch.dtype,
    compile_models: bool,
    benchmark_inference_speed: bool = True,
    compile_backend: str = "inductor",
) -> AutoModel | SamPredictor | SamHQPredictor | EfficientViTSamPredictor | SAM2ImagePredictor:
    """This method optimizes a model by quantizing it and compiling it.

    Args:
        model: The model to optimize.
        device: The device to use for the model.
        precision: The precision to use for the model.
        compile_models: Whether to compile the model.
        benchmark_inference_speed: Whether to show the inference time.
        compile_backend: The backend to use for the model.

    Returns:
        The optimized model.
    """
    # Initial inference
    if benchmark_inference_speed:
        logger.debug("Model initial inference:")
        initial_inference_duration = benchmark_inference(model, precision)

    # Quantize
    if precision != torch.float32:
        if is_sam_model(model):
            model.model.to(precision)
            # Patch SAM model to use the correct dtype
            _monkey_patch_dtype(model)
        else:
            model = model.to(dtype=precision)
        if benchmark_inference_speed:
            logger.debug("Quantized inference:")
            quantized_inference_duration = benchmark_inference(model, precision)
            logger.debug(f"Quantization speedup: {initial_inference_duration / quantized_inference_duration:.2f}x")

    # Compile
    if compile_models:
        logger.debug("Compiling model, this can take a while...")
        if torch.cuda.is_available() and torch.cuda.get_device_capability() in {(7, 0), (8, 0), (9, 0)}:
            if is_sam_model(model):
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
            if benchmark_inference_speed:
                logger.debug("Compiled model inference:")
                compiled_inference_duration = benchmark_inference(model, precision)
                logger.debug(
                    f"Quantization + Compilation speedup: "
                    f"{initial_inference_duration / compiled_inference_duration:.2f}x"
                )
        else:
            logger.warning("GPU is not NVIDIA V100, A100, or H100. Compilation will be skipped.")

    return model


@torch.inference_mode()
def benchmark_inference(
    model: SamPredictor | SamHQPredictor | EfficientViTSamPredictor | SAM2ImagePredictor | AutoModel,
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

    if not is_sam_model(model):
        image = get_dummy_input(model, precision, model.device)
    dtype = (next(model.model.parameters()) if is_sam_model(model) else next(model.parameters())).dtype

    with torch.autocast(model.device.type, dtype=dtype):
        start = time.time()
        for _ in range(repeat):
            torch.cuda.synchronize()
            if is_sam_model(model):
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
    logger.debug(
        f"Inference time: {avg_duration:.2f} seconds, FPS: {1 / avg_duration:.2f}, "
        f"Memory allocated: {model_size:.2f} MB"
    )
    return avg_duration


def _monkey_patch_transform(predictor: EfficientViTSamPredictor, dtype: torch.dtype) -> None:
    """Monkey patch the forward method of the EfficientViTSamImageEncoder to use the correct dtype."""
    original_forward = predictor.model.image_encoder.forward

    def forward_dtype_wrapper(x: torch.Tensor) -> torch.Tensor:
        # The input tensor must be converted to the same dtype as the model's weights.
        return original_forward(x.to(dtype))

    # Replace the original forward method with the wrapper.
    predictor.model.image_encoder.forward = forward_dtype_wrapper


def _monkey_patch_upsample_layers(model: nn.Module, dtype: torch.dtype) -> None:
    """Recursively find all UpSampleLayer instances and patch their forward method.

    This is necessary because F.interpolate with mode='bicubic' can upcast to float32.
    """
    for module in model.modules():
        if isinstance(module, UpSampleLayer):
            original_forward = module.forward

            def new_forward(
                x: torch.Tensor,
                original_forward: Callable[[torch.Tensor], torch.Tensor] = original_forward,
            ) -> torch.Tensor:
                return original_forward(x).to(dtype)

            module.forward = new_forward


def _monkey_patch_preprocess(predictor: SamPredictor | SamHQPredictor, dtype: torch.dtype) -> None:
    """Monkey patch the preprocess method to use the correct dtype."""
    original_preprocess = predictor.model.preprocess

    def preprocess_dtype_wrapper(input_tensor: torch.Tensor) -> torch.Tensor:
        output_from_original_preprocess = original_preprocess(input_tensor)
        return output_from_original_preprocess.to(dtype)

    predictor.model.preprocess = preprocess_dtype_wrapper


def _monkey_patch_prompt_encoder(prompt_encoder: nn.Module, dtype: torch.dtype) -> None:
    """Monkey patch the prompt encoder methods to use the correct dtype."""
    original_pe_encoding = prompt_encoder.pe_layer._pe_encoding  # noqa: SLF001

    def pe_encoding_dtype_wrapper(*args, **kwargs) -> torch.Tensor:
        processed_args = []
        for arg in args:
            if isinstance(arg, torch.Tensor) and arg.dtype == torch.float:
                processed_args.append(arg.to(dtype))
            else:
                processed_args.append(arg)
        args = tuple(processed_args)

        for key, value in kwargs.items():
            if isinstance(value, torch.Tensor) and value.dtype == torch.float:
                kwargs[key] = value.to(dtype)

        return original_pe_encoding(*args, **kwargs)

    prompt_encoder.pe_layer._pe_encoding = pe_encoding_dtype_wrapper  # noqa: SLF001

    original_prompt_encoder_forward = prompt_encoder.forward

    def prompt_encoder_dtype_wrapper(*args, **kwargs) -> torch.Tensor:
        outputs = original_prompt_encoder_forward(*args, **kwargs)
        return [output.to(dtype) for output in outputs]

    prompt_encoder.forward = prompt_encoder_dtype_wrapper


def _monkey_patch_predict_torch(predictor: SamPredictor | SamHQPredictor, dtype: torch.dtype) -> None:
    """Monkey patch the predict_torch method to use the correct dtype."""
    original_predict_torch = predictor.predict_torch

    def predict_torch_dtype_wrapper(*args, **kwargs) -> torch.Tensor:
        processed_args = []
        for arg in args:
            if isinstance(arg, torch.Tensor) and arg.dtype == torch.float:
                processed_args.append(arg.to(dtype))
            else:
                processed_args.append(arg)
        args = tuple(processed_args)

        for key, value in kwargs.items():
            if isinstance(value, torch.Tensor) and value.dtype == torch.float:
                kwargs[key] = value.to(dtype)
        outputs = original_predict_torch(*args, **kwargs)
        # SamPredictor.predict internally converts the outputs of predict_torch to numpy, which requires floats
        processed_outputs = []
        for output in outputs:
            if isinstance(output, torch.Tensor) and output.dtype == dtype:
                processed_outputs.append(output.to(torch.float))
            else:
                processed_outputs.append(output)
        return processed_outputs

    predictor.predict_torch = predict_torch_dtype_wrapper


def _monkey_patch_tinyvit_architecture(predictor: SamPredictor | SamHQPredictor, dtype: torch.dtype) -> None:
    """Change model.forward to use x.to_dtype before calling the original forward function.

    The 'ab' attribute of the Attention layers are not correctly set to the correct dtype.
    """
    if not isinstance(predictor.model.image_encoder, TinyViT):
        return
    original_forward = predictor.model.image_encoder.forward

    def forward_dtype_wrapper(
        self_tinyvit: TinyViT,
        x_input_to_tinyvit: torch.Tensor,
        *args_tinyvit,
        **kwargs_tinyvit,
    ) -> torch.Tensor:
        x_input_to_tinyvit = x_input_to_tinyvit.to(dtype)

        # The 'ab' attribute of the Attention layers are not correctly set to the correct dtype.
        if not self_tinyvit.training:
            for module in self_tinyvit.modules():
                if (
                    isinstance(module, Attention)
                    and hasattr(module, "ab")
                    and module.ab is not None
                    and module.ab.dtype != dtype
                ):
                    module.ab = module.ab.to(dtype)

        return original_forward(x_input_to_tinyvit, *args_tinyvit, **kwargs_tinyvit)

    predictor.model.image_encoder.forward = forward_dtype_wrapper.__get__(
        predictor.model.image_encoder, predictor.model.image_encoder.__class__
    )


def _monkey_patch_sam2_architecture(predictor: SAM2ImagePredictor, dtype: torch.dtype) -> None:
    """Monkey patch the SAM2 architecture to use the correct dtype.

    We adapt the transforms JIT script used and add a .to(dtype) at the end of the Sequential.
    """
    original_transforms = predictor._transforms.transforms  # noqa: SLF001

    class DtypeWrapper(nn.Module):
        """Wrapper to apply dtype conversion after a transformation in a JIT-compatible way."""

        def __init__(self, transform: nn.Module, target_dtype_tensor: torch.Tensor):  # noqa: ANN204
            super().__init__()
            self.transform = transform
            # Store a tensor with the target dtype. Using its dtype is JIT-compatible.
            self.register_buffer("target_dtype_tensor", target_dtype_tensor)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Forward pass with dtype conversion."""
            return self.transform(x).to(self.target_dtype_tensor.dtype)

    # Get device from the model to ensure tensors are on the same device.
    device = next(predictor.model.parameters()).device
    dummy_tensor = torch.tensor([], dtype=dtype, device=device)

    new_transforms_module = DtypeWrapper(original_transforms, dummy_tensor)
    scripted_new_transforms = torch.jit.script(new_transforms_module)

    predictor._transforms.transforms = scripted_new_transforms  # noqa: SLF001


def _monkey_patch_dtype(
    predictor: SamPredictor | SamHQPredictor | EfficientViTSamPredictor | SAM2ImagePredictor,
) -> None:
    """Monkey patch the predictor to use the correct dtype for the model.

    The input to the model has to be transformed to the correct dtype before being passed to the model.

    Args:
        predictor: The predictor to monkey patch.
    """
    if isinstance(predictor, SAM2ImagePredictor):
        dtype = predictor.model.sam_mask_decoder.iou_prediction_head.layers[0].weight.dtype
        _monkey_patch_sam2_architecture(predictor, dtype)
        _monkey_patch_prompt_encoder(predictor.model.sam_prompt_encoder, dtype)
        return

    dtype = predictor.model.mask_decoder.iou_prediction_head.layers[0].weight.dtype
    _monkey_patch_prompt_encoder(predictor.model.prompt_encoder, dtype)
    _monkey_patch_predict_torch(predictor, dtype)
    if isinstance(predictor, EfficientViTSamPredictor):
        _monkey_patch_prompt_encoder(predictor.model.prompt_encoder, dtype)
        _monkey_patch_transform(predictor, dtype)
        _monkey_patch_upsample_layers(predictor.model, dtype)
        _monkey_patch_predict_torch(predictor, dtype)
        return
    _monkey_patch_preprocess(predictor, dtype)
    _monkey_patch_tinyvit_architecture(predictor, dtype)


def is_sam_model(model: Any) -> bool:  # noqa: ANN401
    """Check if the model is a SAM model."""
    return hasattr(model, "set_image")
