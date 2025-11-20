# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Factory function for creating image encoders with different backends."""

from logging import getLogger
from pathlib import Path

from torch import nn

logger = getLogger("Geti Prompt")


def load_image_encoder(
    model_id: str = "dinov3_large",
    device: str = "cuda",
    backend: str = "pytorch",
    model_path: Path | None = None,
    precision: str = "bf16",
    compile_models: bool = False,
    input_size: int = 518,
) -> nn.Module:
    """Load an image encoder with specified backend.

    This factory function creates an image encoder using either PyTorch
    or OpenVINO backend. The PyTorch backend is used for training and
    flexibility, while OpenVINO provides optimized inference.

    Args:
        model_id: The DINO model variant to use. Options:
            - "dinov2_small", "dinov2_base", "dinov2_large", "dinov2_giant"
            - "dinov3_small", "dinov3_small_plus", "dinov3_base", "dinov3_large", "dinov3_huge"
        device: Device to run inference on. For PyTorch: "cuda" or "cpu".
            For OpenVINO: "CPU", "GPU", or "AUTO".
        backend: Which backend to use: "pytorch" or "openvino".
        model_path: Path to model weights/IR files.
            - PyTorch: Path to .pth checkpoint (optional, will download if None)
            - OpenVINO: Path to exported IR directory (required)
        precision: Precision for PyTorch backend: "fp32", "fp16", or "bf16".
            Ignored for OpenVINO.
        compile_models: Whether to compile PyTorch model with torch.compile.
            Ignored for OpenVINO.
        input_size: Input image size (height and width).

    Returns:
        Image encoder instance (PyTorchImageEncoder or OpenVINOImageEncoder).

    Raises:
        ValueError: If backend is not "pytorch" or "openvino".
        FileNotFoundError: If OpenVINO model_path doesn't exist.

    Examples:
        >>> # PyTorch backend (training/flexibility)
        >>> encoder = load_image_encoder(
        ...     model_id="dinov2_large",
        ...     device="cuda",
        ...     backend="pytorch"
        ... )
        >>>
        >>> # Export to OpenVINO
        >>> ov_path = encoder.export(Path("./exported/dinov2_large"))
        >>>
        >>> # OpenVINO backend (optimized inference)
        >>> ov_encoder = load_image_encoder(
        ...     model_id="dinov2_large",
        ...     device="CPU",
        ...     backend="openvino",
        ...     model_path=ov_path
        ... )
    """
    if backend == "pytorch":
        from .pytorch_image_encoder import PyTorchImageEncoder

        return PyTorchImageEncoder(
            model_id=model_id,
            device=device,
            precision=precision,
            compile_models=compile_models,
            input_size=input_size,
        )
    if backend == "openvino":
        from .openvino_image_encoder import OpenVINOImageEncoder

        if model_path is None:
            msg = (
                "model_path is required for OpenVINO backend. "
                "Please export a PyTorch model first:\n"
                "  encoder = load_image_encoder(model_id='...', backend='pytorch')\n"
                "  ov_path = encoder.export(Path('./exported'))\n"
                "  ov_encoder = load_image_encoder(backend='openvino', model_path=ov_path)"
            )
            raise ValueError(msg)

        return OpenVINOImageEncoder(
            model_path=model_path,
            model_id=model_id,
            device=device,
            input_size=input_size,
        )
    msg = f"Invalid backend: {backend}. Must be 'pytorch' or 'openvino'."
    raise ValueError(msg)
