# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Load SAM models."""

from logging import getLogger
from pathlib import Path
from typing import Literal

from getiprompt.components.sam import BaseSAMPredictor, OpenVINOSAMPredictor, PyTorchSAMPredictor
from getiprompt.utils.constants import MODEL_MAP, SAMModelName
from getiprompt.utils.optimization import optimize_model
from getiprompt.utils.utils import precision_to_torch_dtype

logger = getLogger("Geti Prompt")


def load_sam_model(
    sam: SAMModelName,
    device: str = "cuda",
    precision: str = "bf16",
    compile_models: bool = False,
    backend: Literal["pytorch", "openvino"] = "pytorch",
    model_path: Path | None = None,
) -> BaseSAMPredictor:
    """Load and return a SAM predictor with specified backend.

    This function provides a unified interface for loading SAM models with
    different backends (PyTorch, OpenVINO, etc.). The backend parameter
    determines which implementation to use.

    Args:
        sam: The SAM model architecture to load (e.g., SAM_HQ_TINY, SAM2_BASE)
        device: Device to run inference on:
            - PyTorch backend: "cuda", "cpu"
            - OpenVINO backend: "CPU", "GPU", "AUTO"
        precision: Model precision for PyTorch backend ("bf16", "fp32", "fp16")
            Ignored for OpenVINO backend (precision is baked into IR)
        compile_models: Whether to compile model (PyTorch only)
            Ignored for OpenVINO backend
        backend: Which backend to use:
            - "pytorch": Use PyTorch for inference (default)
            - "openvino": Use OpenVINO Runtime for inference
        model_path: Optional path to model weights:
            - PyTorch: Path to .pth checkpoint (auto-downloads if None)
            - OpenVINO: Path to .xml IR file (uses default path if None)

    Returns:
        BaseSAMPredictor: A predictor implementing the common interface

    Raises:
        ValueError: If the model type is invalid
        FileNotFoundError: If OpenVINO model path doesn't exist

    Examples:
        >>> # PyTorch backend with auto-download
        >>> predictor = load_sam_model(
        ...     SAMModelName.SAM_HQ_TINY,
        ...     device="cuda",
        ...     backend="pytorch"
        ... )

        >>> # PyTorch backend with custom checkpoint
        >>> predictor = load_sam_model(
        ...     SAMModelName.SAM_HQ_TINY,
        ...     backend="pytorch",
        ...     model_path=Path("custom_weights.pth")
        ... )

        >>> # OpenVINO backend
        >>> predictor = load_sam_model(
        ...     SAMModelName.SAM_HQ_TINY,
        ...     device="CPU",
        ...     backend="openvino",
        ...     model_path=Path("exported/sam_hq_tiny.xml")
        ... )
    """
    if sam not in MODEL_MAP:
        msg = f"Invalid model type: {sam}"
        raise ValueError(msg)

    if backend == "pytorch":
        # Create PyTorch predictor
        predictor = PyTorchSAMPredictor(
            sam_model_name=sam,
            device=device,
            model_path=model_path,
        )

        # Apply PyTorch-specific optimizations
        predictor._predictor = optimize_model(
            model=predictor._predictor,
            device=device,
            precision=precision_to_torch_dtype(precision),
            compile_models=compile_models,
        )
        return predictor

    if backend == "openvino":
        # Create OpenVINO predictor (no additional optimization needed)
        return OpenVINOSAMPredictor(
            sam_model_name=sam,
            device=device,
            model_path=model_path,
        )

    msg = f"Unknown backend: {backend}. Must be 'pytorch' or 'openvino'"
    raise ValueError(msg)
