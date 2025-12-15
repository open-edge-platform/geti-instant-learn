# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""SAM Predictor factory function and wrapper for multi-backend support."""

from logging import getLogger
from pathlib import Path

import torch

from getiprompt.components.sam.openvino import OpenVINOSAMPredictor
from getiprompt.components.sam.pytorch import PyTorchSAMPredictor
from getiprompt.utils.constants import MODEL_MAP, Backend, SAMModelName
from getiprompt.utils.optimization import optimize_model
from getiprompt.utils.utils import precision_to_torch_dtype

logger = getLogger("Geti Prompt")


def load_sam_model(
    sam: SAMModelName,
    device: str = "cuda",
    precision: str = "bf16",
    compile_models: bool = False,
    backend: Backend = Backend.PYTORCH,
    model_path: Path | None = None,
) -> PyTorchSAMPredictor | OpenVINOSAMPredictor:
    """Load and return a SAM predictor with specified backend.

    This function provides a unified interface for loading SAM models with
    different backends (PyTorch, OpenVINO). The backend parameter determines
    which implementation to use.

    Args:
        sam: The SAM model architecture to load (e.g., SAM_HQ_TINY, SAM2_BASE)
        device: Device to run inference on:
            - PyTorch backend: "cuda", "cpu"
            - OpenVINO backend: "CPU", "GPU", "AUTO"
        precision: Model precision for PyTorch backend ("bf16", "fp32", "fp16").
            Ignored for OpenVINO backend (precision is baked into IR).
        compile_models: Whether to compile model (PyTorch only).
            Ignored for OpenVINO backend.
        backend: Which backend to use:
            - Backend.PYTORCH: Use PyTorch for inference (default)
            - Backend.OPENVINO: Use OpenVINO Runtime for inference
        model_path: Optional path to model weights:
            - PyTorch: Path to .pth checkpoint (auto-downloads if None)
            - OpenVINO: Path to .xml IR file (required)

    Returns:
        A SAM predictor instance (PyTorchSAMPredictor or OpenVINOSAMPredictor).

    Raises:
        ValueError: If the model type or backend is invalid.

    Examples:
        >>> # PyTorch backend with auto-download
        >>> predictor = load_sam_model(
        ...     SAMModelName.SAM_HQ_TINY,
        ...     device="cuda",
        ...     backend=Backend.PYTORCH
        ... )

        >>> # PyTorch backend with custom checkpoint
        >>> predictor = load_sam_model(
        ...     SAMModelName.SAM_HQ_TINY,
        ...     backend=Backend.PYTORCH,
        ...     model_path=Path("custom_weights.pth")
        ... )

        >>> # OpenVINO backend
        >>> predictor = load_sam_model(
        ...     SAMModelName.SAM_HQ_TINY,
        ...     device="CPU",
        ...     backend=Backend.OPENVINO,
        ...     model_path=Path("exported/sam_hq_tiny.xml")
        ... )
    """
    if sam not in MODEL_MAP:
        msg = f"Invalid model type: {sam}"
        raise ValueError(msg)

    if backend == Backend.PYTORCH:
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

    if backend == Backend.OPENVINO:
        if model_path is None:
            msg = (
                "model_path is required for OpenVINO backend. "
                "Please export a PyTorch model first:\n"
                "  predictor = load_sam_model(sam=..., backend=Backend.PYTORCH)\n"
                "  ov_path = predictor.export(Path('./exported'))\n"
                "  ov_predictor = load_sam_model(backend=Backend.OPENVINO, model_path=ov_path)"
            )
            raise ValueError(msg)

        return OpenVINOSAMPredictor(
            sam_model_name=sam,
            device=device,
            model_path=model_path,
            precision=precision,
        )

    msg = f"Unknown backend: {backend}. Must be Backend.PYTORCH or Backend.OPENVINO"
    raise ValueError(msg)


class SAMPredictor:
    """Unified SAM predictor wrapper supporting multiple backends.

    This class provides a unified interface for SAM prediction using different
    backends (PyTorch, OpenVINO). It wraps the underlying predictor implementation
    and exposes common properties and methods.

    Examples:
        >>> from getiprompt.components.sam import SAMPredictor
        >>> from getiprompt.utils.constants import SAMModelName
        >>> import torch
        >>>
        >>> # Create predictor with PyTorch backend
        >>> predictor = SAMPredictor(
        ...     sam_model_name=SAMModelName.SAM_HQ_TINY,
        ...     backend=Backend.PYTORCH
        ... )
        >>>
        >>> # Export to OpenVINO
        >>> ov_path = predictor.export(Path("./exported"))
        >>>
        >>> # Load with OpenVINO backend
        >>> ov_predictor = SAMPredictor(
        ...     sam_model_name=SAMModelName.SAM_HQ_TINY,
        ...     backend=Backend.OPENVINO,
        ...     model_path=ov_path
        ... )
    """

    def __init__(
        self,
        sam_model_name: SAMModelName,
        backend: Backend = Backend.PYTORCH,
        device: str = "cuda",
        precision: str = "bf16",
        compile_models: bool = False,
        model_path: Path | None = None,
    ) -> None:
        """Initialize the SAM predictor.

        Args:
            sam_model_name: The SAM model architecture (e.g., SAM_HQ_TINY, SAM2_BASE)
            backend: Which backend to use: Backend.PYTORCH or Backend.OPENVINO.
            device: Device to run inference on. For PyTorch: "cuda" or "cpu".
                For OpenVINO: "CPU", "GPU", or "AUTO".
            precision: Precision for PyTorch backend: "fp32", "fp16", or "bf16".
                Ignored for OpenVINO.
            compile_models: Whether to compile model with torch.compile.
                Ignored for OpenVINO.
            model_path: Path to model weights (required for OpenVINO backend).
        """
        self.backend = backend
        self.device = device
        self._predictor: PyTorchSAMPredictor | OpenVINOSAMPredictor = load_sam_model(
            sam=sam_model_name,
            device=device,
            precision=precision,
            compile_models=compile_models,
            backend=backend,
            model_path=model_path,
        )

    def set_image(
        self,
        image: torch.Tensor,
        original_size: tuple[int, int],
    ) -> None:
        """Set the image for prediction and compute embeddings if needed.

        Args:
            image: Preprocessed image tensor of shape (C, H, W)
            original_size: Original image size (H, W) before preprocessing
        """
        return self._predictor.set_image(image, original_size)

    def predict(
        self,
        point_coords: torch.Tensor | None = None,
        point_labels: torch.Tensor | None = None,
        boxes: torch.Tensor | None = None,
        mask_input: torch.Tensor | None = None,
        multimask_output: bool = True,
        return_logits: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Predict segmentation masks given prompts.

        Args:
            point_coords: Point coordinates [B, N, 2] in (x, y) format
            point_labels: Point labels [B, N] (1=foreground, 0=background, -1=padding)
            boxes: Box prompts [B, 4] in (x1, y1, x2, y2) format
            mask_input: Low-res mask input [B, 1, 256, 256]
            multimask_output: Whether to return multiple masks
            return_logits: Whether to return logits instead of binary masks

        Returns:
            Tuple of (masks, iou_predictions, low_res_logits)
        """
        return self._predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            boxes=boxes,
            mask_input=mask_input,
            multimask_output=multimask_output,
            return_logits=return_logits,
        )

    def export(self, output_path: Path, backend: str | Backend = Backend.ONNX) -> Path:
        """Export the predictor to the specified format.

        Only available for PyTorch backend.

        Args:
            output_path: Directory to save exported model.
            backend: Backend format to export to. Can be a Backend enum
                (e.g., Backend.ONNX, Backend.OPENVINO) or a string
                (e.g., "onnx", "openvino").

        Returns:
            Path to the exported model file.

        Raises:
            NotImplementedError: If export is not supported for the backend.
        """
        if isinstance(backend, str):
            backend = Backend(backend.lower())
        if not hasattr(self._predictor, "export"):
            msg = f"Export is not supported for backend '{self.backend}'."
            raise NotImplementedError(msg)
        self._predictor = self._predictor.to(device="cpu")  # Force to CPU for export
        return self._predictor.export(output_path, backend=backend)
