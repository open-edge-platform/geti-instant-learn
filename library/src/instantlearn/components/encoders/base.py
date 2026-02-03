# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Base ImageEncoder class and factory function for creating image encoders."""

from logging import getLogger
from pathlib import Path

import torch
from torch import nn
from torchvision import tv_tensors

from instantlearn.components.encoders.huggingface import HuggingFaceImageEncoder
from instantlearn.components.encoders.timm import TimmImageEncoder
from instantlearn.utils.constants import Backend

logger = getLogger("Geti Instant Learn")


def load_image_encoder(
    model_id: str = "dinov3_large",
    device: str = "cuda",
    backend: Backend = Backend.TIMM,
    model_path: Path | None = None,
    precision: str = "bf16",
    compile_models: bool = False,
    input_size: int = 512,
) -> HuggingFaceImageEncoder | TimmImageEncoder:
    """Load an image encoder with specified backend.

    This factory function creates an image encoder using HuggingFace, TIMM,
    or OpenVINO backend. The HuggingFace/TIMM backends are used for training
    and flexibility, while OpenVINO provides optimized inference.

    Args:
        model_id: The DINO model variant to use. Options:
            - "dinov2_small", "dinov2_base", "dinov2_large", "dinov2_giant" (HuggingFace only)
            - "dinov3_small", "dinov3_small_plus", "dinov3_base", "dinov3_large", "dinov3_huge"
        device: Device to run inference on. For HuggingFace/TIMM: "cuda" or "cpu".
            For OpenVINO: "CPU", "GPU", or "AUTO".
        backend: Which backend to use: Backend.HUGGINGFACE, Backend.TIMM, or Backend.OPENVINO.
        model_path: Path to model weights/IR files.
            - HuggingFace/TIMM: Not used (will download from hub)
            - OpenVINO: Path to exported IR directory (required)
        precision: Precision for HuggingFace/TIMM backend: "fp32", "fp16", or "bf16".
            Ignored for OpenVINO.
        compile_models: Whether to compile model with torch.compile.
            Ignored for OpenVINO.
        input_size: Input image size (height and width).

    Returns:
        Image encoder instance (HuggingFaceImageEncoder, TimmImageEncoder, or OpenVINOImageEncoder).

    Raises:
        ValueError: If backend is not valid.
        FileNotFoundError: If OpenVINO model_path doesn't exist.

    Examples:
        >>> # HuggingFace backend (DINOv2 models)
        >>> encoder = load_image_encoder(
        ...     model_id="dinov2_large",
        ...     device="cuda",
        ...     backend=Backend.HUGGINGFACE
        ... )
        >>>
        >>> # TIMM backend (DINOv3 models)
        >>> encoder = load_image_encoder(
        ...     model_id="dinov3_large",
        ...     device="cuda",
        ...     backend=Backend.TIMM
        ... )
        >>>
        >>> # Export to OpenVINO
        >>> ov_path = encoder.export(Path("./exported/dinov2_large"))
        >>>
        >>> # OpenVINO backend (optimized inference)
        >>> ov_encoder = load_image_encoder(
        ...     model_id="dinov2_large",
        ...     device="CPU",
        ...     backend=Backend.OPENVINO,
        ...     model_path=ov_path
        ... )
    """
    if backend == Backend.HUGGINGFACE:
        return HuggingFaceImageEncoder(
            model_id=model_id,
            device=device,
            precision=precision,
            compile_models=compile_models,
            input_size=input_size,
        )
    if backend == Backend.TIMM:
        return TimmImageEncoder(
            model_id=model_id,
            device=device,
            precision=precision,
            compile_models=compile_models,
            input_size=input_size,
        )

    msg = f"Invalid backend: {backend}. Must be Backend.HUGGINGFACE, Backend.TIMM, or Backend.OPENVINO"
    raise ValueError(msg)


class ImageEncoder(nn.Module):
    """Unified image encoder wrapper supporting multiple backends.

    This class provides a unified interface for image encoding using different
    backends (HuggingFace, TIMM, OpenVINO). It wraps the underlying encoder
    implementation and exposes common properties and methods.

    Examples:
        >>> from instantlearn.components.encoders import ImageEncoder
        >>> import torch
        >>>
        >>> # Create encoder with TIMM backend
        >>> encoder = ImageEncoder(model_id="dinov3_large", backend=Backend.TIMM)
        >>> sample_image = torch.zeros((3, 518, 518))
        >>> features = encoder([sample_image])
        >>> features.shape
        torch.Size([1, 1369, 1024])
        >>>
        >>> # Export to OpenVINO
        >>> ov_path = encoder.export(Path("./exported"))
        >>>
        >>> # Load with OpenVINO backend
        >>> ov_encoder = ImageEncoder(
        ...     model_id="dinov2_large",
        ...     backend=Backend.OPENVINO,
        ...     model_path=ov_path
        ... )
    """

    def __init__(
        self,
        model_id: str = "dinov3_large",
        backend: Backend = Backend.TIMM,
        device: str = "cuda",
        precision: str = "bf16",
        compile_models: bool = False,
        input_size: int = 512,
        model_path: Path | None = None,
    ) -> None:
        """Initialize the image encoder.

        Args:
            model_id: The DINO model variant to use. Options:
                - "dinov2_small", "dinov2_base", "dinov2_large", "dinov2_giant" (HuggingFace)
                - "dinov3_small", "dinov3_small_plus", "dinov3_base", "dinov3_large", "dinov3_huge" (TIMM)
            backend: Which backend to use: Backend.HUGGINGFACE, Backend.TIMM, or Backend.OPENVINO.
            device: Device to run inference on. For HuggingFace/TIMM: "cuda" or "cpu".
                For OpenVINO: "CPU", "GPU", or "AUTO".
            precision: Precision for HuggingFace/TIMM backend: "fp32", "fp16", or "bf16".
                Ignored for OpenVINO.
            compile_models: Whether to compile model with torch.compile.
                Ignored for OpenVINO.
            input_size: Input image size (height and width).
            model_path: Path to exported model (required for OpenVINO backend).
        """
        super().__init__()
        self.backend = backend
        self._model: HuggingFaceImageEncoder | TimmImageEncoder = load_image_encoder(
            model_id=model_id,
            device=device,
            backend=backend,
            model_path=model_path,
            precision=precision,
            compile_models=compile_models,
            input_size=input_size,
        )

    @property
    def model_id(self) -> str:
        """The model ID of the encoder."""
        return self._model.model_id

    @property
    def input_size(self) -> int:
        """The input image size."""
        return self._model.input_size

    @property
    def patch_size(self) -> int:
        """The patch size of the encoder."""
        return self._model.patch_size

    @property
    def feature_size(self) -> int:
        """The feature grid size (input_size // patch_size)."""
        return self._model.feature_size

    @property
    def device(self) -> str:
        """The device the encoder is running on."""
        return self._model.device

    @torch.inference_mode()
    def forward(self, images: list[tv_tensors.Image]) -> torch.Tensor:
        """Encode images into normalized patch embeddings.

        Args:
            images: A list of images (as tv_tensors.Image or torch.Tensor).

        Returns:
            Normalized patch-grid feature tensor of shape
            (batch_size, num_patches, embedding_dim).
        """
        return self._model(images)
