# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from enum import Enum
from logging import getLogger

import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoImageProcessor, AutoModel

from getiprompt.models.model_optimizer import optimize_model
from getiprompt.utils.utils import precision_to_torch_dtype

logger = getLogger("Geti Prompt")


class DinoVersion(Enum):
    """The version of the DINO model."""

    V2 = "v2"
    V3 = "v3"


class DinoSize(Enum):
    """The size variants for DINO models."""

    # DinoV2 sizes
    SMALL = "small"
    BASE = "base"
    LARGE = "large"
    GIANT = "giant"

    # DinoV3 sizes
    SMALL_PLUS = "small_plus"
    HUGE = "huge"

    @classmethod
    def from_str(cls, size: str) -> "DinoSize":
        """Convert a string to a DinoSize."""
        return cls(size.lower())


# Model ID mappings
DINO_V2_MODEL_IDS = {
    "small": "facebook/dinov2-with-registers-small",
    "base": "facebook/dinov2-with-registers-base",
    "large": "facebook/dinov2-with-registers-large",
    "giant": "facebook/dinov2-with-registers-giant",
}

DINO_V3_MODEL_IDS = {
    "small": "facebook/dinov3-vits16-pretrain-lvd1689m",
    "small_plus": "facebook/dinov3-vits16plus-pretrain-lvd1689m",
    "base": "facebook/dinov3-vitb16-pretrain-lvd1689m",
    "large": "facebook/dinov3-vitl16-pretrain-lvd1689m",
    "huge": "facebook/dinov3-vith16plus-pretrain-lvd1689m",
}


class Dino(nn.Module):
    """Unified DINO model for extracting features from images.

    Supports both DinoV2 and DinoV3 models with automatic model ID mapping.

    Examples:
        >>> from getiprompt.models.dino import Dino
        >>> # DinoV2 model
        >>> model = Dino(version="v2", size="large")
        >>> model.forward(torch.randn(1, 3, 224, 224))
        torch.Size([1, 1369, 1024])

        >>> # DinoV3 model
        >>> model = Dino(version="v3", size="large")
        >>> model.forward(torch.randn(1, 3, 224, 224))
        torch.Size([1, 1369, 1024])
    """

    def __init__(
        self,
        version: DinoVersion | str = DinoVersion.V3,
        size: DinoSize | str = DinoSize.LARGE,
        device: str = "cuda",
        precision: str = "bf16",
        compile_models: bool = False,
        benchmark_inference_speed: bool = False,
        input_size: int = 518,
    ) -> None:
        """Initialize the DINO model.

        Args:
            version: The DINO version (v2 or v3).
            size: The size of the DINO model.
            device: The device to use for the model.
            precision: The precision to use for the model.
            compile_models: Whether to compile the model.
            benchmark_inference_speed: Whether to benchmark the inference speed.
            input_size: The size of the input image.
        """
        super().__init__()

        # Convert string inputs to enums
        if isinstance(version, str):
            version = DinoVersion(version.lower())
        if isinstance(size, str):
            size = DinoSize.from_str(size)

        self.version = version
        self.size = size
        self.input_size = input_size
        self.device = device

        # Validate size for version
        self._validate_size_for_version(version, size)

        logger.info(f"Loading DINO{version.value.upper()}-{size.value.upper()} model")

        model_id = self._get_model_id(version, size)
        self.model, self.processor = self._load_hf_model(model_id, input_size)
        self.model = self.model.to(device).eval()
        self.patch_size = self.model.config.patch_size
        self.feature_size = self.input_size // self.patch_size
        # Ignore CLS token and register tokens
        self.ignore_token_length = 1 + self.model.config.num_register_tokens

        self.precision = precision_to_torch_dtype(precision)

        self.model = optimize_model(
            model=self.model,
            precision=self.precision,
            device=device,
            compile_models=compile_models,
            benchmark_inference_speed=benchmark_inference_speed,
        ).eval()

    @staticmethod
    def _validate_size_for_version(version: DinoVersion, size: DinoSize) -> None:
        """Validate that the size is compatible with the version."""
        if version == DinoVersion.V2:
            valid_sizes = {DinoSize.SMALL, DinoSize.BASE, DinoSize.LARGE, DinoSize.GIANT}
            if size not in valid_sizes:
                valid_size_names = [s.value for s in valid_sizes]
                msg = f"Size {size.value} is not valid for DinoV2. Valid sizes: {valid_size_names}"
                raise ValueError(msg)
        elif version == DinoVersion.V3:
            valid_sizes = {DinoSize.SMALL, DinoSize.SMALL_PLUS, DinoSize.BASE, DinoSize.LARGE, DinoSize.HUGE}
            if size not in valid_sizes:
                valid_size_names = [s.value for s in valid_sizes]
                msg = f"Size {size.value} is not valid for DinoV3. Valid sizes: {valid_size_names}"
                raise ValueError(msg)

    @staticmethod
    def _get_model_id(version: DinoVersion, size: DinoSize) -> str:
        """Get the model ID based on version and size."""
        if version == DinoVersion.V2:
            return DINO_V2_MODEL_IDS[size.value]
        if version == DinoVersion.V3:
            return DINO_V3_MODEL_IDS[size.value]
        msg = f"Unsupported version: {version}"
        raise ValueError(msg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Embed images using DINO.

        Args:
            x: The input images.

        Returns:
            The normalized features.
        """
        inputs = self.processor(images=x, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        last_hidden_state = self.model(**inputs).last_hidden_state
        features = last_hidden_state[:, self.ignore_token_length :, :]  # Remove CLS token (and register tokens if used)
        return F.normalize(features, p=2, dim=-1)

    @staticmethod
    def _load_hf_model(model_id: str, input_size: int) -> tuple[nn.Module, AutoImageProcessor]:
        """Load DINOv3 model from HuggingFace with error handling.

        Meta requires huggingface users to access weights by first requesting access on the HuggingFace website.
        This function will raise an error if the user does not have access to the weights.

        Args:
            model_id: The model id of the model.
            input_size: The size of the input image.

        Returns:
            The model and processor.
        """
        err_msg = (
            "User does not have access to the weights of the DinoV3 model.\n"
            "Please follow these steps:\n"
            f"1. Request access on the HuggingFace website: https://huggingface.co/{model_id}\n"
            "2. Set your HuggingFace credentials using one of these methods:\n"
            "   - Run: hf auth login\n"
            "   - Set environment variable: export HUGGINGFACE_HUB_TOKEN=your_token\n"
        )
        try:
            model = AutoModel.from_pretrained(model_id)
            processor = AutoImageProcessor.from_pretrained(
                model_id,
                size={"height": input_size, "width": input_size},
                do_center_crop=False,
                use_fast=True,  # uses Rust based image processor
            )
        except OSError as e:
            # Check if this is specifically a HuggingFace gated repo access error
            if "gated repo" in str(e).lower():
                raise ValueError(err_msg) from None
            raise
        return model, processor
