# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Wrapper class that uses a DINO model to encode images into normalized patch embeddings."""

from dataclasses import dataclass
from enum import Enum
from logging import getLogger

import torch
from torch import nn
from torch.nn import functional
from torchvision import tv_tensors
from transformers import AutoImageProcessor, AutoModel

from getiprompt.utils import precision_to_torch_dtype
from getiprompt.utils.optimization import optimize_model

logger = getLogger("Geti Prompt")


@dataclass(frozen=True)
class ModelConfig:
    """Configuration for a HuggingFace model.

    Attributes:
        model_id: HuggingFace model identifier
        revision: Specific revision (commit SHA, tag, or branch) to pin
    """
    model_id: str
    revision: str | None = None


class ImageEncoderModel(Enum):
    """Available image encoder models from HuggingFace."""

    DINOV2_SMALL = ModelConfig(
        model_id="facebook/dinov2-with-registers-small",
        revision="0d9846e56b43a21fa46d7f3f5070f0506a5795a9",
    )
    DINOV2_BASE = ModelConfig(
        model_id="facebook/dinov2-with-registers-base",
        revision="a1d738ccfa7ae170945f210395d99dde8adb1805",
    )
    DINOV2_LARGE = ModelConfig(
        model_id="facebook/dinov2-with-registers-large",
        revision="e4c89a4e05589de9b3e188688a303d0f3c04d0f3",
    )
    DINOV2_GIANT = ModelConfig(
        model_id="facebook/dinov2-with-registers-giant",
        revision="8d0d49f77fb8b5dd78842496ff14afe7dd4d85cb",
    )
    DINOV3_SMALL = ModelConfig(
        model_id="facebook/dinov3-vits16-pretrain-lvd1689m",
        revision="114c1379950215c8b35dfcd4e90a5c251dde0d32",
    )
    DINOV3_SMALL_PLUS = ModelConfig(
        model_id="facebook/dinov3-vits16plus-pretrain-lvd1689m",
        revision="c93d816fc9e567563bc068f01475bec89cc634a6",
    )
    DINOV3_BASE = ModelConfig(
        model_id="facebook/dinov3-vitb16-pretrain-lvd1689m",
        revision="5931719e67bbdb9737e363e781fb0c67687896bc",
    )
    DINOV3_LARGE = ModelConfig(
        model_id="facebook/dinov3-vitl16-pretrain-lvd1689m",
        revision="ea8dc2863c51be0a264bab82070e3e8836b02d51",
    )
    DINOV3_HUGE = ModelConfig(
        model_id="facebook/dinov3-vith16plus-pretrain-lvd1689m",
        revision="c807c9eeea853df70aec4069e6f56b28ddc82acc",
    )


class ImageEncoder(nn.Module):
    """This encoder uses a model from HuggingFace to encode the images.

    Examples:
        >>> from getiprompt.components.encoders import ImageEncoder
        >>> from torchvision import tv_tensors
        >>> import torch

        >>> # Create a sample image
        >>> sample_image = torch.zeros((3, 518, 518))
        >>> encoder = ImageEncoder(model_id="dinov2_large")
        >>> features = encoder(images=[sample_image])
        >>> features.shape
        torch.Size([1369, 1024])
    """

    def __init__(
        self,
        model_id: str = "dinov3_large",
        device: str = "cuda",
        precision: str = "bf16",
        compile_models: bool = False,
        input_size: int = 518,
    ) -> None:
        """Initialize the encoder.

        Args:
            model_id: The model id to use.
            device: The device to use.
            precision: The precision to use.
            compile_models: Whether to compile the models.
            input_size: The input size to use.

        Raises:
            ValueError: If the model ID is invalid.
        """
        super().__init__()

        try:
            model_enum = ImageEncoderModel[model_id.upper()]
            config = model_enum.value
        except KeyError:
            valid_ids = [e.name.lower() for e in ImageEncoderModel]
            msg = f"Invalid model ID: {model_id}. Valid model IDs: {valid_ids}"
            raise ValueError(msg) from None

        self.model_id = model_id
        self.input_size = input_size
        self.device = device

        msg = f"Loading DINO model {model_id} with revision {config.revision}."
        logger.info(msg)
        self.model, self.processor = self._load_hf_model(config, input_size)
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
        ).eval()

    @staticmethod
    def _load_hf_model(config: ModelConfig, input_size: int) -> tuple[nn.Module, AutoImageProcessor]:
        """Load DINO model from HuggingFace with error handling.

        Meta requires huggingface users to access weights by first requesting access on the HuggingFace website.
        This function will raise an error if the user does not have access to the weights.

        Args:
            config: Model configuration with ID and optional revision
            input_size: The size of the input image.

        Returns:
            The model and processor.

        Raises:
            ValueError: If the user does not have access to the weights of the model.
            OSError: If the model is not found.
        """
        err_msg = (
            "User does not have access to the weights of the DinoV3 model.\n"
            "Please follow these steps:\n"
            f"1. Request access on the HuggingFace website: https://huggingface.co/{config.model_id}\n"
            "2. Set your HuggingFace credentials using one of these methods:\n"
            "   - Run: hf auth login\n"
            "   - Set environment variable: export HUGGINGFACE_HUB_TOKEN=your_token\n"
        )
        try:
            model = AutoModel.from_pretrained(config.model_id, revision=config.revision)
            processor = AutoImageProcessor.from_pretrained(
                config.model_id,
                revision=config.revision,
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

    @torch.inference_mode()
    def forward(self, images: list[tv_tensors.Image]) -> torch.Tensor:
        """Encode images into normalized patch embeddings.

        Args:
            images(list[tv_tensors.Image]): A list of images.

        Returns:
            torch.Tensor: Normalized patch-grid feature tensor of shape
                (batch_size, num_patches, embedding_dim).
        """
        inputs = self.processor(images=images, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        last_hidden_state = self.model(**inputs).last_hidden_state
        features = last_hidden_state[:, self.ignore_token_length :, :]  # Remove CLS token (and register tokens if used)
        return functional.normalize(features, p=2, dim=-1)
