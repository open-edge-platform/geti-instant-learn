# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Image encoder using TIMM models."""

from logging import getLogger

import timm
import torch
from torch import nn
from torch.nn import functional
from torchvision import tv_tensors
from torchvision.transforms.v2 import Compose, Normalize, Resize, ToDtype

from getiprompt.utils import precision_to_torch_dtype

logger = getLogger("Geti Prompt")

AVAILABLE_IMAGE_ENCODERS = {
    "dinov3_small": "timm/vit_small_patch16_dinov3.lvd1689m",
    "dinov3_small_plus": "timm/vit_small_plus_patch16_dinov3.lvd1689m",
    "dinov3_base": "timm/vit_base_patch16_dinov3.lvd1689m",
    "dinov3_large": "timm/vit_large_patch16_dinov3.lvd1689m",
    "dinov3_huge": "timm/vit_huge_plus_patch16_dinov3.lvd1689m",
}


class TimmImageEncoder(nn.Module):
    """This encoder uses a model from timm to encode the images.

    Examples:
        >>> from getiprompt.components.encoders.timm import TimmImageEncoder
        >>> from torchvision import tv_tensors
        >>> import torch

        >>> # Create a sample image
        >>> sample_image = torch.zeros((3, 518, 518))
        >>> encoder = TimmImageEncoder(model_id="dinov2_large")
        >>> features = encoder(images=[sample_image])
        >>> features.shape
        torch.Size([1, 1369, 1024])
    """

    def __init__(
        self,
        model_id: str = "dinov3_large",
        device: str = "cuda",
        precision: str = "bf16",
        compile_models: bool = False,
        input_size: int = 512,
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
        from getiprompt.utils.optimization import optimize_model

        super().__init__()

        if model_id not in AVAILABLE_IMAGE_ENCODERS:
            msg = f"Invalid model ID: {model_id}. Valid model IDs: {list(AVAILABLE_IMAGE_ENCODERS.keys())}"
            raise ValueError(msg)

        self.model_id = model_id
        self.input_size = input_size
        self.device = device

        msg = f"Loading DINO model {model_id}"
        logger.info(msg)
        self.precision = precision_to_torch_dtype(precision)
        self.model, self.processor = self._load_timm_model(
            AVAILABLE_IMAGE_ENCODERS[model_id],
            input_size,
            self.precision,
        )
        self.model = self.model.to(device).eval()
        self.patch_size = self.model.patch_embed.patch_size[0]
        self.feature_size = self.input_size // self.patch_size
        # Ignore CLS token and register tokens
        self.ignore_token_length = self.model.num_prefix_tokens
        self.model = optimize_model(
            model=self.model,
            precision=self.precision,
            device=device,
            compile_models=compile_models,
        ).eval()

    @staticmethod
    def _load_timm_model(model_id: str, input_size: int, precision: torch.dtype) -> tuple[nn.Module, Compose]:
        """Load DINO model from timm with error handling.

        Args:
            model_id: The model id of the model.
            input_size: The size of the input image.
            precision: The precision to use.

        Returns:
            The model and processor.
        """
        # Disable dynamic_img_size to avoid conditional position embedding code
        # that creates ONNX If nodes with dynamic rank outputs (not supported by OpenVINO CPU)
        model = timm.create_model(
            model_id,
            pretrained=True,
            num_classes=0,
            dynamic_img_size=False,
            img_size=input_size,
        )
        data_config = timm.data.resolve_model_data_config(model)
        data_config["input_size"] = (3, input_size, input_size)
        processor = Compose([
            ToDtype(dtype=precision, scale=True),
            Resize(size=(input_size, input_size)),
            Normalize(mean=data_config["mean"], std=data_config["std"]),
        ])
        return model, processor

    @torch.inference_mode()
    def forward(self, images: list[tv_tensors.Image]) -> torch.Tensor:
        """Encode images into patch embeddings.

        Args:
            images(list[tv_tensors.Image]): A list of images.

        Returns:
            torch.Tensor: patch-grid feature tensor of shape (batch_size, num_patches, embedding_dim).
        """
        images = torch.stack([self.processor(image.to(self.device)) for image in images])
        features = self.model.forward_features(images)  # (B, N, D)
        features = features[:, self.ignore_token_length :, :]  # ignore CLS and other tokens
        return functional.normalize(features, p=2, dim=-1)
