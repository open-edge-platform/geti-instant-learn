# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""DINOTxt model."""

from pathlib import Path

import torch
import torchvision
from torch import nn
from torchvision import tv_tensors

from getiprompt.types import Priors
from getiprompt.utils.constants import (
    DINOV3_BACKBONE_MAP,
    DINOV3_TXT_HEAD_FILENAME,
    DINOV3_WEIGHTS_PATH,
    IMAGENET_TEMPLATES,
    DINOv3BackboneSize,
)


class DinoTextEncoder(nn.Module):
    """DINOv3 text encoder for zero-shot classification.

    Usage of DINOv3 model is subject to Meta's terms of use.

    Please download the DINOv3 backbone and text encoder weights from
    Meta's official website (https://ai.meta.com/resources/models-and-libraries/dinov3-downloads)
    and save them to ~/data/dinov3_weights (or specify a different location with the weights_location argument).

    Args:
        image_size: The size of the input image.
        weights_location: The location of the DINOv3 weights.
        backbone_size: The size of the backbone model.
        precision: The precision to use for the model.
        device: The device to use for the model.
        mean: The mean to use for image normalization.
        std: The standard deviation to use for image normalization.

    Examples:
        >>> import torch
        >>> from torchvision import tv_tensors
        >>> from getiprompt.types import Priors
        >>> from getiprompt.models.dinotxt import DinoTextEncoder
        >>> encoder = DinoTextEncoder(device="cuda", weights_location="~/data/dinov3_weights")
        >>> text_embedding = encoder.encode_text(Priors(text={0: "cat", 1: "dog"}))
        >>> image_embedding = encoder.encode_image([tv_tensors.Image(torch.randn(224, 224, 3))])
    """

    def __init__(
        self,
        image_size: int = 512,
        precision: torch.dtype = torch.bfloat16,
        device: str = "cuda",
        weights_location: str | Path = DINOV3_WEIGHTS_PATH,
        backbone_size: DINOv3BackboneSize = DINOv3BackboneSize.LARGE,
        mean: tuple[float] = (123.675, 116.28, 103.53),
        std: tuple[float] = (58.395, 57.12, 57.375),
    ) -> None:
        """Initialize the DinoTextEncoder."""
        super().__init__()

        # Load model and tokenizer from local weights
        self.device = device
        self.precision = precision
        self.model, self.tokenizer = DinoTextEncoder._load_model(weights_location, backbone_size.value, device)

        self.transforms = torchvision.transforms.Compose([
            torchvision.transforms.v2.Resize(image_size),
            torchvision.transforms.v2.Normalize(mean=mean, std=std),
            torchvision.transforms.v2.ToDtype(dtype=self.precision),
        ])

    @staticmethod
    def _load_model(
        weights_location: str | Path,
        backbone_size: str = "large",
        device: str = "cuda",
    ) -> tuple[torch.nn.Module, object]:
        """Load DINOv3 model and tokenizer from local weights.

        Args:
            weights_location: Path to the DINOv3 weights location, containing the txt head and backbone weights.
            backbone_size: Size of the backbone model ("small", "small-plus", "base", "large", "huge").
            device: The device to use for the model.

        Returns:
            Tuple of (model, tokenizer).

        Raises:
            FileNotFoundError: If weights files don't exist.
            RuntimeError: If weights loading fails.
            ValueError: If the backbone size is invalid.
        """
        weights_location = Path(weights_location) if isinstance(weights_location, str) else weights_location
        weights_location = weights_location.expanduser()
        txt_head_path = weights_location / DINOV3_TXT_HEAD_FILENAME
        backbone_filename = DINOV3_BACKBONE_MAP.get(backbone_size)
        if not backbone_filename:
            valid_sizes = list(DINOV3_BACKBONE_MAP.keys())
            msg = f"Invalid backbone size: {backbone_size}. Must be one of: {valid_sizes}"
            raise ValueError(msg)
        backbone_path = weights_location / backbone_filename

        # Check if txt head weights exist
        if not txt_head_path.exists():
            msg = (
                f"DINOv3 txt head weights not found at {txt_head_path}.\n"
                f"Please download the DINOv3 weights from Meta's official website:\n"
                f"https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/\n"
                f"Save the weights file 'dinov3_vitl16_dinotxt_vision_head_and_text_encoder-a442d8f5.pth' "
                f"in the directory: {txt_head_path.parent}\n"
                f"Then rerun geti-prompt."
            )
            raise FileNotFoundError(msg)

        # Check if backbone weights exist
        if not backbone_path.exists():
            msg = (
                f"DINOv3 backbone weights not found at {backbone_path}.\n"
                f"Please download the DINOv3 backbone weights from Meta's official website:\n"
                f"https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/\n"
                f"Save the weights file '{backbone_filename}' "
                f"in the directory: {backbone_path.parent}\n"
                f"Then rerun geti-prompt."
            )
            raise FileNotFoundError(msg)

        try:
            # Initialize model architecture using torch.hub.load with both weights
            model, tokenizer = torch.hub.load(
                "facebookresearch/dinov3",
                "dinov3_vitl16_dinotxt_tet1280d20h24l",
                pretrained=False,  # weights are loaded from local weights
                weights=str(txt_head_path),
                backbone_weights=str(backbone_path),
            )
            model = model.to(device)

        except Exception as e:
            msg = (
                f"Failed to load DINOv3 weights from {txt_head_path} and {backbone_path}.\n"
                f"Error: {e!s}\n"
                f"Please ensure the weights files are valid and try again."
            )
            raise RuntimeError(msg) from e

        return model, tokenizer

    @torch.no_grad()
    def encode_text(
        self,
        reference_prior: Priors,
        prompt_template: list[str] = IMAGENET_TEMPLATES,
    ) -> torch.Tensor:
        """Encode the class text prompt to text embedding.

        Args:
            reference_prior: The prior to encode.
            prompt_template: The prompt template to use for the model.

        Returns:
            The text embedding.

        Examples:
            >>> from getiprompt.models.dinotxt import DinoTextEncoder
            >>> from getiprompt.types import Priors
            >>> encoder = DinoTextEncoder()
            >>> prior = Priors(text={0: "cat", 1: "dog"})
            >>> text_embedding = encoder.encode_text(prior)
            >>> text_embedding.shape
            torch.Size([2048, 2])
        """
        zero_shot_weights = []
        for label_name in reference_prior.text.values():
            texts = [template.format(label_name) for template in prompt_template]
            texts = self.tokenizer.tokenize(texts)
            texts = texts.to(self.device)
            with torch.autocast(device_type=self.device, dtype=self.precision):
                class_embeddings = self.model.encode_text(texts)
                class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
                class_embedding = class_embeddings.mean(dim=0)
                class_embedding /= class_embedding.norm()
            zero_shot_weights.append(class_embedding)
        return torch.stack(zero_shot_weights, dim=1)

    @torch.no_grad()
    def encode_image(
        self,
        target_images: list[tv_tensors.Image],
    ) -> torch.Tensor:
        """Encode the reference images to image embedding.

        Args:
            target_images: A list of target images.

        Returns:
            The image embedding.

        Examples:
            >>> from getiprompt.models.dinotxt import DinoTextEncoder
            >>> from torchvision import tv_tensors
            >>> encoder = DinoTextEncoder()
            >>> image = tv_tensors.Image(torch.randn(224, 224, 3))
            >>> image_embedding = encoder.encode_image([image])
        """
        images = [self.transforms(image.to(dtype=self.precision)) for image in target_images]
        images = torch.stack(images, dim=0)
        images = images.to(self.device)
        with torch.autocast(device_type=self.device, dtype=self.precision):
            image_features = self.model.encode_image(images)
            image_features /= image_features.norm(dim=-1, keepdim=True)
        return image_features.to(self.precision)
