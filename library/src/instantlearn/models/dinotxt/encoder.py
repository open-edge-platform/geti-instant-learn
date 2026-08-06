# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""DINOTxt model."""

from __future__ import annotations

import logging
from pathlib import Path

import torch
import torchvision
from torch import nn
from torchvision import tv_tensors

from instantlearn.utils.constants import (
    IMAGENET_TEMPLATES,
    DINOv3BackboneSize,
)

logger = logging.getLogger(__name__)

_DINOV3_HUB_REPO = "facebookresearch/dinov3"
_DINOV3_HUB_ENTRYPOINT = "dinov3_vitl16_dinotxt_tet1280d20h24l"

_ACCESS_URL = "https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/"


class DinoTextEncoder(nn.Module):
    """DINOv3 text encoder for zero-shot classification.

    DINOv3 DinoTxt weights are gated by Meta and require requesting access at
    `<https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/>`_.
    Once accepted, Meta sends download URLs via email. Download the backbone
    and text-head ``.pth`` files, place them in a directory, and pass that
    path as ``weights_location``.

    Unlike Matcher (which uses timm/HuggingFace for publicly hosted backbones),
    the DinoTxt text-head weights are only available from Meta's gated servers.

    Args:
        image_size: The size of the input image.
        precision: The precision to use for the model.
        device: The device to use for the model.
        backbone_size: The size of the backbone model (only "large" is supported).
        weights_location: Path to directory containing the downloaded ``.pth``
            weight files, or a direct URL obtained from Meta after access approval.
        mean: The mean to use for image normalization.
        std: The standard deviation to use for image normalization.

    Examples:
        >>> import torch
        >>> from torchvision import tv_tensors
        >>> from instantlearn.models.dinotxt import DinoTextEncoder
        >>> encoder = DinoTextEncoder(device="cpu", weights_location="~/data/dinov3_weights")
        >>> category_mapping = {0: "cat", 1: "dog"}
        >>> text_embedding = encoder.encode_text(category_mapping)
        >>> image_embedding = encoder.encode_image([tv_tensors.Image(torch.randn(224, 224, 3))])
    """

    def __init__(
        self,
        image_size: tuple[int, int] | int | None = (512, 512),
        precision: torch.dtype = torch.bfloat16,
        device: str = "cuda",
        backbone_size: DINOv3BackboneSize = DINOv3BackboneSize.LARGE,  # noqa: ARG002
        weights_location: str | Path | None = None,
        mean: tuple[float] = (123.675, 116.28, 103.53),
        std: tuple[float] = (58.395, 57.12, 57.375),
    ) -> None:
        """Initialize the DinoTextEncoder."""
        super().__init__()

        self.device = device
        self.precision = precision
        self.model, self.tokenizer = self._load_model(
            device=device,
            weights_location=weights_location,
        )

        # Handle image_size: if tuple, use first dimension; if int, use as is; if None, default to 512
        resize_size = (
            image_size[0] if isinstance(image_size, tuple) else (image_size if image_size is not None else 512)
        )

        self.transforms = torchvision.transforms.Compose([
            torchvision.transforms.v2.Resize(resize_size),
            torchvision.transforms.v2.Normalize(mean=mean, std=std),
            torchvision.transforms.v2.ToDtype(dtype=self.precision),
        ])

    @staticmethod
    def _load_model(
        device: str = "cuda",
        weights_location: str | Path | None = None,
    ) -> tuple[torch.nn.Module, object]:
        """Load DINOv3 DinoTxt model and tokenizer via ``torch.hub``.

        Args:
            device: The device to place the model on.
            weights_location: Directory containing pre-downloaded ``.pth``
                files, or a direct URL from Meta. When ``None``, attempts
                auto-download (works only if Meta has granted access).

        Returns:
            Tuple of (model, tokenizer).

        Raises:
            RuntimeError: If download fails (likely 403 — access not granted).
        """
        hub_kwargs: dict[str, object] = {"pretrained": True}

        if weights_location is not None:
            weights_str = str(Path(weights_location).expanduser())
            hub_kwargs["dinotxt_weights"] = weights_str
            hub_kwargs["backbone_weights"] = weights_str
            logger.info("Loading DINOv3 DinoTxt from local weights: %s", weights_str)
        else:
            logger.info("Loading DINOv3 DinoTxt (downloading weights if needed)...")

        try:
            model, tokenizer = torch.hub.load(
                _DINOV3_HUB_REPO,
                _DINOV3_HUB_ENTRYPOINT,
                **hub_kwargs,
            )
        except Exception as exc:
            if "403" in str(exc) or "Forbidden" in str(exc):
                msg = (
                    "DINOv3 DinoTxt weight download failed (HTTP 403 Forbidden).\n"
                    "Meta gates these weights behind an access request.\n\n"
                    "To fix:\n"
                    f"  1. Request access at {_ACCESS_URL}\n"
                    "  2. Once approved, download the backbone + text-head .pth files\n"
                    "  3. Place them in a directory and pass weights_location='path/to/dir'\n"
                )
                raise RuntimeError(msg) from exc
            raise

        return model.to(device), tokenizer

    @torch.no_grad()
    def encode_text(
        self,
        category_mapping: dict[int, str],
        prompt_template: list[str] = IMAGENET_TEMPLATES,
    ) -> torch.Tensor:
        """Encode the class text prompt to text embedding.

        Args:
            category_mapping: Dictionary mapping class IDs to category names (e.g., {0: "cat", 1: "dog"}).
            prompt_template: The prompt template to use for the model.

        Returns:
            The text embedding tensor with shape (embedding_dim, num_classes).

        Examples:
            >>> from instantlearn.models.dinotxt import DinoTextEncoder
            >>> encoder = DinoTextEncoder(device="cpu")
            >>> category_mapping = {0: "cat", 1: "dog"}
            >>> text_embedding = encoder.encode_text(category_mapping)
            >>> text_embedding.shape[1] == len(category_mapping)
            True
        """
        zero_shot_weights = []
        # Sort by class_id to ensure consistent ordering
        for class_id in sorted(category_mapping.keys()):
            label_name = category_mapping[class_id]
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
            >>> from instantlearn.models.dinotxt import DinoTextEncoder
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
