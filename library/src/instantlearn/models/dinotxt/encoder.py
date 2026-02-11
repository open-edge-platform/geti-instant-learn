# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""DINOTxt model."""

import os
from pathlib import Path

import torch
import torchvision
from torch import nn
from torchvision import tv_tensors

from instantlearn.utils.constants import (
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
        >>> from instantlearn.models.foundation.dinotxt import DinoTextEncoder
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
    def _find_weights_in_cache(filename: str) -> Path | None:
        """Search for weight files in common cache directories.

        Uses official PyTorch and Hugging Face APIs to get cache directories,
        then searches recursively for the specified filename.

        Searches in:
        - PyTorch Hub cache (via torch.hub.get_dir())
        - Hugging Face cache (via HF_HOME or default location)
        - General cache directory (~/.cache)

        Args:
            filename: The name of the weight file to search for.

        Returns:
            Path to the found file, or None if not found.
        """
        cache_dirs = []

        # PyTorch hub cache - use official API
        try:
            torch_hub_dir = torch.hub.get_dir()
            cache_dirs.append(Path(torch_hub_dir))
        except Exception:
            # Fallback to manual construction if API fails
            torch_home = os.environ.get("TORCH_HOME")
            if torch_home:
                cache_dirs.append(Path(torch_home).expanduser() / "hub")
            else:
                cache_dirs.append(Path.home() / ".cache" / "torch" / "hub")

        # Hugging Face cache
        # Note: huggingface_hub doesn't have a direct get_dir() equivalent,
        # but we can check for HF_HOME or use default location
        hf_home = os.environ.get("HF_HOME")
        if hf_home:
            cache_dirs.append(Path(hf_home).expanduser())
        else:
            # Default Hugging Face cache location
            cache_dirs.append(Path.home() / ".cache" / "huggingface")

        # General cache directory
        cache_dirs.append(Path.home() / ".cache")

        # Search recursively in each cache directory
        for cache_dir in cache_dirs:
            if not cache_dir.exists():
                continue
            try:
                # Use rglob to search recursively
                for found_file in cache_dir.rglob(filename):
                    if found_file.is_file():
                        return found_file
            except (PermissionError, OSError):
                # Skip directories we can't access
                continue

        return None

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

        # Check if txt head weights exist, search cache if not found
        if not txt_head_path.exists():
            cached_txt_head = DinoTextEncoder._find_weights_in_cache(DINOV3_TXT_HEAD_FILENAME)
            if cached_txt_head:
                txt_head_path = cached_txt_head
            else:
                msg = (
                    f"DINOv3 txt head weights not found at {weights_location / DINOV3_TXT_HEAD_FILENAME}.\n"
                    f"Searched cache directories (~/.cache/torch/hub, ~/.cache/huggingface, ~/.cache) but not found.\n"
                    f"Please download the DINOv3 weights from Meta's official website:\n"
                    f"https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/\n"
                    f"Save the weights file '{DINOV3_TXT_HEAD_FILENAME}' "
                    f"in the directory: {weights_location}\n"
                    f"Then rerun instant-learn."
                )
                raise FileNotFoundError(msg)

        # Check if backbone weights exist, search cache if not found
        if not backbone_path.exists():
            cached_backbone = DinoTextEncoder._find_weights_in_cache(backbone_filename)
            if cached_backbone:
                backbone_path = cached_backbone
            else:
                msg = (
                    f"DINOv3 backbone weights not found at {weights_location / backbone_filename}.\n"
                    f"Searched cache directories (~/.cache/torch/hub, ~/.cache/huggingface, ~/.cache) but not found.\n"
                    f"Please download the DINOv3 backbone weights from Meta's official website:\n"
                    f"https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/\n"
                    f"Save the weights file '{backbone_filename}' "
                    f"in the directory: {weights_location}\n"
                    f"Then rerun instant-learn."
                )
                raise FileNotFoundError(msg)

        try:
            # Initialize model architecture using torch.hub.load with both weights
            model, tokenizer = torch.hub.load(
                "facebookresearch/dinov3",
                "dinov3_vitl16_dinotxt_tet1280d20h24l",
                dinotxt_weights=str(txt_head_path),
                backbone_weights=str(backbone_path),
            )  # nosec B614 (trusted model id is hardcoded)
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
            >>> from instantlearn.models.foundation.dinotxt import DinoTextEncoder
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
